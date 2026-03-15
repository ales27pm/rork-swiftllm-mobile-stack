import Foundation

actor KVCacheManager {
    private let arena: KVCacheArena
    private let pageTable: KVPageTable
    private let prefixCache: PromptPrefixCache
    private var snapshots: [UUID: CacheSnapshot] = [:]
    private let slidingWindowSize: Int
    private let evictionThreshold: Int

    init(
        pageSize: Int = 128,
        layerCount: Int = 32,
        memoryBudgetMB: Int = 256,
        evictionPolicy: EvictionPolicy = .lruWithPrefixProtection,
        slidingWindowSize: Int = 1536,
        evictionThreshold: Int = 1800
    ) {
        self.arena = KVCacheArena(
            pageSize: pageSize,
            layerCount: layerCount,
            memoryBudgetMB: memoryBudgetMB,
            evictionPolicy: evictionPolicy
        )
        self.pageTable = KVPageTable()
        self.prefixCache = PromptPrefixCache()
        self.slidingWindowSize = slidingWindowSize
        self.evictionThreshold = evictionThreshold
    }

    func beginSequence(prefixHash: String? = nil) async -> UUID {
        await pageTable.createSequence(prefixHash: prefixHash)
    }

    func allocatePages(
        sequenceID: UUID,
        tokens: [Int],
        startPosition: Int
    ) async -> [KVPage] {
        let pageSize = 128
        var allocated: [KVPage] = []
        var offset = 0

        while offset < tokens.count {
            let chunkSize = min(pageSize, tokens.count - offset)
            let page = await arena.allocatePage(
                tokenStart: startPosition + offset,
                tokenCount: chunkSize,
                sequenceID: sequenceID
            )
            await pageTable.appendPage(page.id, to: sequenceID, tokenEnd: startPosition + offset + chunkSize)
            allocated.append(page)
            offset += chunkSize
        }

        return allocated
    }

    func touchSequence(_ sequenceID: UUID) async {
        let pageIDs = await pageTable.pageIDs(for: sequenceID)
        for id in pageIDs {
            await arena.touchPage(id)
        }
        await pageTable.touchSequence(sequenceID)
    }

    func forkSequence(_ sourceID: UUID) async -> UUID? {
        guard let forkedID = await pageTable.forkSequence(sourceID) else { return nil }
        let sourcePages = await pageTable.pageIDs(for: sourceID)
        for pageID in sourcePages {
            await arena.retainPage(pageID)
        }
        return forkedID
    }

    func slidingWindowEvict(
        sequenceID: UUID,
        systemTokenCount: Int,
        currentLength: Int
    ) async -> SlidingWindowResult {
        guard currentLength > evictionThreshold else {
            return SlidingWindowResult(evictedTokens: 0, pagesFreed: 0, newLength: currentLength)
        }

        let keepCount = slidingWindowSize
        let preservePrefix = systemTokenCount
        let prefixPages = (preservePrefix + 127) / 128
        let totalPages = await pageTable.pageIDs(for: sequenceID).count

        let keepPages = prefixPages + (keepCount + 127) / 128
        guard totalPages > keepPages else {
            return SlidingWindowResult(evictedTokens: 0, pagesFreed: 0, newLength: currentLength)
        }

        let removedIDs = await pageTable.truncateSequence(sequenceID, keepPages: keepPages)
        for id in removedIDs {
            await arena.releasePage(id)
        }

        let evictedTokens = removedIDs.count * 128
        let newLength = currentLength - evictedTokens

        return SlidingWindowResult(
            evictedTokens: evictedTokens,
            pagesFreed: removedIDs.count,
            newLength: max(newLength, preservePrefix + keepCount)
        )
    }

    func releaseSequence(_ sequenceID: UUID) async {
        let pageIDs = await pageTable.removeSequence(sequenceID)
        for id in pageIDs {
            await arena.releasePage(id)
        }
    }

    func createSnapshot(sequenceID: UUID, tokens: [Int]) async -> UUID {
        let snapshotID = UUID()
        let pageIDs = await pageTable.pageIDs(for: sequenceID)

        for id in pageIDs {
            await arena.retainPage(id)
        }

        snapshots[snapshotID] = CacheSnapshot(
            id: snapshotID,
            sequenceID: sequenceID,
            pageIDs: pageIDs,
            tokenSnapshot: tokens,
            createdAt: Date()
        )

        return snapshotID
    }

    func restoreSnapshot(_ snapshotID: UUID) async -> CacheSnapshot? {
        guard let snapshot = snapshots[snapshotID] else { return nil }
        return snapshot
    }

    func discardSnapshot(_ snapshotID: UUID) async {
        guard let snapshot = snapshots.removeValue(forKey: snapshotID) else { return }
        for id in snapshot.pageIDs {
            await arena.releasePage(id)
        }
    }

    func lookupPrefix(key: PromptPrefixKey) async -> CachedPrefix? {
        await prefixCache.lookup(key: key)
    }

    func storePrefix(_ prefix: CachedPrefix) async {
        await prefixCache.store(prefix: prefix)
    }

    func handleMemoryPressure() async -> MemoryPressureResponse {
        let stats = await arena.statistics
        let targetFree = stats.memoryBudgetBytes / 4

        let staleSequences = await pageTable.evictableSequences(olderThan: 30)
        var pagesFreed = 0

        for seqID in staleSequences {
            let pageIDs = await pageTable.removeSequence(seqID)
            await arena.freePages(pageIDs)
            pagesFreed += pageIDs.count
        }

        if await arena.estimatedMemoryBytes > stats.memoryBudgetBytes / 2 {
            let evicted = await arena.evictPagesForBudget(targetFreeBytes: targetFree)
            pagesFreed += evicted.count
        }

        let defragResult = await arena.defragment()

        return MemoryPressureResponse(
            sequencesEvicted: staleSequences.count,
            pagesFreed: pagesFreed + defragResult.pagesReclaimed,
            bytesReclaimed: Int64(pagesFreed * 128 * 32 * 2 * MemoryLayout<Float>.size) + defragResult.bytesReclaimed
        )
    }

    func statistics() async -> KVCacheStatistics {
        await arena.statistics
    }

    func activePageCount() async -> Int {
        await arena.activePageCount
    }

    func estimatedMemoryBytes() async -> Int64 {
        await arena.estimatedMemoryBytes
    }

    func reset() async {
        await arena.reset()
        await pageTable.reset()
        await prefixCache.clear()

        for (_, snapshot) in snapshots {
            for id in snapshot.pageIDs {
                await arena.releasePage(id)
            }
        }
        snapshots.removeAll()
    }
}

nonisolated struct CacheSnapshot: Sendable {
    let id: UUID
    let sequenceID: UUID
    let pageIDs: [UUID]
    let tokenSnapshot: [Int]
    let createdAt: Date
}

nonisolated struct SlidingWindowResult: Sendable {
    let evictedTokens: Int
    let pagesFreed: Int
    let newLength: Int
}

nonisolated struct MemoryPressureResponse: Sendable {
    let sequencesEvicted: Int
    let pagesFreed: Int
    let bytesReclaimed: Int64
}
