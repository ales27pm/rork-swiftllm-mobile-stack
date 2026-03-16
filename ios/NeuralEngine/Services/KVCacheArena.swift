import Foundation

actor KVCacheArena {
    private var pages: [UUID: KVPage] = [:]
    private var freePageIDs: [UUID] = []
    private var evictionOrder: [UUID] = []
    private let pageSize: Int
    private let layerCount: Int
    private let memoryBudgetBytes: Int64
    private var peakMemoryBytes: Int64 = 0
    private var blockHashIndex: [String: UUID] = [:]
    private var pageHashReverseIndex: [UUID: Set<String>] = [:]

    nonisolated let evictionPolicy: EvictionPolicy

    init(
        pageSize: Int = 128,
        layerCount: Int = 32,
        memoryBudgetMB: Int = 256,
        evictionPolicy: EvictionPolicy = .lru
    ) {
        self.pageSize = pageSize
        self.layerCount = layerCount
        self.memoryBudgetBytes = Int64(memoryBudgetMB) * 1_048_576
        self.evictionPolicy = evictionPolicy
    }

    func allocatePage(
        tokenStart: Int,
        tokenCount: Int,
        sequenceID: UUID? = nil
    ) -> KVPage {
        if let freeID = freePageIDs.popLast(), var recycled = pages[freeID] {
            removeHashMappings(for: freeID)
            recycled = KVPage(
                id: recycled.id,
                tokenStart: tokenStart,
                tokenCount: tokenCount,
                layerCount: layerCount,
                pageSize: pageSize,
                isActive: true,
                sequenceID: sequenceID
            )
            pages[freeID] = recycled
            evictionOrder.append(freeID)
            updatePeakMemory()
            return recycled
        }

        if estimatedMemoryBytes >= memoryBudgetBytes {
            evictLRUPage()
        }

        let page = KVPage(
            tokenStart: tokenStart,
            tokenCount: tokenCount,
            layerCount: layerCount,
            pageSize: pageSize,
            isActive: true,
            sequenceID: sequenceID
        )
        pages[page.id] = page
        evictionOrder.append(page.id)
        updatePeakMemory()
        return page
    }

    func touchPage(_ pageID: UUID) {
        guard var page = pages[pageID] else { return }
        page.touch()
        pages[pageID] = page
        evictionOrder.removeAll { $0 == pageID }
        evictionOrder.append(pageID)
    }

    func retainPage(_ pageID: UUID) {
        guard var page = pages[pageID] else { return }
        page.retain()
        pages[pageID] = page
    }

    func releasePage(_ pageID: UUID) {
        guard var page = pages[pageID] else { return }
        let freed = page.release()
        if freed {
            page.isActive = false
            pages[pageID] = page
            freePageIDs.append(pageID)
            evictionOrder.removeAll { $0 == pageID }
            removeHashMappings(for: pageID)
        } else {
            pages[pageID] = page
        }
    }

    func freePage(_ pageID: UUID) {
        guard var page = pages[pageID] else { return }
        page.isActive = false
        page.referenceCount = 0
        pages[pageID] = page
        freePageIDs.append(pageID)
        evictionOrder.removeAll { $0 == pageID }
        removeHashMappings(for: pageID)
    }

    func freePages(_ pageIDs: [UUID]) {
        for id in pageIDs {
            freePage(id)
        }
    }

    func getPage(_ pageID: UUID) -> KVPage? {
        pages[pageID]
    }

    private func evictLRUPage() {
        while estimatedMemoryBytes >= memoryBudgetBytes, let candidateID = findEvictionCandidate() {
            freePage(candidateID)
        }
    }

    private func findEvictionCandidate() -> UUID? {
        switch evictionPolicy {
        case .lru:
            for id in evictionOrder {
                if let page = pages[id], page.isActive, !page.isShared {
                    return id
                }
            }
        case .lruWithPrefixProtection:
            for id in evictionOrder {
                if let page = pages[id], page.isActive, !page.isShared, page.tokenStart > 0 {
                    return id
                }
            }
            for id in evictionOrder {
                if let page = pages[id], page.isActive, !page.isShared {
                    return id
                }
            }
        case .oldestFirst:
            let sorted = pages.values
                .filter { $0.isActive && !$0.isShared }
                .sorted { $0.creationDate < $1.creationDate }
            return sorted.first?.id
        }
        return nil
    }

    func evictPagesForBudget(targetFreeBytes: Int64) -> [UUID] {
        var evicted: [UUID] = []
        var freed: Int64 = 0

        while freed < targetFreeBytes, let candidateID = findEvictionCandidate() {
            if let page = pages[candidateID] {
                freed += Int64(page.memoryEstimateBytes)
                evicted.append(candidateID)
                freePage(candidateID)
            }
        }

        return evicted
    }

    func evictSequencePages(sequenceID: UUID) -> Int {
        let sequencePages = pages.values.filter { $0.sequenceID == sequenceID && $0.isActive }
        for page in sequencePages {
            freePage(page.id)
        }
        return sequencePages.count
    }

    func defragment() -> DefragmentResult {
        let beforeCount = pages.count
        let deadIDs = pages.filter { !$0.value.isActive }.map(\.key)

        for id in deadIDs {
            removeHashMappings(for: id)
            pages.removeValue(forKey: id)
        }
        freePageIDs.removeAll()
        evictionOrder = evictionOrder.filter { pages[$0] != nil }

        return DefragmentResult(
            pagesReclaimed: deadIDs.count,
            pagesBefore: beforeCount,
            pagesAfter: pages.count,
            bytesReclaimed: Int64(deadIDs.count * layerCount * pageSize * 2 * MemoryLayout<Float>.size)
        )
    }

    private func updatePeakMemory() {
        let current = estimatedMemoryBytes
        if current > peakMemoryBytes {
            peakMemoryBytes = current
        }
    }

    func reset() {
        pages.removeAll(keepingCapacity: true)
        freePageIDs.removeAll(keepingCapacity: true)
        evictionOrder.removeAll(keepingCapacity: true)
        peakMemoryBytes = 0
        blockHashIndex.removeAll(keepingCapacity: true)
        pageHashReverseIndex.removeAll(keepingCapacity: true)
    }

    var activePageCount: Int {
        pages.values.filter(\.isActive).count
    }

    var totalPageCount: Int {
        pages.count
    }

    var freePageCount: Int {
        freePageIDs.count
    }

    var sharedPageCount: Int {
        pages.values.filter { $0.isActive && $0.isShared }.count
    }

    var estimatedMemoryBytes: Int64 {
        Int64(pages.values.filter(\.isActive).reduce(0) { $0 + $1.memoryEstimateBytes })
    }

    var peakMemoryUsageBytes: Int64 {
        peakMemoryBytes
    }

    var memoryBudgetUtilization: Double {
        guard memoryBudgetBytes > 0 else { return 0 }
        return Double(estimatedMemoryBytes) / Double(memoryBudgetBytes)
    }

    var statistics: KVCacheStatistics {
        let activePages = pages.values.filter(\.isActive)
        let avgRefCount = activePages.isEmpty ? 0.0 : Double(activePages.reduce(0) { $0 + $1.referenceCount }) / Double(activePages.count)
        let dirtyCount = activePages.filter(\.isDirty).count

        return KVCacheStatistics(
            activePages: activePages.count,
            freePages: freePageIDs.count,
            totalPages: pages.count,
            sharedPages: activePages.filter(\.isShared).count,
            dirtyPages: dirtyCount,
            estimatedMemoryBytes: estimatedMemoryBytes,
            peakMemoryBytes: peakMemoryBytes,
            memoryBudgetBytes: memoryBudgetBytes,
            budgetUtilization: memoryBudgetUtilization,
            averageReferenceCount: avgRefCount,
            evictionPolicy: evictionPolicy
        )
    }

    var configuredPageSize: Int {
        pageSize
    }

    func pageID(forBlockHash blockHash: String) -> UUID? {
        guard let pageID = blockHashIndex[blockHash], let page = pages[pageID], page.isActive else {
            blockHashIndex.removeValue(forKey: blockHash)
            return nil
        }
        return pageID
    }

    func register(blockHash: String, pageID: UUID) {
        blockHashIndex[blockHash] = pageID
        pageHashReverseIndex[pageID, default: []].insert(blockHash)
    }

    func referenceCount(for pageID: UUID) -> Int? {
        pages[pageID]?.referenceCount
    }

    private func removeHashMappings(for pageID: UUID) {
        guard let hashes = pageHashReverseIndex.removeValue(forKey: pageID) else { return }
        for hash in hashes where blockHashIndex[hash] == pageID {
            blockHashIndex.removeValue(forKey: hash)
        }
    }
}

nonisolated enum EvictionPolicy: String, Sendable {
    case lru = "LRU"
    case lruWithPrefixProtection = "LRU+PrefixProtect"
    case oldestFirst = "OldestFirst"
}

nonisolated struct KVCacheStatistics: Sendable {
    let activePages: Int
    let freePages: Int
    let totalPages: Int
    let sharedPages: Int
    let dirtyPages: Int
    let estimatedMemoryBytes: Int64
    let peakMemoryBytes: Int64
    let memoryBudgetBytes: Int64
    let budgetUtilization: Double
    let averageReferenceCount: Double
    let evictionPolicy: EvictionPolicy
}

nonisolated struct DefragmentResult: Sendable {
    let pagesReclaimed: Int
    let pagesBefore: Int
    let pagesAfter: Int
    let bytesReclaimed: Int64
}
