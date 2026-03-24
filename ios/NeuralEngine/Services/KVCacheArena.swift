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
    private let softPressureUtilization: Double = 0.92
    private let recoveryUtilization: Double = 0.75

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
        let requiredBytes = Int64(pageFootprintBytes)
        ensureCapacity(forAdditionalBytes: requiredBytes)

        if let recycledID = nextReusablePageID(), let recycledPage = pages[recycledID] {
            let recycled = KVPage(
                id: recycledPage.id,
                tokenStart: tokenStart,
                tokenCount: tokenCount,
                layerCount: layerCount,
                pageSize: pageSize,
                isActive: true,
                sequenceID: sequenceID
            )
            pages[recycledID] = recycled
            evictionOrder.removeAll { $0 == recycledID }
            evictionOrder.append(recycledID)
            updatePeakMemory()
            return recycled
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
        guard var page = pages[pageID], page.isActive else { return }
        page.touch()
        pages[pageID] = page
        evictionOrder.removeAll { $0 == pageID }
        evictionOrder.append(pageID)
    }

    func retainPage(_ pageID: UUID) {
        guard var page = pages[pageID], page.isActive else { return }
        page.retain()
        page.touch()
        pages[pageID] = page
        evictionOrder.removeAll { $0 == pageID }
        evictionOrder.append(pageID)
    }

    func releasePage(_ pageID: UUID) {
        guard var page = pages[pageID], page.isActive else { return }
        let freed = page.release()
        if freed {
            page.isActive = false
            page.isDirty = false
            pages[pageID] = page
            reclaimPage(pageID)
        } else {
            pages[pageID] = page
        }
    }

    func freePage(_ pageID: UUID) {
        guard var page = pages[pageID], page.isActive else { return }
        page.isActive = false
        page.referenceCount = 0
        page.isDirty = false
        pages[pageID] = page
        reclaimPage(pageID)
    }

    func freePages(_ pageIDs: [UUID]) {
        for id in pageIDs {
            freePage(id)
        }
    }

    func getPage(_ pageID: UUID) -> KVPage? {
        pages[pageID]
    }

    func markPageClean(_ pageID: UUID) {
        guard var page = pages[pageID], page.isActive else { return }
        page.markClean()
        pages[pageID] = page
    }

    func evictPagesForBudget(targetFreeBytes: Int64) -> [UUID] {
        guard targetFreeBytes > 0 else { return [] }

        var evicted: [UUID] = []
        var reclaimedBytes: Int64 = 0

        while reclaimedBytes < targetFreeBytes, let candidateID = findEvictionCandidate() {
            guard let page = pages[candidateID], page.isActive else { break }
            reclaimedBytes += Int64(page.memoryEstimateBytes)
            evicted.append(candidateID)
            freePage(candidateID)
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

        let liveIDs = Set(pages.keys)
        freePageIDs = freePageIDs.filter { liveIDs.contains($0) }
        evictionOrder = evictionOrder.filter { liveIDs.contains($0) }

        return DefragmentResult(
            pagesReclaimed: deadIDs.count,
            pagesBefore: beforeCount,
            pagesAfter: pages.count,
            bytesReclaimed: Int64(deadIDs.count * pageFootprintBytes)
        )
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
        let averageReferenceCount = activePages.isEmpty ? 0.0 : Double(activePages.reduce(0) { $0 + $1.referenceCount }) / Double(activePages.count)
        let dirtyPages = activePages.filter(\.isDirty).count

        return KVCacheStatistics(
            activePages: activePages.count,
            freePages: freePageIDs.count,
            totalPages: pages.count,
            sharedPages: activePages.filter(\.isShared).count,
            dirtyPages: dirtyPages,
            estimatedMemoryBytes: estimatedMemoryBytes,
            peakMemoryBytes: peakMemoryBytes,
            memoryBudgetBytes: memoryBudgetBytes,
            budgetUtilization: memoryBudgetUtilization,
            averageReferenceCount: averageReferenceCount,
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
        guard let page = pages[pageID], page.isActive else { return }
        blockHashIndex[blockHash] = pageID
        pageHashReverseIndex[pageID, default: []].insert(blockHash)
    }

    func referenceCount(for pageID: UUID) -> Int? {
        pages[pageID]?.referenceCount
    }

    private var pageFootprintBytes: Int {
        layerCount * pageSize * 2 * MemoryLayout<Float>.size
    }

    private func nextReusablePageID() -> UUID? {
        while let candidateID = freePageIDs.popLast() {
            guard let page = pages[candidateID] else { continue }
            guard !page.isActive else { continue }
            removeHashMappings(for: candidateID)
            return candidateID
        }
        return nil
    }

    private func ensureCapacity(forAdditionalBytes requiredBytes: Int64) {
        guard memoryBudgetBytes > 0 else { return }

        while estimatedMemoryBytes + requiredBytes > memoryBudgetBytes,
              let candidateID = findEvictionCandidate()
        {
            freePage(candidateID)
        }

        let projectedUtilization = Double(estimatedMemoryBytes + requiredBytes) / Double(memoryBudgetBytes)
        if activePageCount >= 3 && projectedUtilization > softPressureUtilization {
            evictUntilUtilizationBelow(recoveryUtilization)
        }
    }

    private func evictUntilUtilizationBelow(_ targetUtilization: Double) {
        guard memoryBudgetBytes > 0 else { return }
        while memoryBudgetUtilization > targetUtilization,
              let candidateID = findEvictionCandidate()
        {
            freePage(candidateID)
        }
    }

    private func reclaimPage(_ pageID: UUID) {
        evictionOrder.removeAll { $0 == pageID }
        removeHashMappings(for: pageID)
        if !freePageIDs.contains(pageID) {
            freePageIDs.append(pageID)
        }
    }

    private func findEvictionCandidate() -> UUID? {
        switch evictionPolicy {
        case .lru:
            return evictionOrder.first(where: isEvictablePage)
        case .lruWithPrefixProtection:
            if let protected = evictionOrder.first(where: { pageID in
                isEvictablePage(pageID) && (pages[pageID]?.tokenStart ?? 0) > 0
            }) {
                return protected
            }
            return evictionOrder.first(where: isEvictablePage)
        case .oldestFirst:
            return pages.values
                .filter { $0.isActive && !$0.isShared }
                .sorted { lhs, rhs in
                    if lhs.creationDate == rhs.creationDate {
                        return lhs.tokenStart < rhs.tokenStart
                    }
                    return lhs.creationDate < rhs.creationDate
                }
                .first?.id
        }
    }

    private func isEvictablePage(_ pageID: UUID) -> Bool {
        guard let page = pages[pageID] else { return false }
        return page.isActive && !page.isShared
    }

    private func updatePeakMemory() {
        let current = estimatedMemoryBytes
        if current > peakMemoryBytes {
            peakMemoryBytes = current
        }
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
