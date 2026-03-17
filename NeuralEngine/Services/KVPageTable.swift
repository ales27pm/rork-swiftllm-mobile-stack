import Foundation

nonisolated enum KVPageOrigin: Sendable, Equatable {
    case sharedPrefix
    case materialized
}

actor KVPageTable {
    private var sequences: [UUID: SequenceEntry] = [:]
    private var sharedPrefixMap: [String: [UUID]] = [:]

    struct PageMapping: Sendable {
        let pageID: UUID
        let tokenStart: Int
        let tokenEnd: Int
        let origin: KVPageOrigin

        var tokenCount: Int {
            max(0, tokenEnd - tokenStart)
        }
    }

    struct SequenceEntry: Sendable {
        let sequenceID: UUID
        var pageMappings: [PageMapping]
        var tokenRange: Range<Int>
        var prefixHash: String?
        var createdAt: Date
        var lastAccessed: Date
        var isActive: Bool

        init(sequenceID: UUID, prefixHash: String? = nil) {
            self.sequenceID = sequenceID
            self.pageMappings = []
            self.tokenRange = 0..<0
            self.prefixHash = prefixHash
            self.createdAt = Date()
            self.lastAccessed = Date()
            self.isActive = true
        }

        var pageIDs: [UUID] {
            pageMappings.map(\.pageID)
        }
    }

    func createSequence(prefixHash: String? = nil) -> UUID {
        let id = UUID()
        var entry = SequenceEntry(sequenceID: id, prefixHash: prefixHash)
        entry.isActive = true
        sequences[id] = entry

        if let hash = prefixHash {
            sharedPrefixMap[hash, default: []].append(id)
        }

        return id
    }

    func appendPage(
        _ pageID: UUID,
        to sequenceID: UUID,
        tokenStart: Int,
        tokenEnd: Int,
        origin: KVPageOrigin = .materialized
    ) {
        guard var entry = sequences[sequenceID] else { return }
        entry.pageMappings.append(PageMapping(pageID: pageID, tokenStart: tokenStart, tokenEnd: tokenEnd, origin: origin))

        if let first = entry.pageMappings.first {
            entry.tokenRange = first.tokenStart..<tokenEnd
        } else {
            entry.tokenRange = tokenStart..<tokenEnd
        }
        entry.lastAccessed = Date()
        sequences[sequenceID] = entry
    }

    func pageIDs(for sequenceID: UUID) -> [UUID] {
        sequences[sequenceID]?.pageIDs ?? []
    }

    func pageMappings(for sequenceID: UUID) -> [PageMapping] {
        sequences[sequenceID]?.pageMappings ?? []
    }

    func sequencesSharing(prefixHash: String) -> [UUID] {
        sharedPrefixMap[prefixHash] ?? []
    }

    func forkSequence(_ sourceID: UUID) -> UUID? {
        guard let source = sequences[sourceID] else { return nil }
        let newID = UUID()
        var forked = SequenceEntry(sequenceID: newID, prefixHash: source.prefixHash)
        forked.pageMappings = source.pageMappings
        forked.tokenRange = source.tokenRange
        sequences[newID] = forked

        if let hash = source.prefixHash {
            sharedPrefixMap[hash, default: []].append(newID)
        }

        return newID
    }

    func truncateSequence(_ sequenceID: UUID, keepPages: Int) -> [UUID] {
        guard var entry = sequences[sequenceID] else { return [] }
        guard entry.pageMappings.count > keepPages else { return [] }

        let removed = Array(entry.pageMappings[keepPages...].map(\.pageID))
        entry.pageMappings = Array(entry.pageMappings.prefix(keepPages))
        entry.lastAccessed = Date()
        sequences[sequenceID] = entry
        return removed
    }

    func truncateMiddlePages(
        _ sequenceID: UUID,
        prefixPages: Int,
        tailPages: Int
    ) -> [PageMapping] {
        guard var entry = sequences[sequenceID] else { return [] }

        let totalPages = entry.pageMappings.count
        let safePrefix = max(0, prefixPages)
        let safeTail = max(0, tailPages)
        guard totalPages > safePrefix + safeTail else { return [] }

        let removeStart = min(safePrefix, totalPages)
        let keepTailStart = max(removeStart, totalPages - safeTail)
        guard removeStart < keepTailStart else { return [] }

        let removed = Array(entry.pageMappings[removeStart..<keepTailStart])
        let keptPrefix = entry.pageMappings.prefix(removeStart)
        let keptTail = entry.pageMappings.suffix(totalPages - keepTailStart)
        entry.pageMappings = Array(keptPrefix + keptTail)

        if let first = entry.pageMappings.first, let last = entry.pageMappings.last {
            entry.tokenRange = first.tokenStart..<last.tokenEnd
        } else {
            entry.tokenRange = 0..<0
        }

        entry.lastAccessed = Date()
        sequences[sequenceID] = entry

        return removed
    }

    func removeSequence(_ sequenceID: UUID) -> [UUID] {
        guard let entry = sequences.removeValue(forKey: sequenceID) else { return [] }

        if let hash = entry.prefixHash {
            sharedPrefixMap[hash]?.removeAll { $0 == sequenceID }
            if sharedPrefixMap[hash]?.isEmpty == true {
                sharedPrefixMap.removeValue(forKey: hash)
            }
        }

        return entry.pageIDs
    }

    func deactivateSequence(_ sequenceID: UUID) {
        sequences[sequenceID]?.isActive = false
    }

    func touchSequence(_ sequenceID: UUID) {
        sequences[sequenceID]?.lastAccessed = Date()
    }

    func evictableSequences(olderThan threshold: TimeInterval) -> [UUID] {
        let cutoff = Date().addingTimeInterval(-threshold)
        return sequences.values
            .filter { !$0.isActive && $0.lastAccessed < cutoff }
            .sorted { $0.lastAccessed < $1.lastAccessed }
            .map(\.sequenceID)
    }

    var activeSequenceCount: Int {
        sequences.values.filter(\.isActive).count
    }

    var totalSequenceCount: Int {
        sequences.count
    }

    var totalPageCount: Int {
        sequences.values.reduce(0) { $0 + $1.pageMappings.count }
    }

    func sequenceInfo(_ sequenceID: UUID) -> SequenceEntry? {
        sequences[sequenceID]
    }

    func reset() {
        sequences.removeAll()
        sharedPrefixMap.removeAll()
    }
}
