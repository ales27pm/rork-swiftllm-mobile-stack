import Foundation

actor KVPageTable {
    private var sequences: [UUID: SequenceEntry] = [:]
    private var sharedPrefixMap: [String: [UUID]] = [:]

    struct SequenceEntry: Sendable {
        let sequenceID: UUID
        var pageIDs: [UUID]
        var tokenRange: Range<Int>
        var prefixHash: String?
        var createdAt: Date
        var lastAccessed: Date
        var isActive: Bool

        init(sequenceID: UUID, prefixHash: String? = nil) {
            self.sequenceID = sequenceID
            self.pageIDs = []
            self.tokenRange = 0..<0
            self.prefixHash = prefixHash
            self.createdAt = Date()
            self.lastAccessed = Date()
            self.isActive = true
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

    func appendPage(_ pageID: UUID, to sequenceID: UUID, tokenEnd: Int) {
        guard var entry = sequences[sequenceID] else { return }
        entry.pageIDs.append(pageID)
        entry.tokenRange = entry.tokenRange.lowerBound..<tokenEnd
        entry.lastAccessed = Date()
        sequences[sequenceID] = entry
    }

    func pageIDs(for sequenceID: UUID) -> [UUID] {
        sequences[sequenceID]?.pageIDs ?? []
    }

    func sequencesSharing(prefixHash: String) -> [UUID] {
        sharedPrefixMap[prefixHash] ?? []
    }

    func forkSequence(_ sourceID: UUID) -> UUID? {
        guard let source = sequences[sourceID] else { return nil }
        let newID = UUID()
        var forked = SequenceEntry(sequenceID: newID, prefixHash: source.prefixHash)
        forked.pageIDs = source.pageIDs
        forked.tokenRange = source.tokenRange
        sequences[newID] = forked

        if let hash = source.prefixHash {
            sharedPrefixMap[hash, default: []].append(newID)
        }

        return newID
    }

    func truncateSequence(_ sequenceID: UUID, keepPages: Int) -> [UUID] {
        guard var entry = sequences[sequenceID] else { return [] }
        guard entry.pageIDs.count > keepPages else { return [] }

        let removed = Array(entry.pageIDs[keepPages...])
        entry.pageIDs = Array(entry.pageIDs.prefix(keepPages))
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
        sequences.values.reduce(0) { $0 + $1.pageIDs.count }
    }

    func sequenceInfo(_ sequenceID: UUID) -> SequenceEntry? {
        sequences[sequenceID]
    }

    func reset() {
        sequences.removeAll()
        sharedPrefixMap.removeAll()
    }
}
