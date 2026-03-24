import Foundation

@MainActor
@Observable
class MemoryConsolidationScheduler {
    private let memoryService: MemoryService
    private let keyValueStore: KeyValueStore
    private var consolidationTask: Task<Void, Never>?
    private let consolidationIntervalHours: Double = 6
    private let decayUpdateInterval: Double = 3600
    private let lexicalMergeThreshold: Double = 0.7
    private let semanticMergeThreshold: Float = 0.88
    private let clusterSimilarityThreshold: Float = 0.8
    private let crossCategoryClusterThreshold: Float = 0.9
    private let minimumMergeScore: Double = 0.82
    private let maxNeighborsPerMemory: Int = 6

    var lastConsolidationDate: Date?
    var consolidationCount: Int = 0
    var isConsolidating: Bool = false

    init(memoryService: MemoryService, keyValueStore: KeyValueStore) {
        self.memoryService = memoryService
        self.keyValueStore = keyValueStore
        if let timestamp = keyValueStore.getDouble("last_consolidation_timestamp") {
            lastConsolidationDate = Date(timeIntervalSince1970: timestamp)
        }
        consolidationCount = keyValueStore.getInt("consolidation_count") ?? 0
    }

    func startScheduledConsolidation() {
        consolidationTask?.cancel()
        consolidationTask = Task {
            while !Task.isCancelled {
                if shouldConsolidate() {
                    await performConsolidation()
                }
                try? await Task.sleep(for: .seconds(decayUpdateInterval))
            }
        }
    }

    func stopScheduledConsolidation() {
        consolidationTask?.cancel()
        consolidationTask = nil
    }

    func performConsolidation() async {
        guard !isConsolidating else { return }
        isConsolidating = true
        defer { isConsolidating = false }

        applyDecayToAll()
        mergeRelatedMemories()
        buildSemanticClusters()
        pruneWeakMemories()
        reinforceAssociativeLinks()

        let now = Date()
        lastConsolidationDate = now
        consolidationCount += 1
        keyValueStore.setDouble(now.timeIntervalSince1970, forKey: "last_consolidation_timestamp")
        keyValueStore.setInt(consolidationCount, forKey: "consolidation_count")
    }

    private func shouldConsolidate() -> Bool {
        guard let last = lastConsolidationDate else { return true }
        let hoursSince = Date().timeIntervalSince(last) / 3600
        return hoursSince >= consolidationIntervalHours
    }

    private func applyDecayToAll() {
        let now = Date().timeIntervalSince1970 * 1000
        let snapshot = memoryService.memories

        for memory in snapshot {
            let hoursSinceAccess = (now - memory.lastAccessed) / (1000 * 60 * 60)
            let accessBoost = min(Double(memory.accessCount) * 0.1, 0.5)
            let importanceBoost = Double(memory.importance) / 5.0 * 0.3
            let halfLife = 168.0 * (1.0 + accessBoost + importanceBoost)
            let newDecay = max(0.05, min(1.0, pow(0.5, hoursSinceAccess / halfLife)))
            let newActivation = max(0, memory.activationLevel - 0.05)

            guard abs(newDecay - memory.decay) > 0.01 || abs(newActivation - memory.activationLevel) > 0.01 else {
                continue
            }

            var updated = memory
            updated.decay = newDecay
            updated.activationLevel = newActivation
            memoryService.addMemory(updated)
        }
    }

    private func mergeRelatedMemories() {
        let snapshot = memoryService.memories
        let memoriesByID = Dictionary(uniqueKeysWithValues: snapshot.map { ($0.id, $0) })
        let ordered = snapshot.sorted { lhs, rhs in
            memoryRank(lhs) > memoryRank(rhs)
        }
        var retiredIDs: Set<String> = []

        for memory in ordered {
            guard !retiredIDs.contains(memory.id) else { continue }
            guard let candidate = bestMergeCandidate(for: memory, memoriesByID: memoriesByID, retiredIDs: retiredIDs) else { continue }

            let merged = mergedMemory(primaryCandidate: memory, secondaryCandidate: candidate.memory)
            retiredIDs.insert(merged.retiredID)

            memoryService.addMemory(merged.memory)
            memoryService.deleteMemory(merged.retiredID)
        }
    }

    private func buildSemanticClusters() {
        let snapshot = memoryService.memories
        let memoriesByID = Dictionary(uniqueKeysWithValues: snapshot.map { ($0.id, $0) })
        var processedPairs: Set<String> = []
        let now = Date().timeIntervalSince1970 * 1000

        for memory in snapshot {
            guard let vector = memoryService.vectorStore.getVector(for: memory.id) else { continue }
            let neighbors = memoryService.vectorStore.searchByVector(vector, maxResults: maxNeighborsPerMemory + 1, minScore: min(clusterSimilarityThreshold, crossCategoryClusterThreshold))

            for neighbor in neighbors {
                guard neighbor.id != memory.id,
                      let other = memoriesByID[neighbor.id] else { continue }

                let pairKey = canonicalPairKey(memory.id, other.id)
                guard processedPairs.insert(pairKey).inserted else { continue }

                let sameCategory = memory.category == other.category
                let threshold = sameCategory ? Double(clusterSimilarityThreshold) : Double(crossCategoryClusterThreshold)
                let lexical = jaccardSimilarity(memory.content, other.content)
                let combined = combinedSimilarity(lexical: lexical, semantic: Double(neighbor.score))
                guard combined >= threshold else { continue }

                let orderedIDs = orderedPair(memory.id, other.id)
                let existing = memoryService.associativeLinks.first {
                    ($0.sourceId == orderedIDs.0 && $0.targetId == orderedIDs.1) ||
                    ($0.sourceId == orderedIDs.1 && $0.targetId == orderedIDs.0)
                }

                let link = AssociativeLink(
                    sourceId: orderedIDs.0,
                    targetId: orderedIDs.1,
                    strength: min(1.0, max(existing?.strength ?? 0, combined)),
                    type: sameCategory ? .topical : .semantic,
                    createdAt: existing?.createdAt ?? now,
                    reinforcements: (existing?.reinforcements ?? 0) + 1
                )
                memoryService.saveLink(link)
            }
        }
    }

    private func pruneWeakMemories() {
        let weakMemories = memoryService.memories.filter { memory in
            memory.decay < 0.1 &&
            memory.importance <= 2 &&
            memory.accessCount <= 1 &&
            memory.source != .manual
        }

        for memory in weakMemories {
            memoryService.deleteMemory(memory.id)
        }
    }

    private func reinforceAssociativeLinks() {
        let memoryMap = Dictionary(uniqueKeysWithValues: memoryService.memories.map { ($0.id, $0) })

        for link in memoryService.associativeLinks {
            guard let source = memoryMap[link.sourceId],
                  let target = memoryMap[link.targetId] else { continue }

            let avgDecay = (source.decay + target.decay) / 2.0
            let lexical = jaccardSimilarity(source.content, target.content)
            var updated = link

            if avgDecay < 0.2 {
                updated.strength = max(0.05, updated.strength * 0.8)
            } else {
                let targetStrength = combinedSimilarity(lexical: lexical, semantic: updated.strength)
                let reinforcementBoost = updated.reinforcements > 2 ? 0.04 : 0.02
                updated.strength = min(1.0, max(updated.strength, targetStrength) + reinforcementBoost)
            }

            memoryService.saveLink(updated)
        }
    }

    private func bestMergeCandidate(
        for memory: MemoryEntry,
        memoriesByID: [String: MemoryEntry],
        retiredIDs: Set<String>
    ) -> MergeCandidate? {
        guard let vector = memoryService.vectorStore.getVector(for: memory.id) else {
            return lexicalMergeCandidate(for: memory, memoriesByID: memoriesByID, retiredIDs: retiredIDs)
        }

        let neighbors = memoryService.vectorStore.searchByVector(vector, maxResults: maxNeighborsPerMemory + 1, minScore: semanticMergeThreshold)
        var best: MergeCandidate?

        for neighbor in neighbors {
            guard neighbor.id != memory.id,
                  !retiredIDs.contains(neighbor.id),
                  let other = memoriesByID[neighbor.id],
                  other.category == memory.category else { continue }

            let lexical = jaccardSimilarity(memory.content, other.content)
            let combined = combinedSimilarity(lexical: lexical, semantic: Double(neighbor.score))
            guard lexical >= lexicalMergeThreshold || combined >= minimumMergeScore else { continue }

            let candidate = MergeCandidate(memory: other, score: combined)
            if best == nil || candidate.score > best!.score {
                best = candidate
            }
        }

        return best ?? lexicalMergeCandidate(for: memory, memoriesByID: memoriesByID, retiredIDs: retiredIDs)
    }

    private func lexicalMergeCandidate(
        for memory: MemoryEntry,
        memoriesByID: [String: MemoryEntry],
        retiredIDs: Set<String>
    ) -> MergeCandidate? {
        var best: MergeCandidate?

        for other in memoriesByID.values {
            guard other.id != memory.id,
                  !retiredIDs.contains(other.id),
                  other.category == memory.category else { continue }

            let lexical = jaccardSimilarity(memory.content, other.content)
            guard lexical >= lexicalMergeThreshold else { continue }

            let candidate = MergeCandidate(memory: other, score: lexical)
            if best == nil || candidate.score > best!.score {
                best = candidate
            }
        }

        return best
    }

    private func mergedMemory(primaryCandidate: MemoryEntry, secondaryCandidate: MemoryEntry) -> MergeResult {
        let primary: MemoryEntry
        let secondary: MemoryEntry

        if memoryRank(primaryCandidate) >= memoryRank(secondaryCandidate) {
            primary = primaryCandidate
            secondary = secondaryCandidate
        } else {
            primary = secondaryCandidate
            secondary = primaryCandidate
        }

        var merged = primary
        let primaryNormalized = primary.content.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let secondaryNormalized = secondary.content.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if secondary.content.count > primary.content.count && !primaryNormalized.contains(secondaryNormalized) {
            merged.content = secondary.content
        }

        merged.keywords = Array(Set(primary.keywords + secondary.keywords)).sorted().prefix(10).map { $0 }
        merged.importance = max(primary.importance, secondary.importance)
        merged.accessCount = primary.accessCount + secondary.accessCount
        merged.lastAccessed = max(primary.lastAccessed, secondary.lastAccessed)
        merged.timestamp = max(primary.timestamp, secondary.timestamp)
        merged.relations = Array(Set(primary.relations + secondary.relations + [secondary.id])).sorted()
        merged.consolidated = true
        merged.decay = max(primary.decay, secondary.decay)
        merged.activationLevel = max(primary.activationLevel, secondary.activationLevel)
        merged.emotionalValence = (primary.emotionalValence + secondary.emotionalValence) / 2.0

        return MergeResult(memory: merged, retiredID: secondary.id)
    }

    private func memoryRank(_ memory: MemoryEntry) -> Double {
        let recency = memory.lastAccessed / 1_000_000_000_000
        return (Double(memory.importance) * 0.35) +
            (Double(memory.accessCount) * 0.15) +
            (memory.decay * 0.2) +
            (memory.activationLevel * 0.15) +
            recency
    }

    private func combinedSimilarity(lexical: Double, semantic: Double) -> Double {
        max(lexical, (semantic * 0.75) + (lexical * 0.25))
    }

    private func canonicalPairKey(_ lhs: String, _ rhs: String) -> String {
        orderedPair(lhs, rhs).0 + "::" + orderedPair(lhs, rhs).1
    }

    private func orderedPair(_ lhs: String, _ rhs: String) -> (String, String) {
        lhs < rhs ? (lhs, rhs) : (rhs, lhs)
    }

    private func jaccardSimilarity(_ lhs: String, _ rhs: String) -> Double {
        let left = Set(lhs.lowercased().split(separator: " ").map(String.init))
        let right = Set(rhs.lowercased().split(separator: " ").map(String.init))
        guard !left.isEmpty || !right.isEmpty else { return 0 }
        let intersection = left.intersection(right).count
        let union = left.union(right).count
        return union > 0 ? Double(intersection) / Double(union) : 0
    }
}

private nonisolated struct MergeCandidate: Sendable {
    let memory: MemoryEntry
    let score: Double
}

private nonisolated struct MergeResult: Sendable {
    let memory: MemoryEntry
    let retiredID: String
}
