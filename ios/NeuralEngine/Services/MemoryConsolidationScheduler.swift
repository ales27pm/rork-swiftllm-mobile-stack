import Foundation

@MainActor
@Observable
class MemoryConsolidationScheduler {
    private let memoryService: MemoryService
    private let keyValueStore: KeyValueStore
    private var consolidationTask: Task<Void, Never>?
    private let consolidationIntervalHours: Double = 6
    private let mergeThreshold: Double = 0.7
    private let decayUpdateInterval: Double = 3600

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

        applyDecayToAll()
        mergeRelatedMemories()
        pruneWeakMemories()
        reinforceAssociativeLinks()

        lastConsolidationDate = Date()
        consolidationCount += 1
        keyValueStore.setDouble(Date().timeIntervalSince1970, forKey: "last_consolidation_timestamp")
        keyValueStore.setInt(consolidationCount, forKey: "consolidation_count")

        isConsolidating = false
    }

    private func shouldConsolidate() -> Bool {
        guard let last = lastConsolidationDate else { return true }
        let hoursSince = Date().timeIntervalSince(last) / 3600
        return hoursSince >= consolidationIntervalHours
    }

    private func applyDecayToAll() {
        let now = Date().timeIntervalSince1970 * 1000
        for i in memoryService.memories.indices {
            let memory = memoryService.memories[i]
            let hoursSinceAccess = (now - memory.lastAccessed) / (1000 * 60 * 60)
            let accessBoost = min(Double(memory.accessCount) * 0.1, 0.5)
            let importanceBoost = Double(memory.importance) / 5.0 * 0.3
            let halfLife = 168.0 * (1.0 + accessBoost + importanceBoost)
            let newDecay = max(0.05, min(1.0, pow(0.5, hoursSinceAccess / halfLife)))

            if abs(newDecay - memory.decay) > 0.01 {
                var updated = memory
                updated.decay = newDecay
                updated.activationLevel = max(0, updated.activationLevel - 0.05)
                memoryService.addMemory(updated)
            }
        }
    }

    private func mergeRelatedMemories() {
        let memories = memoryService.memories
        var merged: Set<String> = []

        for i in 0..<memories.count {
            guard !merged.contains(memories[i].id) else { continue }
            for j in (i + 1)..<memories.count {
                guard !merged.contains(memories[j].id) else { continue }
                guard memories[i].category == memories[j].category else { continue }

                let similarity = jaccardSimilarity(memories[i].content, memories[j].content)
                if similarity > mergeThreshold {
                    let newer = memories[i].timestamp > memories[j].timestamp ? memories[i] : memories[j]
                    let older = memories[i].timestamp > memories[j].timestamp ? memories[j] : memories[i]

                    var consolidated = newer
                    consolidated.keywords = Array(Set(newer.keywords + older.keywords)).prefix(8).map { $0 }
                    consolidated.importance = max(newer.importance, older.importance)
                    consolidated.accessCount = newer.accessCount + older.accessCount
                    consolidated.consolidated = true

                    memoryService.addMemory(consolidated)
                    memoryService.deleteMemory(older.id)
                    merged.insert(older.id)
                }
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
        var links = memoryService.associativeLinks
        let memoryIds = Set(memoryService.memories.map(\.id))

        for i in links.indices {
            guard memoryIds.contains(links[i].sourceId),
                  memoryIds.contains(links[i].targetId) else { continue }

            let sourceDecay = memoryService.memories.first { $0.id == links[i].sourceId }?.decay ?? 0
            let targetDecay = memoryService.memories.first { $0.id == links[i].targetId }?.decay ?? 0
            let avgDecay = (sourceDecay + targetDecay) / 2.0

            if avgDecay < 0.2 {
                links[i].strength = max(0.05, links[i].strength * 0.8)
            } else if avgDecay > 0.7 && links[i].reinforcements > 2 {
                links[i].strength = min(1.0, links[i].strength * 1.05)
            }
        }
    }

    private func jaccardSimilarity(_ a: String, _ b: String) -> Double {
        let tokensA = Set(a.lowercased().split(separator: " ").map(String.init))
        let tokensB = Set(b.lowercased().split(separator: " ").map(String.init))
        guard !tokensA.isEmpty || !tokensB.isEmpty else { return 0 }
        let intersection = tokensA.intersection(tokensB).count
        let union = tokensA.union(tokensB).count
        return union > 0 ? Double(intersection) / Double(union) : 0
    }
}
