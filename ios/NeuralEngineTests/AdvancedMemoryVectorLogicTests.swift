import Foundation
import Testing
@testable import NeuralEngine

struct AdvancedMemoryVectorLogicTests {
    @Test @MainActor
    func vectorStore_upsertWithVectorNormalizesAndRanksSemanticNeighbors() {
        let database = DatabaseService(name: "vector-store-\(UUID().uuidString).sqlite3")
        defer { _ = database.deleteDatabase() }

        let store = VectorStore(database: database)

        #expect(store.upsertWithVector(id: "alpha", vector: [1, 0, 0]))
        #expect(store.upsertWithVector(id: "beta", vector: [0.96, 0.04, 0]))
        #expect(store.upsertWithVector(id: "gamma", vector: [0, 1, 0]))

        let results = store.searchByVector([0.92, 0.08, 0], maxResults: 3, minScore: 0.1)
        let stored = store.getVector(for: "alpha")
        let magnitude = sqrt((stored ?? []).reduce(0) { partial, value in
            partial + Double(value * value)
        })

        #expect(results.count == 3)
        #expect(results[0].id == "alpha")
        #expect(results[1].id == "beta")
        #expect(results[0].score > results[1].score)
        #expect(results[1].score > results[2].score)
        #expect(stored?.count == VectorEmbeddingService.dimensions)
        #expect(abs(magnitude - 1.0) < 0.0001)
    }

    @Test @MainActor
    func consolidationScheduler_mergesSemanticallyDuplicateMemories() async {
        let database = DatabaseService(name: "memory-merge-\(UUID().uuidString).sqlite3")
        let store = KeyValueStore(suiteName: "tests.memory.merge.\(UUID().uuidString)")
        defer {
            store.removeAll()
            _ = database.deleteDatabase()
        }

        let memoryService = MemoryService(database: database)
        memoryService.clearAllMemories()
        let scheduler = MemoryConsolidationScheduler(memoryService: memoryService, keyValueStore: store)

        let earlier = MemoryEntry(
            content: "User preference: enjoys fruit smoothies after workouts",
            keywords: ["smoothies", "fruit"],
            category: .preference,
            importance: 3,
            source: .conversation
        )
        let later = MemoryEntry(
            content: "User preference: likes mango banana shakes after training",
            keywords: ["mango", "banana", "shakes"],
            category: .preference,
            importance: 5,
            source: .conversation
        )

        memoryService.addMemory(earlier)
        memoryService.addMemory(later)
        #expect(memoryService.vectorStore.upsertWithVector(id: earlier.id, vector: [1, 0, 0]))
        #expect(memoryService.vectorStore.upsertWithVector(id: later.id, vector: [0.999, 0.001, 0]))

        await scheduler.performConsolidation()

        let preferenceMemories = memoryService.memories.filter { $0.category == .preference }
        let merged = preferenceMemories.first

        #expect(preferenceMemories.count == 1)
        #expect(merged?.consolidated == true)
        #expect(Set(merged?.keywords ?? []).contains("smoothies"))
        #expect(Set(merged?.keywords ?? []).contains("mango"))
        #expect(memoryService.vectorStore.count == 1)
        #expect(memoryService.vectorStore.hasVector(for: later.id))
        #expect(!memoryService.vectorStore.hasVector(for: earlier.id))
        #expect(scheduler.consolidationCount == 1)
        #expect(store.getInt("consolidation_count") == 1)
    }

    @Test @MainActor
    func consolidationScheduler_buildsAssociativeLinksForStrongSemanticClusters() async {
        let database = DatabaseService(name: "memory-cluster-\(UUID().uuidString).sqlite3")
        let store = KeyValueStore(suiteName: "tests.memory.cluster.\(UUID().uuidString)")
        defer {
            store.removeAll()
            _ = database.deleteDatabase()
        }

        let memoryService = MemoryService(database: database)
        memoryService.clearAllMemories()
        let scheduler = MemoryConsolidationScheduler(memoryService: memoryService, keyValueStore: store)

        let fact = MemoryEntry(
            content: "User fact: neural network training for finance forecasting",
            keywords: ["neural", "finance", "forecasting"],
            category: .fact,
            importance: 4,
            source: .conversation
        )
        let skill = MemoryEntry(
            content: "User skill: neural network training for finance forecasting in production",
            keywords: ["training", "finance", "production"],
            category: .skill,
            importance: 4,
            source: .conversation
        )

        memoryService.addMemory(fact)
        memoryService.addMemory(skill)
        #expect(memoryService.vectorStore.upsertWithVector(id: fact.id, vector: [1, 0, 0]))
        #expect(memoryService.vectorStore.upsertWithVector(id: skill.id, vector: [1, 0, 0]))

        await scheduler.performConsolidation()

        let links = memoryService.associativeLinks.filter {
            ($0.sourceId == fact.id && $0.targetId == skill.id) ||
            ($0.sourceId == skill.id && $0.targetId == fact.id)
        }

        #expect(memoryService.memories.count == 2)
        #expect(links.count == 1)
        #expect(links.first?.type == .semantic)
        #expect((links.first?.strength ?? 0) >= 0.9)
        #expect((links.first?.reinforcements ?? 0) >= 1)
    }

    @Test
    func kvCacheArena_reusesEvictedPagesAndProtectsPrefixPagesUnderPressure() async {
        let arena = KVCacheArena(
            pageSize: 128,
            layerCount: 1024,
            memoryBudgetMB: 2,
            evictionPolicy: .lruWithPrefixProtection
        )

        let prefixPage = await arena.allocatePage(tokenStart: 0, tokenCount: 128)
        let middlePage = await arena.allocatePage(tokenStart: 128, tokenCount: 128)
        let replacementPage = await arena.allocatePage(tokenStart: 256, tokenCount: 128)

        let prefixState = await arena.getPage(prefixPage.id)
        let recycledState = await arena.getPage(middlePage.id)
        let stats = await arena.statistics

        #expect(prefixPage.id != replacementPage.id)
        #expect(replacementPage.id == middlePage.id)
        #expect(prefixState?.isActive == true)
        #expect(prefixState?.tokenStart == 0)
        #expect(recycledState?.tokenStart == 256)
        #expect(recycledState?.isActive == true)
        #expect(stats.activePages == 2)
        #expect(stats.totalPages == 2)
        #expect(stats.freePages == 0)
        #expect(stats.budgetUtilization <= 1.0)
    }
}
