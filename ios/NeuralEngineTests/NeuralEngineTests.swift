import Testing
@testable import NeuralEngine

struct NeuralEngineTests {

    @Test func allocatePages_reusesAllPagesForExactPrefixHit() async throws {
        let manager = KVCacheManager(pageSize: 4, layerCount: 2, memoryBudgetMB: 8)

        let firstSequence = await manager.beginSequence()
        let firstPages = await manager.allocatePages(sequenceID: firstSequence, tokens: [1, 2, 3, 4, 5, 6, 7, 8], startPosition: 0)

        let secondSequence = await manager.beginSequence()
        let secondPages = await manager.allocatePages(sequenceID: secondSequence, tokens: [1, 2, 3, 4, 5, 6, 7, 8], startPosition: 0)

        #expect(firstPages.map(\.id) == secondPages.map(\.id))

        let mappings = await manager.debugPageMappings(sequenceID: secondSequence)
        #expect(mappings.count == 2)
        #expect(mappings.allSatisfy { $0.origin == .sharedPrefix })

        for page in firstPages {
            let refcount = await manager.debugReferenceCount(pageID: page.id)
            #expect(refcount == 2)
        }
    }

    @Test func allocatePages_reusesLeadingPagesForPartialPrefixHit() async throws {
        let manager = KVCacheManager(pageSize: 4, layerCount: 2, memoryBudgetMB: 8)

        let firstSequence = await manager.beginSequence()
        let firstPages = await manager.allocatePages(sequenceID: firstSequence, tokens: [1, 2, 3, 4, 5, 6, 7, 8], startPosition: 0)

        let secondSequence = await manager.beginSequence()
        let secondPages = await manager.allocatePages(sequenceID: secondSequence, tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], startPosition: 0)

        #expect(secondPages.count == 3)
        #expect(secondPages[0].id == firstPages[0].id)
        #expect(secondPages[1].id == firstPages[1].id)
        #expect(secondPages[2].id != firstPages[1].id)

        let mappings = await manager.debugPageMappings(sequenceID: secondSequence)
        #expect(mappings.map(\.origin) == [.sharedPrefix, .sharedPrefix, .materialized])

        #expect(await manager.debugReferenceCount(pageID: firstPages[0].id) == 2)
        #expect(await manager.debugReferenceCount(pageID: firstPages[1].id) == 2)
        #expect(await manager.debugReferenceCount(pageID: secondPages[2].id) == 1)
    }

    @Test func allocatePages_materializesPagesForNoHit() async throws {
        let manager = KVCacheManager(pageSize: 4, layerCount: 2, memoryBudgetMB: 8)

        let firstSequence = await manager.beginSequence()
        let firstPages = await manager.allocatePages(sequenceID: firstSequence, tokens: [1, 2, 3, 4, 5, 6, 7, 8], startPosition: 0)

        let secondSequence = await manager.beginSequence()
        let secondPages = await manager.allocatePages(sequenceID: secondSequence, tokens: [9, 10, 11, 12, 13, 14, 15, 16], startPosition: 0)

        #expect(Set(firstPages.map(\.id)).isDisjoint(with: Set(secondPages.map(\.id))))

        let mappings = await manager.debugPageMappings(sequenceID: secondSequence)
        #expect(mappings.allSatisfy { $0.origin == .materialized })

        for page in firstPages + secondPages {
            #expect(await manager.debugReferenceCount(pageID: page.id) == 1)
        }
    }
}
