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


    @Test func slidingWindowEvict_noEvictionWhenTotalPagesWithinKeepBudget() async throws {
        let manager = KVCacheManager(pageSize: 4, layerCount: 2, memoryBudgetMB: 8, slidingWindowSize: 8, evictionThreshold: 7)
        let sequenceID = await manager.beginSequence()
        let allocated = await manager.allocatePages(sequenceID: sequenceID, tokens: Array(1...12), startPosition: 0)

        let result = await manager.slidingWindowEvict(sequenceID: sequenceID, systemTokenCount: 4, currentLength: 12)

        #expect(result.evictedTokens == 0)
        #expect(result.pagesFreed == 0)
        #expect(result.newLength == 12)

        let remainingMappings = await manager.debugPageMappings(sequenceID: sequenceID)
        #expect(remainingMappings.count == 3)
        #expect(remainingMappings.map(\.pageID) == allocated.map(\.id))
    }

    @Test func slidingWindowEvict_exactThresholdCrossingEvictsUsingExactTokenSpans() async throws {
        let manager = KVCacheManager(pageSize: 4, layerCount: 2, memoryBudgetMB: 8, slidingWindowSize: 0, evictionThreshold: 11)
        let sequenceID = await manager.beginSequence()
        _ = await manager.allocatePages(sequenceID: sequenceID, tokens: Array(1...13), startPosition: 0)

        let atThreshold = await manager.slidingWindowEvict(sequenceID: sequenceID, systemTokenCount: 0, currentLength: 11)
        #expect(atThreshold.evictedTokens == 0)
        #expect(atThreshold.pagesFreed == 0)

        let afterCrossing = await manager.slidingWindowEvict(sequenceID: sequenceID, systemTokenCount: 0, currentLength: 13)
        #expect(afterCrossing.pagesFreed == 3)
        #expect(afterCrossing.evictedTokens == 9)
        #expect(afterCrossing.newLength == 4)

        let remainingMappings = await manager.debugPageMappings(sequenceID: sequenceID)
        #expect(remainingMappings.count == 1)
        #expect(remainingMappings[0].tokenStart == 0)
        #expect(remainingMappings[0].tokenEnd == 4)
    }

    @Test func slidingWindowEvict_largePrefixWithSmallTailRemovesMiddleOnly() async throws {
        let manager = KVCacheManager(pageSize: 4, layerCount: 2, memoryBudgetMB: 8, slidingWindowSize: 4, evictionThreshold: 20)
        let sequenceID = await manager.beginSequence()
        let allocated = await manager.allocatePages(sequenceID: sequenceID, tokens: Array(1...24), startPosition: 0)

        let result = await manager.slidingWindowEvict(sequenceID: sequenceID, systemTokenCount: 12, currentLength: 24)

        #expect(result.pagesFreed == 2)
        #expect(result.evictedTokens == 8)
        #expect(result.newLength == 16)

        let remainingMappings = await manager.debugPageMappings(sequenceID: sequenceID)
        #expect(remainingMappings.count == 4)
        #expect(remainingMappings.map(\.pageID) == [allocated[0].id, allocated[1].id, allocated[2].id, allocated[5].id])
    }

}
