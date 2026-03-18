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

private final class MockLogitsRunner: LogitsPredicting {
    let spanLogits: [[Float]]
    private var cursor: Int = 0

    init(spanLogits: [[Float]]) {
        self.spanLogits = spanLogits
    }

    func predictLogits(inputIDs: [Int]) throws -> [Float] {
        defer { cursor += 1 }
        return spanLogits[min(cursor, max(spanLogits.count - 1, 0))]
    }

    func predictLogitsSpan(inputIDs: [Int]) throws -> [[Float]] {
        Array(spanLogits.prefix(inputIDs.count))
    }
}

extension NeuralEngineTests {
    @Test func verifySpeculativeTokens_rejectsAtBoundaryAndSamplesCorrectionFromResidual() throws {
        let decodeEngine = DecodeEngine()
        let sampler = Sampler(config: SamplingConfig(
            temperature: 1.0,
            topK: 16,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 16,
            stopSequences: [],
            samplerSeed: 7
        ))

        let draft = DraftEngine.DraftSequence(
            tokens: [1, 2, 3],
            logitSnapshots: [
                [0, 4, -4, -4],
                [0, -4, 4, -4],
                [0, -4, -4, 4]
            ],
            confidenceScores: [0.9, 0.9, 0.9],
            draftTokenProbabilities: [0.5, 0.9, 0.9],
            draftLatencyMS: 1
        )

        let runner = MockLogitsRunner(spanLogits: [
            [-4, 5, -4, -4],
            [5, -4, -10, -10],
            [-4, -4, -4, 5]
        ])

        let verification = try decodeEngine.verifySpeculativeTokens(
            draftSequence: draft,
            runner: runner,
            sampler: sampler,
            recentTokens: []
        )

        #expect(verification.accepted == [1])
        #expect(verification.rejected == [2, 3])
        #expect(verification.correctionToken == 0)
        #expect(verification.correctionSampled)
    }

    @Test func verifySpeculativeTokens_acceptsEntirePrefixWhenAllAccepted() throws {
        let decodeEngine = DecodeEngine()
        let sampler = Sampler(config: SamplingConfig(
            temperature: 1.0,
            topK: 16,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 16,
            stopSequences: [],
            samplerSeed: 9
        ))

        let draft = DraftEngine.DraftSequence(
            tokens: [1, 2],
            logitSnapshots: [
                [0, 4, -4],
                [0, -4, 4]
            ],
            confidenceScores: [0.8, 0.8],
            draftTokenProbabilities: [0.4, 0.3],
            draftLatencyMS: 1
        )

        let runner = MockLogitsRunner(spanLogits: [
            [0, 5, -5],
            [0, -5, 5]
        ])

        let verification = try decodeEngine.verifySpeculativeTokens(
            draftSequence: draft,
            runner: runner,
            sampler: sampler,
            recentTokens: []
        )

        #expect(verification.accepted == [1, 2])
        #expect(verification.rejected.isEmpty)
        #expect(verification.correctionToken == nil)
        #expect(!verification.correctionSampled)
    }

    @Test func verifySpeculativeTokens_spanMatchesSingleTokenBaseline() throws {
        let decodeEngine = DecodeEngine()
        let sampler = Sampler(config: SamplingConfig(
            temperature: 1.0,
            topK: 16,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 16,
            stopSequences: [],
            samplerSeed: 11
        ))

        let draft = DraftEngine.DraftSequence(
            tokens: [1, 2, 3],
            logitSnapshots: [
                [0, 4, -4, -4],
                [0, -4, 4, -4],
                [0, -4, -4, 4]
            ],
            confidenceScores: [0.9, 0.9, 0.9],
            draftTokenProbabilities: [0.3, 0.3, 0.3],
            draftLatencyMS: 1
        )

        let logits: [[Float]] = [
            [0, 5, -5, -5],
            [0, -5, 5, -5],
            [0, -5, -5, 5]
        ]

        let spanVerification = try decodeEngine.verifySpeculativeTokensSpan(
            draftSequence: draft,
            runner: MockLogitsRunner(spanLogits: logits),
            sampler: sampler,
            recentTokens: []
        )

        let baselineVerification = try decodeEngine.verifySpeculativeTokensBaseline(
            draftSequence: draft,
            runner: MockLogitsRunner(spanLogits: logits),
            sampler: sampler,
            recentTokens: []
        )

        #expect(spanVerification.accepted == baselineVerification.accepted)
        #expect(spanVerification.rejected == baselineVerification.rejected)
        #expect(spanVerification.correctionToken == baselineVerification.correctionToken)
        #expect(spanVerification.mismatchIndex == baselineVerification.mismatchIndex)
    }

    @Test func verifySpeculativeTokens_throwsWhenSpanLengthDoesNotMatchDraftLength() {
        let decodeEngine = DecodeEngine()
        let sampler = Sampler(config: SamplingConfig(
            temperature: 1.0,
            topK: 16,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 16,
            stopSequences: [],
            samplerSeed: 5
        ))

        let draft = DraftEngine.DraftSequence(
            tokens: [1, 2, 3],
            logitSnapshots: [[0, 4, -4], [0, -4, 4], [0, -4, -4]],
            confidenceScores: [0.9, 0.9, 0.9],
            draftTokenProbabilities: [0.5, 0.5, 0.5],
            draftLatencyMS: 1
        )

        #expect(throws: DecodeError.self) {
            _ = try decodeEngine.verifySpeculativeTokensSpan(
                draftSequence: draft,
                runner: MockLogitsRunner(spanLogits: [[0, 5, -5]]),
                sampler: sampler,
                recentTokens: []
            )
        }
    }

    @Test func verifySpeculativeTokens_rejectsWhenDraftProbabilityIsZero() throws {
        let decodeEngine = DecodeEngine()
        let sampler = Sampler(config: SamplingConfig(
            temperature: 1.0,
            topK: 16,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 16,
            stopSequences: [],
            samplerSeed: 42
        ))

        let draft = DraftEngine.DraftSequence(
            tokens: [1],
            logitSnapshots: [[0, 5, -5]],
            confidenceScores: [0.9],
            draftTokenProbabilities: [0],
            draftLatencyMS: 1
        )

        let runner = MockLogitsRunner(spanLogits: [[0, 5, -5]])

        let verification = try decodeEngine.verifySpeculativeTokens(
            draftSequence: draft,
            runner: runner,
            sampler: sampler,
            recentTokens: []
        )

        #expect(verification.accepted.isEmpty)
        #expect(verification.rejected == [1])
        #expect(verification.correctionSampled)
    }

    @Test func speculationPolicy_disablesSpeculationAfterRepeatedPoorSpanEfficiency() {
        var policy = SpeculationPolicy()

        for _ in 0..<4 {
            policy.recordVerification(
                draftCount: 6,
                acceptedCount: 0,
                rejectedCount: 6,
                correctionCount: 1,
                draftLatencyMS: 12,
                verifyLatencyMS: 6,
                committedCount: 1,
                mismatchIndex: 0
            )
        }

        #expect(!policy.shouldUseSpeculation)
        #expect(policy.k <= 2)
    }

    @Test func speculationPolicyVerificationMetrics_handlesZeroCountsSafely() {
        let metrics = SpeculationPolicy.VerificationMetrics.from(
            draftCount: 0,
            acceptedCount: 0,
            draftLatencyMS: 12,
            verifyLatencyMS: 6,
            committedCount: 0,
            mismatchIndex: nil
        )

        #expect(metrics.acceptanceRate == 0)
        #expect(metrics.acceptedLatencyMS == 18)
        #expect(abs(metrics.latencyEfficiency - 0.3333333333333333) < 0.000_001)
        #expect(metrics.mismatchPenalty == 0)
    }

    @Test func speculationPolicyVerificationMetrics_penalizesEarlyMismatch() {
        let metrics = SpeculationPolicy.VerificationMetrics.from(
            draftCount: 6,
            acceptedCount: 2,
            draftLatencyMS: 12,
            verifyLatencyMS: 6,
            committedCount: 3,
            mismatchIndex: 0
        )

        #expect(abs(metrics.acceptanceRate - (2.0 / 6.0)) < 0.000_001)
        #expect(metrics.acceptedLatencyMS == 6)
        #expect(metrics.mismatchPenalty == 1)
        #expect(abs(metrics.latencyEfficiency - 0.1875) < 0.000_001)
    }

}

private final class PrefixSnapshotTestRunner {
    private var processedTokens: [Int] = []
    private var snapshots: [UUID: [Int]] = [:]

    func resetState() {
        processedTokens.removeAll()
    }

    func prefill(_ tokens: [Int]) {
        processedTokens.append(contentsOf: tokens)
    }

    func exportPrefillState(for prefixTokens: [Int]) -> PrefixStateSnapshotAvailability {
        guard processedTokens == prefixTokens else {
            return .unavailable(reason: "prefix not currently loaded")
        }
        let handleID = UUID()
        snapshots[handleID] = processedTokens
        return .available(.runnerOwned(handleID: handleID, createdAt: Date()))
    }

    func restorePrefillState(from snapshot: PrefixStateSnapshot, expectedPrefixTokens: [Int]) -> Bool {
        switch snapshot {
        case .unavailable:
            return false
        case .runnerOwned(let handleID, _):
            guard let restored = snapshots[handleID], restored == expectedPrefixTokens else {
                return false
            }
            processedTokens = restored
            return true
        }
    }

    func nextTokenProbabilities(for completionTokens: [Int]) -> [Double] {
        let combined = processedTokens + completionTokens
        let seed = combined.enumerated().reduce(0) { partial, item in
            partial + ((item.offset + 1) * item.element)
        }
        return (0..<6).map { index in
            Double(seed + ((index + 1) * 7)) / 100.0
        }
    }
}

extension NeuralEngineTests {
    @Test func promptPrefixCache_restoredSnapshotMatchesColdPrefillProbabilities() async throws {
        let cache = PromptPrefixCache()
        let runner = PrefixSnapshotTestRunner()
        let systemTokens = [11, 22, 33]
        let promptTokens = systemTokens + [44, 55]
        let prefixKey = PromptPrefixKey(modelID: "test-model", tokenizerID: "test-tokenizer", prefix: systemTokens)

        runner.resetState()
        runner.prefill(promptTokens)
        let coldProbabilities = runner.nextTokenProbabilities(for: [])

        runner.resetState()
        runner.prefill(systemTokens)
        let snapshot: PrefixStateSnapshot
        switch runner.exportPrefillState(for: systemTokens) {
        case .available(let exportedSnapshot):
            snapshot = exportedSnapshot
        case .unavailable(let reason):
            Issue.record("Expected snapshot export to succeed: \(reason)")
            return
        }

        await cache.store(prefix: CachedPrefix(
            key: prefixKey,
            tokenizedPrefix: systemTokens,
            pageCount: 1,
            sequencePosition: systemTokens.count,
            stateSnapshot: snapshot,
            timestamp: Date()
        ))

        guard let cached = await cache.lookup(key: prefixKey) else {
            Issue.record("Expected cached prefix")
            return
        }

        #expect(runner.restorePrefillState(from: cached.stateSnapshot, expectedPrefixTokens: systemTokens))
        runner.prefill(Array(promptTokens.dropFirst(systemTokens.count)))
        let restoredProbabilities = runner.nextTokenProbabilities(for: [])

        #expect(restoredProbabilities == coldProbabilities)
    }

    @Test func promptPrefixCache_unavailableSnapshotFallsBackToColdPrefillProbabilities() async throws {
        let cache = PromptPrefixCache()
        let runner = PrefixSnapshotTestRunner()
        let systemTokens = [5, 6, 7]
        let promptTokens = systemTokens + [8, 9]
        let prefixKey = PromptPrefixKey(modelID: "test-model", tokenizerID: "test-tokenizer", prefix: systemTokens)

        runner.resetState()
        runner.prefill(promptTokens)
        let coldProbabilities = runner.nextTokenProbabilities(for: [])

        await cache.store(prefix: CachedPrefix(
            key: prefixKey,
            tokenizedPrefix: systemTokens,
            pageCount: 1,
            sequencePosition: systemTokens.count,
            stateSnapshot: .unavailable(reason: "snapshotting disabled"),
            timestamp: Date()
        ))

        guard let cached = await cache.lookup(key: prefixKey) else {
            Issue.record("Expected cached prefix")
            return
        }

        #expect(!runner.restorePrefillState(from: cached.stateSnapshot, expectedPrefixTokens: systemTokens))
        runner.resetState()
        runner.prefill(promptTokens)
        let fallbackProbabilities = runner.nextTokenProbabilities(for: [])

        #expect(fallbackProbabilities == coldProbabilities)
    }
}
