import Foundation
import UIKit
import CoreML
import LlamaSwift

nonisolated final class LlamaModelRunner: DraftLogitsPredicting, @unchecked Sendable {
    private struct StoredPrefixStateRecord {
        let snapshot: LlamaSessionSnapshot
        let metadata: PrefixStateSnapshotMetadata
    }

    private struct GenerationState {
        let context: OpaquePointer
        let vocab: OpaquePointer
        let vocabSize: Int
    }

    private struct SpeculativeStepResult {
        let committedTokens: [Int]
        let acceptedCount: Int
        let rejectedCount: Int
        let hitEOS: Bool
        let nextTargetLogits: [Float]
        let nextDraftLogits: [Float]?
        let draftLatencyMS: Double
        let verificationLatencyMS: Double
        let firstTokenTimeMS: Double?
    }

    // MARK: – Locking
    //
    // All mutable state is protected by stateCondition (an NSCondition, which
    // is also a mutex). Two extra fields manage the generation-drain protocol:
    //
    //   activeGenerationCount – number of "generation regions" currently in
    //     flight that hold raw C pointers (context / model). Any path that
    //     calls llama_decode / llama_get_logits_ith etc. must hold a generation
    //     token for the duration.
    //
    //   pendingUnload – set to true by unload()/loadModel() BEFORE they wait
    //     for the drain. It makes tryAcquireGenerationToken() return false so
    //     no new generation can start, and it is checked cooperatively inside
    //     the generation loop so the active generator exits quickly.
    //
    // Invariants:
    //   • stateCondition is NEVER held while calling any llama_* C function.
    //   • llama_free / llama_model_free are ONLY called after
    //     activeGenerationCount has reached 0.
    //   • pendingUnload is cleared only after the new model is fully loaded
    //     (or after freeNativeResources() for a plain unload).

    private let stateCondition = NSCondition()

    private var activeGenerationCount: Int = 0
    private var pendingUnload: Bool = false

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var state: RunnerState = .idle
    private var isBackgrounded: Bool = false
    private var backgroundObserver: NSObjectProtocol?
    private var foregroundObserver: NSObjectProtocol?
    private var lastContextPath: String?
    private var lastNCtx: Int32 = 2048
    private var lastNGPULayers: Int32 = 99
    private var backgroundTimestamp: Date?
    private var sessionTokenCount: Int = 0
    private var tokenHistory: [Int] = []
    private var nBatch: Int32 = 512
    private var consecutiveFailures: Int = 0
    private var lastSuccessfulPrediction: Date?
    private var totalRecoveries: Int = 0
    private var lastRecoveryAttempt: Date?
    private let recoveryBackoffBase: TimeInterval = 0.5
    private let healthCheckInterval: TimeInterval = 30
    private var activeComputeUnits: MLComputeUnits = .cpuAndGPU
    private var prefixSnapshotStore: [UUID: StoredPrefixStateRecord] = [:]
    private var prefixSnapshotModelID: String?
    private var prefixSnapshotTokenizerID: String = "unknown-tokenizer"
    private var modelSessionID: UUID = UUID()

    // MARK: – State-condition helpers

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        return try body()
    }

    // MARK: – Generation-token protocol

    /// Increment the generation reference count, preventing any concurrent
    /// unload from freeing native pointers while we hold a token.
    /// Returns false if an unload is already pending (caller must not proceed).
    func tryAcquireGenerationToken() -> Bool {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        guard !pendingUnload else { return false }
        activeGenerationCount += 1
        return true
    }

    /// Decrement the generation reference count.  Broadcasts on the condition
    /// so any thread blocked in waitForGenerationDrainLocked() wakes up.
    func releaseGenerationToken() {
        stateCondition.lock()
        activeGenerationCount -= 1
        if activeGenerationCount == 0 {
            stateCondition.broadcast()
        }
        stateCondition.unlock()
    }

    /// Must be called while stateCondition IS locked.
    /// Sets pendingUnload = true and blocks until activeGenerationCount == 0.
    private func waitForGenerationDrainLocked() {
        pendingUnload = true
        let deadline = Date(timeIntervalSinceNow: 10.0)
        while activeGenerationCount > 0 {
            stateCondition.wait(until: deadline)
        }
    }

    /// Checks whether an external cancellation (unload / reload) has been
    /// requested.  Must be called without holding stateCondition.
    private func isCancellationPending() -> Bool {
        stateCondition.lock()
        let pending = pendingUnload
        stateCondition.unlock()
        return pending
    }

    // MARK: – Public API

    var isLoaded: Bool {
        withLock {
            model != nil && context != nil && (state == .ready || state == .recovering)
        }
    }

    var currentState: RunnerState {
        withLock { state }
    }

    var currentPrefixSnapshotModelID: String? {
        withLock { prefixSnapshotModelID ?? lastContextPath }
    }

    var recoveryCount: Int {
        withLock { totalRecoveries }
    }

    func configurePrefixSnapshotContext(modelID: String, tokenizerID: String) {
        withLock {
            prefixSnapshotModelID = modelID
            prefixSnapshotTokenizerID = tokenizerID
        }
    }

    func loadModel(at path: String, nCtx: Int32 = 2048, nGPULayers: Int32 = 99) throws {
        let gpuLayers: Int32
#if targetEnvironment(simulator)
        gpuLayers = 0
#else
        gpuLayers = nGPULayers
#endif

        // Drain any in-flight generation before we free the old context.
        stateCondition.lock()
        waitForGenerationDrainLocked()
        stateCondition.unlock()

        stateCondition.lock()
        defer { stateCondition.unlock() }

        freeNativeResources()
        state = .loading

        var modelParameters = llama_model_default_params()
        modelParameters.n_gpu_layers = gpuLayers

        guard let loadedModel = llama_model_load_from_file(path, modelParameters) else {
            state = .idle
            pendingUnload = false
            throw LlamaRunnerError.modelLoadFailed
        }

        let batchSize = min(max(nCtx, 1), 512)
        var contextParameters = llama_context_default_params()
        contextParameters.n_ctx = UInt32(nCtx)
        contextParameters.n_batch = UInt32(batchSize)
        let threadCount = max(ProcessInfo.processInfo.activeProcessorCount - 2, 1)
        contextParameters.n_threads = Int32(threadCount)
        contextParameters.n_threads_batch = Int32(threadCount)

        guard let loadedContext = llama_init_from_model(loadedModel, contextParameters) else {
            llama_model_free(loadedModel)
            state = .idle
            pendingUnload = false
            throw LlamaRunnerError.contextCreationFailed
        }

        model = loadedModel
        context = loadedContext
        nBatch = batchSize
        lastContextPath = path
        lastNCtx = nCtx
        lastNGPULayers = gpuLayers
        sessionTokenCount = 0
        tokenHistory.removeAll(keepingCapacity: true)
        consecutiveFailures = 0
        lastSuccessfulPrediction = Date()
        lastRecoveryAttempt = nil
        prefixSnapshotStore.removeAll()
        modelSessionID = UUID()
        activeComputeUnits = gpuLayers > 0 ? .cpuAndGPU : .cpuOnly
        state = .ready

        // Clear the pending flag now that a new model is ready.
        pendingUnload = false

        setupLifecycleObservers()
    }

    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int32,
        topP: Float,
        repetitionPenalty: Float,
        onToken: @escaping (String) -> Void,
        shouldStop: @escaping () -> Bool
    ) throws -> LlamaGenerationResult {
        try generateWithDraft(
            prompt: prompt,
            samplingConfig: SamplingConfig(
                temperature: temperature,
                topK: Int(topK),
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens
            ),
            draftRunner: nil,
            draftCount: 0,
            onToken: onToken,
            shouldStop: shouldStop
        )
    }

    func generateWithDraft(
        prompt: String,
        samplingConfig: SamplingConfig,
        draftRunner: LlamaModelRunner?,
        draftCount: Int,
        onToken: @escaping (String) -> Void,
        shouldStop: @escaping () -> Bool
    ) throws -> LlamaGenerationResult {

        // ── Acquire generation tokens ──────────────────────────────────────
        // We must increment the reference count BEFORE capturing any raw C
        // pointer.  This prevents a concurrent unload from freeing the context
        // while we hold it.

        guard tryAcquireGenerationToken() else {
            throw LlamaRunnerError.invalidState(currentState)
        }
        defer { releaseGenerationToken() }

        // Acquire a token for the draft runner if it is healthy.  If it can't
        // be acquired (e.g. mid-deactivation) we fall back to non-speculative.
        let hasDraftToken: Bool
        let effectiveDraftRunner: LlamaModelRunner?
        if let dr = draftRunner, draftCount > 0, dr.tryAcquireGenerationToken() {
            hasDraftToken = true
            effectiveDraftRunner = dr
        } else {
            hasDraftToken = false
            effectiveDraftRunner = nil
        }
        defer {
            if hasDraftToken { draftRunner?.releaseGenerationToken() }
        }

        // ── Snapshot raw pointers under the lock ──────────────────────────
        // After this point we own both tokens so the pointers cannot be freed.
        let generationState = try captureGenerationState()
        let promptTokens = tokenize(vocab: generationState.vocab, text: prompt, addBOS: true)
        guard !promptTokens.isEmpty else {
            throw LlamaRunnerError.tokenizationFailed
        }

        do {
            let memory = llama_get_memory(generationState.context)
            llama_memory_clear(memory, true)

            let prefillStart = Date()
            try decodeTokens(promptTokens, context: generationState.context)
            var currentLogits = try readCurrentLogits(context: generationState.context, vocabSize: generationState.vocabSize)
            let prefillDuration = Date().timeIntervalSince(prefillStart)
            let prefillTPS = Double(promptTokens.count) / max(prefillDuration, 0.001)

            var currentDraftLogits: [Float]? = nil
            if let dr = effectiveDraftRunner {
                try dr.prime(with: promptTokens)
                currentDraftLogits = try dr.captureCurrentLogits()
            }

            let sampler = Sampler(config: samplingConfig)
            var allTokens = promptTokens
            var generatedCount = 0
            var acceptedSpeculativeTokens = 0
            var rejectedSpeculativeTokens = 0
            var cumulativeDraftLatencyMS: Double = 0
            var cumulativeVerificationLatencyMS: Double = 0
            var verificationCount = 0
            var firstTokenTimeMS: Double = 0
            let decodeStart = Date()

            while generatedCount < samplingConfig.maxTokens && !shouldStop() {

                // Cooperative cancellation: exit the loop if an unload /
                // reload has been requested so the drain can proceed quickly.
                if isCancellationPending() { break }

                if let dr = effectiveDraftRunner,
                   let draftLogits = currentDraftLogits {
                    let maxDraft = min(draftCount, samplingConfig.maxTokens - generatedCount)
                    if maxDraft > 0 {
                        let speculative = try performSpeculativeStep(
                            sampler: sampler,
                            draftRunner: dr,
                            draftLogits: draftLogits,
                            targetContext: generationState.context,
                            targetVocab: generationState.vocab,
                            targetVocabSize: generationState.vocabSize,
                            allTokens: allTokens,
                            targetCurrentLogits: currentLogits,
                            maxDraft: maxDraft,
                            prefillStart: prefillStart,
                            generatedCount: generatedCount,
                            onToken: onToken
                        )

                        if speculative.acceptedCount + speculative.rejectedCount > 0 {
                            verificationCount += 1
                            acceptedSpeculativeTokens += speculative.acceptedCount
                            rejectedSpeculativeTokens += speculative.rejectedCount
                            cumulativeDraftLatencyMS += speculative.draftLatencyMS
                            cumulativeVerificationLatencyMS += speculative.verificationLatencyMS
                        }

                        if firstTokenTimeMS == 0, let t = speculative.firstTokenTimeMS {
                            firstTokenTimeMS = t
                        }

                        currentLogits = speculative.nextTargetLogits
                        currentDraftLogits = speculative.nextDraftLogits

                        if !speculative.committedTokens.isEmpty {
                            allTokens.append(contentsOf: speculative.committedTokens)
                            generatedCount += speculative.committedTokens.count
                            if speculative.hitEOS || generatedCount >= samplingConfig.maxTokens {
                                break
                            }
                            continue
                        }

                        if speculative.hitEOS { break }
                    }
                }

                let recentTokens = Array(allTokens.suffix(64))
                let sampledToken = sampler.sample(logits: currentLogits, recentTokens: recentTokens)
                if llama_vocab_is_eog(generationState.vocab, llama_token(sampledToken)) {
                    break
                }

                if generatedCount == 0 {
                    firstTokenTimeMS = Date().timeIntervalSince(prefillStart) * 1000
                }

                let piece = tokenToPiece(vocab: generationState.vocab, token: llama_token(sampledToken))
                onToken(piece)

                allTokens.append(sampledToken)
                generatedCount += 1
                try decodeTokens([sampledToken], context: generationState.context)
                currentLogits = try readCurrentLogits(context: generationState.context, vocabSize: generationState.vocabSize)

                if let dr = effectiveDraftRunner {
                    let draftState = try dr.captureGenerationState()
                    try dr.decodeTokens([sampledToken], context: draftState.context)
                    currentDraftLogits = try dr.readCurrentLogits(context: draftState.context, vocabSize: draftState.vocabSize)
                }
            }

            let totalDuration = Date().timeIntervalSince(prefillStart)
            let decodeDuration = Date().timeIntervalSince(decodeStart)
            let decodeTPS = Double(generatedCount) / max(decodeDuration, 0.001)

            withLock {
                sessionTokenCount = allTokens.count
                tokenHistory = allTokens
                lastSuccessfulPrediction = Date()
                consecutiveFailures = 0
                state = .ready
            }

            return LlamaGenerationResult(
                promptTokenCount: promptTokens.count,
                generatedTokenCount: generatedCount,
                prefillTokensPerSecond: prefillTPS,
                decodeTokensPerSecond: decodeTPS,
                timeToFirstTokenMS: firstTokenTimeMS,
                totalDuration: totalDuration,
                acceptedSpeculativeTokens: acceptedSpeculativeTokens,
                rejectedSpeculativeTokens: rejectedSpeculativeTokens,
                speculativeDraftLatencyMS: verificationCount > 0 ? cumulativeDraftLatencyMS / Double(verificationCount) : 0,
                speculativeVerifyLatencyMS: verificationCount > 0 ? cumulativeVerificationLatencyMS / Double(verificationCount) : 0,
                speculativeVerificationCount: verificationCount
            )
        } catch {
            try recordPredictionFailure(error)
        }
    }

    func predictLogits(inputIDs: [Int]) throws -> [Float] {
        let span = try predictLogitsSpan(inputIDs: inputIDs)
        return span.last ?? []
    }

    func predictLogitsSpan(inputIDs: [Int]) throws -> [[Float]] {
        guard !inputIDs.isEmpty else {
            throw LlamaRunnerError.emptyInput
        }

        guard tryAcquireGenerationToken() else {
            throw LlamaRunnerError.invalidState(currentState)
        }
        defer { releaseGenerationToken() }

        let generationState = try captureGenerationState()

        do {
            let logits = try decodeAndCollectLogits(
                inputIDs,
                context: generationState.context,
                vocabSize: generationState.vocabSize
            )
            withLock {
                sessionTokenCount += inputIDs.count
                tokenHistory.append(contentsOf: inputIDs)
                lastSuccessfulPrediction = Date()
                consecutiveFailures = 0
                state = .ready
            }
            return logits
        } catch {
            try recordPredictionFailure(error)
        }
    }

    func resetContext() {
        // If generation is active, skip: generateWithDraft clears the KV cache
        // at the very start of every run, so skipping here is safe.
        stateCondition.lock()
        let generationActive = activeGenerationCount > 0
        stateCondition.unlock()
        if generationActive { return }

        withLock {
            guard let context else {
                sessionTokenCount = 0
                tokenHistory.removeAll(keepingCapacity: true)
                state = .idle
                return
            }

            let memory = llama_get_memory(context)
            llama_memory_clear(memory, true)
            sessionTokenCount = 0
            tokenHistory.removeAll(keepingCapacity: true)
            lastSuccessfulPrediction = Date()
            consecutiveFailures = 0
            state = model != nil ? .ready : .idle
        }
    }

    func resetState() {
        resetContext()
    }

    func saveState() -> LlamaSessionSnapshot? {
        withLock {
            guard let context, model != nil, state == .ready || state == .recovering else {
                return nil
            }

            let requiredSize = Int(llama_state_get_size(context))
            guard requiredSize > 0 else { return nil }

            var bytes = [UInt8](repeating: 0, count: requiredSize)
            let written = bytes.withUnsafeMutableBufferPointer { pointer -> Int in
                guard let baseAddress = pointer.baseAddress else { return 0 }
                return Int(llama_state_get_data(context, baseAddress, pointer.count))
            }
            guard written > 0 else { return nil }

            return LlamaSessionSnapshot(
                modelPath: lastContextPath ?? "",
                nCtx: lastNCtx,
                nGPULayers: lastNGPULayers,
                sessionTokenCount: sessionTokenCount,
                tokenHistory: tokenHistory,
                serializedState: Data(bytes.prefix(written)),
                modelID: prefixSnapshotModelID ?? lastContextPath ?? "unknown-model",
                tokenizerID: prefixSnapshotTokenizerID,
                modelSessionID: modelSessionID,
                timestamp: Date()
            )
        }
    }

    func restoreFromSnapshot(_ snapshot: LlamaSessionSnapshot) throws {
        guard snapshot.isValid else {
            throw LlamaRunnerError.snapshotExpired
        }

        let currentPath = withLock { lastContextPath }
        if currentPath != snapshot.modelPath || !isLoaded {
            try loadModel(at: snapshot.modelPath, nCtx: snapshot.nCtx, nGPULayers: snapshot.nGPULayers)
        }

        try withLock {
            guard let context, model != nil else {
                throw LlamaRunnerError.modelNotLoaded
            }

            let restored = snapshot.serializedState.withUnsafeBytes { buffer -> Int in
                guard let baseAddress = buffer.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                return Int(llama_state_set_data(context, baseAddress, buffer.count))
            }
            guard restored == snapshot.serializedState.count else {
                throw LlamaRunnerError.stateRestoreFailed
            }

            sessionTokenCount = snapshot.sessionTokenCount
            tokenHistory = snapshot.tokenHistory
            prefixSnapshotModelID = snapshot.modelID
            prefixSnapshotTokenizerID = snapshot.tokenizerID
            modelSessionID = snapshot.modelSessionID
            lastSuccessfulPrediction = Date()
            consecutiveFailures = 0
            state = .ready
        }
    }

    func runZeroTokenProbe() -> ZeroTokenProbeResult {
        let start = Date()
        guard isLoaded else {
            return ZeroTokenProbeResult(passed: false, state: currentState, latencyMS: 0)
        }

        let snapshot = saveState()
        let probeToken = withLock { tokenHistory.last ?? 1 }

        do {
            _ = try predictLogits(inputIDs: [probeToken])
            if let snapshot {
                try restoreFromSnapshot(snapshot)
            } else {
                resetContext()
            }

            let latency = Date().timeIntervalSince(start) * 1000
            return ZeroTokenProbeResult(passed: true, state: .ready, latencyMS: latency)
        } catch {
            if let snapshot {
                try? restoreFromSnapshot(snapshot)
            }
            let latency = Date().timeIntervalSince(start) * 1000
            return ZeroTokenProbeResult(passed: false, state: currentState, latencyMS: latency)
        }
    }

    func healthCheck() -> HealthStatus {
        withLock {
            let staleDuration = lastSuccessfulPrediction.map { Date().timeIntervalSince($0) }
            let isStale = staleDuration.map { $0 > healthCheckInterval } ?? false
            let isHealthy = model != nil && context != nil && (state == .ready || state == .recovering) && consecutiveFailures == 0 && !isStale
            return HealthStatus(
                isHealthy: isHealthy,
                state: state,
                failures: consecutiveFailures,
                recoveries: totalRecoveries,
                computeUnits: activeComputeUnits,
                staleDuration: staleDuration
            )
        }
    }

    func attemptRecovery() async throws {
        let snapshot = saveState()
        let recoveryContext = withLock { () -> (path: String, nCtx: Int32, nGPULayers: Int32)? in
            guard let path = lastContextPath else { return nil }

            if let lastRecoveryAttempt {
                let backoff = recoveryBackoffBase * pow(2.0, Double(min(totalRecoveries, 5)))
                let elapsed = Date().timeIntervalSince(lastRecoveryAttempt)
                if elapsed < backoff { return nil }
            }

            state = .recovering
            self.lastRecoveryAttempt = Date()
            return (path, lastNCtx, lastNGPULayers)
        }

        guard let recoveryContext else {
            if withLock({ lastContextPath == nil }) {
                throw LlamaRunnerError.modelNotLoaded
            }
            return
        }

        do {
            try loadModel(
                at: recoveryContext.path,
                nCtx: recoveryContext.nCtx,
                nGPULayers: recoveryContext.nGPULayers
            )
            if let snapshot {
                try? restoreFromSnapshot(snapshot)
            }
            withLock {
                totalRecoveries += 1
                consecutiveFailures = 0
                state = .ready
            }
        } catch {
            withLock { state = .evicted }
            throw error
        }
    }

    func switchToCPUOnly() async throws {
        let snapshot = saveState()
        let reloadContext = withLock { () -> (path: String, nCtx: Int32)? in
            guard let path = lastContextPath else { return nil }
            state = .recovering
            return (path, lastNCtx)
        }

        guard let reloadContext else {
            throw LlamaRunnerError.modelNotLoaded
        }

        do {
            try loadModel(at: reloadContext.path, nCtx: reloadContext.nCtx, nGPULayers: 0)
            if let snapshot {
                try? restoreFromSnapshot(snapshot.withGPULayers(0))
            }
            withLock {
                totalRecoveries += 1
                consecutiveFailures = 0
                activeComputeUnits = .cpuOnly
                state = .ready
            }
        } catch {
            withLock { state = .evicted }
            throw error
        }
    }

    func exportPrefillState(for prefixTokens: [Int]) -> PrefixStateSnapshotAvailability {
        guard !prefixTokens.isEmpty else {
            return .unavailable(reason: "Prefix snapshot requires at least one token")
        }
        guard let snapshot = saveState() else {
            return .unavailable(reason: "GGUF state snapshot is unavailable")
        }

        let metadata = PrefixStateSnapshotMetadata(
            modelID: currentPrefixSnapshotModelID ?? snapshot.modelPath,
            tokenizerID: snapshot.tokenizerID,
            computeUnits: computeUnitsIdentifier(activeComputeUnits),
            prefixTokens: prefixTokens,
            modelSessionID: snapshot.modelSessionID
        )
        let handleID = UUID()

        withLock {
            prefixSnapshotStore[handleID] = StoredPrefixStateRecord(snapshot: snapshot, metadata: metadata)
            prunePrefixSnapshotStoreIfNeededLocked(maxEntries: 16)
        }

        return .available(.runnerOwned(
            RunnerOwnedPrefixStateSnapshot(
                handleID: handleID,
                createdAt: Date(),
                metadata: metadata
            )
        ))
    }

    func restorePrefillState(from snapshot: PrefixStateSnapshot, expectedPrefixTokens: [Int]) -> Bool {
        switch snapshot {
        case .unavailable:
            return false
        case .runnerOwned(let runnerSnapshot):
            let record = withLock { prefixSnapshotStore[runnerSnapshot.handleID] }
            guard let record else { return false }
            guard record.metadata.prefixTokens == expectedPrefixTokens else { return false }
            guard record.metadata.modelID == (currentPrefixSnapshotModelID ?? record.snapshot.modelPath) else { return false }
            guard record.metadata.tokenizerID == prefixSnapshotTokenizerID else { return false }
            guard record.metadata.computeUnits == computeUnitsIdentifier(activeComputeUnits) else { return false }
            do {
                try restoreFromSnapshot(record.snapshot)
                return true
            } catch {
                return false
            }
        }
    }

    func unload() {
        // 1. Signal pending unload and drain all in-flight generation.
        //    waitForGenerationDrainLocked sets pendingUnload = true and waits
        //    until activeGenerationCount reaches 0.
        stateCondition.lock()
        removeLifecycleObserversInternal()
        waitForGenerationDrainLocked()
        stateCondition.unlock()

        // 2. Free native resources — safe because no generation is in flight.
        stateCondition.lock()
        freeNativeResources()
        pendingUnload = false
        stateCondition.unlock()
    }

    // MARK: – Private helpers

    /// Free llama C objects and reset all derived state.
    /// Must be called while stateCondition IS locked.
    private func freeNativeResources() {
        if let context {
            llama_free(context)
        }
        if let model {
            llama_model_free(model)
        }
        context = nil
        model = nil
        sessionTokenCount = 0
        tokenHistory.removeAll(keepingCapacity: true)
        consecutiveFailures = 0
        lastSuccessfulPrediction = nil
        lastRecoveryAttempt = nil
        prefixSnapshotStore.removeAll()
        modelSessionID = UUID()
        activeComputeUnits = .cpuAndGPU
        state = .idle
    }

    private func captureGenerationState() throws -> GenerationState {
        stateCondition.lock()
        defer { stateCondition.unlock() }

        guard let model, let context else {
            throw LlamaRunnerError.modelNotLoaded
        }
        guard state == .ready || state == .recovering else {
            throw LlamaRunnerError.invalidState(state)
        }
        guard let vocab = llama_model_get_vocab(model) else {
            throw LlamaRunnerError.tokenizationFailed
        }

        return GenerationState(
            context: context,
            vocab: vocab,
            vocabSize: Int(llama_vocab_n_tokens(vocab))
        )
    }

    private func prime(with tokens: [Int]) throws {
        let generationState = try captureGenerationState()
        let memory = llama_get_memory(generationState.context)
        llama_memory_clear(memory, true)
        try decodeTokens(tokens, context: generationState.context)
        withLock {
            sessionTokenCount = tokens.count
            tokenHistory = tokens
            lastSuccessfulPrediction = Date()
            consecutiveFailures = 0
            state = .ready
        }
    }

    private func captureCurrentLogits() throws -> [Float] {
        let generationState = try captureGenerationState()
        return try readCurrentLogits(context: generationState.context, vocabSize: generationState.vocabSize)
    }

    private func performSpeculativeStep(
        sampler: Sampler,
        draftRunner: LlamaModelRunner,
        draftLogits: [Float],
        targetContext: OpaquePointer,
        targetVocab: OpaquePointer,
        targetVocabSize: Int,
        allTokens: [Int],
        targetCurrentLogits: [Float],
        maxDraft: Int,
        prefillStart: Date,
        generatedCount: Int,
        onToken: @escaping (String) -> Void
    ) throws -> SpeculativeStepResult {
        guard let targetSnapshot = saveState(), let draftSnapshot = draftRunner.saveState() else {
            return SpeculativeStepResult(
                committedTokens: [],
                acceptedCount: 0,
                rejectedCount: 0,
                hitEOS: false,
                nextTargetLogits: targetCurrentLogits,
                nextDraftLogits: draftLogits,
                draftLatencyMS: 0,
                verificationLatencyMS: 0,
                firstTokenTimeMS: nil
            )
        }

        // Snapshot draft state – draft token already has a generation token
        // held by the caller (generateWithDraft), so captureGenerationState is safe.
        let draftState = try draftRunner.captureGenerationState()
        let draftStart = Date()
        var proposedTokens: [Int] = []
        var draftLogitSnapshots: [[Float]] = []
        var draftTokenProbabilities: [Float] = []
        var rollingDraftLogits = draftLogits
        var draftHistory = allTokens

        for index in 0..<maxDraft {
            let context = Array(draftHistory.suffix(64))
            let draftDistribution = sampler.prepareDistribution(logits: rollingDraftLogits, recentTokens: context)
            let token = sampler.sample(from: draftDistribution)
            proposedTokens.append(token)
            draftLogitSnapshots.append(rollingDraftLogits)
            draftTokenProbabilities.append(draftDistribution.probability(of: token))
            draftHistory.append(token)

            if llama_vocab_is_eog(draftState.vocab, llama_token(token)) {
                break
            }

            if index < maxDraft - 1 {
                try draftRunner.decodeTokens([token], context: draftState.context)
                rollingDraftLogits = try draftRunner.readCurrentLogits(context: draftState.context, vocabSize: draftState.vocabSize)
            }
        }

        let draftLatencyMS = Date().timeIntervalSince(draftStart) * 1000
        guard !proposedTokens.isEmpty else {
            return SpeculativeStepResult(
                committedTokens: [],
                acceptedCount: 0,
                rejectedCount: 0,
                hitEOS: false,
                nextTargetLogits: targetCurrentLogits,
                nextDraftLogits: draftLogits,
                draftLatencyMS: draftLatencyMS,
                verificationLatencyMS: 0,
                firstTokenTimeMS: nil
            )
        }

        let verificationStart = Date()
        var targetLogitSpan: [[Float]] = []
        targetLogitSpan.reserveCapacity(proposedTokens.count)
        var rollingTargetLogits = targetCurrentLogits
        for index in proposedTokens.indices {
            targetLogitSpan.append(rollingTargetLogits)
            if index < proposedTokens.count - 1 {
                try decodeTokens([proposedTokens[index]], context: targetContext)
                rollingTargetLogits = try readCurrentLogits(context: targetContext, vocabSize: targetVocabSize)
            }
        }

        try restoreFromSnapshot(targetSnapshot)
        try draftRunner.restoreFromSnapshot(draftSnapshot)

        var acceptedTokens: [Int] = []
        var rejectedCount = 0
        var correctionToken: Int?
        var hitEOS = false
        var verificationContext = allTokens

        for index in proposedTokens.indices {
            let proposedToken = proposedTokens[index]
            let recentTokens = Array(verificationContext.suffix(64))
            let targetDistribution = sampler.prepareDistribution(logits: targetLogitSpan[index], recentTokens: recentTokens)
            let targetProbability = targetDistribution.probability(of: proposedToken)
            let draftProbability = draftTokenProbabilities[index]
            let acceptance = draftProbability > 0 ? min(1.0, targetProbability / draftProbability) : 0

            if sampler.uniformSample() <= acceptance {
                acceptedTokens.append(proposedToken)
                verificationContext.append(proposedToken)
                if llama_vocab_is_eog(targetVocab, llama_token(proposedToken)) {
                    hitEOS = true
                    break
                }
                continue
            }

            rejectedCount = proposedTokens.count - index
            let draftDistribution = sampler.probabilityDistribution(logits: draftLogitSnapshots[index], recentTokens: recentTokens)
            correctionToken = sampler.sampleResidual(target: targetDistribution.probabilities, draft: draftDistribution)
            if let correctionToken, llama_vocab_is_eog(targetVocab, llama_token(correctionToken)) {
                hitEOS = true
            }
            break
        }

        let verificationLatencyMS = Date().timeIntervalSince(verificationStart) * 1000

        var committedTokens: [Int] = []
        for token in acceptedTokens {
            if llama_vocab_is_eog(targetVocab, llama_token(token)) {
                hitEOS = true
                break
            }
            committedTokens.append(token)
        }
        if !hitEOS, let correctionToken {
            if llama_vocab_is_eog(targetVocab, llama_token(correctionToken)) {
                hitEOS = true
            } else {
                committedTokens.append(correctionToken)
            }
        }

        var nextTargetLogits = targetCurrentLogits
        var nextDraftLogits = draftLogits
        var firstTokenTimeMS: Double?

        if !committedTokens.isEmpty {
            try decodeTokens(committedTokens, context: targetContext)
            nextTargetLogits = try readCurrentLogits(context: targetContext, vocabSize: targetVocabSize)

            try draftRunner.decodeTokens(committedTokens, context: draftState.context)
            nextDraftLogits = try draftRunner.readCurrentLogits(context: draftState.context, vocabSize: draftState.vocabSize)

            for (offset, token) in committedTokens.enumerated() {
                if firstTokenTimeMS == nil && generatedCount + offset == 0 {
                    firstTokenTimeMS = Date().timeIntervalSince(prefillStart) * 1000
                }
                let piece = tokenToPiece(vocab: targetVocab, token: llama_token(token))
                onToken(piece)
            }
        }

        return SpeculativeStepResult(
            committedTokens: committedTokens,
            acceptedCount: acceptedTokens.filter { !llama_vocab_is_eog(targetVocab, llama_token($0)) }.count,
            rejectedCount: rejectedCount,
            hitEOS: hitEOS,
            nextTargetLogits: nextTargetLogits,
            nextDraftLogits: nextDraftLogits,
            draftLatencyMS: draftLatencyMS,
            verificationLatencyMS: verificationLatencyMS,
            firstTokenTimeMS: firstTokenTimeMS
        )
    }

    private func setupLifecycleObservers() {
        removeLifecycleObserversInternal()

        backgroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleEnterBackground()
        }

        foregroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleEnterForeground()
        }
    }

    private func removeLifecycleObserversInternal() {
        if let observer = backgroundObserver {
            NotificationCenter.default.removeObserver(observer)
            backgroundObserver = nil
        }
        if let observer = foregroundObserver {
            NotificationCenter.default.removeObserver(observer)
            foregroundObserver = nil
        }
    }

    private func handleEnterBackground() {
        withLock {
            isBackgrounded = true
            backgroundTimestamp = Date()
        }
    }

    private func handleEnterForeground() {
        let enteredBackgroundAt: Date? = withLock {
            isBackgrounded = false
            let ts = backgroundTimestamp
            backgroundTimestamp = nil
            return ts
        }

        if let enteredBackgroundAt {
            let duration = Date().timeIntervalSince(enteredBackgroundAt)
            if duration > 120 {
                resetContext()
            }
        }
    }

    private func tokenize(vocab: OpaquePointer, text: String, addBOS: Bool) -> [Int] {
        let utf8Count = Int32(text.utf8.count)
        let maxTokens = utf8Count + (addBOS ? 1 : 0) + 16
        var tokens = [llama_token](repeating: 0, count: Int(maxTokens))

        let count = text.withCString { cString in
            llama_tokenize(vocab, cString, utf8Count, &tokens, maxTokens, addBOS, true)
        }
        guard count >= 0 else { return [] }

        return Array(tokens.prefix(Int(count))).map(Int.init)
    }

    private func tokenToPiece(vocab: OpaquePointer, token: llama_token) -> String {
        var buffer = [CChar](repeating: 0, count: 256)
        let count = llama_token_to_piece(vocab, token, &buffer, 256, 0, true)
        if count > 0 {
            buffer[Int(count)] = 0
            return String(cString: buffer)
        }
        return ""
    }

    func decodeTokens(_ tokens: [Int], context: OpaquePointer) throws {
        for token in tokens {
            var mutableToken = llama_token(token)
            let batch = withUnsafeMutablePointer(to: &mutableToken) { pointer in
                llama_batch_get_one(pointer, 1)
            }
            let result = llama_decode(context, batch)
            if result != 0 {
                throw LlamaRunnerError.decodeFailed
            }
        }
    }

    func readCurrentLogits(context: OpaquePointer, vocabSize: Int) throws -> [Float] {
        guard let logitsPointer = llama_get_logits_ith(context, -1) else {
            throw LlamaRunnerError.invalidLogits
        }
        return Array(UnsafeBufferPointer(start: logitsPointer, count: vocabSize))
    }

    private func decodeAndCollectLogits(_ tokens: [Int], context: OpaquePointer, vocabSize: Int) throws -> [[Float]] {
        var collected: [[Float]] = []
        collected.reserveCapacity(tokens.count)

        for token in tokens {
            try decodeTokens([token], context: context)
            collected.append(try readCurrentLogits(context: context, vocabSize: vocabSize))
        }

        return collected
    }

    private func recordPredictionFailure(_ error: Error) throws -> Never {
        withLock {
            consecutiveFailures += 1
            if consecutiveFailures >= 3 {
                state = .evicted
            }
        }

        let failures = withLock { consecutiveFailures }
        if failures >= 3 {
            throw LlamaRunnerError.modelEvicted
        }
        throw error
    }

    private func prunePrefixSnapshotStoreIfNeededLocked(maxEntries: Int) {
        guard prefixSnapshotStore.count > maxEntries else { return }
        let sortedIDs = prefixSnapshotStore
            .sorted { $0.value.metadata.prefixTokens.count > $1.value.metadata.prefixTokens.count }
            .map(\.key)
        for id in sortedIDs.dropFirst(maxEntries) {
            prefixSnapshotStore.removeValue(forKey: id)
        }
    }

    private func computeUnitsIdentifier(_ units: MLComputeUnits) -> String {
        switch units {
        case .all: return "all"
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown"
        }
    }

    deinit {
        removeLifecycleObserversInternal()
        stateCondition.lock()
        waitForGenerationDrainLocked()
        freeNativeResources()
        pendingUnload = false
        stateCondition.unlock()
    }
}

nonisolated struct LlamaGenerationResult: Sendable {
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let timeToFirstTokenMS: Double
    let totalDuration: Double
    let acceptedSpeculativeTokens: Int
    let rejectedSpeculativeTokens: Int
    let speculativeDraftLatencyMS: Double
    let speculativeVerifyLatencyMS: Double
    let speculativeVerificationCount: Int

    init(
        promptTokenCount: Int,
        generatedTokenCount: Int,
        prefillTokensPerSecond: Double,
        decodeTokensPerSecond: Double,
        timeToFirstTokenMS: Double,
        totalDuration: Double,
        acceptedSpeculativeTokens: Int = 0,
        rejectedSpeculativeTokens: Int = 0,
        speculativeDraftLatencyMS: Double = 0,
        speculativeVerifyLatencyMS: Double = 0,
        speculativeVerificationCount: Int = 0
    ) {
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.timeToFirstTokenMS = timeToFirstTokenMS
        self.totalDuration = totalDuration
        self.acceptedSpeculativeTokens = acceptedSpeculativeTokens
        self.rejectedSpeculativeTokens = rejectedSpeculativeTokens
        self.speculativeDraftLatencyMS = speculativeDraftLatencyMS
        self.speculativeVerifyLatencyMS = speculativeVerifyLatencyMS
        self.speculativeVerificationCount = speculativeVerificationCount
    }
}

nonisolated struct LlamaSessionSnapshot: Sendable {
    let modelPath: String
    let nCtx: Int32
    let nGPULayers: Int32
    let sessionTokenCount: Int
    let tokenHistory: [Int]
    let serializedState: Data
    let modelID: String
    let tokenizerID: String
    let modelSessionID: UUID
    let timestamp: Date

    var isValid: Bool {
        Date().timeIntervalSince(timestamp) < 300
    }

    func withGPULayers(_ nGPULayers: Int32) -> LlamaSessionSnapshot {
        LlamaSessionSnapshot(
            modelPath: modelPath,
            nCtx: nCtx,
            nGPULayers: nGPULayers,
            sessionTokenCount: sessionTokenCount,
            tokenHistory: tokenHistory,
            serializedState: serializedState,
            modelID: modelID,
            tokenizerID: tokenizerID,
            modelSessionID: modelSessionID,
            timestamp: timestamp
        )
    }
}

nonisolated enum LlamaRunnerError: Error, Sendable, LocalizedError {
    case modelNotLoaded
    case modelLoadFailed
    case contextCreationFailed
    case tokenizationFailed
    case decodeFailed
    case emptyInput
    case invalidLogits
    case invalidState(RunnerState)
    case snapshotExpired
    case stateRestoreFailed
    case modelEvicted

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No GGUF model loaded"
        case .modelLoadFailed: return "Failed to load GGUF model file"
        case .contextCreationFailed: return "Failed to create llama context"
        case .tokenizationFailed: return "Failed to tokenize input"
        case .decodeFailed: return "Decode step failed"
        case .emptyInput: return "Empty input token sequence"
        case .invalidLogits: return "Failed to read GGUF logits"
        case .invalidState(let state): return "Runner in invalid state: \(state.rawValue)"
        case .snapshotExpired: return "Session snapshot has expired"
        case .stateRestoreFailed: return "Failed to restore GGUF state snapshot"
        case .modelEvicted: return "GGUF context became unavailable and requires reload"
        }
    }
}
