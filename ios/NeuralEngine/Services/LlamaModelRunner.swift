import Foundation
import UIKit
import CoreML
import OSLog
import LlamaSwift

final class LlamaModelRunner: DraftLogitsPredicting, @unchecked Sendable {
    private final class CooperativeCancellationToken: @unchecked Sendable {
        private let lock = NSLock()
        private var cancelled = false
        private(set) var signaledAt: Date?

        var isCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }

        func signal(at date: Date = Date()) {
            lock.lock()
            if !cancelled {
                cancelled = true
                signaledAt = date
            }
            lock.unlock()
        }
    }

    private static let logger: Logger = {
        let subsystem = Bundle.main.bundleIdentifier ?? "NeuralEngine"
        return Logger(subsystem: subsystem, category: "LlamaModelRunner")
    }()

    private struct StoredPrefixStateRecord {
        let snapshot: LlamaSessionSnapshot
        let metadata: PrefixStateSnapshotMetadata
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
    // All mutable state is protected by stateCondition (an NSCondition).
    //
    //   activeGenerationCount – 0 or 1. Exactly ONE generation region may be
    //     active per runner at a time. Any path that calls llama_decode or reads
    //     raw C pointers must hold the generation token for its full duration.
    //
    //   pendingUnload – set to true by unload()/loadModel() BEFORE they wait
    //     for the drain. It makes tryAcquireGenerationToken() return false so
    //     no new generation can start, and it is checked cooperatively inside
    //     the generation loop so the active generator exits quickly.
    //
    // Invariants:
    //   • llama_free / llama_model_free are ONLY called after
    //     activeGenerationCount has reached 0.
    //   • pendingUnload is cleared only after the new model is fully loaded
    //     (or after freeNativeResources() for a plain unload).
    //   • Long-running decode paths do NOT hold stateCondition while calling
    //     llama_decode. Short vocabulary/state helpers may run under the lock so
    //     model/context lifetime cannot change while they touch native state.
    //   • Raw OpaquePointer values NEVER escape a runner instance. Decode/read
    //     operations borrow pointers only while a generation token is held.

    private let stateCondition = NSCondition()

    private var activeGenerationCount: Int = 0
    private var activeGenerationID: UUID?
    private var activeGenerationLabel: String?
    private var pendingUnload: Bool = false
    private var generationCancellationRequested: Bool = false
    private var generationCancellationReason: String?
    private var activeCancellationToken: CooperativeCancellationToken?
    private var forcedGenerationOwnershipRelease: Bool = false

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
    private let decodeExecutionKey = DispatchSpecificKey<UInt8>()
    private lazy var decodeExecutionQueue: DispatchQueue = {
        let queue = DispatchQueue(label: "com.neuralengine.llama.decode", qos: .userInitiated)
        queue.setSpecific(key: decodeExecutionKey, value: 1)
        return queue
    }()
#if DEBUG
    typealias SyntheticLoadConfigurationProvider = @Sendable (String, Int32, Int32) -> LlamaSyntheticTestingConfiguration?
    private var syntheticTestingState: LlamaSyntheticTestingState?
    private var syntheticLoadConfigurationProvider: SyntheticLoadConfigurationProvider?
#endif

    // MARK: – Lock helpers

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        return try body()
    }

    private func logDebug(_ message: String) {
        Self.logger.debug("\(message, privacy: .public)")
    }

    private func logNotice(_ message: String) {
        Self.logger.notice("\(message, privacy: .public)")
    }

    private func logError(_ message: String) {
        Self.logger.error("\(message, privacy: .public)")
    }

    private func runOnDecodeQueue<T>(_ work: () throws -> T) rethrows -> T {
        if DispatchQueue.getSpecific(key: decodeExecutionKey) != nil {
            return try work()
        }
        return try decodeExecutionQueue.sync(execute: work)
    }

    private func debugGuardDecodeThread(_ operation: String) {
#if DEBUG
        if Thread.isMainThread {
            let message = "Decode thread misuse: \(operation) invoked from main thread"
            logError(message)
            assertionFailure(message)
        }
#endif
    }

    private func ownershipSummaryLocked() -> String {
        let modelPath = lastContextPath ?? "none"
        let generationID = activeGenerationID?.uuidString ?? "none"
        let generationLabel = activeGenerationLabel ?? "none"
        return "model=\(modelPath) session=\(modelSessionID.uuidString) state=\(state.rawValue) pendingUnload=\(pendingUnload) activeGenerationCount=\(activeGenerationCount) generationID=\(generationID) label=\(generationLabel)"
    }

    private func cancellationReasonLocked() -> String {
        generationCancellationReason ?? (pendingUnload ? "unload" : "none")
    }

    private func assertDecodeAccessLocked(_ operation: String) {
#if DEBUG
        if activeGenerationCount == 0 {
            assertionFailure("GGUF \(operation) attempted without a generation token")
        }
        if pendingUnload || generationCancellationRequested {
            assertionFailure("GGUF \(operation) attempted while runner is unloading or cancelled")
        }
        if state == .evicted {
            assertionFailure("GGUF \(operation) attempted on an evicted runner")
        }
#endif
    }

    private func throwIfDecodeDisallowedLocked(_ operation: String) throws {
        assertDecodeAccessLocked(operation)
        if pendingUnload || generationCancellationRequested {
            throw LlamaRunnerError.generationCancelled
        }
        if state == .evicted {
            throw LlamaRunnerError.modelEvicted
        }
    }

    // MARK: – Generation-token protocol

    /// Increment the generation reference count, preventing any concurrent
    /// unload from freeing native pointers while we hold a token.
    /// Returns false if an unload is pending or another generation is active.
    /// Each runner allows at most ONE active generation at a time.
    func tryAcquireGenerationToken(reason: String = "work") -> Bool {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        guard !pendingUnload, activeGenerationCount == 0 else { return false }
        activeGenerationCount = 1
        activeGenerationID = UUID()
        activeGenerationLabel = reason
        generationCancellationRequested = false
        generationCancellationReason = nil
        forcedGenerationOwnershipRelease = false
        activeCancellationToken = CooperativeCancellationToken()
        return true
    }

    /// Decrement the generation reference count. Broadcasts on the condition
    /// so any thread blocked in waitForGenerationDrainLocked() wakes up.
    func releaseGenerationToken() {
        stateCondition.lock()
        if activeGenerationCount == 0 {
            let wasForcedRelease = forcedGenerationOwnershipRelease
            forcedGenerationOwnershipRelease = false
            stateCondition.unlock()
#if DEBUG
            if !wasForcedRelease {
                assertionFailure("releaseGenerationToken() called without an active generation")
            }
#endif
            return
        }
        activeGenerationCount = max(activeGenerationCount - 1, 0)
        let didDrain = activeGenerationCount == 0
        let drainedGenerationID = activeGenerationID?.uuidString ?? "none"
        let drainedReason = cancellationReasonLocked()
        let shouldLogDrain = didDrain && (pendingUnload || generationCancellationRequested)
        if didDrain {
            activeGenerationID = nil
            activeGenerationLabel = nil
            generationCancellationRequested = false
            generationCancellationReason = nil
            activeCancellationToken = nil
            stateCondition.broadcast()
        }
        stateCondition.unlock()

        if shouldLogDrain {
            logNotice("Cancellation drained generationID=\(drainedGenerationID) reason=\(drainedReason)")
        }
    }

    /// Must be called while stateCondition IS locked.
    /// Sets pendingUnload = true and blocks until activeGenerationCount == 0.
    private func waitForGenerationDrainLocked(reason: String) {
        pendingUnload = true
        if activeGenerationCount > 0 {
            generationCancellationRequested = true
            generationCancellationReason = reason
        }
        if state != .idle {
            state = .disposing
        }
        while activeGenerationCount > 0 {
            stateCondition.wait(until: Date(timeIntervalSinceNow: 0.1))
        }
    }

    /// Checks whether an unload / reload cancellation has been requested.
    /// Must be called WITHOUT holding stateCondition.
    private func isCancellationPending() -> Bool {
        stateCondition.lock()
        let pending = pendingUnload || generationCancellationRequested || (activeCancellationToken?.isCancelled ?? false)
        stateCondition.unlock()
        return pending
    }

    func requestGenerationCancellation(reason: String) {
        stateCondition.lock()
        let hasActiveGeneration = activeGenerationCount > 0
        let generationID = activeGenerationID?.uuidString ?? "none"
        let token = activeCancellationToken
        if hasActiveGeneration {
            generationCancellationRequested = true
            generationCancellationReason = reason
        }
        stateCondition.unlock()

        token?.signal()

        if hasActiveGeneration {
            logNotice("Cancellation requested generationID=\(generationID) reason=\(reason)")
        }
    }

    func awaitGenerationDrain(timeoutMS: UInt64) async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let deadline = Date().addingTimeInterval(Double(timeoutMS) / 1000)
                self.stateCondition.lock()
                while self.activeGenerationCount > 0 && Date() < deadline {
                    self.stateCondition.wait(until: Date(timeIntervalSinceNow: 0.01))
                }
                let drained = self.activeGenerationCount == 0
                self.stateCondition.unlock()
                continuation.resume(returning: drained)
            }
        }
    }

    func forceStopSchedulingAndReleaseOwnership(reason: String) {
        stateCondition.lock()
        let hadActiveGeneration = activeGenerationCount > 0
        let generationID = activeGenerationID?.uuidString ?? "none"
        let token = activeCancellationToken
        generationCancellationRequested = true
        generationCancellationReason = reason
        token?.signal()
        if hadActiveGeneration {
            activeGenerationCount = 0
            activeGenerationID = nil
            activeGenerationLabel = nil
            forcedGenerationOwnershipRelease = true
            stateCondition.broadcast()
        }
        stateCondition.unlock()

        if hadActiveGeneration {
            logNotice("Forced generation ownership release generationID=\(generationID) reason=\(reason)")
        }
    }

    private func throwIfCancellationPending(_ otherRunner: LlamaModelRunner? = nil) throws {
        if isCancellationPending() || otherRunner?.isCancellationPending() == true {
            throw LlamaRunnerError.generationCancelled
        }
    }

    private func waitForOutstandingBackendWorkLocked(reason: String) {
#if DEBUG
        if let syntheticTestingState {
            syntheticTestingState.waitForPendingDecodeCompletion()
            return
        }
#endif
        guard let context else { return }
        logDebug("Synchronizing backend work reason=\(reason)")
        llama_synchronize(context)
    }

#if DEBUG
    func installSyntheticModelForTesting(
        at path: String = "synthetic://gguf",
        nCtx: Int32 = 2048,
        nGPULayers: Int32 = 0,
        configuration: LlamaSyntheticTestingConfiguration
    ) {
        withLock {
            installSyntheticModelLocked(path: path, nCtx: nCtx, nGPULayers: nGPULayers, configuration: configuration)
        }
        setupLifecycleObservers()
    }

    func setSyntheticLoadConfigurationProviderForTesting(_ provider: SyntheticLoadConfigurationProvider?) {
        withLock {
            syntheticLoadConfigurationProvider = provider
        }
    }

    private func installSyntheticModelLocked(
        path: String,
        nCtx: Int32,
        nGPULayers: Int32,
        configuration: LlamaSyntheticTestingConfiguration
    ) {
        syntheticTestingState = LlamaSyntheticTestingState(configuration: configuration)
        nBatch = min(max(nCtx, 1), 512)
        lastContextPath = path
        lastNCtx = nCtx
        lastNGPULayers = nGPULayers
        sessionTokenCount = 0
        tokenHistory.removeAll(keepingCapacity: true)
        consecutiveFailures = 0
        lastSuccessfulPrediction = Date()
        lastRecoveryAttempt = nil
        prefixSnapshotStore.removeAll()
        modelSessionID = UUID()
        activeComputeUnits = nGPULayers > 0 ? .cpuAndGPU : .cpuOnly
        state = .ready
        pendingUnload = false
        generationCancellationRequested = false
        generationCancellationReason = nil
        activeGenerationID = nil
        activeGenerationLabel = nil
    }

    private func syntheticVocabularySize(for configuration: LlamaSyntheticTestingConfiguration) -> Int {
        let maxToken = [configuration.plannedTokens.max(), configuration.eogTokens.max(), configuration.tokenPieces.keys.max(), configuration.vocabSize]
            .compactMap { $0 }
            .max() ?? 2
        return max(maxToken + 1, 3)
    }

    private func syntheticLogits(for syntheticState: LlamaSyntheticTestingState) -> [Float] {
        let configuration = syntheticState.configuration
        let vocabSize = syntheticVocabularySize(for: configuration)
        let fallbackToken = configuration.eogTokens.sorted().first ?? 0
        let generationCursor = syntheticState.generationCursorValue
        let token = configuration.plannedTokens.indices.contains(generationCursor)
            ? configuration.plannedTokens[generationCursor]
            : fallbackToken
        var logits = [Float](repeating: -12, count: vocabSize)
        if token >= 0, token < logits.count {
            logits[token] = 12
        }
        return logits
    }
#endif

    // MARK: – Safe pointer borrowing
    //
    // These helpers read the live C pointers under the condition lock, then
    // release the lock before returning. The caller MUST hold a generation
    // token; that token prevents llama_free from running concurrently, so the
    // pointers are valid for the entire duration of the caller's token.
    //
    // The helpers are private. No raw OpaquePointer value is ever returned
    // from a method accessible by external call-sites or by other runner
    // instances. The only cross-instance operations are the higher-level
    // decode / readLogits / isEOG / tokenPiece methods below, which perform
    // the complete llama_* operation inside the method body.

    private func borrowContextPtr(for operation: String) throws -> OpaquePointer {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        try throwIfDecodeDisallowedLocked(operation)
        guard let c = context else { throw LlamaRunnerError.modelNotLoaded }
        return c
    }

    private func borrowModelAndContextPtr(for operation: String) throws -> (model: OpaquePointer, context: OpaquePointer) {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        try throwIfDecodeDisallowedLocked(operation)
        guard let m = model, let c = context else { throw LlamaRunnerError.modelNotLoaded }
        let isOperationalState = state == .ready || state == .recovering || (state == .disposing && activeGenerationCount > 0)
        guard isOperationalState else { throw LlamaRunnerError.invalidState(state) }
        return (m, c)
    }

    // MARK: – Self-contained llama operations
    //
    // Every method in this section reads self.context / self.model under the
    // lock at the start, releases the lock, then performs the llama_* call.
    // No OpaquePointer parameter is accepted or returned.

    /// Decode a sequence of tokens using self's live context.
    /// Caller must hold a generation token for self.
    private func decode(_ tokens: [Int]) throws {
        debugGuardDecodeThread("decode")
#if DEBUG
        if let syntheticState = try withLock({ () throws -> LlamaSyntheticTestingState? in
            try throwIfDecodeDisallowedLocked("decode")
            return syntheticTestingState
        }) {
            for token in tokens {
                try throwIfCancellationPending()
                syntheticState.beginDecode()
                defer { syntheticState.endDecode() }
                let delaySeconds = syntheticState.configuration.decodeDelaySeconds
                if delaySeconds > 0 {
                    Thread.sleep(forTimeInterval: delaySeconds)
                }
                try throwIfCancellationPending()
                syntheticState.recordDecodedToken(token)
                syntheticState.scheduleDecodeCompletion()
            }
            return
        }
#endif
        let ctx = try borrowContextPtr(for: "decode")
        for token in tokens {
            try throwIfCancellationPending()
            var mutableToken = llama_token(token)
            let batch = withUnsafeMutablePointer(to: &mutableToken) { ptr in
                llama_batch_get_one(ptr, 1)
            }
            let result = try runOnDecodeQueue {
                debugGuardDecodeThread("llama_decode")
                return llama_decode(ctx, batch)
            }
            guard result == 0 else { throw LlamaRunnerError.decodeFailed }
        }
    }

    /// Read the current logits from self's live context.
    /// Caller must hold a generation token for self.
    private func readLogits() throws -> [Float] {
#if DEBUG
        if let syntheticState = try withLock({ () throws -> LlamaSyntheticTestingState? in
            try throwIfDecodeDisallowedLocked("readLogits")
            return syntheticTestingState
        }) {
            return syntheticLogits(for: syntheticState)
        }
#endif
        let (m, ctx) = try borrowModelAndContextPtr(for: "readLogits")
        guard let vocab = llama_model_get_vocab(m) else { throw LlamaRunnerError.tokenizationFailed }
        let vocabSize = Int(llama_vocab_n_tokens(vocab))
        guard let ptr = llama_get_logits_ith(ctx, -1) else { throw LlamaRunnerError.invalidLogits }
        return Array(UnsafeBufferPointer(start: ptr, count: vocabSize))
    }

    /// Clear the KV-cache of self's live context. No-op if model not loaded.
    /// Caller must hold a generation token for self.
    private func clearKVCache() {
#if DEBUG
        if let syntheticState = withLock({ () -> LlamaSyntheticTestingState? in
            assertDecodeAccessLocked("clearKVCache")
            return syntheticTestingState
        }) {
            syntheticState.resetForNewSequence()
            return
        }
#endif
        stateCondition.lock()
        assertDecodeAccessLocked("clearKVCache")
        let ctx = context
        stateCondition.unlock()
        guard let ctx else { return }
        let memory = llama_get_memory(ctx)
        llama_memory_clear(memory, true)
    }

    /// Returns true when `token` is an end-of-generation token according to
    /// self's vocabulary. Safe to call from within the generation loop.
    func isEOG(_ token: Int) -> Bool {
        withLock {
#if DEBUG
            if let syntheticTestingState {
                return syntheticTestingState.configuration.eogTokens.contains(token)
            }
#endif
            guard let m = model, let vocab = llama_model_get_vocab(m) else { return false }
            return llama_vocab_is_eog(vocab, llama_token(token))
        }
    }

    /// Convert a token ID to its UTF-8 string piece using self's vocabulary.
    private func tokenPiece(_ token: Int) -> String {
        withLock {
#if DEBUG
            if let syntheticTestingState {
                return syntheticTestingState.configuration.tokenPieces[token] ?? "<\(token)>"
            }
#endif
            guard let m = model, let vocab = llama_model_get_vocab(m) else { return "" }
            var buffer = [CChar](repeating: 0, count: 256)
            let count = llama_token_to_piece(vocab, llama_token(token), &buffer, 256, 0, true)
            guard count > 0 else { return "" }
            buffer[Int(count)] = 0
            return String(cString: buffer)
        }
    }

    /// Tokenize `text` using self's vocabulary.
    /// Returns an empty array when the model is not loaded.
    private func tokenize(_ text: String, addBOS: Bool) -> [Int] {
#if DEBUG
        if let syntheticState = withLock({ syntheticTestingState }) {
            let upperBound = max(syntheticVocabularySize(for: syntheticState.configuration) - 1, 1)
            let mappedTokens = text.utf8.map { Int($0 % UInt8(upperBound)) + 1 }
            if addBOS {
                return [1] + mappedTokens
            }
            return mappedTokens
        }
#endif
        let utf8Count = Int32(text.utf8.count)
        let maxTokens = utf8Count + (addBOS ? 1 : 0) + 16
        var tokens = [llama_token](repeating: 0, count: Int(maxTokens))

        return withLock {
            guard let m = model, let vocab = llama_model_get_vocab(m) else { return [] }
            let count = text.withCString { cString in
                llama_tokenize(vocab, cString, utf8Count, &tokens, maxTokens, addBOS, true)
            }
            guard count >= 0 else { return [] }
            return Array(tokens.prefix(Int(count))).map(Int.init)
        }
    }

    /// Decode each token and collect the resulting logits.
    /// Caller must hold a generation token for self.
    private func decodeAndCollectAllLogits(_ tokens: [Int]) throws -> [[Float]] {
        var collected: [[Float]] = []
        collected.reserveCapacity(tokens.count)
        for token in tokens {
            try decode([token])
            collected.append(try readLogits())
        }
        return collected
    }

    // MARK: – Public API

    var isLoaded: Bool {
        withLock {
#if DEBUG
            if syntheticTestingState != nil {
                return state == .ready || state == .recovering
            }
#endif
            return model != nil && context != nil && (state == .ready || state == .recovering)
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

#if DEBUG
    func isSyntheticDecodeActiveForTesting() -> Bool {
        withLock { syntheticTestingState?.hasActiveDecode ?? false }
    }

    func isSyntheticBackendCompletionPendingForTesting() -> Bool {
        withLock { syntheticTestingState?.hasPendingDecodeCompletion ?? false }
    }

    func syntheticDecodeCallCountForTesting() -> Int {
        withLock { syntheticTestingState?.decodeCallCount ?? 0 }
    }
#endif

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

        let loadRequestSummary = withLock { ownershipSummaryLocked() }
        logNotice("Model load requested path=\(path) nCtx=\(nCtx) gpuLayers=\(gpuLayers) \(loadRequestSummary)")

        stateCondition.lock()
        let unloadBlockedByActiveWork = activeGenerationCount > 0
        let blockedSummary = ownershipSummaryLocked()
        waitForGenerationDrainLocked(reason: "modelLoad")
        stateCondition.unlock()

        if unloadBlockedByActiveWork {
            logNotice("Unload blocked by active work reason=modelLoad \(blockedSummary)")
        }

        stateCondition.lock()
        defer { stateCondition.unlock() }

        freeNativeResources()
        state = .loading

#if DEBUG
        if let configuration = syntheticLoadConfigurationProvider?(path, nCtx, gpuLayers) {
            let delaySeconds = configuration.loadDelaySeconds
            if delaySeconds > 0 {
                Thread.sleep(forTimeInterval: delaySeconds)
            }
            installSyntheticModelLocked(path: path, nCtx: nCtx, nGPULayers: gpuLayers, configuration: configuration)
            setupLifecycleObservers()
            logNotice("Model load completed path=\(path) \(ownershipSummaryLocked())")
            return
        }
#endif

        var modelParameters = llama_model_default_params()
        modelParameters.n_gpu_layers = gpuLayers

        guard let loadedModel = llama_model_load_from_file(path, modelParameters) else {
            state = .idle
            pendingUnload = false
            logError("Model load failed path=\(path) error=\(LlamaRunnerError.modelLoadFailed.localizedDescription)")
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
            logError("Context creation failed path=\(path) error=\(LlamaRunnerError.contextCreationFailed.localizedDescription)")
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
        pendingUnload = false

        setupLifecycleObservers()
        logNotice("Model load completed path=\(path) \(ownershipSummaryLocked())")
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

        // Acquire generation token BEFORE reading any raw pointer.
        // This prevents a concurrent unload from freeing the context
        // while we are decoding.
        guard tryAcquireGenerationToken(reason: draftRunner == nil ? "generation" : "generation+speculation") else {
            throw LlamaRunnerError.invalidState(currentState)
        }
        defer { releaseGenerationToken() }

        // Acquire a token for the draft runner. If unavailable (e.g.
        // mid-deactivation) fall back to non-speculative generation.
        let hasDraftToken: Bool
        let effectiveDraftRunner: LlamaModelRunner?
        if let dr = draftRunner, draftCount > 0, dr.tryAcquireGenerationToken(reason: "draft-speculation") {
            hasDraftToken = true
            effectiveDraftRunner = dr
        } else {
            hasDraftToken = false
            effectiveDraftRunner = nil
        }
        defer {
            if hasDraftToken { draftRunner?.releaseGenerationToken() }
        }

        let generationStartSummary = withLock { ownershipSummaryLocked() }
        logNotice("Generation start draft=\(effectiveDraftRunner != nil) promptChars=\(prompt.count) maxTokens=\(samplingConfig.maxTokens) \(generationStartSummary)")

        // Tokenize the prompt using self's own vocab (no raw pointer escapes).
        let promptTokens = tokenize(prompt, addBOS: true)
        guard !promptTokens.isEmpty else {
            throw LlamaRunnerError.tokenizationFailed
        }

        do {
            try throwIfCancellationPending(effectiveDraftRunner)
            clearKVCache()
            let prefillStart = Date()
            try decode(promptTokens)
            try throwIfCancellationPending(effectiveDraftRunner)
            var currentLogits = try readLogits()
            let prefillDuration = Date().timeIntervalSince(prefillStart)
            let prefillTPS = Double(promptTokens.count) / max(prefillDuration, 0.001)

            var currentDraftLogits: [Float]? = nil
            if let dr = effectiveDraftRunner {
                try throwIfCancellationPending(dr)
                try dr.prime(with: promptTokens)
                try throwIfCancellationPending(dr)
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
                try throwIfCancellationPending(effectiveDraftRunner)

                if let dr = effectiveDraftRunner,
                   let draftLogits = currentDraftLogits {
                    let maxDraft = min(draftCount, samplingConfig.maxTokens - generatedCount)
                    if maxDraft > 0 {
                        // performSpeculativeStep uses self and draftRunner exclusively
                        // through their higher-level decode/readLogits/isEOG methods.
                        // No raw OpaquePointer is passed between runners.
                        let speculative = try performSpeculativeStep(
                            sampler: sampler,
                            draftRunner: dr,
                            draftLogits: draftLogits,
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
                if isEOG(sampledToken) { break }

                if generatedCount == 0 {
                    firstTokenTimeMS = Date().timeIntervalSince(prefillStart) * 1000
                }

                try throwIfCancellationPending(effectiveDraftRunner)
                onToken(tokenPiece(sampledToken))

                allTokens.append(sampledToken)
                generatedCount += 1
                try throwIfCancellationPending(effectiveDraftRunner)
                try decode([sampledToken])
                currentLogits = try readLogits()

                if let dr = effectiveDraftRunner {
                    try throwIfCancellationPending(dr)
                    try dr.decode([sampledToken])
                    currentDraftLogits = try dr.readLogits()
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

            let result = LlamaGenerationResult(
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
            let completionSummary = withLock { ownershipSummaryLocked() }
            logNotice("Generation stop status=success generatedTokens=\(generatedCount) acceptedSpeculative=\(acceptedSpeculativeTokens) rejectedSpeculative=\(rejectedSpeculativeTokens) \(completionSummary)")
            return result
        } catch let error as LlamaRunnerError {
            if case .generationCancelled = error {
                let cancellationReason = withLock { cancellationReasonLocked() }
                let cancellationSummary = withLock { ownershipSummaryLocked() }
                logNotice("Generation stop status=cancelled reason=\(cancellationReason) \(cancellationSummary)")
                throw error
            }
            let failureSummary = withLock { ownershipSummaryLocked() }
            logError("Generation stop status=failed error=\(error.localizedDescription) \(failureSummary)")
            try recordPredictionFailure(error)
        } catch {
            let failureSummary = withLock { ownershipSummaryLocked() }
            logError("Generation stop status=failed error=\(error.localizedDescription) \(failureSummary)")
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

        guard tryAcquireGenerationToken(reason: "predictLogitsSpan") else {
            throw LlamaRunnerError.invalidState(currentState)
        }
        defer { releaseGenerationToken() }

        do {
            let logits = try decodeAndCollectAllLogits(inputIDs)
            withLock {
                sessionTokenCount += inputIDs.count
                tokenHistory.append(contentsOf: inputIDs)
                lastSuccessfulPrediction = Date()
                consecutiveFailures = 0
                state = .ready
            }
            return logits
        } catch let error as LlamaRunnerError {
            if case .generationCancelled = error {
                let cancellationReason = withLock { cancellationReasonLocked() }
                let cancellationSummary = withLock { ownershipSummaryLocked() }
                logNotice("Prediction stop status=cancelled reason=\(cancellationReason) \(cancellationSummary)")
                throw error
            }
            let failureSummary = withLock { ownershipSummaryLocked() }
            logError("Prediction stop status=failed error=\(error.localizedDescription) \(failureSummary)")
            try recordPredictionFailure(error)
        } catch {
            let failureSummary = withLock { ownershipSummaryLocked() }
            logError("Prediction stop status=failed error=\(error.localizedDescription) \(failureSummary)")
            try recordPredictionFailure(error)
        }
    }

    func resetContext() {
        withLock {
            guard activeGenerationCount == 0, !pendingUnload else { return }
#if DEBUG
            if let syntheticTestingState {
                syntheticTestingState.waitForPendingDecodeCompletion()
                syntheticTestingState.resetForNewSequence()
                sessionTokenCount = 0
                tokenHistory.removeAll(keepingCapacity: true)
                lastSuccessfulPrediction = Date()
                consecutiveFailures = 0
                state = .ready
                return
            }
#endif
            guard let ctx = context else {
                sessionTokenCount = 0
                tokenHistory.removeAll(keepingCapacity: true)
                state = .idle
                return
            }
            waitForOutstandingBackendWorkLocked(reason: "resetContext")
            let memory = llama_get_memory(ctx)
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

    private func saveStateWhileHoldingGenerationToken() -> LlamaSessionSnapshot? {
        withLock {
            guard activeGenerationCount > 0 else { return nil }
#if DEBUG
            if let syntheticTestingState, state == .ready || state == .recovering {
                let data = syntheticTestingState.serializedState()
                return LlamaSessionSnapshot(
                    modelPath: lastContextPath ?? "",
                    nCtx: lastNCtx,
                    nGPULayers: lastNGPULayers,
                    sessionTokenCount: sessionTokenCount,
                    tokenHistory: tokenHistory,
                    serializedState: data,
                    modelID: prefixSnapshotModelID ?? lastContextPath ?? "unknown-model",
                    tokenizerID: prefixSnapshotTokenizerID,
                    modelSessionID: modelSessionID,
                    timestamp: Date()
                )
            }
#endif
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

    func saveState() -> LlamaSessionSnapshot? {
        guard tryAcquireGenerationToken(reason: "saveState") else { return nil }
        defer { releaseGenerationToken() }
        return saveStateWhileHoldingGenerationToken()
    }

    private func restoreStateToCurrentContextWhileHoldingGenerationToken(_ snapshot: LlamaSessionSnapshot) throws {
        try withLock {
            guard activeGenerationCount > 0 else {
                throw LlamaRunnerError.generationInProgress
            }
#if DEBUG
            if let syntheticTestingState {
                let restored = syntheticTestingState.restore(from: snapshot.serializedState)
                guard restored else {
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
                return
            }
#endif
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

    func restoreFromSnapshot(_ snapshot: LlamaSessionSnapshot) throws {
        guard snapshot.isValid else {
            throw LlamaRunnerError.snapshotExpired
        }
        try throwIfCancellationPending()

        guard tryAcquireGenerationToken(reason: "restoreState") else {
            throw LlamaRunnerError.generationInProgress
        }

        let needsLoad = withLock { () -> Bool in
#if DEBUG
            let hasResidentModel = syntheticTestingState != nil || (model != nil && context != nil)
#else
            let hasResidentModel = model != nil && context != nil
#endif
            return lastContextPath != snapshot.modelPath || !hasResidentModel
        }

        if !needsLoad {
            defer { releaseGenerationToken() }
            try restoreStateToCurrentContextWhileHoldingGenerationToken(snapshot)
            return
        }

        releaseGenerationToken()
        try loadModel(at: snapshot.modelPath, nCtx: snapshot.nCtx, nGPULayers: snapshot.nGPULayers)

        guard tryAcquireGenerationToken(reason: "restoreState") else {
            throw LlamaRunnerError.generationInProgress
        }
        defer { releaseGenerationToken() }
        try restoreStateToCurrentContextWhileHoldingGenerationToken(snapshot)
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
#if DEBUG
            let hasSyntheticModel = syntheticTestingState != nil
#else
            let hasSyntheticModel = false
#endif
            let isHealthy = (hasSyntheticModel || (model != nil && context != nil)) && (state == .ready || state == .recovering) && consecutiveFailures == 0 && !isStale
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

    var configuredContextLength: Int {
        withLock { Int(lastNCtx) }
    }

    var configuredBatchSize: Int {
        withLock { Int(nBatch) }
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
        let unloadRequestSummary = withLock { ownershipSummaryLocked() }
        logNotice("Unload requested reason=explicit \(unloadRequestSummary)")

        stateCondition.lock()
        removeLifecycleObserversInternal()
        let unloadBlockedByActiveWork = activeGenerationCount > 0
        let blockedSummary = ownershipSummaryLocked()
        waitForGenerationDrainLocked(reason: "unload")
#if DEBUG
        let syntheticUnloadDelaySeconds = syntheticTestingState?.configuration.unloadDelaySeconds ?? 0
        if syntheticUnloadDelaySeconds > 0 {
            Thread.sleep(forTimeInterval: syntheticUnloadDelaySeconds)
        }
#endif
        stateCondition.unlock()

        if unloadBlockedByActiveWork {
            logNotice("Unload blocked by active work reason=explicit \(blockedSummary)")
        }

        stateCondition.lock()
        freeNativeResources()
        pendingUnload = false
        let completionSummary = ownershipSummaryLocked()
        stateCondition.unlock()
        logNotice("Unload completed \(completionSummary)")
    }

    // MARK: – Private helpers

    private func freeNativeResources() {
        waitForOutstandingBackendWorkLocked(reason: "freeNativeResources")
        if let context {
            llama_free(context)
        }
        if let model {
            llama_model_free(model)
        }
        context = nil
        model = nil
#if DEBUG
        syntheticTestingState = nil
#endif
        sessionTokenCount = 0
        tokenHistory.removeAll(keepingCapacity: true)
        consecutiveFailures = 0
        lastSuccessfulPrediction = nil
        lastRecoveryAttempt = nil
        prefixSnapshotStore.removeAll()
        modelSessionID = UUID()
        activeComputeUnits = .cpuAndGPU
        generationCancellationRequested = false
        generationCancellationReason = nil
        activeCancellationToken = nil
        activeGenerationID = nil
        activeGenerationLabel = nil
        forcedGenerationOwnershipRelease = false
        state = .idle
    }

    /// Prime the draft runner: clear its KV-cache and decode `tokens`.
    /// Must be called while the draft runner's generation token is held.
    private func prime(with tokens: [Int]) throws {
        clearKVCache()
        try decode(tokens)
        withLock {
            sessionTokenCount = tokens.count
            tokenHistory = tokens
            lastSuccessfulPrediction = Date()
            consecutiveFailures = 0
            state = .ready
        }
    }

    /// Read the current logits from this runner's context.
    /// Must be called while this runner's generation token is held.
    private func captureCurrentLogits() throws -> [Float] {
        return try readLogits()
    }

    // MARK: – Speculative decoding
    //
    // performSpeculativeStep takes no raw pointer parameters.
    // All llama_* operations on the target (self) and the draft runner go
    // through the self-contained decode / readLogits / isEOG / tokenPiece
    // methods defined above.  Both generation tokens (target + draft) are
    // held by the caller (generateWithDraft) for the entire duration.

    private func performSpeculativeStep(
        sampler: Sampler,
        draftRunner: LlamaModelRunner,
        draftLogits: [Float],
        allTokens: [Int],
        targetCurrentLogits: [Float],
        maxDraft: Int,
        prefillStart: Date,
        generatedCount: Int,
        onToken: @escaping (String) -> Void
    ) throws -> SpeculativeStepResult {
        try throwIfCancellationPending(draftRunner)
        guard let targetSnapshot = saveStateWhileHoldingGenerationToken(),
              let draftSnapshot = draftRunner.saveStateWhileHoldingGenerationToken() else {
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

        let draftStart = Date()
        var proposedTokens: [Int] = []
        var draftLogitSnapshots: [[Float]] = []
        var draftTokenProbabilities: [Float] = []
        var rollingDraftLogits = draftLogits
        var draftHistory = allTokens

        for index in 0..<maxDraft {
            try throwIfCancellationPending(draftRunner)
            let context = Array(draftHistory.suffix(64))
            let draftDistribution = sampler.prepareDistribution(logits: rollingDraftLogits, recentTokens: context)
            let token = sampler.sample(from: draftDistribution)
            proposedTokens.append(token)
            draftLogitSnapshots.append(rollingDraftLogits)
            draftTokenProbabilities.append(draftDistribution.probability(of: token))
            draftHistory.append(token)

            if draftRunner.isEOG(token) { break }

            if index < maxDraft - 1 {
                try draftRunner.decode([token])
                rollingDraftLogits = try draftRunner.readLogits()
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
            try throwIfCancellationPending(draftRunner)
            targetLogitSpan.append(rollingTargetLogits)
            if index < proposedTokens.count - 1 {
                try decode([proposedTokens[index]])
                rollingTargetLogits = try readLogits()
            }
        }

        try throwIfCancellationPending(draftRunner)
        try restoreStateToCurrentContextWhileHoldingGenerationToken(targetSnapshot)
        try draftRunner.restoreStateToCurrentContextWhileHoldingGenerationToken(draftSnapshot)
        try throwIfCancellationPending(draftRunner)

        var acceptedTokens: [Int] = []
        var rejectedCount = 0
        var correctionToken: Int?
        var hitEOS = false
        var verificationContext = allTokens

        for index in proposedTokens.indices {
            try throwIfCancellationPending(draftRunner)
            let proposedToken = proposedTokens[index]
            let recentTokens = Array(verificationContext.suffix(64))
            let targetDistribution = sampler.prepareDistribution(logits: targetLogitSpan[index], recentTokens: recentTokens)
            let targetProbability = targetDistribution.probability(of: proposedToken)
            let draftProbability = draftTokenProbabilities[index]
            let acceptance = draftProbability > 0 ? min(1.0, targetProbability / draftProbability) : 0

            if sampler.uniformSample() <= acceptance {
                acceptedTokens.append(proposedToken)
                verificationContext.append(proposedToken)
                if isEOG(proposedToken) {
                    hitEOS = true
                    break
                }
                continue
            }

            rejectedCount = proposedTokens.count - index
            let draftDistribution = sampler.probabilityDistribution(logits: draftLogitSnapshots[index], recentTokens: recentTokens)
            correctionToken = sampler.sampleResidual(target: targetDistribution.probabilities, draft: draftDistribution)
            if let correctionToken, isEOG(correctionToken) {
                hitEOS = true
            }
            break
        }

        let verificationLatencyMS = Date().timeIntervalSince(verificationStart) * 1000

        var committedTokens: [Int] = []
        for token in acceptedTokens {
            if isEOG(token) {
                hitEOS = true
                break
            }
            committedTokens.append(token)
        }
        if !hitEOS, let correctionToken {
            if isEOG(correctionToken) {
                hitEOS = true
            } else {
                committedTokens.append(correctionToken)
            }
        }

        var nextTargetLogits = targetCurrentLogits
        var nextDraftLogits = draftLogits
        var firstTokenTimeMS: Double?

        if !committedTokens.isEmpty {
            try throwIfCancellationPending(draftRunner)
            try decode(committedTokens)
            nextTargetLogits = try readLogits()

            try throwIfCancellationPending(draftRunner)
            try draftRunner.decode(committedTokens)
            nextDraftLogits = try draftRunner.readLogits()

            for (offset, token) in committedTokens.enumerated() {
                try throwIfCancellationPending(draftRunner)
                if firstTokenTimeMS == nil && generatedCount + offset == 0 {
                    firstTokenTimeMS = Date().timeIntervalSince(prefillStart) * 1000
                }
                onToken(tokenPiece(token))
            }
        }

        return SpeculativeStepResult(
            committedTokens: committedTokens,
            acceptedCount: acceptedTokens.filter { !isEOG($0) }.count,
            rejectedCount: rejectedCount,
            hitEOS: hitEOS,
            nextTargetLogits: nextTargetLogits,
            nextDraftLogits: nextDraftLogits,
            draftLatencyMS: draftLatencyMS,
            verificationLatencyMS: verificationLatencyMS,
            firstTokenTimeMS: firstTokenTimeMS
        )
    }

    // MARK: – Lifecycle observers

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

    // MARK: – Misc private helpers

    private func recordPredictionFailure(_ error: Error) throws -> Never {
        let failureSummary = withLock {
            consecutiveFailures += 1
            if consecutiveFailures >= 3 {
                state = .evicted
            }
            return ownershipSummaryLocked()
        }
        logError("Prediction failure error=\(error.localizedDescription) \(failureSummary)")

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
        waitForGenerationDrainLocked(reason: "deinit")
        freeNativeResources()
        pendingUnload = false
        stateCondition.unlock()
    }
}

// MARK: – Supporting value types

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
    case generationCancelled
    case generationInProgress

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
        case .generationCancelled: return "GGUF generation was cancelled during a runner transition"
        case .generationInProgress: return "GGUF runner is busy with an active generation"
        }
    }
}
