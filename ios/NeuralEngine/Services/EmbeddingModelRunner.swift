import Foundation
import Accelerate
import LlamaSwift

nonisolated final class EmbeddingModelRunner: @unchecked Sendable {
    private let stateCondition = NSCondition()
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var embeddingDimensions: Int = 0
    private var modelPath: String?
    private var activeOperationCount: Int = 0
    private var pendingUnload: Bool = false
#if DEBUG
    private var syntheticConfiguration: EmbeddingSyntheticTestingConfiguration?
    private var syntheticEmbedCallCount: Int = 0
    private var syntheticEmbedActive: Bool = false
#endif

    var isLoaded: Bool {
        stateCondition.lock()
        defer { stateCondition.unlock() }
#if DEBUG
        return (model != nil && context != nil) || syntheticConfiguration != nil
#else
        return model != nil && context != nil
#endif
    }

    var dimensions: Int {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        return embeddingDimensions
    }

    func loadModel(at path: String, nCtx: Int32 = 512) throws {
        stateCondition.lock()
        waitForOperationDrainLocked()
        defer {
            pendingUnload = false
            stateCondition.broadcast()
            stateCondition.unlock()
        }

        unloadInternalLocked()

        var modelParams = llama_model_default_params()
#if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
#else
        modelParams.n_gpu_layers = 99
#endif

        guard let loadedModel = llama_model_load_from_file(path, modelParams) else {
            throw EmbeddingRunnerError.modelLoadFailed
        }

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(nCtx)
        ctxParams.n_batch = UInt32(nCtx)
        ctxParams.embeddings = true
        let threadCount = max(ProcessInfo.processInfo.activeProcessorCount - 2, 1)
        ctxParams.n_threads = Int32(threadCount)
        ctxParams.n_threads_batch = Int32(threadCount)
        ctxParams.pooling_type = LLAMA_POOLING_TYPE_MEAN

        guard let loadedContext = llama_init_from_model(loadedModel, ctxParams) else {
            llama_model_free(loadedModel)
            throw EmbeddingRunnerError.contextCreationFailed
        }

        let nEmbd = Int(llama_model_n_embd(loadedModel))
        guard nEmbd > 0 else {
            llama_free(loadedContext)
            llama_model_free(loadedModel)
            throw EmbeddingRunnerError.invalidEmbeddingDimensions
        }

        model = loadedModel
        context = loadedContext
        embeddingDimensions = nEmbd
        modelPath = path
    }

    func embed(_ text: String) -> [Float]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        guard let borrowedState = beginEmbeddingOperation() else { return nil }
        defer { endEmbeddingOperation(borrowedState) }

        switch borrowedState {
        case .native(let context, let vocab, let dimensions):
            let tokens = tokenize(vocab: vocab, text: trimmed, addBOS: true)
            guard !tokens.isEmpty else { return nil }

            let memory = llama_get_memory(context)
            llama_memory_clear(memory, true)

            for token in tokens {
                var mutableToken = llama_token(token)
                let batch = withUnsafeMutablePointer(to: &mutableToken) { pointer in
                    llama_batch_get_one(pointer, 1)
                }
                let result = llama_encode(context, batch)
                guard result == 0 else { return nil }
            }

            guard let embPtr = llama_get_embeddings_seq(context, 0) else {
                guard let embIth = llama_get_embeddings_ith(context, -1) else {
                    return nil
                }
                let raw = Array(UnsafeBufferPointer(start: embIth, count: dimensions))
                return l2Normalize(raw)
            }

            let raw = Array(UnsafeBufferPointer(start: embPtr, count: dimensions))
            return l2Normalize(raw)
#if DEBUG
        case .synthetic(let configuration):
            return syntheticEmbedding(for: trimmed, dimensions: configuration.dimensions)
#endif
        }
    }

    func embedBatch(_ texts: [String]) -> [[Float]?] {
        texts.map { embed($0) }
    }

    func unload() {
        stateCondition.lock()
        waitForOperationDrainLocked()
        unloadInternalLocked()
        pendingUnload = false
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    private func waitForOperationDrainLocked() {
        pendingUnload = true
        while activeOperationCount > 0 {
            stateCondition.wait()
        }
        synchronizeContextLocked()
    }

    private func synchronizeContextLocked() {
        guard let context else { return }
        llama_synchronize(context)
    }

    private func unloadInternalLocked() {
        if let context {
            llama_free(context)
        }
        if let model {
            llama_model_free(model)
        }
        context = nil
        model = nil
        embeddingDimensions = 0
        modelPath = nil
#if DEBUG
        syntheticConfiguration = nil
        syntheticEmbedCallCount = 0
        syntheticEmbedActive = false
#endif
    }

    private func beginEmbeddingOperation() -> BorrowedEmbeddingState? {
        stateCondition.lock()
        while activeOperationCount > 0 && !pendingUnload {
            stateCondition.wait()
        }

        guard !pendingUnload else {
            stateCondition.unlock()
            return nil
        }

#if DEBUG
        if let syntheticConfiguration {
            activeOperationCount = 1
            syntheticEmbedCallCount += 1
            syntheticEmbedActive = true
            stateCondition.unlock()
            if syntheticConfiguration.embedDelaySeconds > 0 {
                Thread.sleep(forTimeInterval: syntheticConfiguration.embedDelaySeconds)
            }
            return .synthetic(configuration: syntheticConfiguration)
        }
#endif

        guard let model, let context else {
            stateCondition.unlock()
            return nil
        }

        activeOperationCount = 1
        let dimensions = embeddingDimensions
        let vocab = llama_model_get_vocab(model)
        stateCondition.unlock()

        guard let vocab else {
            endEmbeddingOperation(nil)
            return nil
        }

        return .native(context: context, vocab: vocab, dimensions: dimensions)
    }

    private func endEmbeddingOperation(_ borrowedState: BorrowedEmbeddingState?) {
        if case .native(let context, _, _) = borrowedState {
            llama_synchronize(context)
        }

        stateCondition.lock()
        activeOperationCount = max(activeOperationCount - 1, 0)
#if DEBUG
        syntheticEmbedActive = false
#endif
        stateCondition.broadcast()
        stateCondition.unlock()
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

    private func l2Normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return vector }
        var magnitude: Float = 0
        vDSP_svesq(vector, 1, &magnitude, vDSP_Length(vector.count))
        magnitude = sqrtf(magnitude)
        guard magnitude > 0 else { return vector }
        var normalized = [Float](repeating: 0, count: vector.count)
        var divisor = magnitude
        vDSP_vsdiv(vector, 1, &divisor, &normalized, 1, vDSP_Length(vector.count))
        return normalized
    }

#if DEBUG
    func installSyntheticModelForTesting(dimensions: Int = 8, embedDelaySeconds: TimeInterval = 0) {
        stateCondition.lock()
        waitForOperationDrainLocked()
        unloadInternalLocked()
        syntheticConfiguration = EmbeddingSyntheticTestingConfiguration(
            dimensions: max(dimensions, 1),
            embedDelaySeconds: max(embedDelaySeconds, 0)
        )
        embeddingDimensions = max(dimensions, 1)
        modelPath = "synthetic://embedding"
        pendingUnload = false
        stateCondition.broadcast()
        stateCondition.unlock()
    }

    func isSyntheticEmbedActiveForTesting() -> Bool {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        return syntheticEmbedActive
    }

    func syntheticEmbedCallCountForTesting() -> Int {
        stateCondition.lock()
        defer { stateCondition.unlock() }
        return syntheticEmbedCallCount
    }

    private func syntheticEmbedding(for text: String, dimensions: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: max(dimensions, 1))
        for (index, byte) in text.utf8.enumerated() {
            vector[index % vector.count] += Float(byte) / 255.0
        }
        return l2Normalize(vector)
    }
#endif

    deinit {
        stateCondition.lock()
        waitForOperationDrainLocked()
        unloadInternalLocked()
        pendingUnload = false
        stateCondition.broadcast()
        stateCondition.unlock()
    }
}

private enum BorrowedEmbeddingState {
    case native(context: OpaquePointer, vocab: OpaquePointer, dimensions: Int)
#if DEBUG
    case synthetic(configuration: EmbeddingSyntheticTestingConfiguration)
#endif
}

#if DEBUG
nonisolated private struct EmbeddingSyntheticTestingConfiguration: Sendable {
    let dimensions: Int
    let embedDelaySeconds: TimeInterval
}
#endif

nonisolated enum EmbeddingRunnerError: Error, Sendable, LocalizedError {
    case modelLoadFailed
    case contextCreationFailed
    case invalidEmbeddingDimensions
    case encodeFailed

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed: return "Failed to load GGUF embedding model"
        case .contextCreationFailed: return "Failed to create embedding context"
        case .invalidEmbeddingDimensions: return "Model returned invalid embedding dimensions"
        case .encodeFailed: return "Embedding encode step failed"
        }
    }
}
