import Foundation
import Accelerate
import LlamaSwift

nonisolated final class EmbeddingModelRunner: @unchecked Sendable {
    private let lock = NSLock()
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var embeddingDimensions: Int = 0
    private var modelPath: String?

    var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return model != nil && context != nil
    }

    var dimensions: Int {
        lock.lock()
        defer { lock.unlock() }
        return embeddingDimensions
    }

    func loadModel(at path: String, nCtx: Int32 = 512) throws {
        lock.lock()
        defer { lock.unlock() }

        unloadInternal()

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
        lock.lock()
        guard let model, let context else {
            lock.unlock()
            return nil
        }
        let dims = embeddingDimensions
        lock.unlock()

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        guard let vocab = llama_model_get_vocab(model) else { return nil }

        let tokens = tokenize(vocab: vocab, text: trimmed, addBOS: true)
        guard !tokens.isEmpty else { return nil }

        lock.lock()
        defer { lock.unlock() }

        guard let ctx = self.context else { return nil }

        let memory = llama_get_memory(ctx)
        llama_memory_clear(memory, true)

        for (index, token) in tokens.enumerated() {
            var mutableToken = llama_token(token)
            let batch = withUnsafeMutablePointer(to: &mutableToken) { pointer in
                llama_batch_get_one(pointer, 1)
            }

            let result: Int32
            if index < tokens.count - 1 {
                result = llama_encode(ctx, batch)
            } else {
                result = llama_encode(ctx, batch)
            }

            if result != 0 {
                return nil
            }
        }

        guard let embPtr = llama_get_embeddings_seq(ctx, 0) else {
            if let embIth = llama_get_embeddings_ith(ctx, -1) {
                let raw = Array(UnsafeBufferPointer(start: embIth, count: dims))
                return l2Normalize(raw)
            }
            return nil
        }

        let raw = Array(UnsafeBufferPointer(start: embPtr, count: dims))
        return l2Normalize(raw)
    }

    func embedBatch(_ texts: [String]) -> [[Float]?] {
        texts.map { embed($0) }
    }

    func unload() {
        lock.lock()
        defer { lock.unlock() }
        unloadInternal()
    }

    private func unloadInternal() {
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        context = nil
        model = nil
        embeddingDimensions = 0
        modelPath = nil
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

    deinit {
        lock.lock()
        unloadInternal()
        lock.unlock()
    }
}

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
