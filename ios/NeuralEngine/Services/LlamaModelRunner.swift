import Foundation
import LlamaSwift

nonisolated final class LlamaModelRunner: @unchecked Sendable {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private let lock = NSLock()

    var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return model != nil && context != nil
    }

    func loadModel(at path: String, nCtx: Int32 = 2048, nGPULayers: Int32 = 99) throws {
        lock.lock()
        defer { lock.unlock() }

        unloadInternal()

        var mParams = llama_model_default_params()
        mParams.n_gpu_layers = nGPULayers

        guard let loadedModel = llama_model_load_from_file(path, mParams) else {
            throw LlamaRunnerError.modelLoadFailed
        }

        var cParams = llama_context_default_params()
        cParams.n_ctx = UInt32(nCtx)
        cParams.n_batch = UInt32(min(nCtx, 512))
        let threadCount = max(ProcessInfo.processInfo.activeProcessorCount - 2, 1)
        cParams.n_threads = Int32(threadCount)
        cParams.n_threads_batch = Int32(threadCount)

        guard let ctx = llama_init_from_model(loadedModel, cParams) else {
            llama_model_free(loadedModel)
            throw LlamaRunnerError.contextCreationFailed
        }

        model = loadedModel
        context = ctx
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
        lock.lock()
        guard let model, let context else {
            lock.unlock()
            throw LlamaRunnerError.modelNotLoaded
        }
        let vocab = llama_model_get_vocab(model)
        lock.unlock()

        guard let vocab else {
            throw LlamaRunnerError.tokenizationFailed
        }

        let promptTokens = tokenize(vocab: vocab, text: prompt, addBOS: true)
        guard !promptTokens.isEmpty else {
            throw LlamaRunnerError.tokenizationFailed
        }

        let mem = llama_get_memory(context)
        llama_memory_clear(mem, true)

        let prefillStart = Date()

        var tokens = promptTokens
        let batch = tokens.withUnsafeMutableBufferPointer { buf in
            llama_batch_get_one(buf.baseAddress, Int32(buf.count))
        }

        let prefillResult = llama_decode(context, batch)
        guard prefillResult == 0 else {
            throw LlamaRunnerError.decodeFailed
        }

        let prefillDuration = Date().timeIntervalSince(prefillStart)
        let prefillTPS = Double(promptTokens.count) / max(prefillDuration, 0.001)

        let sampler = createSampler(
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty
        )
        defer { llama_sampler_free(sampler) }

        var nCur = Int32(promptTokens.count)
        var generatedCount = 0
        let decodeStart = Date()
        var firstTokenTime: Double = 0

        while generatedCount < maxTokens && !shouldStop() {
            let newToken = llama_sampler_sample(sampler, context, -1)

            if llama_vocab_is_eog(vocab, newToken) {
                break
            }

            if generatedCount == 0 {
                firstTokenTime = Date().timeIntervalSince(prefillStart) * 1000
            }

            let piece = tokenToPiece(vocab: vocab, token: newToken)
            onToken(piece)
            generatedCount += 1

            var singleToken = newToken
            let decodeBatch = withUnsafeMutablePointer(to: &singleToken) { ptr in
                llama_batch_get_one(ptr, 1)
            }

            let decodeResult = llama_decode(context, decodeBatch)
            if decodeResult != 0 {
                break
            }

            nCur += 1
        }

        let totalDuration = Date().timeIntervalSince(prefillStart)
        let decodeDuration = Date().timeIntervalSince(decodeStart)
        let decodeTPS = Double(generatedCount) / max(decodeDuration, 0.001)

        return LlamaGenerationResult(
            promptTokenCount: promptTokens.count,
            generatedTokenCount: generatedCount,
            prefillTokensPerSecond: prefillTPS,
            decodeTokensPerSecond: decodeTPS,
            timeToFirstTokenMS: firstTokenTime,
            totalDuration: totalDuration
        )
    }

    func resetContext() {
        lock.lock()
        defer { lock.unlock() }
        if let context {
            let mem = llama_get_memory(context)
            llama_memory_clear(mem, true)
        }
    }

    func unload() {
        lock.lock()
        defer { lock.unlock() }
        unloadInternal()
    }

    private func unloadInternal() {
        if let context {
            llama_free(context)
        }
        if let model {
            llama_model_free(model)
        }
        context = nil
        model = nil
    }

    private func tokenize(vocab: OpaquePointer, text: String, addBOS: Bool) -> [llama_token] {
        let utf8Count = Int32(text.utf8.count)
        let maxTokens = utf8Count + (addBOS ? 1 : 0) + 16
        var tokens = [llama_token](repeating: 0, count: Int(maxTokens))

        let nTokens = text.withCString { cStr in
            llama_tokenize(vocab, cStr, utf8Count, &tokens, maxTokens, addBOS, true)
        }
        guard nTokens >= 0 else { return [] }

        return Array(tokens.prefix(Int(nTokens)))
    }

    private func tokenToPiece(vocab: OpaquePointer, token: llama_token) -> String {
        var buf = [CChar](repeating: 0, count: 256)
        let nChars = llama_token_to_piece(vocab, token, &buf, 256, 0, true)
        if nChars > 0 {
            buf[Int(nChars)] = 0
            return String(cString: buf)
        }
        return ""
    }

    private func createSampler(
        temperature: Float,
        topK: Int32,
        topP: Float,
        repetitionPenalty: Float
    ) -> UnsafeMutablePointer<llama_sampler> {
        let sparams = llama_sampler_chain_default_params()
        let chain = llama_sampler_chain_init(sparams)!

        if repetitionPenalty != 1.0 {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, repetitionPenalty, 0.0, 0.0))
        }

        if temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK))
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 0...UInt32.max)))
        }

        return chain
    }

    deinit {
        unloadInternal()
    }
}

nonisolated struct LlamaGenerationResult: Sendable {
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let timeToFirstTokenMS: Double
    let totalDuration: Double
}

nonisolated enum LlamaRunnerError: Error, Sendable, LocalizedError {
    case modelNotLoaded
    case modelLoadFailed
    case contextCreationFailed
    case tokenizationFailed
    case decodeFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No GGUF model loaded"
        case .modelLoadFailed: return "Failed to load GGUF model file"
        case .contextCreationFailed: return "Failed to create llama context"
        case .tokenizationFailed: return "Failed to tokenize input"
        case .decodeFailed: return "Decode step failed"
        }
    }
}
