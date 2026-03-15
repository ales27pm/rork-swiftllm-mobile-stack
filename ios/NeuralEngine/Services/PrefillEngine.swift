import Foundation
import CoreML

nonisolated final class PrefillEngine: @unchecked Sendable {
    private let lock = NSLock()
    private var isActive: Bool = false
    private let chunkSize: Int
    private let maxParallelChunks: Int

    init(chunkSize: Int = 512, maxParallelChunks: Int = 4) {
        self.chunkSize = chunkSize
        self.maxParallelChunks = maxParallelChunks
    }

    struct PrefillResult: Sendable {
        let processedTokenCount: Int
        let durationSeconds: Double
        let tokensPerSecond: Double
        let chunksProcessed: Int
        let finalLogits: [Float]
    }

    func prefill(
        tokens: [Int],
        runner: CoreMLModelRunner
    ) throws -> PrefillResult {
        lock.lock()
        guard !isActive else {
            lock.unlock()
            throw PrefillError.alreadyActive
        }
        isActive = true
        lock.unlock()

        defer {
            lock.lock()
            isActive = false
            lock.unlock()
        }

        guard !tokens.isEmpty else {
            throw PrefillError.emptyInput
        }

        let startTime = Date()
        var chunks: [[Int]] = []
        var offset = 0

        while offset < tokens.count {
            let end = min(offset + chunkSize, tokens.count)
            chunks.append(Array(tokens[offset..<end]))
            offset = end
        }

        var lastLogits: [Float] = []
        var totalProcessed = 0

        for chunk in chunks {
            let logits = try runner.predictLogits(inputIDs: chunk)
            totalProcessed += chunk.count
            lastLogits = logits
        }

        let duration = Date().timeIntervalSince(startTime)
        let tps = Double(totalProcessed) / max(duration, 0.001)

        return PrefillResult(
            processedTokenCount: totalProcessed,
            durationSeconds: duration,
            tokensPerSecond: tps,
            chunksProcessed: chunks.count,
            finalLogits: lastLogits
        )
    }

    func prefillBatched(
        tokens: [Int],
        runner: CoreMLModelRunner,
        onChunkComplete: @Sendable (Int, Int) -> Void
    ) throws -> PrefillResult {
        lock.lock()
        guard !isActive else {
            lock.unlock()
            throw PrefillError.alreadyActive
        }
        isActive = true
        lock.unlock()

        defer {
            lock.lock()
            isActive = false
            lock.unlock()
        }

        guard !tokens.isEmpty else {
            throw PrefillError.emptyInput
        }

        let startTime = Date()
        var chunks: [[Int]] = []
        var offset = 0

        while offset < tokens.count {
            let batchEnd = min(offset + chunkSize * maxParallelChunks, tokens.count)
            chunks.append(Array(tokens[offset..<batchEnd]))
            offset = batchEnd
        }

        var lastLogits: [Float] = []
        var totalProcessed = 0

        for (index, chunk) in chunks.enumerated() {
            var subOffset = 0
            while subOffset < chunk.count {
                let subEnd = min(subOffset + chunkSize, chunk.count)
                let subChunk = Array(chunk[subOffset..<subEnd])
                let logits = try runner.predictLogits(inputIDs: subChunk)
                totalProcessed += subChunk.count
                lastLogits = logits
                subOffset = subEnd
            }
            onChunkComplete(index + 1, chunks.count)
        }

        let duration = Date().timeIntervalSince(startTime)
        let tps = Double(totalProcessed) / max(duration, 0.001)

        return PrefillResult(
            processedTokenCount: totalProcessed,
            durationSeconds: duration,
            tokensPerSecond: tps,
            chunksProcessed: chunks.count,
            finalLogits: lastLogits
        )
    }

    func cancel() {
        lock.lock()
        isActive = false
        lock.unlock()
    }

    var running: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isActive
    }
}

nonisolated enum PrefillError: Error, Sendable, LocalizedError {
    case alreadyActive
    case emptyInput
    case chunkFailed(Int, Error)

    var errorDescription: String? {
        switch self {
        case .alreadyActive: return "Prefill engine is already processing"
        case .emptyInput: return "Empty token sequence for prefill"
        case .chunkFailed(let idx, let err): return "Prefill chunk \(idx) failed: \(err.localizedDescription)"
        }
    }
}
