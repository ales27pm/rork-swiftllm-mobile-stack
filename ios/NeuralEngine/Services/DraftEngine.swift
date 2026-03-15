import Foundation
import CoreML

nonisolated final class DraftEngine: @unchecked Sendable {
    private let lock = NSLock()
    private var draftRunner: CoreMLModelRunner?
    private var isActive: Bool = false

    struct DraftSequence: Sendable {
        let tokens: [Int]
        let logitSnapshots: [[Float]]
        let confidenceScores: [Float]
        let draftLatencyMS: Double
    }

    var hasDraftModel: Bool {
        lock.lock()
        defer { lock.unlock() }
        return draftRunner?.isLoaded ?? false
    }

    func attachRunner(_ runner: CoreMLModelRunner) {
        lock.lock()
        draftRunner = runner
        lock.unlock()
    }

    func detachRunner() {
        lock.lock()
        draftRunner?.unload()
        draftRunner = nil
        lock.unlock()
    }

    func generateDraftTokens(
        seedToken: Int,
        count: Int,
        sampler: Sampler,
        recentTokens: [Int],
        minConfidence: Float = 0.3
    ) throws -> DraftSequence {
        lock.lock()
        guard let runner = draftRunner, runner.isLoaded else {
            lock.unlock()
            throw DraftError.noDraftModel
        }
        guard !isActive else {
            lock.unlock()
            throw DraftError.alreadyActive
        }
        isActive = true
        lock.unlock()

        defer {
            lock.lock()
            isActive = false
            lock.unlock()
        }

        let startTime = Date()
        var draftTokens: [Int] = []
        var logitSnapshots: [[Float]] = []
        var confidenceScores: [Float] = []
        var currentToken = seedToken
        var contextWindow = recentTokens

        for _ in 0..<count {
            let logits = try runner.predictLogits(inputIDs: [currentToken])

            let confidence = computeTopTokenConfidence(logits: logits)

            if confidence < minConfidence && !draftTokens.isEmpty {
                break
            }

            let token = sampler.sample(logits: logits, recentTokens: Array(contextWindow.suffix(64)))
            draftTokens.append(token)
            logitSnapshots.append(logits)
            confidenceScores.append(confidence)
            currentToken = token
            contextWindow.append(token)
        }

        let latency = Date().timeIntervalSince(startTime) * 1000

        return DraftSequence(
            tokens: draftTokens,
            logitSnapshots: logitSnapshots,
            confidenceScores: confidenceScores,
            draftLatencyMS: latency
        )
    }

    func generateDraftBurst(
        seedToken: Int,
        count: Int,
        sampler: Sampler,
        recentTokens: [Int],
        entropyThreshold: Float = 2.5
    ) throws -> DraftSequence {
        lock.lock()
        guard let runner = draftRunner, runner.isLoaded else {
            lock.unlock()
            throw DraftError.noDraftModel
        }
        guard !isActive else {
            lock.unlock()
            throw DraftError.alreadyActive
        }
        isActive = true
        lock.unlock()

        defer {
            lock.lock()
            isActive = false
            lock.unlock()
        }

        let startTime = Date()
        var draftTokens: [Int] = []
        var logitSnapshots: [[Float]] = []
        var confidenceScores: [Float] = []
        var currentToken = seedToken
        var contextWindow = recentTokens

        for _ in 0..<count {
            let logits = try runner.predictLogits(inputIDs: [currentToken])

            let entropy = computeShannonEntropy(logits: logits)
            let confidence = 1.0 - min(Float(entropy) / entropyThreshold, 1.0)

            if entropy > Double(entropyThreshold) && !draftTokens.isEmpty {
                break
            }

            let token = sampler.sample(logits: logits, recentTokens: Array(contextWindow.suffix(64)))
            draftTokens.append(token)
            logitSnapshots.append(logits)
            confidenceScores.append(confidence)
            currentToken = token
            contextWindow.append(token)
        }

        let latency = Date().timeIntervalSince(startTime) * 1000

        return DraftSequence(
            tokens: draftTokens,
            logitSnapshots: logitSnapshots,
            confidenceScores: confidenceScores,
            draftLatencyMS: latency
        )
    }

    private func computeTopTokenConfidence(logits: [Float]) -> Float {
        guard !logits.isEmpty else { return 0 }

        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(Float(0), +)

        guard sumExps > 0 else { return 0 }

        let topProb = exp(logits.max()! - maxLogit) / sumExps
        return topProb
    }

    private func computeShannonEntropy(logits: [Float]) -> Double {
        guard !logits.isEmpty else { return 0 }

        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(Float(0), +)

        guard sumExps > 0 else { return 0 }

        var entropy: Double = 0
        for e in exps {
            let p = Double(e / sumExps)
            if p > 1e-10 {
                entropy -= p * log(p)
            }
        }

        return entropy
    }

    func resetDraftState() {
        lock.lock()
        draftRunner?.resetState()
        lock.unlock()
    }

    var running: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isActive
    }
}

nonisolated enum DraftError: Error, Sendable, LocalizedError {
    case noDraftModel
    case alreadyActive
    case lowConfidence

    var errorDescription: String? {
        switch self {
        case .noDraftModel: return "No draft model loaded for speculative decoding"
        case .alreadyActive: return "Draft engine is already generating"
        case .lowConfidence: return "Draft sequence confidence below threshold"
        }
    }
}
