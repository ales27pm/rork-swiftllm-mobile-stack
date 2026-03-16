import Foundation
import CoreML

nonisolated final class DecodeEngine: @unchecked Sendable {
    private let lock = NSLock()
    private var isActive: Bool = false
    private var shouldStop: Bool = false

    struct DecodeStep: Sendable {
        let token: Int
        let logits: [Float]
        let latencyMS: Double
    }

    struct DecodeSession: Sendable {
        let totalTokens: Int
        let durationSeconds: Double
        let tokensPerSecond: Double
        let averageLatencyMS: Double
        let peakLatencyMS: Double
    }

    func decodeToken(
        inputToken: Int,
        runner: CoreMLModelRunner
    ) throws -> DecodeStep {
        let start = Date()
        let logits = try runner.predictLogits(inputIDs: [inputToken])
        let latency = Date().timeIntervalSince(start) * 1000
        return DecodeStep(token: inputToken, logits: logits, latencyMS: latency)
    }

    func decodeLoop(
        initialToken: Int,
        maxTokens: Int,
        runner: CoreMLModelRunner,
        sampler: Sampler,
        eosTokens: Set<Int>,
        recentTokens: [Int],
        thermalDelay: () -> Double,
        shouldCancel: () -> Bool,
        onToken: (Int, [Float], Double) -> Void
    ) throws -> DecodeSession {
        lock.lock()
        guard !isActive else {
            lock.unlock()
            throw DecodeError.alreadyActive
        }
        isActive = true
        shouldStop = false
        lock.unlock()

        defer {
            lock.lock()
            isActive = false
            lock.unlock()
        }

        let startTime = Date()
        var currentToken = initialToken
        var generated = 0
        var peakLatency: Double = 0
        var totalLatency: Double = 0
        var recentWindow = recentTokens

        while generated < maxTokens {
            if shouldCancel() || isStopped { break }

            let delay = thermalDelay()
            if delay > 0 {
                Thread.sleep(forTimeInterval: delay)
            }

            let stepStart = Date()
            let logits = try runner.predictLogits(inputIDs: [currentToken])
            let stepLatency = Date().timeIntervalSince(stepStart) * 1000
            totalLatency += stepLatency
            peakLatency = max(peakLatency, stepLatency)

            let sampledToken = sampler.sample(
                logits: logits,
                recentTokens: Array(recentWindow.suffix(64))
            )

            if eosTokens.contains(sampledToken) { break }

            currentToken = sampledToken
            recentWindow.append(sampledToken)
            generated += 1

            onToken(sampledToken, logits, stepLatency)
        }

        let duration = Date().timeIntervalSince(startTime)
        let tps = Double(generated) / max(duration, 0.001)
        let avgLatency = generated > 0 ? totalLatency / Double(generated) : 0

        return DecodeSession(
            totalTokens: generated,
            durationSeconds: duration,
            tokensPerSecond: tps,
            averageLatencyMS: avgLatency,
            peakLatencyMS: peakLatency
        )
    }

    func verifySpeculativeTokens(
        draftTokens: [Int],
        runner: CoreMLModelRunner,
        sampler: Sampler,
        recentTokens: [Int]
    ) throws -> SpeculativeVerification {
        let startTime = Date()
        let logits = try runner.predictLogits(inputIDs: draftTokens)

        var accepted: [Int] = []
        var rejected: [Int] = []
        var correctionToken: Int?

        let vocabSize = logits.count
        guard vocabSize > 0 else {
            return SpeculativeVerification(
                accepted: [], rejected: draftTokens, correctionToken: nil,
                verificationLatencyMS: Date().timeIntervalSince(startTime) * 1000
            )
        }

        let verifiedToken = sampler.sample(logits: logits, recentTokens: recentTokens)

        if let lastDraft = draftTokens.last, lastDraft == verifiedToken {
            accepted = draftTokens
        } else {
            for (i, draft) in draftTokens.enumerated() {
                let singleLogits = try runner.predictLogits(inputIDs: [draft])
                let verified = sampler.sample(logits: singleLogits, recentTokens: recentTokens + accepted)

                if verified == draft || (i < draftTokens.count - 1 && draftTokens[i + 1] == verified) {
                    accepted.append(draft)
                } else {
                    rejected = Array(draftTokens[i...])
                    correctionToken = verified
                    break
                }
            }
        }

        let latency = Date().timeIntervalSince(startTime) * 1000

        return SpeculativeVerification(
            accepted: accepted,
            rejected: rejected,
            correctionToken: correctionToken,
            verificationLatencyMS: latency
        )
    }

    func stop() {
        lock.lock()
        shouldStop = true
        lock.unlock()
    }

    private var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return shouldStop
    }

    var running: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isActive
    }
}

nonisolated struct SpeculativeVerification: Sendable {
    let accepted: [Int]
    let rejected: [Int]
    let correctionToken: Int?
    let verificationLatencyMS: Double

    var acceptanceRate: Double {
        let total = accepted.count + rejected.count
        guard total > 0 else { return 0 }
        return Double(accepted.count) / Double(total)
    }
}

nonisolated enum DecodeError: Error, Sendable, LocalizedError {
    case alreadyActive
    case verificationFailed(String)

    var errorDescription: String? {
        switch self {
        case .alreadyActive: return "Decode engine is already active"
        case .verificationFailed(let reason): return "Speculative verification failed: \(reason)"
        }
    }
}
