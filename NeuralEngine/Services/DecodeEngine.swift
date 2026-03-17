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
        draftSequence: DraftEngine.DraftSequence,
        runner: LogitsPredicting,
        sampler: Sampler,
        recentTokens: [Int]
    ) throws -> SpeculativeVerification {
        let startTime = Date()
        let draftTokens = draftSequence.tokens

        guard !draftTokens.isEmpty else {
            return SpeculativeVerification(
                accepted: [],
                rejected: [],
                correctionToken: nil,
                correctionSampled: false,
                verificationLatencyMS: Date().timeIntervalSince(startTime) * 1000
            )
        }

        let targetLogitsSpan = try runner.predictLogitsSpan(inputIDs: draftTokens)
        let spanCount = min(draftTokens.count, targetLogitsSpan.count)

        var accepted: [Int] = []
        var rejected: [Int] = []
        var correctionToken: Int?
        var rejectionIndex: Int?
        var contextWindow = recentTokens

        for i in 0..<spanCount {
            let token = draftTokens[i]
            let targetDist = sampler.probabilityDistribution(logits: targetLogitsSpan[i], recentTokens: Array(contextWindow.suffix(64)))
            let targetProb = token < targetDist.count ? targetDist[token] : 0
            let draftProb = i < draftSequence.draftTokenProbabilities.count ? draftSequence.draftTokenProbabilities[i] : 0
            let acceptance = draftProb > 0 ? min(1.0, targetProb / draftProb) : (targetProb > 0 ? 1.0 : 0)

            if sampler.uniformSample() <= acceptance {
                accepted.append(token)
                contextWindow.append(token)
                continue
            }

            rejectionIndex = i
            rejected = Array(draftTokens[i...])

            if i < draftSequence.logitSnapshots.count {
                let draftDist = sampler.probabilityDistribution(logits: draftSequence.logitSnapshots[i], recentTokens: Array(contextWindow.suffix(64)))
                correctionToken = sampler.sampleResidual(target: targetDist, draft: draftDist)
            } else {
                correctionToken = sampler.sample(logits: targetLogitsSpan[i], recentTokens: Array(contextWindow.suffix(64)))
            }
            break
        }

        if rejectionIndex == nil && draftTokens.count > spanCount {
            rejected = Array(draftTokens[spanCount...])
        }

        return SpeculativeVerification(
            accepted: accepted,
            rejected: rejected,
            correctionToken: correctionToken,
            correctionSampled: correctionToken != nil,
            verificationLatencyMS: Date().timeIntervalSince(startTime) * 1000
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
    let correctionSampled: Bool
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
