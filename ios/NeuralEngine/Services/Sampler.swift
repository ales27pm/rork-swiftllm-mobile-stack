import Foundation

nonisolated final class SeededRandomSource: @unchecked Sendable {
    private let lock = NSLock()
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0xA5A5_A5A5_A5A5_A5A5 : seed
    }

    func nextFloat() -> Float {
        lock.lock()
        defer { lock.unlock() }

        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z = z ^ (z >> 31)

        let maxValue = Double(UInt64.max)
        return Float(Double(z) / maxValue)
    }
}

nonisolated struct SamplingDistribution: Sendable {
    let probabilities: [Float]

    func probability(of token: Int) -> Float {
        guard token >= 0, token < probabilities.count else { return 0 }
        return probabilities[token]
    }
}

nonisolated struct Sampler: Sendable {
    let config: SamplingConfig
    private let randomSource: SeededRandomSource?

    init(config: SamplingConfig) {
        self.config = config
        if let seed = config.samplerSeed {
            self.randomSource = SeededRandomSource(seed: seed)
        } else {
            self.randomSource = nil
        }
    }

    func sample(logits: [Float], recentTokens: [Int]) -> Int {
        guard !logits.isEmpty else { return 0 }
        let distribution = prepareDistribution(logits: logits, recentTokens: recentTokens)
        return sample(from: distribution)
    }

    func probability(of token: Int, logits: [Float], recentTokens: [Int]) -> Float {
        guard token >= 0 else { return 0 }
        let distribution = prepareDistribution(logits: logits, recentTokens: recentTokens)
        return distribution.probability(of: token)
    }

    func probabilityDistribution(logits: [Float], recentTokens: [Int]) -> [Float] {
        prepareDistribution(logits: logits, recentTokens: recentTokens).probabilities
    }

    func prepareDistribution(logits: [Float], recentTokens: [Int]) -> SamplingDistribution {
        guard !logits.isEmpty else { return SamplingDistribution(probabilities: []) }

        var processed = logits
        applyRepetitionPenalty(&processed, recentTokens: recentTokens)
        applyTemperature(&processed)

        let topKIndices = topKIndexSet(processed)
        let topPIndices = topPIndexSet(processed, restrictedTo: topKIndices)

        var filtered = [Float](repeating: -.greatestFiniteMagnitude, count: processed.count)
        for idx in topPIndices {
            filtered[idx] = processed[idx]
        }

        return SamplingDistribution(probabilities: softmax(filtered))
    }

    func sampleResidual(target: [Float], draft: [Float]) -> Int {
        guard target.count == draft.count, !target.isEmpty else { return 0 }
        var residual = [Float](repeating: 0, count: target.count)
        var total: Float = 0

        for i in residual.indices {
            let value = max(0, target[i] - draft[i])
            residual[i] = value
            total += value
        }

        guard total > 0 else {
            return sample(from: SamplingDistribution(probabilities: target))
        }

        for i in residual.indices {
            residual[i] /= total
        }

        return sample(from: SamplingDistribution(probabilities: residual))
    }

    func uniformSample() -> Float {
        randomSource?.nextFloat() ?? Float.random(in: 0..<1)
    }

    func sample(from distribution: SamplingDistribution) -> Int {
        sample(fromProbabilities: distribution.probabilities)
    }

    private func sample(fromProbabilities probs: [Float]) -> Int {
        guard !probs.isEmpty else { return 0 }
        let r = uniformSample()
        var cumulative: Float = 0

        for (i, prob) in probs.enumerated() where prob > 0 {
            cumulative += prob
            if r <= cumulative {
                return i
            }
        }

        return probs.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }

    private func applyRepetitionPenalty(_ logits: inout [Float], recentTokens: [Int]) {
        let penalty = config.repetitionPenalty
        guard penalty != 1.0 else { return }

        let recentSet = Set(recentTokens.suffix(64))
        for tokenID in recentSet where tokenID < logits.count {
            if logits[tokenID] > 0 {
                logits[tokenID] /= penalty
            } else {
                logits[tokenID] *= penalty
            }
        }
    }

    private func applyTemperature(_ logits: inout [Float]) {
        let temp = max(config.temperature, 1e-7)
        guard temp != 1.0 else { return }
        for i in logits.indices {
            logits[i] /= temp
        }
    }

    private func topKIndexSet(_ logits: [Float]) -> Set<Int> {
        let limit = min(max(config.topK, 1), logits.count)
        let sorted = logits.enumerated().sorted { $0.element > $1.element }
        return Set(sorted.prefix(limit).map(\.offset))
    }

    private func topPIndexSet(_ logits: [Float], restrictedTo indices: Set<Int>) -> Set<Int> {
        let sorted = indices
            .map { ($0, logits[$0]) }
            .sorted { $0.1 > $1.1 }

        guard !sorted.isEmpty else { return [] }

        let maxLogit = sorted.first?.1 ?? 0
        let exps = sorted.map { exp($0.1 - maxLogit) }
        let sumExps = exps.reduce(0, +)

        var cumulative: Float = 0
        var accepted: Set<Int> = []

        for (i, entry) in sorted.enumerated() {
            let prob = exps[i] / max(sumExps, 1e-12)
            cumulative += prob
            accepted.insert(entry.0)
            if cumulative >= config.topP {
                break
            }
        }

        return accepted
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { logit -> Float in
            if logit == -.greatestFiniteMagnitude { return 0 }
            return exp(logit - maxLogit)
        }
        let sumExps = exps.reduce(0, +)
        guard sumExps > 0 else { return [Float](repeating: 0, count: logits.count) }
        return exps.map { $0 / sumExps }
    }
}
