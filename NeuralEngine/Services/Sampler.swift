import Foundation

nonisolated struct Sampler: Sendable {
    let config: SamplingConfig

    func sample(logits: [Float], recentTokens: [Int]) -> Int {
        guard !logits.isEmpty else { return 0 }

        var processed = logits

        applyRepetitionPenalty(&processed, recentTokens: recentTokens)
        applyTemperature(&processed)

        let topKFiltered = applyTopK(processed)
        let topPFiltered = applyTopP(topKFiltered)

        return weightedSample(topPFiltered)
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

    private func applyTopK(_ logits: [Float]) -> [(index: Int, value: Float)] {
        let indexed = logits.enumerated().map { (index: $0.offset, value: $0.element) }
        let sorted = indexed.sorted { $0.value > $1.value }
        return Array(sorted.prefix(config.topK))
    }

    private func applyTopP(_ candidates: [(index: Int, value: Float)]) -> [(index: Int, value: Float)] {
        let maxLogit = candidates.first?.value ?? 0
        let exps = candidates.map { exp($0.value - maxLogit) }
        let sumExps = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExps }

        var cumulative: Float = 0
        var filtered: [(index: Int, value: Float)] = []

        for (i, prob) in probs.enumerated() {
            cumulative += prob
            filtered.append(candidates[i])
            if cumulative >= config.topP { break }
        }

        return filtered
    }

    private func weightedSample(_ candidates: [(index: Int, value: Float)]) -> Int {
        guard !candidates.isEmpty else { return 0 }

        let maxLogit = candidates.max(by: { $0.value < $1.value })?.value ?? 0
        let exps = candidates.map { exp($0.value - maxLogit) }
        let sumExps = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExps }

        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0

        for (i, prob) in probs.enumerated() {
            cumulative += prob
            if r < cumulative {
                return candidates[i].index
            }
        }

        return candidates.last?.index ?? 0
    }
}
