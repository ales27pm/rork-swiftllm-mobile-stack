import Foundation
import NaturalLanguage
import Accelerate

nonisolated final class VectorEmbeddingService: Sendable {
    static let shared = VectorEmbeddingService()
    static let dimensions = 512

    private init() {}

    func embed(_ text: String, languageHint: String? = nil) -> [Float]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let language = resolveLanguage(for: trimmed, hint: languageHint)

        if let vec = sentenceVector(for: trimmed, language: language) {
            return vec
        }

        return averagedWordVector(for: trimmed, language: language)
    }

    func embedBatch(_ texts: [String], languageHint: String? = nil) -> [[Float]?] {
        texts.map { embed($0, languageHint: languageHint) }
    }

    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var magA: Float = 0
        var magB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        vDSP_svesq(a, 1, &magA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &magB, vDSP_Length(b.count))
        let denom = sqrtf(magA) * sqrtf(magB)
        return denom > 0 ? dot / denom : 0
    }

    func l2Distance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return Float.greatestFiniteMagnitude }
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        var sumSq: Float = 0
        vDSP_svesq(diff, 1, &sumSq, vDSP_Length(diff.count))
        return sqrtf(sumSq)
    }

    private func resolveLanguage(for text: String, hint: String?) -> NLLanguage {
        if let hint { return NLLanguage(rawValue: hint) }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage ?? .english
    }

    private func sentenceVector(for text: String, language: NLLanguage) -> [Float]? {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else { return nil }
        guard let vector = embedding.vector(for: text) else { return nil }
        return normalize(vector.map { Float($0) })
    }

    private func averagedWordVector(for text: String, language: NLLanguage) -> [Float]? {
        guard let embedding = NLEmbedding.wordEmbedding(for: language) else { return nil }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var vectors: [[Float]] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range]).lowercased()
            if let vec = embedding.vector(for: word) {
                vectors.append(vec.map { Float($0) })
            }
            return true
        }
        guard !vectors.isEmpty else { return nil }
        let dim = vectors[0].count
        var mean = [Float](repeating: 0, count: dim)
        for vec in vectors {
            guard vec.count == dim else { continue }
            vDSP_vadd(mean, 1, vec, 1, &mean, 1, vDSP_Length(dim))
        }
        var count = Float(vectors.count)
        vDSP_vsdiv(mean, 1, &count, &mean, 1, vDSP_Length(dim))
        if dim != Self.dimensions {
            return resizeVector(mean, targetDim: Self.dimensions)
        }
        return normalize(mean)
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        var magnitude: Float = 0
        vDSP_svesq(vector, 1, &magnitude, vDSP_Length(vector.count))
        magnitude = sqrtf(magnitude)
        guard magnitude > 0 else { return vector }
        var result = [Float](repeating: 0, count: vector.count)
        var mag = magnitude
        vDSP_vsdiv(vector, 1, &mag, &result, 1, vDSP_Length(vector.count))
        return result
    }

    private func resizeVector(_ vector: [Float], targetDim: Int) -> [Float] {
        if vector.count == targetDim { return normalize(vector) }
        if vector.count > targetDim {
            return normalize(Array(vector.prefix(targetDim)))
        }
        var padded = vector
        padded.append(contentsOf: [Float](repeating: 0, count: targetDim - vector.count))
        return normalize(padded)
    }
}
