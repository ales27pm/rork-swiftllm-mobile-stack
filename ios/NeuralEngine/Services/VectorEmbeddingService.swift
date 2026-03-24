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

        if let vector = sentenceVector(for: trimmed, language: language) {
            return vector
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
        let denominator = sqrtf(magA) * sqrtf(magB)
        return denominator > 0 ? dot / denominator : 0
    }

    func l2Distance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return Float.greatestFiniteMagnitude }
        var difference = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &difference, 1, vDSP_Length(a.count))
        var sumOfSquares: Float = 0
        vDSP_svesq(difference, 1, &sumOfSquares, vDSP_Length(difference.count))
        return sqrtf(sumOfSquares)
    }

    private func resolveLanguage(for text: String, hint: String?) -> NLLanguage {
        if let hint, let mappedHint = mapLanguageHint(hint) {
            return mappedHint
        }

        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return bestAvailableLanguage(for: recognizer.dominantLanguage ?? .english)
    }

    private func mapLanguageHint(_ hint: String) -> NLLanguage? {
        let normalized = hint
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")

        let supportedLanguages: [String: NLLanguage] = [
            "en": .english,
            "fr": .french,
            "fr-ca": .french,
            "es": .spanish,
            "it": .italian,
            "de": .german,
            "pt": .portuguese,
            "pt-br": .portuguese,
            "zh": .simplifiedChinese,
            "zh-cn": .simplifiedChinese,
            "zh-hans": .simplifiedChinese
        ]

        if let exact = supportedLanguages[normalized] {
            return exact
        }

        if let base = normalized.split(separator: "-").first,
           let mapped = supportedLanguages[String(base)]
        {
            return mapped
        }

        return nil
    }

    private func bestAvailableLanguage(for language: NLLanguage) -> NLLanguage {
        mapLanguageHint(language.rawValue) ?? .english
    }

    private func sentenceVector(for text: String, language: NLLanguage) -> [Float]? {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else { return nil }
        guard let vector = embedding.vector(for: text) else { return nil }
        return resizeVector(vector.map { Float($0) }, targetDim: Self.dimensions)
    }

    private func averagedWordVector(for text: String, language: NLLanguage) -> [Float]? {
        guard let embedding = NLEmbedding.wordEmbedding(for: language) else { return nil }

        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text

        var vectors: [[Float]] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range]).lowercased()
            if let vector = embedding.vector(for: word) {
                vectors.append(vector.map { Float($0) })
            }
            return true
        }

        guard !vectors.isEmpty else { return nil }

        let dimensions = vectors[0].count
        var mean = [Float](repeating: 0, count: dimensions)

        for vector in vectors where vector.count == dimensions {
            vDSP_vadd(mean, 1, vector, 1, &mean, 1, vDSP_Length(dimensions))
        }

        var count = Float(vectors.count)
        vDSP_vsdiv(mean, 1, &count, &mean, 1, vDSP_Length(dimensions))
        return resizeVector(mean, targetDim: Self.dimensions)
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        var magnitude: Float = 0
        vDSP_svesq(vector, 1, &magnitude, vDSP_Length(vector.count))
        magnitude = sqrtf(magnitude)
        guard magnitude > 0 else { return vector }

        var normalized = [Float](repeating: 0, count: vector.count)
        var divisor = magnitude
        vDSP_vsdiv(vector, 1, &divisor, &normalized, 1, vDSP_Length(vector.count))
        return normalized
    }

    private func resizeVector(_ vector: [Float], targetDim: Int) -> [Float] {
        guard !vector.isEmpty else { return [] }

        if vector.count == targetDim {
            return normalize(vector)
        }

        if vector.count > targetDim {
            return normalize(Array(vector.prefix(targetDim)))
        }

        var padded = vector
        padded.append(contentsOf: [Float](repeating: 0, count: targetDim - vector.count))
        return normalize(padded)
    }
}
