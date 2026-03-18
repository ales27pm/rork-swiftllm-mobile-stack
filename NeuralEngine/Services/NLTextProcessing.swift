import Foundation
import NaturalLanguage

struct NLTextProcessing {
    struct ProcessedText: Sendable {
        let originalText: String
        let normalizedText: String
        let language: NLLanguage?
        let sentences: [String]
        let tokens: [String]
        let lemmas: [String]
        let namedEntities: [String]

        var searchableTerms: [String] {
            let combined = lemmas.isEmpty ? tokens : lemmas + tokens
            var seen = Set<String>()
            return combined.filter { seen.insert($0).inserted }
        }
    }

    private static let fallbackStopWords: Set<String> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "don", "now", "and", "but", "or", "if", "while", "this", "that", "these", "those", "it", "its", "i", "me", "my", "you", "your", "he", "she", "we", "they", "them", "his", "her", "our", "their", "what", "which", "who", "whom"
    ]

    static func detectLanguage(for text: String, preferredHint: String? = nil) -> NLLanguage? {
        if let preferredHint {
            return NLLanguage(rawValue: preferredHint)
        }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage
    }

    static func process(text: String, languageHint: String? = nil) -> ProcessedText {
        let language = detectLanguage(for: text, preferredHint: languageHint)
        let sentences = tokenize(text: text, unit: .sentence)
        let tokens = tokenize(text: text, unit: .word)
            .map { normalizeSurface($0) }
            .filter { !$0.isEmpty }

        let tagger = NLTagger(tagSchemes: [.lemma, .nameTypeOrLexicalClass])
        tagger.string = text
        if let language {
            tagger.setLanguage(language, range: text.startIndex..<text.endIndex)
        }

        var lemmas: [String] = []
        var namedEntities: [String] = []

        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lemma, options: options) { tag, tokenRange in
            let token = String(text[tokenRange])
            let normalized = normalizeSurface(token)
            if normalized.isEmpty { return true }
            let lemma = normalizeSurface(tag?.rawValue ?? token)
            lemmas.append(lemma.isEmpty ? normalized : lemma)
            return true
        }

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameTypeOrLexicalClass, options: options) { tag, tokenRange in
            let token = String(text[tokenRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let tag, [.personalName, .placeName, .organizationName].contains(tag), !token.isEmpty {
                namedEntities.append(token)
            }
            return true
        }

        let normalizedText = normalizeSurface((lemmas.isEmpty ? tokens : lemmas).joined(separator: " "))
        return ProcessedText(
            originalText: text,
            normalizedText: normalizedText,
            language: language,
            sentences: sentences,
            tokens: tokens,
            lemmas: lemmas,
            namedEntities: namedEntities
        )
    }

    static func normalizeForMatching(_ text: String, languageHint: String? = nil) -> String {
        process(text: text, languageHint: languageHint).normalizedText
    }

    static func stemmedTerms(_ text: String, languageHint: String? = nil, droppingStopWords: Bool = false) -> [String] {
        let processed = process(text: text, languageHint: languageHint)
        let source = processed.lemmas.isEmpty ? processed.tokens : processed.lemmas
        if !droppingStopWords { return source }
        return source.filter { !fallbackStopWords.contains($0) && $0.count > 2 }
    }

    static func embeddingSimilarity(query: String, document: String, languageHint: String? = nil) -> Double? {
        let language = detectLanguage(for: query + "\n" + document, preferredHint: languageHint)
        guard let language, let embedding = NLEmbedding.sentenceEmbedding(for: language) ?? NLEmbedding.wordEmbedding(for: language) else {
            return nil
        }

        let sentenceDistance = embedding.distance(between: query, and: document)
        if sentenceDistance.isFinite {
            return max(0, 1 - Double(sentenceDistance))
        }

        let queryTerms = stemmedTerms(query, languageHint: language.rawValue, droppingStopWords: true)
        let documentTerms = stemmedTerms(document, languageHint: language.rawValue, droppingStopWords: true)
        guard !queryTerms.isEmpty, !documentTerms.isEmpty else { return nil }

        let queryVectors = queryTerms.compactMap { embedding.vector(for: $0) }
        let documentVectors = documentTerms.compactMap { embedding.vector(for: $0) }
        guard let queryMean = meanVector(queryVectors), let docMean = meanVector(documentVectors) else { return nil }
        return cosineSimilarity(queryMean, docMean)
    }

    private static func tokenize(text: String, unit: NLTokenUnit) -> [String] {
        let tokenizer = NLTokenizer(unit: unit)
        tokenizer.string = text
        var parts: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let piece = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { parts.append(piece) }
            return true
        }
        return parts
    }

    private static func normalizeSurface(_ text: String) -> String {
        text.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
            .lowercased()
            .replacingOccurrences(of: "[^\\p{L}\\p{N}\\s]", with: " ", options: .regularExpression)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func meanVector(_ vectors: [[Double]]) -> [Double]? {
        guard let first = vectors.first else { return nil }
        var accum = Array(repeating: 0.0, count: first.count)
        for vector in vectors {
            guard vector.count == accum.count else { continue }
            for (index, value) in vector.enumerated() {
                accum[index] += value
            }
        }
        let count = Double(vectors.count)
        return accum.map { $0 / count }
    }

    private static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return 0 }
        var dot = 0.0
        var magA = 0.0
        var magB = 0.0
        for index in a.indices {
            dot += a[index] * b[index]
            magA += a[index] * a[index]
            magB += b[index] * b[index]
        }
        let denom = sqrt(magA) * sqrt(magB)
        return denom > 0 ? max(0, dot / denom) : 0
    }
}
