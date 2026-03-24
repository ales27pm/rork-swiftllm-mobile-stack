import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

nonisolated struct EmbeddingLLMInput: Sendable, Hashable {
    let content: String
    let keywords: [String]
    let category: String
}

@MainActor
final class EmbeddingLLMService {
    typealias Enhancer = @Sendable (EmbeddingLLMInput) async -> String?

    static let shared = EmbeddingLLMService()

    private let enhancer: Enhancer?
    private var cachedOutputs: [EmbeddingLLMInput: String] = [:]

    init(enhancer: Enhancer? = nil) {
        self.enhancer = enhancer
    }

    func generateSemanticAugmentation(for input: EmbeddingLLMInput) async -> String? {
        guard !input.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }

        if let cached = cachedOutputs[input] {
            return cached
        }

        let rawOutput: String?
        if let enhancer {
            rawOutput = await enhancer(input)
        } else {
            rawOutput = await generateWithFoundationModels(for: input)
        }

        guard let sanitized = sanitize(rawOutput) else {
            return nil
        }

        cachedOutputs[input] = sanitized
        return sanitized
    }

    private func sanitize(_ text: String?) -> String? {
        guard let text else { return nil }

        let cleaned = text
            .replacingOccurrences(of: "\n", with: ", ")
            .replacingOccurrences(of: "•", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard cleaned.count >= 12 else { return nil }
        if cleaned.count <= 360 {
            return cleaned
        }
        return String(cleaned.prefix(360)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func generateWithFoundationModels(for input: EmbeddingLLMInput) async -> String? {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            return await generateWithFoundationModels26(for: input)
        }
        #endif
        return nil
    }

    #if canImport(FoundationModels)
    @available(iOS 26.0, *)
    private func generateWithFoundationModels26(for input: EmbeddingLLMInput) async -> String? {
        guard SystemLanguageModel.default.isAvailable else {
            return nil
        }

        let session = LanguageModelSession(
            instructions: "You create compact semantic retrieval expansions for vector search. Return a single line of comma-separated concepts, entities, synonyms, and related search terms. Do not explain. Do not use numbering."
        )

        let keywordText = input.keywords.isEmpty ? "none" : input.keywords.joined(separator: ", ")
        let prompt = """
        Build a semantic retrieval expansion for this memory item.

        Category: \(input.category)
        Keywords: \(keywordText)
        Content: \(input.content)

        Requirements:
        - one line only
        - 8 to 24 short terms or short phrases
        - include canonical topic, likely paraphrases, related domain vocabulary, and named entities when relevant
        - keep it dense for vector search, not conversational
        """

        do {
            let response = try await session.respond(to: prompt)
            return response.content
        } catch {
            return nil
        }
    }
    #endif
}
