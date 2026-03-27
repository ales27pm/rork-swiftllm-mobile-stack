import Foundation
import OSLog
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

    nonisolated private static let logger: Logger = {
        let subsystem = Bundle.main.bundleIdentifier ?? "NeuralEngine"
        return Logger(subsystem: subsystem, category: "EmbeddingLLMService")
    }()

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
            Self.logger.notice("Apple Foundation model unavailable for semantic augmentation")
            return nil
        }

        guard SystemLanguageModel.default.supportsLocale() else {
            Self.logger.notice("Locale unsupported for semantic augmentation locale=\(Locale.current.identifier, privacy: .public)")
            return nil
        }

        let session = LanguageModelSession(
            instructions: semanticAugmentationInstructions(for: Locale.current)
        )

        let keywordText = input.keywords.isEmpty ? "none" : input.keywords.joined(separator: ", ")
        let prompt = semanticAugmentationPrompt(for: input, keywordText: keywordText)

        do {
            let response = try await session.respond(to: prompt)
            return response.content
        } catch LanguageModelSession.GenerationError.guardrailViolation(_) {
            Self.logger.notice("Semantic augmentation blocked by guardrails")
            return nil
        } catch LanguageModelSession.GenerationError.unsupportedLanguageOrLocale(_) {
            Self.logger.notice("Semantic augmentation failed with unsupported language/locale")
            return nil
        } catch {
            Self.logger.error("Semantic augmentation failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    @available(iOS 26.0, *)
    private func semanticAugmentationInstructions(for locale: Locale) -> String {
        let localeInstruction = localeInstructionPrefix(for: locale)

        if #available(iOS 26.4, *) {
            return """
            \(localeInstruction)
            You create compact semantic retrieval expansions for vector search.
            Return ONE line of comma-separated concepts, entities, synonyms, and related search terms.
            MUST keep output factual and grounded in the provided content.
            MUST NOT include harmful, sexual, violent, or self-harm content.
            Do not explain. Do not use numbering.
            """
        } else {
            return """
            \(localeInstruction)
            You create compact semantic retrieval expansions for vector search.
            Return one line of comma-separated concepts, entities, synonyms, and related search terms.
            Do not explain. Do not use numbering.
            """
        }
    }

    @available(iOS 26.0, *)
    private func semanticAugmentationPrompt(for input: EmbeddingLLMInput, keywordText: String) -> String {
        if #available(iOS 26.4, *) {
            return """
            Build a semantic retrieval expansion for this memory item.
            Category: \(input.category)
            Keywords: \(keywordText)
            Content: \(input.content)

            Requirements:
            - one line only
            - 10 to 24 short terms or short phrases
            - include canonical topic, likely paraphrases, related domain vocabulary, and named entities when relevant
            - include only terms grounded in the content and keywords
            - keep it dense for vector search, not conversational
            """
        }

        return """
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
    }

    @available(iOS 26.0, *)
    private func localeInstructionPrefix(for locale: Locale) -> String {
        if Locale.Language(identifier: "en_US").isEquivalent(to: locale.language) {
            return "The person's locale is en_US."
        }
        return "The person's locale is \(locale.identifier)."
    }
    #endif
}
