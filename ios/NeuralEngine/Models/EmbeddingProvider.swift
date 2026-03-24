import Foundation

nonisolated enum EmbeddingProvider: String, Codable, Sendable {
    case naturalLanguage = "natural_language"
    case llmAugmented = "llm_augmented"
    case externalVector = "external_vector"
    case ggufEmbedding = "gguf_embedding"
}
