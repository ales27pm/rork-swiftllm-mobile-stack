import Foundation

nonisolated struct StoredEmbeddingMetadata: Sendable {
    let id: String
    let sourceText: String
    let augmentationText: String?
    let provider: EmbeddingProvider
    let updatedAt: Double
}
