import Foundation
import Testing
@testable import NeuralEngine

struct EmbeddingModelManifestTests {
    @Test func embeddingPoolingDefaultsToMeanWhenMissingFromRegistry() throws {
        let json = #"""
        {
          "id": "embed-default-pooling",
          "name": "Embedding Fixture",
          "variant": "Q8",
          "parameterCount": "22M",
          "quantization": "Q8_0",
          "sizeBytes": 24,
          "contextLength": 512,
          "architecture": "bert",
          "repoID": "fixture/repo",
          "tokenizerRepoID": null,
          "modelFilePattern": "fixture.gguf",
          "isDraft": false,
          "format": "gguf",
          "recommendation": null,
          "isEmbedding": true,
          "embeddingDimensions": 384
        }
        """#

        let manifest = try JSONDecoder().decode(ModelManifest.self, from: Data(json.utf8))

        #expect(manifest.isEmbedding)
        #expect(manifest.embeddingPooling == .mean)
    }

    @Test func embeddingPoolingDecodesLastTokenForQwenEmbeddingManifest() throws {
        let json = #"""
        {
          "id": "qwen3-embed-fixture",
          "name": "Qwen3 Embedding",
          "variant": "0.6B Q3_K_M imat GGUF",
          "parameterCount": "0.6B",
          "quantization": "Q3_K_M",
          "sizeBytes": 331000000,
          "contextLength": 32768,
          "architecture": "qwen",
          "repoID": "PeterAM4/Qwen3-Embedding-0.6B-GGUF",
          "tokenizerRepoID": null,
          "modelFilePattern": "Qwen3-Embedding-0.6B-Q3_K_M-imat.gguf",
          "isDraft": false,
          "format": "gguf",
          "recommendation": null,
          "isEmbedding": true,
          "embeddingDimensions": 1024,
          "embeddingPooling": "last_token"
        }
        """#

        let manifest = try JSONDecoder().decode(ModelManifest.self, from: Data(json.utf8))

        #expect(manifest.embeddingPooling == .lastToken)
    }
}
