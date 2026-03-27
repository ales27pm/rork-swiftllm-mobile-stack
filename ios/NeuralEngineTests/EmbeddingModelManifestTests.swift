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

    @Test func embeddingPoolingFallsBackToMeanForUnknownRawValue() throws {
        let json = #"""
        {
          "id": "unknown-pooling-fixture",
          "name": "Embedding Fixture",
          "variant": "Q4",
          "parameterCount": "22M",
          "quantization": "Q4_K_M",
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
          "embeddingDimensions": 384,
          "embeddingPooling": "cls"
        }
        """#

        let manifest = try JSONDecoder().decode(ModelManifest.self, from: Data(json.utf8))

        #expect(manifest.embeddingPooling == .mean)
    }

    @Test func nonEmbeddingManifestWithLastTokenPoolingFallsBackToMean() throws {
        let manifest = ModelManifest(
            id: "non-embedding-last-token",
            name: "Fixture",
            variant: "Q4",
            parameterCount: "1B",
            quantization: "Q4_K_M",
            sizeBytes: 42,
            contextLength: 1024,
            architecture: .llama,
            repoID: "fixture/repo",
            tokenizerRepoID: nil,
            modelFilePattern: "fixture.gguf",
            isDraft: false,
            format: .gguf,
            recommendation: nil,
            isEmbedding: false,
            embeddingDimensions: nil,
            embeddingPooling: .lastToken
        )

        #expect(manifest.embeddingPooling == .mean)
    }

    @Test func modelFormatDecodesAppleFoundation() throws {
        let payload = #""appleFoundation""#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelFormat.self, from: payload)
        #expect(decoded == .appleFoundation)
    }

    @Test func modelFormatCaseIterableIncludesAppleFoundation() {
        #expect(ModelFormat.allCases.contains(.appleFoundation))
    }
}
