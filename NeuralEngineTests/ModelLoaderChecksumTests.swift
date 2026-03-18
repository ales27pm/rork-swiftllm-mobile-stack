import Foundation
import Testing
@testable import NeuralEngine

@MainActor
struct ModelLoaderChecksumTests {
    private struct FixtureManifest: Decodable {
        let id: String
        let checksum: String
    }

    private static let checksumFixture = #"""
    [
      {"id":"missing-checksum","checksum":""},
      {"id":"mismatched-checksum","checksum":"0000000000000000000000000000000000000000000000000000000000000000"}
    ]
    """#

    @Test func registryManifestWithoutChecksumIsUnsupported() throws {
        let entries = try JSONDecoder().decode([FixtureManifest].self, from: Data(Self.checksumFixture.utf8))
        let missing = try #require(entries.first(where: { $0.id == "missing-checksum" }))

        let manifest = ModelManifest(
            id: missing.id,
            name: "Fixture",
            variant: "Missing checksum",
            parameterCount: "1",
            quantization: "Q4",
            sizeBytes: 4,
            contextLength: 128,
            architecture: .llama,
            repoID: "fixture/repo",
            tokenizerRepoID: nil,
            modelFilePattern: "fixture.gguf",
            checksum: missing.checksum,
            isDraft: false,
            format: .gguf
        )

        #expect(ModelLoaderService.registryIssue(for: manifest) == "Missing required model checksum in registry.")
        #expect(ModelLoaderService().resolveRestoredStatus(for: manifest, modelURL: nil, tokenizerURL: nil) == .unsupported("Checksum unavailable. Missing required model checksum in registry. Delete any local copy and wait for an updated registry."))
    }

    @Test func checksumMismatchNeverBecomesReady() throws {
        let entries = try JSONDecoder().decode([FixtureManifest].self, from: Data(Self.checksumFixture.utf8))
        let mismatch = try #require(entries.first(where: { $0.id == "mismatched-checksum" }))

        let tempFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("gguf")
        try Data("fixture-gguf".utf8).write(to: tempFile)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let manifest = ModelManifest(
            id: mismatch.id,
            name: "Fixture",
            variant: "Checksum mismatch",
            parameterCount: "1",
            quantization: "Q4",
            sizeBytes: Int64(Data("fixture-gguf".utf8).count),
            contextLength: 128,
            architecture: .llama,
            repoID: "fixture/repo",
            tokenizerRepoID: nil,
            modelFilePattern: tempFile.lastPathComponent,
            checksum: mismatch.checksum,
            isDraft: false,
            format: .gguf
        )

        let loader = ModelLoaderService()
        let status = loader.resolveRestoredStatus(for: manifest, modelURL: tempFile, tokenizerURL: nil)

        if case .ready = status {
            Issue.record("Checksum-mismatched assets must never transition to ready")
        }

        #expect(status == .checksumFailed("Checksum mismatch detected. Delete and re-download."))
    }
}
