import Foundation
import CryptoKit
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

    @Test func directoryHashRemainsStableAcrossNestedFiles() throws {
        let fileSystem = FileSystemService()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let nestedDirectories = [
            root.appendingPathComponent("weights", isDirectory: true),
            root.appendingPathComponent("weights/part-0001", isDirectory: true),
            root.appendingPathComponent("metadata/config", isDirectory: true)
        ]
        for directory in nestedDirectories {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        let files: [(String, Data)] = [
            ("Manifest.json", Data("{\"model\":\"fixture\"}".utf8)),
            ("weights/part-0001/tensor-a.bin", Data([0, 1, 2, 3, 4, 5])),
            ("weights/part-0001/tensor-b.bin", Data(repeating: 0xAB, count: 4096)),
            ("metadata/config/tokenizer.json", Data("{\"bos_token\":\"<s>\"}".utf8))
        ]

        for (relativePath, data) in files {
            let fileURL = root.appendingPathComponent(relativePath)
            try data.write(to: fileURL)
        }

        let expectedHash = try legacyDirectoryHash(at: root)
        let actualHash = try #require(fileSystem.computeDirectorySHA256(at: root))
        let assetHash = try #require(fileSystem.computeAssetSHA256(for: root))

        #expect(actualHash == expectedHash)
        #expect(assetHash == expectedHash)
    }

    @Test func persistAssetsCopiesSnapshotsIntoManagedStorage() throws {
        let fileSystem = FileSystemService()

        let sourceFile = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("gguf")
        try Data([0x47, 0x47, 0x55, 0x46, 0x01, 0x02, 0x03, 0x04]).write(to: sourceFile)
        defer { try? FileManager.default.removeItem(at: sourceFile) }

        let sourceTokenizer = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: sourceTokenizer, withIntermediateDirectories: true)
        try Data("{\"hello\":\"world\"}".utf8).write(to: sourceTokenizer.appendingPathComponent("tokenizer.json"))
        defer { try? FileManager.default.removeItem(at: sourceTokenizer) }

        let modelID = "persist-fixture-\(UUID().uuidString)"
        let persistedModel = try fileSystem.persistModelAsset(from: sourceFile, forModelID: modelID)
        let persistedTokenizer = try fileSystem.persistTokenizerAsset(from: sourceTokenizer, forModelID: modelID)
        defer { fileSystem.deleteModelAssets(forModelID: modelID) }

        #expect(persistedModel.path.hasPrefix(fileSystem.modelStorageDirectory.path))
        #expect(persistedTokenizer.path.hasPrefix(fileSystem.tokenizerStorageDirectory.path))
        #expect(FileManager.default.fileExists(atPath: persistedModel.path))
        #expect(FileManager.default.fileExists(atPath: persistedTokenizer.appendingPathComponent("tokenizer.json").path))
        #expect(fileSystem.computeAssetSHA256(for: sourceFile) == fileSystem.computeAssetSHA256(for: persistedModel))
        #expect(fileSystem.computeAssetSHA256(for: sourceTokenizer) == fileSystem.computeAssetSHA256(for: persistedTokenizer))
    }

    private func legacyDirectoryHash(at directory: URL) throws -> String {
        let enumerator = try #require(FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ))

        var fileURLs: [URL] = []
        while let url = enumerator.nextObject() as? URL {
            let isFile = (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
            if isFile {
                fileURLs.append(url)
            }
        }

        fileURLs.sort { $0.path < $1.path }

        var hasher = SHA256()
        for fileURL in fileURLs {
            let relativePath = fileURL.path.replacingOccurrences(of: directory.path, with: "")
            hasher.update(data: Data(relativePath.utf8))

            let fileData = try Data(contentsOf: fileURL)
            let fileDigest = SHA256.hash(data: fileData)
            hasher.update(data: Data(fileDigest.map { String(format: "%02x", $0) }.joined().utf8))
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
