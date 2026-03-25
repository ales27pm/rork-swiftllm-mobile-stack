import Foundation
import Testing
@testable import NeuralEngine

@MainActor
struct ModelLoaderGenerationDrainTests {
    @Test func deleteModel_waitsForDrainHandlerBeforeRemovingAssets() async throws {
        let suiteName = "model-loader-delete-drain-\(UUID().uuidString)"
        let keyValueStore = KeyValueStore(suiteName: suiteName)
        let loader = ModelLoaderService(keyValueStore: keyValueStore)
        let fileSystem = FileSystemService()
        let modelID = "fixture-delete-drain-\(UUID().uuidString)"
        let manifest = ModelManifest(
            id: modelID,
            name: "Fixture",
            variant: "Delete Drain",
            parameterCount: "1B",
            quantization: "Q4",
            sizeBytes: 4096,
            contextLength: 128,
            architecture: .llama,
            repoID: "fixture/repo",
            tokenizerRepoID: nil,
            modelFilePattern: "fixture.gguf",
            checksum: String(repeating: "a", count: 64),
            isDraft: false,
            format: .gguf
        )

        loader.availableModels.append(manifest)
        loader.modelStatuses[modelID] = .ready
        loader.activeModelID = modelID
        loader.activeFormat = .gguf

        let modelContainer = fileSystem.modelContainerDirectory(forModelID: modelID)
        try FileManager.default.createDirectory(at: modelContainer, withIntermediateDirectories: true)
        let modelURL = modelContainer.appendingPathComponent("fixture.gguf")
        try Data([0x47, 0x47, 0x55, 0x46, 0x01, 0x02, 0x03, 0x04]).write(to: modelURL)
        try fileSystem.saveModelPath(modelURL, forModelID: modelID)

        let drainState = DrainState()
        loader.setGenerationDrainHandler {
            await drainState.markStarted()
            try? await Task.sleep(for: .milliseconds(80))
            await drainState.markFinished()
        }

        loader.deleteModel(modelID)

        try await Task.sleep(for: .milliseconds(20))
        #expect(fileSystem.exists(at: modelURL))
        #expect(await drainState.started)
        #expect(!(await drainState.finished))

        try await Task.sleep(for: .milliseconds(140))
        #expect(await drainState.finished)
        #expect(!fileSystem.exists(at: modelURL))
        #expect(loader.activeModelID == nil)
        #expect(loader.modelStatuses[modelID] == .notDownloaded)

        keyValueStore.removeAll()
    }

    @Test func ensureModelLoaded_forceReloadAwaitsDrainHandler() async throws {
        let suiteName = "model-loader-force-reload-\(UUID().uuidString)"
        let keyValueStore = KeyValueStore(suiteName: suiteName)
        let loader = ModelLoaderService(keyValueStore: keyValueStore)
        let modelID = "fixture-force-reload-\(UUID().uuidString)"
        let manifest = ModelManifest(
            id: modelID,
            name: "Fixture",
            variant: "Force Reload",
            parameterCount: "1B",
            quantization: "Q4",
            sizeBytes: 4096,
            contextLength: 128,
            architecture: .llama,
            repoID: "fixture/repo",
            tokenizerRepoID: nil,
            modelFilePattern: "missing.gguf",
            checksum: String(repeating: "b", count: 64),
            isDraft: false,
            format: .gguf
        )

        loader.availableModels.append(manifest)
        loader.modelStatuses[modelID] = .ready

        let drainState = DrainState()
        loader.setGenerationDrainHandler {
            await drainState.markStarted()
            try? await Task.sleep(for: .milliseconds(80))
            await drainState.markFinished()
        }

        let start = Date()
        let didLoad = await loader.ensureModelLoaded(modelID, forceReload: true, persistSelection: false)
        let elapsed = Date().timeIntervalSince(start)

        #expect(!didLoad)
        #expect(await drainState.started)
        #expect(await drainState.finished)
        #expect(elapsed >= 0.07)

        keyValueStore.removeAll()
    }

    @Test func activateDraftModel_waitsForDrainHandlerBeforeLoadAttempt() async throws {
        let suiteName = "model-loader-draft-drain-\(UUID().uuidString)"
        let keyValueStore = KeyValueStore(suiteName: suiteName)
        let loader = ModelLoaderService(keyValueStore: keyValueStore)
        let fileSystem = FileSystemService()
        let modelID = "fixture-draft-drain-\(UUID().uuidString)"
        let manifest = ModelManifest(
            id: modelID,
            name: "Fixture Draft",
            variant: "Draft",
            parameterCount: "0.5B",
            quantization: "Q8",
            sizeBytes: 4096,
            contextLength: 128,
            architecture: .llama,
            repoID: "fixture/draft",
            tokenizerRepoID: nil,
            modelFilePattern: "draft.gguf",
            checksum: "",
            isDraft: true,
            format: .gguf
        )

        loader.availableModels.append(manifest)
        loader.modelStatuses[modelID] = .ready

        let modelContainer = fileSystem.modelContainerDirectory(forModelID: modelID)
        try FileManager.default.createDirectory(at: modelContainer, withIntermediateDirectories: true)
        let modelURL = modelContainer.appendingPathComponent("draft.gguf")
        try Data([0x47, 0x47, 0x55, 0x46, 0x01, 0x02, 0x03, 0x04]).write(to: modelURL)
        try fileSystem.saveModelPath(modelURL, forModelID: modelID)

        let drainState = DrainState()
        loader.setGenerationDrainHandler {
            await drainState.markStarted()
            try? await Task.sleep(for: .milliseconds(80))
            await drainState.markFinished()
        }

        loader.activateDraftModel(modelID)

        try await Task.sleep(for: .milliseconds(20))
        #expect(await drainState.started)
        #expect(!(await drainState.finished))
        #expect(loader.activeDraftModelID == nil)

        try await Task.sleep(for: .milliseconds(140))
        #expect(await drainState.finished)
        #expect(loader.activeDraftModelID == nil)

        keyValueStore.removeAll()
        fileSystem.deleteModelAssets(forModelID: modelID)
    }
}

private actor DrainState {
    private(set) var started: Bool = false
    private(set) var finished: Bool = false

    func markStarted() {
        started = true
    }

    func markFinished() {
        finished = true
    }
}
