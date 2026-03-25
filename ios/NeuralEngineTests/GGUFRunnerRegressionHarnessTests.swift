import Foundation
import Testing
@testable import NeuralEngine

@MainActor
struct GGUFRunnerRegressionHarnessTests {
    @Test func unloadDuringActiveDecode_cancelsGenerationAndUnloadsRunner() async {
        let runner = LlamaModelRunner()
        runner.installSyntheticModelForTesting(
            configuration: LlamaSyntheticTestingConfiguration(
                plannedTokens: [5, 6, 0],
                eogTokens: [0],
                tokenPieces: [5: "A", 6: "B"],
                decodeDelaySeconds: 0.03,
                unloadDelaySeconds: 0.01,
                vocabSize: 16
            )
        )

        let generationTask = makeGenerationTask(runner: runner, prompt: longPrompt)
        let enteredDecode = await waitUntil {
            runner.isSyntheticDecodeActiveForTesting()
        }
        #expect(enteredDecode)
        guard enteredDecode else { return }

        let unloadTask = Task.detached(priority: .userInitiated) {
            runner.unload()
        }

        let generationResult = await generationTask.value
        _ = await unloadTask.value

        #expect(isGenerationCancelled(generationResult))
        #expect(runner.syntheticDecodeCallCountForTesting() > 0)
        #expect(!runner.isLoaded)
        #expect(runner.currentState == .idle)
    }

    @Test func forceReloadDuringActiveDecode_drainsThenReloads() async throws {
        let context = try makeLoaderContext(prefix: "force-reload")
        defer {
            context.keyValueStore.removeAll()
            context.fileSystem.deleteModelAssets(forModelID: context.targetManifest.id)
        }

        context.loader.llamaRunner.setSyntheticLoadConfigurationProviderForTesting { path, _, _ in
            path == context.targetURL.path ? context.targetConfiguration : nil
        }

        let initialLoad = await context.loader.ensureModelLoaded(context.targetManifest.id, persistSelection: false)
        #expect(initialLoad)
        guard initialLoad else { return }

        let generationTask = makeGenerationTask(runner: context.loader.llamaRunner, prompt: longPrompt)
        let enteredDecode = await waitUntil {
            context.loader.llamaRunner.isSyntheticDecodeActiveForTesting()
        }
        #expect(enteredDecode)
        guard enteredDecode else { return }

        let drainState = DrainRecorder()
        context.loader.setGenerationDrainHandler {
            await drainState.markStarted()
            generationTask.cancel()
            _ = await generationTask.value
            await drainState.markFinished()
        }

        let reloaded = await context.loader.ensureModelLoaded(context.targetManifest.id, forceReload: true, persistSelection: false)
        let generationResult = await generationTask.value

        #expect(reloaded)
        #expect(await drainState.started)
        #expect(await drainState.finished)
        #expect(isGenerationCancelled(generationResult))
        #expect(context.loader.llamaRunner.isLoaded)
        #expect(context.loader.llamaRunner.currentState == .ready)
        #expect(context.loader.activeModelID == context.targetManifest.id)
    }

    @Test func deleteCurrentModelDuringActiveDecode_waitsForDrainThenRemovesAssets() async throws {
        let context = try makeLoaderContext(prefix: "delete-current")
        defer {
            context.keyValueStore.removeAll()
            context.fileSystem.deleteModelAssets(forModelID: context.targetManifest.id)
        }

        context.loader.llamaRunner.setSyntheticLoadConfigurationProviderForTesting { path, _, _ in
            path == context.targetURL.path ? context.targetConfiguration : nil
        }

        let initialLoad = await context.loader.ensureModelLoaded(context.targetManifest.id, persistSelection: false)
        #expect(initialLoad)
        guard initialLoad else { return }

        let generationTask = makeGenerationTask(runner: context.loader.llamaRunner, prompt: longPrompt)
        let enteredDecode = await waitUntil {
            context.loader.llamaRunner.isSyntheticDecodeActiveForTesting()
        }
        #expect(enteredDecode)
        guard enteredDecode else { return }

        let drainState = DrainRecorder()
        context.loader.setGenerationDrainHandler {
            await drainState.markStarted()
            generationTask.cancel()
            _ = await generationTask.value
            await drainState.markFinished()
        }

        context.loader.deleteModel(context.targetManifest.id)

        let settled = await waitUntil(timeout: .seconds(5)) {
            await drainState.finished &&
            context.loader.activeModelID == nil &&
            context.loader.modelStatuses[context.targetManifest.id] == .notDownloaded &&
            !context.fileSystem.exists(at: context.targetURL)
        }
        let generationResult = await generationTask.value

        #expect(settled)
        #expect(await drainState.started)
        #expect(await drainState.finished)
        #expect(isGenerationCancelled(generationResult))
        #expect(context.loader.activeModelID == nil)
        #expect(context.loader.modelStatuses[context.targetManifest.id] == .notDownloaded)
        #expect(!context.fileSystem.exists(at: context.targetURL))
    }

    @Test func draftRunnerUnloadDuringSpeculativeGeneration_cancelsSafely() async {
        let target = LlamaModelRunner()
        let draft = LlamaModelRunner()

        target.installSyntheticModelForTesting(
            configuration: LlamaSyntheticTestingConfiguration(
                plannedTokens: [9, 10, 11, 0],
                eogTokens: [0],
                tokenPieces: [9: "x", 10: "y", 11: "z"],
                decodeDelaySeconds: 0.02,
                vocabSize: 24
            )
        )
        draft.installSyntheticModelForTesting(
            configuration: LlamaSyntheticTestingConfiguration(
                plannedTokens: [9, 10, 11, 0],
                eogTokens: [0],
                tokenPieces: [9: "x", 10: "y", 11: "z"],
                decodeDelaySeconds: 0.02,
                unloadDelaySeconds: 0.01,
                vocabSize: 24
            )
        )

        let generationTask = Task.detached(priority: .userInitiated) { () -> Result<LlamaGenerationResult, Error> in
            do {
                let result = try target.generateWithDraft(
                    prompt: longPrompt,
                    samplingConfig: SamplingConfig(
                        temperature: 0.7,
                        topK: 8,
                        topP: 1.0,
                        repetitionPenalty: 1.0,
                        maxTokens: 12,
                        stopSequences: [],
                        samplerSeed: 7
                    ),
                    draftRunner: draft,
                    draftCount: 3,
                    onToken: { _ in },
                    shouldStop: { Task.isCancelled }
                )
                return .success(result)
            } catch {
                return .failure(error)
            }
        }

        let enteredDraftDecode = await waitUntil(timeout: .seconds(5)) {
            draft.isSyntheticDecodeActiveForTesting()
        }
        #expect(enteredDraftDecode)
        guard enteredDraftDecode else { return }

        let unloadTask = Task.detached(priority: .userInitiated) {
            draft.unload()
        }

        let generationResult = await generationTask.value
        _ = await unloadTask.value

        #expect(isGenerationCancelled(generationResult))
        #expect(draft.syntheticDecodeCallCountForTesting() > 0)
        #expect(!draft.isLoaded)
        #expect(draft.currentState == .idle)
    }

    @Test func repeatedDraftActivateDeactivateUnderCancellationPressure_staysStable() async throws {
        let context = try makeLoaderContext(prefix: "draft-churn", includeDraft: true)
        defer {
            context.keyValueStore.removeAll()
            context.fileSystem.deleteModelAssets(forModelID: context.targetManifest.id)
            if let draftManifest = context.draftManifest {
                context.fileSystem.deleteModelAssets(forModelID: draftManifest.id)
            }
        }

        context.loader.llamaRunner.setSyntheticLoadConfigurationProviderForTesting { path, _, _ in
            path == context.targetURL.path ? context.targetConfiguration : nil
        }
        if let draftURL = context.draftURL, let draftConfiguration = context.draftConfiguration {
            context.loader.draftLlamaRunner.setSyntheticLoadConfigurationProviderForTesting { path, _, _ in
                path == draftURL.path ? draftConfiguration : nil
            }
        }

        let initialLoad = await context.loader.ensureModelLoaded(context.targetManifest.id, persistSelection: false)
        #expect(initialLoad)
        guard initialLoad, let draftManifest = context.draftManifest else { return }

        for _ in 0..<3 {
            let activated = await runTransitionWhileGenerationIsActive(
                loader: context.loader,
                runner: context.loader.llamaRunner,
                operation: {
                    context.loader.activateDraftModel(draftManifest.id)
                },
                settleCondition: {
                    context.loader.activeDraftModelID == draftManifest.id && context.loader.draftLlamaRunner.isLoaded
                }
            )
            #expect(activated)

            let deactivated = await runTransitionWhileGenerationIsActive(
                loader: context.loader,
                runner: context.loader.llamaRunner,
                operation: {
                    context.loader.deactivateDraftModel()
                },
                settleCondition: {
                    context.loader.activeDraftModelID == nil && !context.loader.draftLlamaRunner.isLoaded
                }
            )
            #expect(deactivated)
        }

        #expect(context.loader.llamaRunner.isLoaded)
        #expect(context.loader.llamaRunner.currentState == .ready)
        #expect(context.loader.activeDraftModelID == nil)
    }

    private func runTransitionWhileGenerationIsActive(
        loader: ModelLoaderService,
        runner: LlamaModelRunner,
        operation: @escaping @MainActor () -> Void,
        settleCondition: @escaping @MainActor () -> Bool
    ) async -> Bool {
        let generationTask = makeGenerationTask(runner: runner, prompt: longPrompt)
        let enteredDecode = await waitUntil(timeout: .seconds(5)) {
            runner.isSyntheticDecodeActiveForTesting()
        }
        guard enteredDecode else {
            generationTask.cancel()
            _ = await generationTask.value
            return false
        }

        let drainState = DrainRecorder()
        loader.setGenerationDrainHandler {
            await drainState.markStarted()
            generationTask.cancel()
            _ = await generationTask.value
            await drainState.markFinished()
        }

        operation()

        let settled = await waitUntil(timeout: .seconds(5)) {
            await drainState.finished && settleCondition()
        }
        let generationResult = await generationTask.value
        return settled && isGenerationCancelled(generationResult)
    }

    private func makeGenerationTask(runner: LlamaModelRunner, prompt: String) -> Task<Result<LlamaGenerationResult, Error>, Never> {
        Task.detached(priority: .userInitiated) { () -> Result<LlamaGenerationResult, Error> in
            do {
                let result = try runner.generate(
                    prompt: prompt,
                    maxTokens: 16,
                    temperature: 0.7,
                    topK: 8,
                    topP: 1.0,
                    repetitionPenalty: 1.0,
                    onToken: { _ in },
                    shouldStop: { Task.isCancelled }
                )
                return .success(result)
            } catch {
                return .failure(error)
            }
        }
    }

    private func makeLoaderContext(prefix: String, includeDraft: Bool = false) throws -> LoaderContext {
        let suiteName = "\(prefix)-\(UUID().uuidString)"
        let keyValueStore = KeyValueStore(suiteName: suiteName)
        let loader = ModelLoaderService(keyValueStore: keyValueStore)
        let fileSystem = FileSystemService()

        let targetID = "\(prefix)-target-\(UUID().uuidString)"
        let targetManifest = makeManifest(id: targetID, fileName: "target.gguf", isDraft: false)
        let targetURL = try createModelFixture(fileSystem: fileSystem, modelID: targetID, fileName: "target.gguf")
        loader.availableModels.append(targetManifest)
        loader.modelStatuses[targetID] = .ready

        let targetConfiguration = LlamaSyntheticTestingConfiguration(
            plannedTokens: [7, 8, 9, 0],
            eogTokens: [0],
            tokenPieces: [7: "m", 8: "n", 9: "o"],
            decodeDelaySeconds: 0.03,
            unloadDelaySeconds: 0.01,
            loadDelaySeconds: 0.01,
            vocabSize: 24
        )

        if includeDraft {
            let draftID = "\(prefix)-draft-\(UUID().uuidString)"
            let draftManifest = makeManifest(id: draftID, fileName: "draft.gguf", isDraft: true)
            let draftURL = try createModelFixture(fileSystem: fileSystem, modelID: draftID, fileName: "draft.gguf")
            loader.availableModels.append(draftManifest)
            loader.modelStatuses[draftID] = .ready
            return LoaderContext(
                loader: loader,
                fileSystem: fileSystem,
                keyValueStore: keyValueStore,
                targetManifest: targetManifest,
                targetURL: targetURL,
                targetConfiguration: targetConfiguration,
                draftManifest: draftManifest,
                draftURL: draftURL,
                draftConfiguration: LlamaSyntheticTestingConfiguration(
                    plannedTokens: [7, 8, 9, 0],
                    eogTokens: [0],
                    tokenPieces: [7: "m", 8: "n", 9: "o"],
                    decodeDelaySeconds: 0.02,
                    unloadDelaySeconds: 0.01,
                    loadDelaySeconds: 0.01,
                    vocabSize: 24
                )
            )
        }

        return LoaderContext(
            loader: loader,
            fileSystem: fileSystem,
            keyValueStore: keyValueStore,
            targetManifest: targetManifest,
            targetURL: targetURL,
            targetConfiguration: targetConfiguration,
            draftManifest: nil,
            draftURL: nil,
            draftConfiguration: nil
        )
    }

    private func makeManifest(id: String, fileName: String, isDraft: Bool) -> ModelManifest {
        ModelManifest(
            id: id,
            name: isDraft ? "Draft Fixture" : "Target Fixture",
            variant: isDraft ? "Draft" : "Target",
            parameterCount: isDraft ? "0.5B" : "1B",
            quantization: isDraft ? "Q8" : "Q4",
            sizeBytes: 4096,
            contextLength: 256,
            architecture: .llama,
            repoID: "fixture/repo",
            tokenizerRepoID: nil,
            modelFilePattern: fileName,
            checksum: String(repeating: isDraft ? "d" : "c", count: 64),
            isDraft: isDraft,
            format: .gguf
        )
    }

    private func createModelFixture(fileSystem: FileSystemService, modelID: String, fileName: String) throws -> URL {
        let modelContainer = fileSystem.modelContainerDirectory(forModelID: modelID)
        try FileManager.default.createDirectory(at: modelContainer, withIntermediateDirectories: true)
        let modelURL = modelContainer.appendingPathComponent(fileName)
        try Data([0x47, 0x47, 0x55, 0x46, 0x01, 0x02, 0x03, 0x04]).write(to: modelURL)
        try fileSystem.saveModelPath(modelURL, forModelID: modelID)
        return modelURL
    }
}

private let longPrompt: String = String(repeating: "abcdefghij", count: 40)

private struct LoaderContext {
    let loader: ModelLoaderService
    let fileSystem: FileSystemService
    let keyValueStore: KeyValueStore
    let targetManifest: ModelManifest
    let targetURL: URL
    let targetConfiguration: LlamaSyntheticTestingConfiguration
    let draftManifest: ModelManifest?
    let draftURL: URL?
    let draftConfiguration: LlamaSyntheticTestingConfiguration?
}

private actor DrainRecorder {
    private(set) var started: Bool = false
    private(set) var finished: Bool = false

    func markStarted() {
        started = true
    }

    func markFinished() {
        finished = true
    }
}

private func waitUntil(
    timeout: Duration = .seconds(3),
    interval: Duration = .milliseconds(10),
    _ condition: @escaping @Sendable () async -> Bool
) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)

    while clock.now < deadline {
        if await condition() {
            return true
        }
        try? await Task.sleep(for: interval)
    }

    return await condition()
}

private func isGenerationCancelled(_ result: Result<LlamaGenerationResult, Error>) -> Bool {
    switch result {
    case .success:
        return false
    case .failure(let error):
        guard let runnerError = error as? LlamaRunnerError else { return false }
        if case .generationCancelled = runnerError {
            return true
        }
        return false
    }
}
