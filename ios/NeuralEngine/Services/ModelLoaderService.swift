import Foundation
import CoreML
import OSLog
import Hub

nonisolated private struct HuggingFaceModelIndexResponse: Decodable, Sendable {
    let siblings: [HuggingFaceRepositorySibling]
}

nonisolated private struct HuggingFaceRepositorySibling: Decodable, Sendable {
    let rfilename: String
}

nonisolated struct HuggingFaceFileInfo: Sendable, Identifiable {
    let fileName: String
    let sizeBytes: Int64
    var id: String { fileName }

    var sizeFormatted: String {
        let gb = Double(sizeBytes) / 1_073_741_824
        if gb >= 1.0 { return String(format: "%.1f GB", gb) }
        let mb = Double(sizeBytes) / 1_048_576
        if mb >= 1.0 { return String(format: "%.0f MB", mb) }
        return "Unknown size"
    }
}

nonisolated private struct HuggingFaceTreeEntry: Decodable, Sendable {
    let path: String
    let size: Int64?
}

@Observable
@MainActor
class ModelLoaderService {
    private static let logger: Logger = {
        let subsystem = Bundle.main.bundleIdentifier ?? "NeuralEngine"
        return Logger(subsystem: subsystem, category: "ModelLoaderService")
    }()
    static let foundationOnlyMode: Bool = true

    var availableModels: [ModelManifest] = []
    var modelStatuses: [String: ModelStatus] = [:]
    var activeModelID: String?
    var isLoadingRegistry: Bool = false

    let modelRunner = CoreMLModelRunner()
    let llamaRunner = LlamaModelRunner()
    let draftLlamaRunner = LlamaModelRunner()
    let embeddingRunner = EmbeddingModelRunner()
    let tokenizer = TokenizerService()
    var activeFormat: ModelFormat = .appleFoundation
    var activeDraftModelID: String?
    var activeEmbeddingModelID: String?
    var embeddingModelStatuses: [String: ModelStatus] = [:]
    var isReembedding: Bool = false
    var reembeddingProgress: Double = 0

    static let preferredActiveModelIDKey: String = "preferred_active_model_id"
    static let preferredEmbeddingModelIDKey: String = "preferred_embedding_model_id"

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var activationTask: Task<Bool, Never>?
    private var activationTargetModelID: String?
    private let fileSystem = FileSystemService()
    private let keyValueStore: KeyValueStore?
    private var generationDrainHandler: (() async -> Void)?

    private func logNotice(_ message: String) {
        Self.logger.notice("\(message, privacy: .public)")
    }

    private func logError(_ message: String) {
        Self.logger.error("\(message, privacy: .public)")
    }

    init(keyValueStore: KeyValueStore? = nil) {
        self.keyValueStore = keyValueStore
        loadBuiltinRegistry()
        restorePreferredModelSelection()
        restorePreferredEmbeddingModelSelection()
        restorePreferredDraftModelSelection()
    }

    func setGenerationDrainHandler(_ handler: @escaping () async -> Void) {
        generationDrainHandler = handler
    }

    private func drainGenerationIfNeeded(reason: String) async {
        logNotice("Generation drain requested reason=\(reason) activeModelID=\(activeModelID ?? "none") activeDraftModelID=\(activeDraftModelID ?? "none")")
        if let generationDrainHandler {
            await generationDrainHandler()
        }
        logNotice("Generation drain completed reason=\(reason) activeModelID=\(activeModelID ?? "none") activeDraftModelID=\(activeDraftModelID ?? "none")")
    }

    var preferredModelID: String? {
        keyValueStore?.getString(Self.preferredActiveModelIDKey)
    }

    private func persistPreferredModelID(_ modelID: String?) {
        guard let keyValueStore else { return }
        if let modelID {
            keyValueStore.setString(modelID, forKey: Self.preferredActiveModelIDKey)
        } else {
            keyValueStore.remove(Self.preferredActiveModelIDKey)
        }
    }

    var preferredEmbeddingModelID: String? {
        keyValueStore?.getString(Self.preferredEmbeddingModelIDKey)
    }

    private func persistPreferredEmbeddingModelID(_ modelID: String?) {
        guard let keyValueStore else { return }
        if let modelID {
            keyValueStore.setString(modelID, forKey: Self.preferredEmbeddingModelIDKey)
        } else {
            keyValueStore.remove(Self.preferredEmbeddingModelIDKey)
        }
    }

    var preferredDraftModelID: String? {
        keyValueStore?.getString(Self.preferredDraftModelIDKey)
    }

    private func persistPreferredDraftModelID(_ modelID: String?) {
        guard let keyValueStore else { return }
        if let modelID {
            keyValueStore.setString(modelID, forKey: Self.preferredDraftModelIDKey)
        } else {
            keyValueStore.remove(Self.preferredDraftModelIDKey)
        }
    }

    private func restorePreferredEmbeddingModelSelection() {
        guard activeEmbeddingModelID == nil,
              let preferredEmbeddingModelID,
              case .some(.ready) = embeddingModelStatuses[preferredEmbeddingModelID] else {
            return
        }
        activateEmbeddingModel(preferredEmbeddingModelID)
    }

    private func restorePreferredDraftModelSelection() {
        guard activeDraftModelID == nil,
              let preferredDraftModelID,
              case .some(.ready) = modelStatuses[preferredDraftModelID] else {
            return
        }
    }

    private func restorePreferredModelSelection() {
        guard activeModelID == nil,
              let preferredModelID,
              case .some(.ready) = modelStatuses[preferredModelID] else {
            return
        }

        activateModel(preferredModelID)
    }

    static let preferredDraftModelIDKey: String = "preferred_draft_model_id"


    private func foundationModelStatus(for manifest: ModelManifest) -> ModelStatus {
        guard manifest.format == .appleFoundation else { return .notDownloaded }

        guard ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 26, minorVersion: 0, patchVersion: 0)
        ) else {
            return .failed("Requires iOS 26.0+")
        }

        return .ready
    }

    private func statusForIntegrityResult(_ result: AssetIntegrityResult) -> ModelStatus {
        switch result {
        case .intact:
            return .ready
        case .missing:
            return .notDownloaded
        case .corrupted(let reason):
            return .failed("Asset corrupted: \(reason). Delete and re-download.")
        }
    }

    private func verifyRestoredModelIntegrity(for manifest: ModelManifest, at url: URL) -> AssetIntegrityResult {
        return fileSystem.verifyModelIntegrity(at: url, format: url.pathExtension)
    }

    func resolveRestoredStatus(for manifest: ModelManifest, modelURL: URL?, tokenizerURL: URL?) -> ModelStatus {
        if manifest.format == .appleFoundation {
            return foundationModelStatus(for: manifest)
        }

        guard let modelURL else {
            return .notDownloaded
        }

        let integrityResult = verifyRestoredModelIntegrity(for: manifest, at: modelURL)
        guard integrityResult.isValid else {
            return statusForIntegrityResult(integrityResult)
        }

        return .ready
    }

    func loadBuiltinRegistry() {
        let loadedModels = ResourceLoader.load([ModelManifest].self, from: "model_registry") ?? []
        if Self.foundationOnlyMode {
            availableModels = loadedModels.filter { $0.format == .appleFoundation && !$0.isEmbedding }
        } else {
            availableModels = loadedModels
        }
        loadCustomModels()

        for model in availableModels {
            if model.isEmbedding {
                embeddingModelStatuses[model.id] = .notDownloaded
            } else {
                modelStatuses[model.id] = .notDownloaded
            }
        }

        restorePreviouslyDownloadedModels()

        if Self.foundationOnlyMode,
           activeModelID == nil,
           let foundationModel = availableModels.first(where: { $0.format == .appleFoundation }) {
            activeModelID = foundationModel.id
            activeFormat = .appleFoundation
            persistPreferredModelID(foundationModel.id)
        }
    }

    private func restorePreviouslyDownloadedModels() {
        for model in availableModels {
            let status = resolveRestoredStatus(
                for: model,
                modelURL: loadModelPath(forModelID: model.id),
                tokenizerURL: loadTokenizerPath(forModelID: model.id)
            )

            if case .notDownloaded = status {
                continue
            }

            if model.isEmbedding {
                embeddingModelStatuses[model.id] = status
            } else {
                modelStatuses[model.id] = status
            }
        }
    }

    func downloadModel(_ modelID: String) {
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else {
            logError("Download requested for unknown modelID=\(modelID)")
            return
        }

        if Self.foundationOnlyMode, manifest.format != .appleFoundation {
            modelStatuses[modelID] = .failed("Disabled in Foundation-only mode.")
            return
        }

        if manifest.isEmbedding {
            downloadEmbeddingModel(modelID)
            return
        }

        if manifest.format == .appleFoundation {
            modelStatuses[modelID] = foundationModelStatus(for: manifest)
            return
        }

        guard modelStatuses[modelID] != .some(.verifying),
              modelStatuses[modelID] != .some(.compiling) else { return }

        if case .downloading = modelStatuses[modelID] { return }

        if loadModelPath(forModelID: modelID) != nil {
            fileSystem.deleteModelAssets(forModelID: modelID)
        }

        modelStatuses[modelID] = .downloading(progress: 0)

        if manifest.format == .gguf {
            downloadGGUFModel(modelID: modelID, manifest: manifest)
            return
        }

        let task = Task {
            do {
                modelStatuses[modelID] = .downloading(progress: 0.05)

                let tokenizerPatterns = ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json", "generation_config.json"]

                modelStatuses[modelID] = .downloading(progress: 0.1)

                let tokenizerDir = try await downloadTokenizerSnapshot(for: manifest, matching: tokenizerPatterns)

                modelStatuses[modelID] = .downloading(progress: 0.3)

                let modelDir = try await downloadCoreMLSnapshot(for: manifest, modelRepo: Hub.Repo(id: manifest.repoID))

                modelStatuses[modelID] = .downloading(progress: 0.7)

                let searchDir = modelDir

                modelStatuses[modelID] = .verifying
                try? await Task.sleep(for: .seconds(0.3))

                let modelURL = findModelFile(in: searchDir)

                if let url = modelURL {
                    let ext = url.pathExtension

                        if ext == "mlmodelc" {
                        let persistedModelURL = try fileSystem.persistModelAsset(from: url, forModelID: modelID)
                        try saveModelPath(persistedModelURL, forModelID: modelID)
                    } else if ext == "mlpackage" {
                        let manifestFile = url.appendingPathComponent("Manifest.json")
                        if FileManager.default.fileExists(atPath: manifestFile.path) {
                            modelStatuses[modelID] = .compiling
                            let compiledURL = try await compileModel(at: url)
                            let persistedModelURL = try fileSystem.persistModelAsset(from: compiledURL, forModelID: modelID)
                            try saveModelPath(persistedModelURL, forModelID: modelID)
                        } else {
                            let innerModel = findMLModelFile(in: url)
                            if let innerModel {
                                modelStatuses[modelID] = .compiling
                                let compiledURL = try await compileModel(at: innerModel)
                                let persistedModelURL = try fileSystem.persistModelAsset(from: compiledURL, forModelID: modelID)
                                try saveModelPath(persistedModelURL, forModelID: modelID)
                            } else {
                                throw ModelLoaderError.invalidPackage("mlpackage is missing Manifest.json. The download may be incomplete — delete and re-download.")
                            }
                        }
                    } else if ext == "mlmodel" {
                        modelStatuses[modelID] = .compiling
                        let compiledURL = try await compileModel(at: url)
                        let persistedModelURL = try fileSystem.persistModelAsset(from: compiledURL, forModelID: modelID)
                        try saveModelPath(persistedModelURL, forModelID: modelID)
                    } else {
                        throw ModelLoaderError.noModelFound("Downloaded files do not contain a valid CoreML model.")
                    }
                } else {
                    throw ModelLoaderError.noModelFound("No CoreML model file found in the downloaded repository. This repo may not contain CoreML models.")
                }

                try verifyTokenizerDependencies(in: tokenizerDir)

                let schemaResult = tokenizer.validateSchema(in: tokenizerDir)
                if !schemaResult.isCompatible {
                    switch schemaResult.diagnosticCode {
                    case .corruptedEncoding:
                        throw ModelLoaderError.integrityCheckFailed("Tokenizer encoding corrupted — BPE schema may have been altered by a coremltools version change. Delete and re-download.")
                    case .schemaMismatch:
                        throw ModelLoaderError.integrityCheckFailed("Unsupported tokenizer class: \(schemaResult.tokenizerClass ?? "unknown"). Asset repair required.")
                    case .missingConfig:
                        logNotice("Tokenizer config missing; proceeding with fallback tokenizer modelID=\(modelID)")
                    default:
                        break
                    }
                }

                let persistedTokenizerURL = try fileSystem.persistTokenizerAsset(from: tokenizerDir, forModelID: modelID)
                try saveTokenizerPath(persistedTokenizerURL, forModelID: modelID)

                if let savedURL = loadModelPath(forModelID: modelID) {
                    let postIntegrity = fileSystem.verifyModelIntegrity(at: savedURL, format: savedURL.pathExtension)
                    guard postIntegrity.isValid else {
                        throw ModelLoaderError.integrityCheckFailed("Post-save integrity check failed: \(postIntegrity.description)")
                    }
                }

                modelStatuses[modelID] = .ready
                autoActivateIfNeeded(modelID)
            } catch {
                if !Task.isCancelled {
                    modelStatuses[modelID] = .failed(error.localizedDescription)
                }
            }
        }

        downloadTasks[modelID] = task
    }

    private func autoActivateIfNeeded(_ modelID: String) {
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else { return }

        if manifest.isDraft {
            autoActivateDraftIfNeeded(modelID)
            return
        }

        guard activeModelID == nil else { return }
        if let preferredModelID, preferredModelID != modelID {
            return
        }
        activateModel(modelID)
    }

    private func autoActivateDraftIfNeeded(_ modelID: String) {
        guard let activeModel, !activeModel.isDraft, activeModel.format == .gguf else { return }
        guard let draftManifest = availableModels.first(where: { $0.id == modelID && $0.isDraft }) else { return }
        guard draftManifest.architecture == activeModel.architecture else { return }

        if activeDraftModelID == nil {
            activateDraftModel(modelID)
        }
    }

    private func buildModelDownloadPatterns(for manifest: ModelManifest) -> [[String]] {
        let pattern = manifest.modelFilePattern

        if pattern.contains(".mlpackage") {
            let cleanBase: String
            if pattern.hasSuffix("/*") {
                cleanBase = String(pattern.dropLast(2))
            } else if pattern.hasSuffix(".mlpackage") {
                cleanBase = pattern
            } else {
                cleanBase = pattern.components(separatedBy: "/").first ?? pattern
            }
            return [
                [
                    cleanBase,
                    "\(cleanBase)/Manifest.json",
                    "\(cleanBase)/Data/*",
                    "\(cleanBase)/Data/*/*",
                    "\(cleanBase)/Data/*/*/*",
                    "\(cleanBase)/Data/*/*/*/*",
                    "\(cleanBase)/Data/*/*/*/*/*"
                ],
                [
                    "\(cleanBase)/*",
                    "\(cleanBase)/*/*",
                    "\(cleanBase)/*/*/*",
                    "\(cleanBase)/*/*/*/*",
                    "\(cleanBase)/*/*/*/*/*"
                ]
            ]
        }

        if pattern.contains(".mlmodelc") {
            let baseName = pattern.components(separatedBy: "/").first ?? pattern
            let cleanBase = baseName.hasSuffix("/*") ? String(baseName.dropLast(2)) : baseName
            return [
                [
                    cleanBase,
                    "\(cleanBase)/*",
                    "\(cleanBase)/*/*",
                    "\(cleanBase)/*/*/*"
                ]
            ]
        }

        return [
            ["*.mlpackage", "*.mlpackage/Manifest.json", "*.mlpackage/Data/*", "*.mlpackage/Data/*/*", "*.mlpackage/Data/*/*/*", "*.mlpackage/Data/*/*/*/*"],
            ["*.mlpackage/*", "*.mlpackage/*/*", "*.mlpackage/*/*/*", "*.mlpackage/*/*/*/*", "*.mlpackage/*/*/*/*/*"],
            ["*.mlmodelc", "*.mlmodelc/*", "*.mlmodelc/*/*", "*.mlmodelc/*/*/*"],
            ["*.mlmodel"]
        ]
    }

    private func inferredTokenizerRepoID(for manifest: ModelManifest) -> String? {
        switch manifest.draftCompatibilityIdentifier {
        case "dolphin-llama3.2":
            return "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
        case "dolphin-qwen2.5":
            return "cognitivecomputations/Dolphin3.0-Qwen2.5-1.5B"
        case "smollm2":
            return "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        case "qwen2.5":
            return "Qwen/Qwen2.5-1.5B-Instruct"
        case "llama3.2":
            return "meta-llama/Llama-3.2-3B-Instruct"
        case "gemma2":
            return "google/gemma-2-2b-it"
        case "phi":
            return "microsoft/Phi-3.5-mini-instruct"
        default:
            return nil
        }
    }

    private func tokenizerRepositoryCandidates(for manifest: ModelManifest) -> [String] {
        var candidates: [String] = []
        if let explicit = manifest.tokenizerRepoID, !explicit.isEmpty {
            candidates.append(explicit)
        }
        if let inferred = inferredTokenizerRepoID(for: manifest), !inferred.isEmpty {
            candidates.append(inferred)
        }
        if !manifest.repoID.isEmpty {
            candidates.append(manifest.repoID)
        }
        var seen = Set<String>()
        return candidates.filter { seen.insert($0).inserted }
    }

    private func loadTokenizerForActivation(_ manifest: ModelManifest, modelID: String) async throws {
        if let tokenizerDir = loadTokenizerPath(forModelID: modelID) {
            try await tokenizer.loadFromDirectory(tokenizerDir)
            logNotice("Tokenizer activated modelID=\(modelID) source=\(tokenizer.cacheIdentifier)")
            return
        }

        var attempted: [String] = []
        for repoID in tokenizerRepositoryCandidates(for: manifest) {
            attempted.append(repoID)
            do {
                try await tokenizer.loadFromHub(repoID: repoID)
                logNotice("Tokenizer activated modelID=\(modelID) source=\(tokenizer.cacheIdentifier)")
                return
            } catch {
                logError("Tokenizer candidate failed modelID=\(modelID) repo=\(repoID) error=\(error.localizedDescription)")
            }
        }

        throw ModelLoaderError.noModelFound("No compatible tokenizer could be loaded. Tried: \(attempted.joined(separator: ", "))")
    }

    private func persistOptionalTokenizerSnapshot(for manifest: ModelManifest, modelID: String) async {
        guard loadTokenizerPath(forModelID: modelID) == nil else { return }

        let tokenizerPatterns = ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json", "generation_config.json", "tokenizer.model"]

        for repoID in tokenizerRepositoryCandidates(for: manifest) {
            do {
                let tokenizerRepo = Hub.Repo(id: repoID)
                let snapshotDir: URL

                if let directSnapshot = try? await Hub.snapshot(from: tokenizerRepo, matching: tokenizerPatterns),
                   (try? verifyTokenizerDependencies(in: directSnapshot)) != nil {
                    snapshotDir = directSnapshot
                } else {
                    let repositoryPaths = try await Self.repositoryPaths(for: repoID)
                    let exactPaths = Self.tokenizerRepositoryPaths(from: repositoryPaths, allowedFileNames: tokenizerPatterns)
                    guard !exactPaths.isEmpty else { continue }
                    snapshotDir = try await Hub.snapshot(from: tokenizerRepo, matching: exactPaths)
                }

                try verifyTokenizerDependencies(in: snapshotDir)
                let persistedTokenizerURL = try fileSystem.persistTokenizerAsset(from: snapshotDir, forModelID: modelID)
                try saveTokenizerPath(persistedTokenizerURL, forModelID: modelID)
                logNotice("Persisted tokenizer snapshot modelID=\(modelID) repo=\(repoID) path=\(persistedTokenizerURL.lastPathComponent)")
                return
            } catch {
                logError("Optional tokenizer snapshot failed modelID=\(modelID) repo=\(repoID) error=\(error.localizedDescription)")
            }
        }
    }

    private func downloadTokenizerSnapshot(for manifest: ModelManifest, matching patterns: [String]) async throws -> URL {
        var attempted: [String] = []

        for tokenizerRepoID in tokenizerRepositoryCandidates(for: manifest) {
            attempted.append(tokenizerRepoID)
            let tokenizerRepo = Hub.Repo(id: tokenizerRepoID)

            if let directSnapshot = try? await Hub.snapshot(from: tokenizerRepo, matching: patterns),
               (try? verifyTokenizerDependencies(in: directSnapshot)) != nil {
                return directSnapshot
            }

            do {
                let repositoryPaths = try await Self.repositoryPaths(for: tokenizerRepoID)
                let exactPaths = Self.tokenizerRepositoryPaths(from: repositoryPaths, allowedFileNames: patterns)
                if exactPaths.isEmpty {
                    continue
                }
                let snapshot = try await Hub.snapshot(from: tokenizerRepo, matching: exactPaths)
                try verifyTokenizerDependencies(in: snapshot)
                return snapshot
            } catch {
                logError("Tokenizer snapshot candidate failed repo=\(tokenizerRepoID) error=\(error.localizedDescription)")
            }
        }

        throw ModelLoaderError.integrityCheckFailed("Missing required tokenizer files. Tried: \(attempted.joined(separator: ", "))")
    }

    private func downloadCoreMLSnapshot(for manifest: ModelManifest, modelRepo: Hub.Repo) async throws -> URL {
        for patternGroup in buildModelDownloadPatterns(for: manifest) {
            if let snapshotDir = try? await Hub.snapshot(from: modelRepo, matching: patternGroup),
               findModelFile(in: snapshotDir) != nil {
                return snapshotDir
            }
        }

        let repositoryPaths = try await Self.repositoryPaths(for: manifest.repoID)
        let exactPaths = Self.coreMLRepositoryPaths(for: manifest, from: repositoryPaths)
        guard !exactPaths.isEmpty else {
            throw ModelLoaderError.noModelFound("No CoreML model file found in the downloaded repository. This repo may not contain CoreML models.")
        }

        let snapshotDir = try await Hub.snapshot(from: modelRepo, matching: exactPaths)
        guard findModelFile(in: snapshotDir) != nil else {
            throw ModelLoaderError.noModelFound("No CoreML model file found in the downloaded repository. This repo may not contain CoreML models.")
        }

        return snapshotDir
    }

    nonisolated private static func repositoryPaths(for repoID: String) async throws -> [String] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoID)") else {
            throw ModelLoaderError.noModelFound("Unable to resolve repository metadata.")
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse, 200..<300 ~= httpResponse.statusCode else {
            throw ModelLoaderError.noModelFound("Unable to load repository metadata from Hugging Face.")
        }

        let decoded = try JSONDecoder().decode(HuggingFaceModelIndexResponse.self, from: data)
        return decoded.siblings.map(\.rfilename)
    }

    nonisolated private static func coreMLRepositoryPaths(for manifest: ModelManifest, from repositoryPaths: [String]) -> [String] {
        let packageName = manifest.modelFilePattern.components(separatedBy: "/").first ?? manifest.modelFilePattern
        let packagePrefix = packageName.hasSuffix("/") ? packageName : "\(packageName)/"
        return repositoryPaths
            .filter { $0 == packageName || $0.hasPrefix(packagePrefix) }
            .sorted()
    }

    nonisolated private static func tokenizerRepositoryPaths(from repositoryPaths: [String], allowedFileNames: [String]) -> [String] {
        let allowedNames = Set(allowedFileNames)
        return repositoryPaths
            .filter { allowedNames.contains(URL(fileURLWithPath: $0).lastPathComponent) }
            .sorted()
    }

    private func downloadGGUFModel(modelID: String, manifest: ModelManifest) {
        let task = Task {
            do {
                modelStatuses[modelID] = .downloading(progress: 0.1)

                let modelRepo = Hub.Repo(id: manifest.repoID)
                let ggufFileName = manifest.modelFilePattern

                modelStatuses[modelID] = .downloading(progress: 0.2)

                let snapshotDir = try await Hub.snapshot(from: modelRepo, matching: [ggufFileName])

                modelStatuses[modelID] = .downloading(progress: 0.8)
                modelStatuses[modelID] = .verifying

                let ggufURL = findGGUFFile(in: snapshotDir, named: ggufFileName)

                guard let url = ggufURL else {
                    throw ModelLoaderError.noModelFound("No GGUF file found in downloaded repository.")
                }

                if manifest.sizeBytes > 0 {
                    if let actualSize = fileSystem.fileSize(at: url) {
                        let tolerance = Double(manifest.sizeBytes) * 0.05
                        let lowerBound = Double(manifest.sizeBytes) - tolerance
                        if Double(actualSize) < lowerBound {
                            throw ModelLoaderError.partialDownload("GGUF file appears truncated: \(actualSize) bytes vs expected ~\(manifest.sizeBytes) bytes")
                        }
                    }
                }

                let persistedModelURL = try fileSystem.persistModelAsset(from: url, forModelID: modelID)
                try saveModelPath(persistedModelURL, forModelID: modelID)

                await persistOptionalTokenizerSnapshot(for: manifest, modelID: modelID)

                modelStatuses[modelID] = .ready
                autoActivateIfNeeded(modelID)
            } catch {
                if !Task.isCancelled {
                    modelStatuses[modelID] = .failed(error.localizedDescription)
                }
            }
        }
        downloadTasks[modelID] = task
    }

    private func findGGUFFile(in directory: URL, named fileName: String) -> URL? {
        let directFile = directory.appendingPathComponent(fileName)
        if FileManager.default.fileExists(atPath: directFile.path) {
            return directFile
        }

        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else {
            return nil
        }
        while let url = enumerator.nextObject() as? URL {
            if url.pathExtension == "gguf" {
                return url
            }
        }
        return nil
    }

    func downloadEmbeddingModel(_ modelID: String) {
        guard let manifest = availableModels.first(where: { $0.id == modelID && $0.isEmbedding }) else { return }

        if case .downloading = embeddingModelStatuses[modelID] { return }

        if loadModelPath(forModelID: modelID) != nil {
            fileSystem.deleteModelAssets(forModelID: modelID)
        }

        embeddingModelStatuses[modelID] = .downloading(progress: 0)

        let task = Task {
            do {
                embeddingModelStatuses[modelID] = .downloading(progress: 0.1)

                let modelRepo = Hub.Repo(id: manifest.repoID)
                let ggufFileName = manifest.modelFilePattern

                embeddingModelStatuses[modelID] = .downloading(progress: 0.2)

                let snapshotDir = try await Hub.snapshot(from: modelRepo, matching: [ggufFileName])

                embeddingModelStatuses[modelID] = .downloading(progress: 0.8)
                embeddingModelStatuses[modelID] = .verifying

                let ggufURL = findGGUFFile(in: snapshotDir, named: ggufFileName)

                guard let url = ggufURL else {
                    throw ModelLoaderError.noModelFound("No GGUF embedding file found in downloaded repository.")
                }

                if manifest.sizeBytes > 0 {
                    if let actualSize = fileSystem.fileSize(at: url) {
                        let tolerance = Double(manifest.sizeBytes) * 0.15
                        let lowerBound = Double(manifest.sizeBytes) - tolerance
                        if Double(actualSize) < lowerBound {
                            throw ModelLoaderError.partialDownload("GGUF embedding file appears truncated: \(actualSize) bytes vs expected ~\(manifest.sizeBytes) bytes")
                        }
                    }
                }

                let persistedModelURL = try fileSystem.persistModelAsset(from: url, forModelID: modelID)
                try saveModelPath(persistedModelURL, forModelID: modelID)

                embeddingModelStatuses[modelID] = .ready
                autoActivateEmbeddingIfNeeded(modelID)
            } catch {
                if !Task.isCancelled {
                    embeddingModelStatuses[modelID] = .failed(error.localizedDescription)
                }
            }
        }
        downloadTasks[modelID] = task
    }

    func deleteEmbeddingModel(_ modelID: String) {
        downloadTasks[modelID]?.cancel()
        downloadTasks.removeValue(forKey: modelID)
        embeddingModelStatuses[modelID] = .notDownloaded

        if activeEmbeddingModelID == modelID {
            activeEmbeddingModelID = nil
            embeddingRunner.unload()
        }

        if preferredEmbeddingModelID == modelID {
            persistPreferredEmbeddingModelID(nil)
        }

        fileSystem.deleteModelAssets(forModelID: modelID)
    }

    func activateEmbeddingModel(_ modelID: String) {
        guard case .some(.ready) = embeddingModelStatuses[modelID],
              let manifest = availableModels.first(where: { $0.id == modelID && $0.isEmbedding }),
              let modelURL = loadModelPath(forModelID: modelID) else {
            return
        }

        do {
            embeddingRunner.unload()
            try embeddingRunner.loadModel(
                at: modelURL.path,
                nCtx: Int32(manifest.contextLength),
                pooling: manifest.embeddingPooling
            )
            activeEmbeddingModelID = modelID
            persistPreferredEmbeddingModelID(modelID)
        } catch {
            embeddingModelStatuses[modelID] = .failed("Failed to load: \(error.localizedDescription)")
            if activeEmbeddingModelID == modelID {
                activeEmbeddingModelID = nil
            }
        }
    }

    private func autoActivateEmbeddingIfNeeded(_ modelID: String) {
        guard activeEmbeddingModelID == nil else { return }
        if let preferredEmbeddingModelID, preferredEmbeddingModelID != modelID {
            return
        }
        activateEmbeddingModel(modelID)
    }

    func autoDownloadEmbeddingModelIfNeeded() {
        let hasAnyEmbeddingModel = availableModels
            .filter(\.isEmbedding)
            .contains { embeddingModelStatuses[$0.id] == .some(.ready) }

        guard !hasAnyEmbeddingModel else {
            if activeEmbeddingModelID == nil {
                restorePreferredEmbeddingModelSelection()
            }
            return
        }

        let targetID = "arctic-embed-xs-q8-gguf"
        guard embeddingModelStatuses[targetID] == .some(.notDownloaded) else { return }

        downloadEmbeddingModel(targetID)
    }

    var activeEmbeddingModel: ModelManifest? {
        guard let id = activeEmbeddingModelID else { return nil }
        return availableModels.first { $0.id == id }
    }

    var activeDraftModel: ModelManifest? {
        guard let id = activeDraftModelID else { return nil }
        return availableModels.first { $0.id == id }
    }

    var embeddingModels: [ModelManifest] {
        availableModels.filter(\.isEmbedding)
    }

    var draftModels: [ModelManifest] {
        availableModels.filter { $0.isDraft && !$0.isEmbedding }
    }

    var chatModels: [ModelManifest] {
        availableModels.filter { !$0.isEmbedding }
    }

    func autoDownloadDraftModelIfNeeded(for targetManifest: ModelManifest) {
        guard !targetManifest.isDraft, targetManifest.format == .gguf else { return }

        let compatibleDrafts = availableModels.filter { manifest in
            manifest.isDraft &&
            manifest.format == .gguf &&
            !manifest.isEmbedding &&
            manifest.draftCompatibilityIdentifier == targetManifest.draftCompatibilityIdentifier
        }

        let hasReadyDraft = compatibleDrafts.contains { modelStatuses[$0.id] == .some(.ready) }
        guard !hasReadyDraft else {
            loadDraftRunnerIfPossible(for: targetManifest)
            return
        }

        guard let bestDraft = compatibleDrafts
            .filter({ modelStatuses[$0.id] == .some(.notDownloaded) })
            .sorted(by: { $0.sizeBytes < $1.sizeBytes })
            .first else { return }

        downloadModel(bestDraft.id)
    }

    func activateDraftModel(_ modelID: String) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.activateDraftModelAfterDraining(modelID)
        }
    }

    private func activateDraftModelAfterDraining(_ modelID: String) async {
        guard let manifest = availableModels.first(where: { $0.id == modelID && $0.isDraft }),
              case .some(.ready) = modelStatuses[modelID],
              let modelURL = loadModelPath(forModelID: modelID) else {
            return
        }

        logNotice("Draft activation requested draftModelID=\(modelID)")
        await drainGenerationIfNeeded(reason: "activateDraft:\(modelID)")
        draftLlamaRunner.unload()

        let targetCtx: Int32
        if let activeModel, !activeModel.isDraft {
            targetCtx = Int32(min(manifest.contextLength, activeModel.contextLength))
        } else {
            targetCtx = Int32(manifest.contextLength)
        }

        do {
            try draftLlamaRunner.loadModel(
                at: modelURL.path,
                nCtx: targetCtx,
                nGPULayers: 0
            )
            activeDraftModelID = modelID
            persistPreferredDraftModelID(modelID)
            logNotice("Draft activation completed draftModelID=\(modelID) nCtx=\(targetCtx)")
        } catch {
            modelStatuses[modelID] = .failed("Failed to load draft: \(error.localizedDescription)")
            activeDraftModelID = nil
            draftLlamaRunner.unload()
            logError("Draft activation failed draftModelID=\(modelID) error=\(error.localizedDescription)")
        }
    }

    func deactivateDraftModel() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.deactivateDraftModelAfterDraining()
        }
    }

    private func deactivateDraftModelAfterDraining() async {
        logNotice("Draft deactivation requested draftModelID=\(activeDraftModelID ?? "none")")
        await drainGenerationIfNeeded(reason: "deactivateDraft")
        activeDraftModelID = nil
        draftLlamaRunner.unload()
        persistPreferredDraftModelID(nil)
        logNotice("Draft deactivation completed")
    }

    func deleteModel(_ modelID: String) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.deleteModelAfterDraining(modelID)
        }
    }

    private func deleteModelAfterDraining(_ modelID: String) async {
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else { return }

        if manifest.isEmbedding {
            deleteEmbeddingModel(modelID)
            return
        }

        logNotice("Delete requested modelID=\(modelID)")
        await drainGenerationIfNeeded(reason: "deleteModel:\(modelID)")
        downloadTasks[modelID]?.cancel()
        downloadTasks.removeValue(forKey: modelID)
        activationTask?.cancel()
        if activationTargetModelID == modelID {
            activationTask = nil
            activationTargetModelID = nil
        }
        if manifest.format == .appleFoundation {
            modelStatuses[modelID] = foundationModelStatus(for: manifest)
        } else {
            modelStatuses[modelID] = .notDownloaded
        }

        if activeModelID == modelID {
            activeModelID = nil
            activeDraftModelID = nil
            modelRunner.unload()
            llamaRunner.unload()
            draftLlamaRunner.unload()
            tokenizer.unloadTokenizer()
            activeFormat = .appleFoundation
        } else if activeDraftModelID == modelID {
            activeDraftModelID = nil
            draftLlamaRunner.unload()
        }

        if preferredModelID == modelID {
            persistPreferredModelID(nil)
        }

        if preferredDraftModelID == modelID {
            persistPreferredDraftModelID(nil)
        }

        fileSystem.deleteModelAssets(forModelID: modelID)
        logNotice("Delete completed modelID=\(modelID)")
    }

    func activateModel(_ modelID: String) {
        if Self.foundationOnlyMode,
           let manifest = availableModels.first(where: { $0.id == modelID }),
           manifest.format != .appleFoundation {
            modelStatuses[modelID] = .failed("Disabled in Foundation-only mode.")
            return
        }
        Task { @MainActor [weak self] in
            guard let self else { return }
            _ = await self.ensureModelLoaded(modelID, persistSelection: true)
        }
    }

    @discardableResult
    func ensureActiveModelLoaded(forceReload: Bool = false) async -> Bool {
        guard let candidateModelID = activeModelID ?? preferredModelID else {
            return false
        }
        return await ensureModelLoaded(candidateModelID, forceReload: forceReload, persistSelection: true)
    }

    @discardableResult
    func ensureModelLoaded(_ modelID: String, forceReload: Bool = false, persistSelection: Bool = true) async -> Bool {
        guard case .some(.ready) = modelStatuses[modelID],
              let manifest = availableModels.first(where: { $0.id == modelID }) else {
            return false
        }

        if Self.foundationOnlyMode, manifest.format != .appleFoundation {
            modelStatuses[modelID] = .failed("Disabled in Foundation-only mode.")
            return false
        }

        if let activationTask, activationTargetModelID == modelID {
            return await activationTask.value
        }

        logNotice("Model activation requested modelID=\(modelID) format=\(manifest.format.rawValue) forceReload=\(forceReload)")

        if !forceReload, isModelReadyForInference(manifest) {
            activeModelID = modelID
            activeFormat = manifest.format
            if persistSelection {
                persistPreferredModelID(modelID)
            }
            return true
        }

        activationTask?.cancel()
        activationTargetModelID = modelID

        let activationTask = Task { @MainActor [weak self] in
            guard let self else { return false }
            return await self.performModelActivation(manifest, forceReload: forceReload, persistSelection: persistSelection)
        }
        self.activationTask = activationTask

        let success = await activationTask.value
        if activationTargetModelID == modelID {
            self.activationTask = nil
            activationTargetModelID = nil
        }
        return success
    }

    func reactivateCurrentModel() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            _ = await self.ensureActiveModelLoaded(forceReload: true)
        }
    }

    var runnerHealthStatus: HealthStatus {
        modelRunner.healthCheck()
    }

    private func isModelReadyForInference(_ manifest: ModelManifest) -> Bool {
        guard activeModelID == manifest.id else { return false }
        switch manifest.format {
        case .coreML:
            return modelRunner.isLoaded
        case .gguf:
            return llamaRunner.isLoaded
        case .appleFoundation:
            if case .ready = foundationModelStatus(for: manifest) {
                return true
            }
            return false
        }
    }

    private func performModelActivation(_ manifest: ModelManifest, forceReload: Bool, persistSelection: Bool) async -> Bool {
        guard !Task.isCancelled else { return false }
        if Self.foundationOnlyMode, manifest.format != .appleFoundation {
            modelStatuses[manifest.id] = .failed("Disabled in Foundation-only mode.")
            return false
        }

        let modelID = manifest.id
        activeModelID = modelID
        activeFormat = manifest.format

        await drainGenerationIfNeeded(reason: forceReload ? "activateModel:\(modelID):forceReload" : "activateModel:\(modelID)")

        do {
            switch manifest.format {
            case .coreML:
                guard let modelURL = loadModelPath(forModelID: modelID) else {
                    throw ModelLoaderError.noModelFound("CoreML model file not found.")
                }

                llamaRunner.unload()
                draftLlamaRunner.unload()
                activeDraftModelID = nil

                if forceReload || modelRunner.isLoaded {
                    modelRunner.unload()
                }

                do {
                    try await loadTokenizerForActivation(manifest, modelID: modelID)
                } catch {
                    logError("Tokenizer load failed modelID=\(modelID) error=\(error.localizedDescription)")
                }

                try await modelRunner.loadModel(at: modelURL, computeUnits: thermalComputeUnits)

            case .gguf:
                guard let modelURL = loadModelPath(forModelID: modelID) else {
                    throw ModelLoaderError.noModelFound("GGUF model file not found.")
                }

                modelRunner.unload()
                activeDraftModelID = nil

                if forceReload || llamaRunner.isLoaded {
                    llamaRunner.unload()
                    draftLlamaRunner.unload()
                }

                do {
                    try await loadTokenizerForActivation(manifest, modelID: modelID)
                } catch {
                    tokenizer.unloadTokenizer()
                    logError("GGUF tokenizer load failed modelID=\(modelID) error=\(error.localizedDescription)")
                }

                try llamaRunner.loadModel(
                    at: modelURL.path,
                    nCtx: Int32(manifest.contextLength)
                )
                await loadDraftRunnerIfPossibleAfterDraining(for: manifest)

            case .appleFoundation:
                modelRunner.unload()
                llamaRunner.unload()
                draftLlamaRunner.unload()
                tokenizer.unloadTokenizer()
                activeDraftModelID = nil

                guard case .ready = foundationModelStatus(for: manifest) else {
                    throw ModelLoaderError.noModelFound("Apple Foundation Models require iOS 26.0+ with on-device model availability.")
                }
            }

            if persistSelection {
                persistPreferredModelID(modelID)
            }
            logNotice("Model activation completed modelID=\(modelID) format=\(manifest.format.rawValue) draftModelID=\(activeDraftModelID ?? "none")")
            return true
        } catch {
            modelStatuses[modelID] = .failed("Failed to load: \(error.localizedDescription)")
            if activeModelID == modelID {
                activeModelID = nil
            }
            if activeDraftModelID == modelID {
                activeDraftModelID = nil
            }
            if manifest.format == .gguf {
                draftLlamaRunner.unload()
                activeFormat = .appleFoundation
            }
            logError("Model activation failed modelID=\(modelID) format=\(manifest.format.rawValue) error=\(error.localizedDescription)")
            return false
        }
    }

    func loadDraftRunnerIfPossible(for targetManifest: ModelManifest) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.loadDraftRunnerIfPossibleAfterDraining(for: targetManifest)
        }
    }

    private func loadDraftRunnerIfPossibleAfterDraining(for targetManifest: ModelManifest) async {
        logNotice("Draft auto-activation requested targetModelID=\(targetManifest.id)")
        await drainGenerationIfNeeded(reason: "autoActivateDraft:\(targetManifest.id)")
        guard !targetManifest.isDraft else {
            activeDraftModelID = nil
            draftLlamaRunner.unload()
            return
        }

        guard let draftManifest = compatibleDraftManifest(for: targetManifest),
              let draftURL = loadModelPath(forModelID: draftManifest.id) else {
            activeDraftModelID = nil
            draftLlamaRunner.unload()
            return
        }

        do {
            draftLlamaRunner.unload()
            try draftLlamaRunner.loadModel(
                at: draftURL.path,
                nCtx: Int32(min(draftManifest.contextLength, targetManifest.contextLength)),
                nGPULayers: 0
            )
            activeDraftModelID = draftManifest.id
            persistPreferredDraftModelID(draftManifest.id)
            logNotice("Draft auto-activation completed targetModelID=\(targetManifest.id) draftModelID=\(draftManifest.id)")
        } catch {
            logError("Draft auto-activation failed targetModelID=\(targetManifest.id) draftModelID=\(draftManifest.id) error=\(error.localizedDescription)")
            activeDraftModelID = nil
            draftLlamaRunner.unload()
        }
    }

    private func compatibleDraftManifest(for targetManifest: ModelManifest) -> ModelManifest? {
        let candidates = availableModels
            .filter { manifest in
                manifest.format == .gguf &&
                manifest.isDraft &&
                manifest.id != targetManifest.id &&
                manifest.draftCompatibilityIdentifier == targetManifest.draftCompatibilityIdentifier &&
                modelStatuses[manifest.id] == .ready
            }
            .sorted { lhs, rhs in
                if lhs.sizeBytes != rhs.sizeBytes {
                    return lhs.sizeBytes < rhs.sizeBytes
                }
                return lhs.contextLength < rhs.contextLength
            }

        if let preferredDraftModelID,
           let preferred = candidates.first(where: { $0.id == preferredDraftModelID }) {
            return preferred
        }

        return candidates.first
    }

    var activeModel: ModelManifest? {
        guard let id = activeModelID else { return nil }
        return availableModels.first { $0.id == id }
    }

    var readyModels: [ModelManifest] {
        availableModels.filter { model in
            if case .ready = modelStatuses[model.id] { return true }
            return false
        }
    }

    private var thermalComputeUnits: MLComputeUnits {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal, .fair: return .all
        case .serious: return .cpuAndNeuralEngine
        case .critical: return .cpuOnly
        @unknown default: return .all
        }
    }

    private func findModelFile(in directory: URL) -> URL? {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: []
        ) else {
            return nil
        }

        var mlmodelcURL: URL?
        var validMlpackageURL: URL?
        var anyMlpackageURL: URL?
        var mlmodelURL: URL?

        while let url = enumerator.nextObject() as? URL {
            let ext = url.pathExtension
            if ext == "mlmodelc" {
                let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if isDir {
                    mlmodelcURL = url
                    enumerator.skipDescendants()
                    break
                }
            } else if ext == "mlpackage" {
                let manifestPath = url.appendingPathComponent("Manifest.json")
                if FileManager.default.fileExists(atPath: manifestPath.path) {
                    validMlpackageURL = url
                } else {
                    anyMlpackageURL = url
                }
                enumerator.skipDescendants()
            } else if ext == "mlmodel" {
                mlmodelURL = url
            }
        }

        return mlmodelcURL ?? validMlpackageURL ?? mlmodelURL ?? anyMlpackageURL
    }

    private func findMLModelFile(in directory: URL) -> URL? {
        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else {
            return nil
        }
        while let url = enumerator.nextObject() as? URL {
            if url.pathExtension == "mlmodel" {
                return url
            }
        }
        return nil
    }

    nonisolated private func compileModel(at url: URL) async throws -> URL {
        do {
            return try await MLModel.compileModel(at: url)
        } catch {
            throw ModelLoaderError.compilationFailed("Failed to compile model: \(error.localizedDescription)")
        }
    }

    private func saveModelPath(_ url: URL, forModelID id: String) throws {
        fileSystem.excludeFromBackup(url)
        try fileSystem.saveModelPath(url, forModelID: id)
    }

    private func saveTokenizerPath(_ url: URL, forModelID id: String) throws {
        try fileSystem.saveTokenizerPath(url, forModelID: id)
    }

    private func loadModelPath(forModelID id: String) -> URL? {
        fileSystem.loadModelPath(forModelID: id)
    }

    private func loadTokenizerPath(forModelID id: String) -> URL? {
        fileSystem.loadTokenizerPath(forModelID: id)
    }

    private func verifyTokenizerDependencies(in directory: URL) throws {
        let requiredFiles = ["tokenizer.json"]
        for fileName in requiredFiles {
            let fileURL = directory.appendingPathComponent(fileName)
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                let contents = fileSystem.listContents(of: directory)
                let subDirs = contents.filter { fileSystem.isDirectory(at: $0) }
                var found = false
                for subDir in subDirs {
                    let nested = subDir.appendingPathComponent(fileName)
                    if FileManager.default.fileExists(atPath: nested.path) {
                        found = true
                        break
                    }
                    let deepContents = fileSystem.listContents(of: subDir)
                    for deepDir in deepContents where fileSystem.isDirectory(at: deepDir) {
                        let deepNested = deepDir.appendingPathComponent(fileName)
                        if FileManager.default.fileExists(atPath: deepNested.path) {
                            found = true
                            break
                        }
                    }
                    if found { break }
                }
                if !found {
                    throw ModelLoaderError.integrityCheckFailed("Missing required tokenizer file: \(fileName)")
                }
            }
        }
    }

    func verifyModelIntegrity(forModelID modelID: String) -> AssetIntegrityResult {
        guard let modelURL = loadModelPath(forModelID: modelID) else {
            return .missing
        }
        return fileSystem.verifyModelIntegrity(at: modelURL, format: modelURL.pathExtension)
    }

    func initiateAssetRepair(forModelID modelID: String) {
        guard availableModels.contains(where: { $0.id == modelID }) else { return }

        fileSystem.deleteModelAssets(forModelID: modelID)
        modelStatuses[modelID] = .notDownloaded

        downloadModel(modelID)
    }

    func runFullIntegrityAudit() -> [String: AssetIntegrityResult] {
        var results: [String: AssetIntegrityResult] = [:]
        for model in availableModels {
            guard modelStatuses[model.id] == .some(.ready) else { continue }
            results[model.id] = verifyModelIntegrity(forModelID: model.id)
        }
        return results
    }

    // MARK: - Custom HuggingFace Model Support

    private static let customModelsKey = "custom_models_v1"

    private func loadCustomModels() {
        guard let data = keyValueStore?.getData(Self.customModelsKey),
              let customs = try? JSONDecoder().decode([ModelManifest].self, from: data) else {
            return
        }
        let existingIDs = Set(availableModels.map(\.id))
        for model in customs where !existingIDs.contains(model.id) {
            if Self.foundationOnlyMode, model.format != .appleFoundation {
                continue
            }
            availableModels.append(model)
        }
    }

    private func persistCustomModels() {
        let customs = availableModels.filter(\.isCustom)
        guard let data = try? JSONEncoder().encode(customs) else { return }
        keyValueStore?.setData(data, forKey: Self.customModelsKey)
    }

    func addCustomModel(_ manifest: ModelManifest) {
        if Self.foundationOnlyMode, manifest.format != .appleFoundation {
            logNotice("Rejected custom model in foundation-only mode id=\(manifest.id) format=\(manifest.format.rawValue)")
            return
        }
        guard !availableModels.contains(where: { $0.id == manifest.id }) else {
            logNotice("Custom model already exists id=\(manifest.id)")
            return
        }
        availableModels.append(manifest)
        if manifest.isEmbedding {
            embeddingModelStatuses[manifest.id] = .notDownloaded
        } else {
            modelStatuses[manifest.id] = .notDownloaded
        }
        persistCustomModels()
        logNotice("Custom model added id=\(manifest.id) repo=\(manifest.repoID)")
    }

    func deleteCustomModel(_ modelID: String) {
        guard availableModels.contains(where: { $0.id == modelID && $0.isCustom }) else { return }

        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.drainGenerationIfNeeded(reason: "deleteCustom:\(modelID)")
            self.downloadTasks[modelID]?.cancel()
            self.downloadTasks.removeValue(forKey: modelID)

            if self.activeModelID == modelID {
                self.activeModelID = nil
                self.activeDraftModelID = nil
                self.modelRunner.unload()
                self.llamaRunner.unload()
                self.draftLlamaRunner.unload()
                self.tokenizer.unloadTokenizer()
                self.activeFormat = .appleFoundation
            }

            if self.activeEmbeddingModelID == modelID {
                self.activeEmbeddingModelID = nil
                self.embeddingRunner.unload()
            }

            self.embeddingModelStatuses.removeValue(forKey: modelID)
            self.modelStatuses.removeValue(forKey: modelID)

            self.fileSystem.deleteModelAssets(forModelID: modelID)
            self.availableModels.removeAll { $0.id == modelID }
            self.persistCustomModels()
            self.logNotice("Custom model deleted id=\(modelID)")
        }
    }

    nonisolated static func probeHuggingFaceRepo(_ repoID: String) async throws -> [HuggingFaceFileInfo] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoID)") else {
            throw ModelLoaderError.noModelFound("Invalid repository ID.")
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse, 200..<300 ~= httpResponse.statusCode else {
            throw ModelLoaderError.noModelFound("Repository not found on HuggingFace.")
        }

        let decoded = try JSONDecoder().decode(HuggingFaceModelIndexResponse.self, from: data)
        return decoded.siblings
            .filter { $0.rfilename.hasSuffix(".gguf") }
            .map { HuggingFaceFileInfo(fileName: $0.rfilename, sizeBytes: 0) }
    }

    nonisolated static func probeHuggingFaceRepoWithSizes(_ repoID: String) async throws -> [HuggingFaceFileInfo] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoID)/tree/main") else {
            throw ModelLoaderError.noModelFound("Invalid repository ID.")
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse, 200..<300 ~= httpResponse.statusCode else {
            return try await probeHuggingFaceRepo(repoID)
        }

        let entries = try JSONDecoder().decode([HuggingFaceTreeEntry].self, from: data)
        return entries
            .filter { $0.path.hasSuffix(".gguf") }
            .map { HuggingFaceFileInfo(fileName: $0.path, sizeBytes: $0.size ?? 0) }
    }

    static func repoNameFromURL(_ urlString: String) -> String? {
        var cleaned = urlString
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "https://huggingface.co/", with: "")
            .replacingOccurrences(of: "http://huggingface.co/", with: "")

        if cleaned.hasSuffix("/") { cleaned = String(cleaned.dropLast()) }

        let parts = cleaned.components(separatedBy: "/")
        guard parts.count >= 2 else { return nil }
        return "\(parts[0])/\(parts[1])"
    }
}

nonisolated enum ModelLoaderError: Error, Sendable, LocalizedError {
    case invalidPackage(String)
    case noModelFound(String)
    case compilationFailed(String)
    case integrityCheckFailed(String)
    case assetRepairFailed(String)
    case partialDownload(String)

    var errorDescription: String? {
        switch self {
        case .invalidPackage(let msg): return msg
        case .noModelFound(let msg): return msg
        case .compilationFailed(let msg): return msg
        case .integrityCheckFailed(let msg): return msg
        case .assetRepairFailed(let msg): return msg
        case .partialDownload(let msg): return msg
        }
    }
}
