import Foundation
import CoreML
import Hub

nonisolated private struct HuggingFaceModelIndexResponse: Decodable, Sendable {
    let siblings: [HuggingFaceRepositorySibling]
}

nonisolated private struct HuggingFaceRepositorySibling: Decodable, Sendable {
    let rfilename: String
}

@Observable
@MainActor
class ModelLoaderService {
    var availableModels: [ModelManifest] = []
    var modelStatuses: [String: ModelStatus] = [:]
    var activeModelID: String?
    var isLoadingRegistry: Bool = false

    let modelRunner = CoreMLModelRunner()
    let llamaRunner = LlamaModelRunner()
    let draftLlamaRunner = LlamaModelRunner()
    let tokenizer = TokenizerService()
    var activeFormat: ModelFormat = .coreML
    var activeDraftModelID: String?

    static let preferredActiveModelIDKey: String = "preferred_active_model_id"

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var activationTask: Task<Bool, Never>?
    private var activationTargetModelID: String?
    private let fileSystem = FileSystemService()
    private let keyValueStore: KeyValueStore?

    init(keyValueStore: KeyValueStore? = nil) {
        self.keyValueStore = keyValueStore
        loadBuiltinRegistry()
        restorePreferredModelSelection()
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

    private func restorePreferredModelSelection() {
        guard activeModelID == nil,
              let preferredModelID,
              case .some(.ready) = modelStatuses[preferredModelID] else {
            return
        }

        activateModel(preferredModelID)
    }

    static func registryIssue(for manifest: ModelManifest) -> String? {
        guard !manifest.checksum.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Missing required model checksum in registry."
        }

        if manifest.tokenizerRepoID != nil, (manifest.tokenizerChecksum?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) {
            return "Missing required tokenizer checksum in registry."
        }

        return nil
    }

    private func statusForUnsupportedManifest(_ manifest: ModelManifest) -> ModelStatus? {
        guard let issue = Self.registryIssue(for: manifest) else { return nil }
        return .unsupported("Checksum unavailable. \(issue) Delete any local copy and wait for an updated registry.")
    }

    private func statusForIntegrityResult(_ result: AssetIntegrityResult) -> ModelStatus {
        switch result {
        case .intact:
            return .ready
        case .missing:
            return .notDownloaded
        case .corrupted(let reason):
            return .checksumFailed("Asset corrupted: \(reason). Delete and re-download.")
        case .checksumMismatch:
            return .checksumFailed("Checksum mismatch detected. Delete and re-download.")
        }
    }

    private func verifyTokenizerIntegrity(for manifest: ModelManifest, at url: URL) -> AssetIntegrityResult {
        guard let checksum = manifest.tokenizerChecksum, !checksum.isEmpty else {
            return manifest.tokenizerRepoID == nil ? .intact : .corrupted("Missing tokenizer checksum")
        }

        guard let actual = fileSystem.computeAssetSHA256(for: url) else {
            return .corrupted("Unable to compute tokenizer SHA-256 hash")
        }

        return actual == checksum ? .intact : .checksumMismatch(expected: checksum, actual: actual)
    }

    private func verifyRestoredModelIntegrity(for manifest: ModelManifest, at url: URL) -> AssetIntegrityResult {
        let structuralResult = fileSystem.verifyModelIntegrity(at: url, format: url.pathExtension)
        guard structuralResult.isValid else {
            return structuralResult
        }

        if manifest.format == .coreML, url.pathExtension == "mlmodelc" {
            guard let storedHash = fileSystem.loadChecksum(forModelID: manifest.id), !storedHash.isEmpty else {
                return .corrupted("Missing recorded source checksum")
            }

            return .intact
        }

        return fileSystem.verifyIntegrity(for: manifest, at: url)
    }

    func resolveRestoredStatus(for manifest: ModelManifest, modelURL: URL?, tokenizerURL: URL?) -> ModelStatus {
        if let unsupportedStatus = statusForUnsupportedManifest(manifest) {
            return unsupportedStatus
        }

        guard let modelURL else {
            return .notDownloaded
        }

        let integrityResult = verifyRestoredModelIntegrity(for: manifest, at: modelURL)
        guard integrityResult.isValid else {
            return statusForIntegrityResult(integrityResult)
        }

        if let tokenizerURL {
            return statusForIntegrityResult(verifyTokenizerIntegrity(for: manifest, at: tokenizerURL))
        }

        return manifest.tokenizerRepoID == nil ? .ready : .checksumFailed("Tokenizer assets are missing. Delete and re-download.")
    }

    func loadBuiltinRegistry() {
        availableModels = [
            ModelManifest(
                id: "dolphin3-3b-q4-gguf",
                name: "Dolphin 3.0",
                variant: "3B Q4 GGUF",
                parameterCount: "3B",
                quantization: "Q4_K_M",
                sizeBytes: 2_019_382_400,
                contextLength: 4096,
                architecture: .dolphin,
                repoID: "bartowski/Dolphin3.0-Llama3.2-3B-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf",
                checksum: "5d6d02eeefa1ab5dbf23f97afdf5c2c95ad3d946dc3b6e9ab72e6c1637d54177",
                isDraft: false,
                format: .gguf,
                recommendation: ModelRecommendation(
                    badge: "Best fit",
                    reason: "Recent Dolphin release with the best 3B quality-to-size balance for on-device GGUF.",
                    rank: 1
                )
            ),
            ModelManifest(
                id: "dolphin3-3b-int4-coreml",
                name: "Dolphin 3.0",
                variant: "3B Int4 CoreML",
                parameterCount: "3B",
                quantization: "Int4-LUT",
                sizeBytes: 1_610_000_000,
                contextLength: 4096,
                architecture: .dolphin,
                repoID: "ales27pm/Dolphin3.0-CoreML",
                tokenizerRepoID: "cognitivecomputations/Dolphin3.0-Llama3.2-3B",
                modelFilePattern: "Dolphin3.0-Llama3.2-3B-int4-lut.mlpackage",
                checksum: "930a399f92118e519eef85060c4862f7432da17dccfb380dd5e13700acf3a21e",
                tokenizerChecksum: "dcb5f6ff03f1140b431a00df71f789f48b093dbc2c15f1e15ab41e4652092afd",
                isDraft: false
            ),
            ModelManifest(
                id: "dolphin3-3b-int8-coreml",
                name: "Dolphin 3.0",
                variant: "3B Int8 CoreML",
                parameterCount: "3B",
                quantization: "Int8",
                sizeBytes: 3_230_000_000,
                contextLength: 4096,
                architecture: .dolphin,
                repoID: "ales27pm/Dolphin3.0-CoreML",
                tokenizerRepoID: "cognitivecomputations/Dolphin3.0-Llama3.2-3B",
                modelFilePattern: "Dolphin3.0-Llama3.2-3B-int8.mlpackage",
                checksum: "1a11a354b6807d422f9511a3c9067f97e88696167aaad2023b13ae146fd6d29c",
                tokenizerChecksum: "dcb5f6ff03f1140b431a00df71f789f48b093dbc2c15f1e15ab41e4652092afd",
                isDraft: false
            ),
            ModelManifest(
                id: "dolphin3-3b-fp16-coreml",
                name: "Dolphin 3.0",
                variant: "3B FP16 CoreML",
                parameterCount: "3B",
                quantization: "Float16",
                sizeBytes: 6_460_000_000,
                contextLength: 4096,
                architecture: .dolphin,
                repoID: "ales27pm/Dolphin3.0-CoreML",
                tokenizerRepoID: "cognitivecomputations/Dolphin3.0-Llama3.2-3B",
                modelFilePattern: "Dolphin3.0-Llama3.2-3B-fp16.mlpackage",
                checksum: "54f3693cf804f98aab7f4d2f51e5e47c3f8abe535eeec973a82fd7aace86e633",
                tokenizerChecksum: "dcb5f6ff03f1140b431a00df71f789f48b093dbc2c15f1e15ab41e4652092afd",
                isDraft: false
            ),
            ModelManifest(
                id: "smollm2-135m-coreml",
                name: "SmolLM2",
                variant: "135M CoreML (Draft)",
                parameterCount: "135M",
                quantization: "Float16",
                sizeBytes: 270_000_000,
                contextLength: 2048,
                architecture: .smolLM,
                repoID: "apple/OpenELM-270M-Instruct",
                tokenizerRepoID: nil,
                modelFilePattern: "*",
                checksum: "",
                isDraft: true
            ),
            ModelManifest(
                id: "openelm-270m-coreml",
                name: "OpenELM",
                variant: "270M Instruct",
                parameterCount: "270M",
                quantization: "Float16",
                sizeBytes: 540_000_000,
                contextLength: 2048,
                architecture: .llama,
                repoID: "apple/OpenELM-270M-Instruct",
                tokenizerRepoID: nil,
                modelFilePattern: "*.mlmodelc/*",
                checksum: "",
                isDraft: false
            ),
            ModelManifest(
                id: "phi3-mini-3.8b-coreml",
                name: "Phi-3 Mini",
                variant: "3.8B CoreML",
                parameterCount: "3.8B",
                quantization: "Int4",
                sizeBytes: 2_362_232_012,
                contextLength: 4096,
                architecture: .phi,
                repoID: "microsoft/Phi-3-mini-4k-instruct",
                tokenizerRepoID: nil,
                modelFilePattern: "*.mlmodelc/*",
                checksum: "",
                isDraft: false
            ),
            ModelManifest(
                id: "llama3.2-3b-coreml",
                name: "Llama 3.2",
                variant: "3B CoreML",
                parameterCount: "3B",
                quantization: "Int4",
                sizeBytes: 1_932_735_283,
                contextLength: 4096,
                architecture: .llama,
                repoID: "meta-llama/Llama-3.2-3B-Instruct",
                tokenizerRepoID: nil,
                modelFilePattern: "*.mlmodelc/*",
                checksum: "",
                isDraft: false
            ),
            ModelManifest(
                id: "gemma2-2b-coreml",
                name: "Gemma 2",
                variant: "2B CoreML",
                parameterCount: "2B",
                quantization: "Int4",
                sizeBytes: 1_503_238_553,
                contextLength: 2048,
                architecture: .gemma,
                repoID: "google/gemma-2-2b-it",
                tokenizerRepoID: nil,
                modelFilePattern: "*.mlmodelc/*",
                checksum: "",
                isDraft: false
            ),
            ModelManifest(
                id: "qwen2.5-1.5b-coreml",
                name: "Qwen 2.5",
                variant: "1.5B CoreML",
                parameterCount: "1.5B",
                quantization: "Int4",
                sizeBytes: 986_710_016,
                contextLength: 2048,
                architecture: .qwen,
                repoID: "Qwen/Qwen2.5-1.5B-Instruct",
                tokenizerRepoID: nil,
                modelFilePattern: "*.mlmodelc/*",
                checksum: "",
                isDraft: false
            ),
            ModelManifest(
                id: "mistral-7b-coreml",
                name: "Mistral",
                variant: "7B CoreML",
                parameterCount: "7B",
                quantization: "Int4",
                sizeBytes: 3_087_007_744,
                contextLength: 8192,
                architecture: .mistral,
                repoID: "mistralai/Mistral-7B-Instruct-v0.3",
                tokenizerRepoID: nil,
                modelFilePattern: "*.mlmodelc/*",
                checksum: "",
                isDraft: false
            ),
            ModelManifest(
                id: "smollm2-360m-gguf",
                name: "SmolLM2",
                variant: "360M Q8 GGUF (Draft)",
                parameterCount: "360M",
                quantization: "Q8_0",
                sizeBytes: 386_000_000,
                contextLength: 2048,
                architecture: .smolLM,
                repoID: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "smollm2-360m-instruct-q8_0.gguf",
                checksum: "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201",
                isDraft: true,
                format: .gguf
            ),
            ModelManifest(
                id: "smollm2-1.7b-gguf",
                name: "SmolLM2",
                variant: "1.7B Q4 GGUF",
                parameterCount: "1.7B",
                quantization: "Q4_K_M",
                sizeBytes: 1_060_000_000,
                contextLength: 2048,
                architecture: .smolLM,
                repoID: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "smollm2-1.7b-instruct-q4_k_m.gguf",
                checksum: "decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33",
                isDraft: false,
                format: .gguf
            ),
            ModelManifest(
                id: "qwen2.5-1.5b-gguf",
                name: "Qwen 2.5",
                variant: "1.5B Q4 GGUF",
                parameterCount: "1.5B",
                quantization: "Q4_K_M",
                sizeBytes: 986_000_000,
                contextLength: 4096,
                architecture: .qwen,
                repoID: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                checksum: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
                isDraft: false,
                format: .gguf
            ),
            ModelManifest(
                id: "llama3.2-3b-gguf",
                name: "Llama 3.2",
                variant: "3B Q4 GGUF",
                parameterCount: "3B",
                quantization: "Q4_K_M",
                sizeBytes: 2_020_000_000,
                contextLength: 4096,
                architecture: .llama,
                repoID: "bartowski/Llama-3.2-3B-Instruct-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                checksum: "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
                isDraft: false,
                format: .gguf
            ),
            ModelManifest(
                id: "phi3-mini-3.8b-gguf",
                name: "Phi-3 Mini",
                variant: "3.8B Q4 GGUF",
                parameterCount: "3.8B",
                quantization: "Q4_K_M",
                sizeBytes: 2_390_000_000,
                contextLength: 4096,
                architecture: .phi,
                repoID: "bartowski/Phi-3.5-mini-instruct-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                checksum: "e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5",
                isDraft: false,
                format: .gguf
            ),
            ModelManifest(
                id: "dolphin3-qwen2.5-1.5b-q4-gguf",
                name: "Dolphin 3.0 Qwen",
                variant: "1.5B Q4 GGUF",
                parameterCount: "1.5B",
                quantization: "Q4_K_M",
                sizeBytes: 986_000_000,
                contextLength: 4096,
                architecture: .dolphin,
                repoID: "bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "Dolphin3.0-Qwen2.5-1.5B-Q4_K_M.gguf",
                checksum: "7caa630a60c8831a509e2663e1761355fa24bcf6ccc03e3cc767e5b5747a3be5",
                isDraft: false,
                format: .gguf,
                recommendation: ModelRecommendation(
                    badge: "Smallest",
                    reason: "Fastest Dolphin option under 1 GB when RAM or storage is tight.",
                    rank: 2
                )
            ),
            ModelManifest(
                id: "dolphin3-qwen2.5-1.5b-q8-gguf",
                name: "Dolphin 3.0 Qwen",
                variant: "1.5B Q8 GGUF",
                parameterCount: "1.5B",
                quantization: "Q8_0",
                sizeBytes: 1_650_000_000,
                contextLength: 4096,
                architecture: .dolphin,
                repoID: "bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "Dolphin3.0-Qwen2.5-1.5B-Q8_0.gguf",
                checksum: "a6e0e6f801428a083b39a25897a08eee3ab8a0f091573299c53d8f85cc8b3e04",
                isDraft: false,
                format: .gguf
            ),
            ModelManifest(
                id: "gemma2-2b-gguf",
                name: "Gemma 2",
                variant: "2B Q4 GGUF",
                parameterCount: "2B",
                quantization: "Q4_K_M",
                sizeBytes: 1_500_000_000,
                contextLength: 2048,
                architecture: .gemma,
                repoID: "bartowski/gemma-2-2b-it-GGUF",
                tokenizerRepoID: nil,
                modelFilePattern: "gemma-2-2b-it-Q4_K_M.gguf",
                checksum: "e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787",
                isDraft: false,
                format: .gguf
            )
        ]

        for model in availableModels {
            if let unsupportedStatus = statusForUnsupportedManifest(model) {
                modelStatuses[model.id] = unsupportedStatus
            } else {
                modelStatuses[model.id] = .notDownloaded
            }
        }

        restorePreviouslyDownloadedModels()
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

            modelStatuses[model.id] = status
        }
    }

    func downloadModel(_ modelID: String) {
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else { return }
        guard modelStatuses[modelID] != .some(.verifying),
              modelStatuses[modelID] != .some(.compiling) else { return }

        if case .downloading = modelStatuses[modelID] { return }
        if case .some(.unsupported(_)) = modelStatuses[modelID] { return }

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

                    let preIntegrity = fileSystem.verifyIntegrity(for: manifest, at: url)
                    guard preIntegrity.isValid else {
                        throw ModelLoaderError.integrityCheckFailed(statusForIntegrityResult(preIntegrity).displayMessage)
                    }

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
                        print("[AssetPipeline] tokenizer_config.json not found — proceeding with fallback tokenizer")
                    default:
                        break
                    }
                }

                let persistedTokenizerURL = try fileSystem.persistTokenizerAsset(from: tokenizerDir, forModelID: modelID)
                try saveTokenizerPath(persistedTokenizerURL, forModelID: modelID)

                let tokenizerIntegrity = verifyTokenizerIntegrity(for: manifest, at: persistedTokenizerURL)
                guard tokenizerIntegrity.isValid else {
                    throw ModelLoaderError.integrityCheckFailed(statusForIntegrityResult(tokenizerIntegrity).displayMessage)
                }

                if let hash = fileSystem.computeAssetSHA256(for: modelURL ?? tokenizerDir) {
                    fileSystem.saveChecksum(hash, forModelID: modelID)
                }

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
        guard activeModelID == nil else { return }
        if let preferredModelID, preferredModelID != modelID {
            return
        }
        activateModel(modelID)
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

    private func downloadTokenizerSnapshot(for manifest: ModelManifest, matching patterns: [String]) async throws -> URL {
        let tokenizerRepoID = manifest.tokenizerRepoID ?? manifest.repoID
        let tokenizerRepo = Hub.Repo(id: tokenizerRepoID)

        if let directSnapshot = try? await Hub.snapshot(from: tokenizerRepo, matching: patterns),
           (try? verifyTokenizerDependencies(in: directSnapshot)) != nil {
            return directSnapshot
        }

        let repositoryPaths = try await Self.repositoryPaths(for: tokenizerRepoID)
        let exactPaths = Self.tokenizerRepositoryPaths(from: repositoryPaths, allowedFileNames: patterns)
        guard !exactPaths.isEmpty else {
            throw ModelLoaderError.integrityCheckFailed("Missing required tokenizer file: tokenizer.json")
        }

        return try await Hub.snapshot(from: tokenizerRepo, matching: exactPaths)
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

                let ggufIntegrity = fileSystem.verifyIntegrity(for: manifest, at: url)
                guard ggufIntegrity.isValid else {
                    throw ModelLoaderError.integrityCheckFailed(statusForIntegrityResult(ggufIntegrity).displayMessage)
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

                if let hash = fileSystem.computeStreamingSHA256(for: persistedModelURL) {
                    fileSystem.saveChecksum(hash, forModelID: modelID)
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

    func deleteModel(_ modelID: String) {
        downloadTasks[modelID]?.cancel()
        downloadTasks.removeValue(forKey: modelID)
        activationTask?.cancel()
        if activationTargetModelID == modelID {
            activationTask = nil
            activationTargetModelID = nil
        }
        modelStatuses[modelID] = .notDownloaded

        if activeModelID == modelID {
            activeModelID = nil
            activeDraftModelID = nil
            modelRunner.unload()
            llamaRunner.unload()
            draftLlamaRunner.unload()
            tokenizer.unloadTokenizer()
            activeFormat = .coreML
        } else if activeDraftModelID == modelID {
            activeDraftModelID = nil
            draftLlamaRunner.unload()
        }

        if preferredModelID == modelID {
            persistPreferredModelID(nil)
        }

        fileSystem.deleteModelAssets(forModelID: modelID)
    }

    func activateModel(_ modelID: String) {
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

        if let activationTask, activationTargetModelID == modelID {
            return await activationTask.value
        }

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
        }
    }

    private func performModelActivation(_ manifest: ModelManifest, forceReload: Bool, persistSelection: Bool) async -> Bool {
        guard !Task.isCancelled else { return false }

        let modelID = manifest.id
        activeModelID = modelID
        activeFormat = manifest.format

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
                    if let tokenizerDir = loadTokenizerPath(forModelID: modelID) {
                        try await tokenizer.loadFromDirectory(tokenizerDir)
                    } else {
                        let tokenizerRepoID = manifest.tokenizerRepoID ?? manifest.repoID
                        try await tokenizer.loadFromHub(repoID: tokenizerRepoID)
                    }
                } catch {
                    print("Tokenizer load failed, using fallback: \(error)")
                }

                try await modelRunner.loadModel(at: modelURL, computeUnits: thermalComputeUnits)

            case .gguf:
                guard let modelURL = loadModelPath(forModelID: modelID) else {
                    throw ModelLoaderError.noModelFound("GGUF model file not found.")
                }

                modelRunner.unload()
                tokenizer.unloadTokenizer()
                activeDraftModelID = nil

                if forceReload || llamaRunner.isLoaded {
                    llamaRunner.unload()
                    draftLlamaRunner.unload()
                }

                try llamaRunner.loadModel(
                    at: modelURL.path,
                    nCtx: Int32(manifest.contextLength)
                )
                loadDraftRunnerIfPossible(for: manifest)
            }

            if persistSelection {
                persistPreferredModelID(modelID)
            }
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
                activeFormat = .coreML
            }
            return false
        }
    }

    private func loadDraftRunnerIfPossible(for targetManifest: ModelManifest) {
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
            try draftLlamaRunner.loadModel(
                at: draftURL.path,
                nCtx: Int32(min(draftManifest.contextLength, targetManifest.contextLength)),
                nGPULayers: 0
            )
            activeDraftModelID = draftManifest.id
        } catch {
            print("Draft GGUF load failed: \(error)")
            activeDraftModelID = nil
            draftLlamaRunner.unload()
        }
    }

    private func compatibleDraftManifest(for targetManifest: ModelManifest) -> ModelManifest? {
        availableModels
            .filter { manifest in
                manifest.format == .gguf &&
                manifest.isDraft &&
                manifest.id != targetManifest.id &&
                manifest.architecture == targetManifest.architecture &&
                modelStatuses[manifest.id] == .ready
            }
            .sorted { lhs, rhs in
                if lhs.sizeBytes != rhs.sizeBytes {
                    return lhs.sizeBytes < rhs.sizeBytes
                }
                return lhs.contextLength < rhs.contextLength
            }
            .first
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
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else {
            return fileSystem.verifyModelIntegrity(at: modelURL, format: modelURL.pathExtension)
        }

        let modelResult = verifyRestoredModelIntegrity(for: manifest, at: modelURL)
        guard modelResult.isValid else { return modelResult }

        if let tokenizerURL = loadTokenizerPath(forModelID: modelID) {
            return verifyTokenizerIntegrity(for: manifest, at: tokenizerURL)
        }

        return manifest.tokenizerRepoID == nil ? .intact : .missing
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
}

nonisolated enum ModelLoaderError: Error, Sendable, LocalizedError {
    case invalidPackage(String)
    case noModelFound(String)
    case compilationFailed(String)
    case integrityCheckFailed(String)
    case checksumMismatch(expected: String, actual: String)
    case assetRepairFailed(String)
    case partialDownload(String)

    var errorDescription: String? {
        switch self {
        case .invalidPackage(let msg): return msg
        case .noModelFound(let msg): return msg
        case .compilationFailed(let msg): return msg
        case .integrityCheckFailed(let msg): return msg
        case .checksumMismatch(let expected, let actual): return "Checksum mismatch: expected \(expected.prefix(12))… got \(actual.prefix(12))…"
        case .assetRepairFailed(let msg): return msg
        case .partialDownload(let msg): return msg
        }
    }
}
