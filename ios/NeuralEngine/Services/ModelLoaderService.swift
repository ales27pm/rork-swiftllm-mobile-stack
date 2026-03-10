import Foundation
import CoreML
import Hub

@Observable
@MainActor
class ModelLoaderService {
    var availableModels: [ModelManifest] = []
    var modelStatuses: [String: ModelStatus] = [:]
    var activeModelID: String?
    var isLoadingRegistry: Bool = false

    let modelRunner = CoreMLModelRunner()
    let tokenizer = TokenizerService()

    private var downloadTasks: [String: Task<Void, Never>] = [:]

    init() {
        loadBuiltinRegistry()
    }

    func loadBuiltinRegistry() {
        availableModels = [
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
                checksum: "",
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
                checksum: "",
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
                checksum: "",
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
            )
        ]

        for model in availableModels {
            modelStatuses[model.id] = .notDownloaded
        }

        restorePreviouslyDownloadedModels()
    }

    private func restorePreviouslyDownloadedModels() {
        for model in availableModels {
            if let modelURL = loadModelPath(forModelID: model.id) {
                let ext = modelURL.pathExtension
                if ext == "mlmodelc" {
                    modelStatuses[model.id] = .ready
                } else if ext == "mlpackage" {
                    let manifestURL = modelURL.appendingPathComponent("Manifest.json")
                    if FileManager.default.fileExists(atPath: manifestURL.path) {
                        modelStatuses[model.id] = .ready
                    } else {
                        modelStatuses[model.id] = .failed("Model package is incomplete. Delete and re-download.")
                    }
                } else {
                    modelStatuses[model.id] = .ready
                }
            }
        }
    }

    func downloadModel(_ modelID: String) {
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else { return }
        guard modelStatuses[modelID] != .some(.verifying),
              modelStatuses[modelID] != .some(.compiling) else { return }

        if case .downloading = modelStatuses[modelID] { return }

        if let existingPath = loadModelPath(forModelID: modelID) {
            try? FileManager.default.removeItem(at: modelPathURL(forModelID: modelID))
            try? FileManager.default.removeItem(at: tokenizerPathURL(forModelID: modelID))
        }

        modelStatuses[modelID] = .downloading(progress: 0)

        let task = Task {
            do {
                modelStatuses[modelID] = .downloading(progress: 0.05)

                let tokenizerRepoID = manifest.tokenizerRepoID ?? manifest.repoID
                let tokenizerRepo = Hub.Repo(id: tokenizerRepoID)

                let tokenizerPatterns = ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json", "generation_config.json"]

                modelStatuses[modelID] = .downloading(progress: 0.1)

                let tokenizerDir = try await Hub.snapshot(from: tokenizerRepo, matching: tokenizerPatterns)

                modelStatuses[modelID] = .downloading(progress: 0.3)

                let modelRepo = Hub.Repo(id: manifest.repoID)
                let modelPatterns = buildModelDownloadPatterns(for: manifest)
                var modelDir: URL?
                for patternGroup in modelPatterns {
                    do {
                        let dir = try await Hub.snapshot(from: modelRepo, matching: patternGroup)
                        modelDir = dir
                        break
                    } catch {
                        continue
                    }
                }

                modelStatuses[modelID] = .downloading(progress: 0.7)

                let searchDir = modelDir ?? tokenizerDir

                modelStatuses[modelID] = .verifying
                try? await Task.sleep(for: .seconds(0.3))

                let modelURL = findModelFile(in: searchDir)

                if let url = modelURL {
                    let ext = url.pathExtension
                    if ext == "mlmodelc" {
                        try saveModelPath(url, forModelID: modelID)
                    } else if ext == "mlpackage" {
                        let manifestFile = url.appendingPathComponent("Manifest.json")
                        if FileManager.default.fileExists(atPath: manifestFile.path) {
                            modelStatuses[modelID] = .compiling
                            let compiledURL = try await compileModel(at: url)
                            try saveModelPath(compiledURL, forModelID: modelID)
                        } else {
                            let innerModel = findMLModelFile(in: url)
                            if let innerModel {
                                modelStatuses[modelID] = .compiling
                                let compiledURL = try await compileModel(at: innerModel)
                                try saveModelPath(compiledURL, forModelID: modelID)
                            } else {
                                throw ModelLoaderError.invalidPackage("mlpackage is missing Manifest.json. The download may be incomplete — delete and re-download.")
                            }
                        }
                    } else if ext == "mlmodel" {
                        modelStatuses[modelID] = .compiling
                        let compiledURL = try await compileModel(at: url)
                        try saveModelPath(compiledURL, forModelID: modelID)
                    } else {
                        throw ModelLoaderError.noModelFound("Downloaded files do not contain a valid CoreML model.")
                    }
                } else {
                    throw ModelLoaderError.noModelFound("No CoreML model file found in the downloaded repository. This repo may not contain CoreML models.")
                }

                try saveTokenizerPath(tokenizerDir, forModelID: modelID)

                modelStatuses[modelID] = .ready
            } catch {
                if !Task.isCancelled {
                    modelStatuses[modelID] = .failed(error.localizedDescription)
                }
            }
        }

        downloadTasks[modelID] = task
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
                    "\(cleanBase)/*",
                    "\(cleanBase)/*/*",
                    "\(cleanBase)/*/*/*"
                ]
            ]
        }

        return [
            ["*.mlpackage/Manifest.json", "*.mlpackage/Data/*", "*.mlpackage/Data/*/*", "*.mlpackage/Data/*/*/*", "*.mlpackage/Data/*/*/*/*"],
            ["*.mlpackage/*", "*.mlpackage/*/*", "*.mlpackage/*/*/*", "*.mlpackage/*/*/*/*", "*.mlpackage/*/*/*/*/*"],
            ["*.mlmodelc/*", "*.mlmodelc/*/*", "*.mlmodelc/*/*/*"],
            ["*.mlmodel"]
        ]
    }

    func deleteModel(_ modelID: String) {
        downloadTasks[modelID]?.cancel()
        downloadTasks.removeValue(forKey: modelID)
        modelStatuses[modelID] = .notDownloaded

        if activeModelID == modelID {
            activeModelID = nil
            modelRunner.unload()
            tokenizer.unloadTokenizer()
        }

        let modelPathFile = modelPathURL(forModelID: modelID)
        try? FileManager.default.removeItem(at: modelPathFile)

        let tokenizerPathFile = tokenizerPathURL(forModelID: modelID)
        try? FileManager.default.removeItem(at: tokenizerPathFile)
    }

    func activateModel(_ modelID: String) {
        guard case .ready = modelStatuses[modelID] else { return }
        guard let manifest = availableModels.first(where: { $0.id == modelID }) else { return }

        activeModelID = modelID

        Task {
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

            do {
                if let modelURL = loadModelPath(forModelID: modelID) {
                    let computeUnits: MLComputeUnits
                    switch thermalComputeUnits {
                    case .all: computeUnits = .all
                    case .cpuAndNeuralEngine: computeUnits = .cpuAndNeuralEngine
                    case .cpuOnly: computeUnits = .cpuOnly
                    default: computeUnits = .all
                    }
                    try await modelRunner.loadModel(at: modelURL, computeUnits: computeUnits)
                }
            } catch {
                print("Model load failed: \(error)")
                modelStatuses[modelID] = .failed("Failed to load: \(error.localizedDescription)")
                activeModelID = nil
            }
        }
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
            let compiledURL = try await MLModel.compileModel(at: url)
            let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            let destURL = cacheDir.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".mlmodelc")
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.moveItem(at: compiledURL, to: destURL)
            return destURL
        } catch {
            throw ModelLoaderError.compilationFailed("Failed to compile model: \(error.localizedDescription)")
        }
    }

    private func modelPathURL(forModelID id: String) -> URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("model_path_\(id).txt")
    }

    private func tokenizerPathURL(forModelID id: String) -> URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("tokenizer_path_\(id).txt")
    }

    private func saveModelPath(_ url: URL, forModelID id: String) throws {
        try url.path.write(to: modelPathURL(forModelID: id), atomically: true, encoding: .utf8)
    }

    private func saveTokenizerPath(_ url: URL, forModelID id: String) throws {
        try url.path.write(to: tokenizerPathURL(forModelID: id), atomically: true, encoding: .utf8)
    }

    private func loadModelPath(forModelID id: String) -> URL? {
        guard let path = try? String(contentsOf: modelPathURL(forModelID: id), encoding: .utf8) else { return nil }
        let url = URL(fileURLWithPath: path)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private func loadTokenizerPath(forModelID id: String) -> URL? {
        guard let path = try? String(contentsOf: tokenizerPathURL(forModelID: id), encoding: .utf8) else { return nil }
        let url = URL(fileURLWithPath: path)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

nonisolated enum ModelLoaderError: Error, Sendable, LocalizedError {
    case invalidPackage(String)
    case noModelFound(String)
    case compilationFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidPackage(let msg): return msg
        case .noModelFound(let msg): return msg
        case .compilationFailed(let msg): return msg
        }
    }
}
