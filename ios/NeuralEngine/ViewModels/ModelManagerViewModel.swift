import SwiftUI

@Observable
@MainActor
class ModelManagerViewModel {
    let modelLoader: ModelLoaderService

    var searchText: String = ""
    var selectedFilter: ModelFilter = .recommended
    var showAddCustomModel: Bool = false
    var customRepoInput: String = ""
    var isProbing: Bool = false
    var probeError: String?
    var probedFiles: [HuggingFaceFileInfo] = []
    var probedRepoID: String?

    init(modelLoader: ModelLoaderService) {
        self.modelLoader = modelLoader
    }

    var filteredModels: [ModelManifest] {
        var models = modelLoader.availableModels

        if !searchText.isEmpty {
            models = models.filter {
                $0.name.localizedStandardContains(searchText) ||
                $0.variant.localizedStandardContains(searchText) ||
                $0.architecture.rawValue.localizedStandardContains(searchText)
            }
        }

        switch selectedFilter {
        case .all:
            models = models.filter { !$0.isEmbedding }
        case .recommended:
            models = models.filter { $0.recommendation != nil && !$0.isEmbedding }
        case .downloaded:
            models = models.filter { model in
                if model.isEmbedding {
                    if case .ready = modelLoader.embeddingModelStatuses[model.id] { return true }
                    return false
                }
                if case .ready = modelLoader.modelStatuses[model.id] { return true }
                return false
            }
        case .custom:
            models = models.filter { $0.isCustom }
        case .draft:
            models = models.filter { $0.isDraft && !$0.isEmbedding }
        case .gguf:
            models = models.filter { $0.format == .gguf && !$0.isEmbedding }
        case .coreml:
            models = models.filter { $0.format == .coreML && !$0.isEmbedding }
        case .embedding:
            models = models.filter(\.isEmbedding)
        }

        return models.sorted { lhs, rhs in
            let lhsRank = lhs.recommendation?.rank ?? Int.max
            let rhsRank = rhs.recommendation?.rank ?? Int.max
            if lhsRank != rhsRank {
                return lhsRank < rhsRank
            }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }
    }

    func download(_ model: ModelManifest) {
        if model.isEmbedding {
            modelLoader.downloadEmbeddingModel(model.id)
        } else {
            modelLoader.downloadModel(model.id)
        }
    }

    func delete(_ model: ModelManifest) {
        if model.isCustom {
            modelLoader.deleteCustomModel(model.id)
        } else {
            modelLoader.deleteModel(model.id)
        }
    }

    func activate(_ model: ModelManifest) {
        if model.isEmbedding {
            modelLoader.activateEmbeddingModel(model.id)
        } else if model.isDraft {
            modelLoader.activateDraftModel(model.id)
        } else {
            modelLoader.activateModel(model.id)
        }
    }

    func deactivateDraft(_ model: ModelManifest) {
        guard model.isDraft else { return }
        modelLoader.deactivateDraftModel()
    }

    func status(for model: ModelManifest) -> ModelStatus {
        if model.isEmbedding {
            return modelLoader.embeddingModelStatuses[model.id] ?? .notDownloaded
        }
        return modelLoader.modelStatuses[model.id] ?? .notDownloaded
    }

    var activeModelID: String? {
        modelLoader.activeModelID
    }

    var activeDraftModelID: String? {
        modelLoader.activeDraftModelID
    }

    var activeEmbeddingModelID: String? {
        modelLoader.activeEmbeddingModelID
    }

    func isActiveModel(_ model: ModelManifest) -> Bool {
        if model.isEmbedding {
            return modelLoader.activeEmbeddingModelID == model.id
        }
        if model.isDraft {
            return modelLoader.activeDraftModelID == model.id
        }
        return modelLoader.activeModelID == model.id
    }

    func isDraftCompatibleWithActiveModel(_ model: ModelManifest) -> Bool {
        guard model.isDraft, let activeModel = modelLoader.activeModel else { return false }
        return model.architecture == activeModel.architecture && activeModel.format == .gguf && !activeModel.isDraft
    }

    var activeDraftModel: ModelManifest? {
        modelLoader.activeDraftModel
    }

    var activeMainModel: ModelManifest? {
        modelLoader.activeModel
    }

    func probeCustomRepo() {
        let input = customRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return }

        let repoID: String
        if let parsed = ModelLoaderService.repoNameFromURL(input) {
            repoID = parsed
        } else if input.contains("/"), input.components(separatedBy: "/").count >= 2 {
            repoID = input
        } else {
            probeError = "Enter a valid HuggingFace repo (e.g. user/model-name)"
            return
        }

        isProbing = true
        probeError = nil
        probedFiles = []
        probedRepoID = repoID

        Task {
            do {
                let files = try await ModelLoaderService.probeHuggingFaceRepoWithSizes(repoID)
                probedFiles = files
                if files.isEmpty {
                    probeError = "No GGUF files found in this repository."
                }
            } catch {
                probeError = error.localizedDescription
            }
            isProbing = false
        }
    }

    func addCustomFile(_ file: HuggingFaceFileInfo) {
        guard let repoID = probedRepoID else { return }
        let repoName = repoID.components(separatedBy: "/").last ?? repoID
        let manifest = ModelManifest.customGGUF(
            repoID: repoID,
            fileName: file.fileName,
            name: repoName,
            sizeBytes: file.sizeBytes
        )
        modelLoader.addCustomModel(manifest)
        showAddCustomModel = false
        resetProbe()
    }

    func resetProbe() {
        customRepoInput = ""
        probedFiles = []
        probedRepoID = nil
        probeError = nil
        isProbing = false
    }
}

nonisolated enum ModelFilter: String, CaseIterable, Sendable {
    case recommended = "Recommended"
    case all = "All"
    case downloaded = "Downloaded"
    case custom = "Custom"
    case gguf = "GGUF"
    case coreml = "CoreML"
    case draft = "Draft"
    case embedding = "Embedding"
}
