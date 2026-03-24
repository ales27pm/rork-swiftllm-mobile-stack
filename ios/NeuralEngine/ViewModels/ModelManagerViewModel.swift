import SwiftUI

@Observable
@MainActor
class ModelManagerViewModel {
    let modelLoader: ModelLoaderService

    var searchText: String = ""
    var selectedFilter: ModelFilter = .recommended

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
        modelLoader.deleteModel(model.id)
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
}

nonisolated enum ModelFilter: String, CaseIterable, Sendable {
    case recommended = "Recommended"
    case all = "All"
    case downloaded = "Downloaded"
    case gguf = "GGUF"
    case coreml = "CoreML"
    case draft = "Draft"
    case embedding = "Embedding"
}
