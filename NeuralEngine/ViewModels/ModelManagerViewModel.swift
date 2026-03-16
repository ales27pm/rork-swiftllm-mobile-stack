import SwiftUI

@Observable
@MainActor
class ModelManagerViewModel {
    let modelLoader: ModelLoaderService

    var searchText: String = ""
    var selectedFilter: ModelFilter = .all

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
        case .all: break
        case .downloaded:
            models = models.filter { model in
                if case .ready = modelLoader.modelStatuses[model.id] { return true }
                return false
            }
        case .draft:
            models = models.filter(\.isDraft)
        case .gguf:
            models = models.filter { $0.format == .gguf }
        case .coreml:
            models = models.filter { $0.format == .coreML }
        }

        return models
    }

    func download(_ model: ModelManifest) {
        modelLoader.downloadModel(model.id)
    }

    func delete(_ model: ModelManifest) {
        modelLoader.deleteModel(model.id)
    }

    func activate(_ model: ModelManifest) {
        modelLoader.activateModel(model.id)
    }

    func status(for model: ModelManifest) -> ModelStatus {
        modelLoader.modelStatuses[model.id] ?? .notDownloaded
    }

    var activeModelID: String? {
        modelLoader.activeModelID
    }
}

enum ModelFilter: String, CaseIterable {
    case all = "All"
    case downloaded = "Downloaded"
    case gguf = "GGUF"
    case coreml = "CoreML"
    case draft = "Draft"
}
