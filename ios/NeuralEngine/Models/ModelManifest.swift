import Foundation

nonisolated struct ModelManifest: Identifiable, Sendable, Codable {
    let id: String
    let name: String
    let variant: String
    let parameterCount: String
    let quantization: String
    let sizeBytes: Int64
    let contextLength: Int
    let architecture: ModelArchitecture
    let repoID: String
    let tokenizerRepoID: String?
    let modelFilePattern: String
    let checksum: String
    let tokenizerChecksum: String?
    let isDraft: Bool
    let format: ModelFormat
    let recommendation: ModelRecommendation?
    let isEmbedding: Bool
    let embeddingDimensions: Int?

    init(id: String, name: String, variant: String, parameterCount: String, quantization: String, sizeBytes: Int64, contextLength: Int, architecture: ModelArchitecture, repoID: String, tokenizerRepoID: String?, modelFilePattern: String, checksum: String, tokenizerChecksum: String? = nil, isDraft: Bool, format: ModelFormat = .coreML, recommendation: ModelRecommendation? = nil, isEmbedding: Bool = false, embeddingDimensions: Int? = nil) {
        self.id = id
        self.name = name
        self.variant = variant
        self.parameterCount = parameterCount
        self.quantization = quantization
        self.sizeBytes = sizeBytes
        self.contextLength = contextLength
        self.architecture = architecture
        self.repoID = repoID
        self.tokenizerRepoID = tokenizerRepoID
        self.modelFilePattern = modelFilePattern
        self.checksum = checksum
        self.tokenizerChecksum = tokenizerChecksum
        self.isDraft = isDraft
        self.format = format
        self.recommendation = recommendation
        self.isEmbedding = isEmbedding
        self.embeddingDimensions = embeddingDimensions
    }

    var sizeFormatted: String {
        let gb = Double(sizeBytes) / 1_073_741_824
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(sizeBytes) / 1_048_576
        return String(format: "%.0f MB", mb)
    }

    var downloadURL: String { "https://huggingface.co/\(repoID)" }

    var draftCompatibilityIdentifier: String {
        let combined = [id, repoID, tokenizerRepoID ?? "", modelFilePattern]
            .joined(separator: " ")
            .lowercased()

        if combined.contains("dolphin3.0-qwen2.5") || (combined.contains("dolphin") && combined.contains("qwen2.5")) {
            return "dolphin-qwen2.5"
        }

        if combined.contains("dolphin3.0-llama3.2") || (combined.contains("dolphin") && (combined.contains("llama-3.2") || combined.contains("llama3.2"))) {
            return "dolphin-llama3.2"
        }

        if combined.contains("smollm2") {
            return "smollm2"
        }

        if combined.contains("qwen2.5") {
            return "qwen2.5"
        }

        if combined.contains("llama-3.2") || combined.contains("llama3.2") {
            return "llama3.2"
        }

        if combined.contains("gemma-2") || combined.contains("gemma 2") {
            return "gemma2"
        }

        return architecture.rawValue
    }

    var ggufChatTemplateStyle: GGUFChatTemplateStyle {
        switch draftCompatibilityIdentifier {
        case "dolphin-llama3.2", "llama3.2":
            return .llama3
        case "gemma2":
            return .gemma2
        default:
            return .chatML
        }
    }
}

nonisolated struct ModelRecommendation: Sendable, Codable, Equatable {
    let badge: String
    let reason: String
    let rank: Int
}

nonisolated enum ModelArchitecture: String, Sendable, Codable {
    case llama
    case phi
    case gemma
    case qwen
    case mistral
    case smolLM = "smollm"
    case dolphin
    case bert
    case lfm2
}

nonisolated enum ModelFormat: String, Sendable, Codable {
    case coreML
    case gguf
}

nonisolated enum GGUFChatTemplateStyle: String, Sendable, Codable {
    case chatML
    case llama3
    case gemma2
}

nonisolated enum ModelStatus: Sendable, Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case verifying
    case compiling
    case ready
    case unsupported(String)
    case checksumFailed(String)
    case failed(String)

    var blocksDownload: Bool {
        if case .unsupported = self { return true }
        return false
    }

    var displayMessage: String {
        switch self {
        case .notDownloaded:
            return "Not downloaded"
        case .downloading:
            return "Downloading"
        case .verifying:
            return "Verifying checksum"
        case .compiling:
            return "Compiling model"
        case .ready:
            return "Ready"
        case .unsupported(let message), .checksumFailed(let message), .failed(let message):
            return message
        }
    }
}
