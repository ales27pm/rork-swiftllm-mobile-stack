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
    let isDraft: Bool
    let format: ModelFormat
    let recommendation: ModelRecommendation?
    let isEmbedding: Bool
    let embeddingDimensions: Int?
    let embeddingPooling: EmbeddingPoolingStrategy
    let isCustom: Bool

    init(id: String, name: String, variant: String, parameterCount: String, quantization: String, sizeBytes: Int64, contextLength: Int, architecture: ModelArchitecture, repoID: String, tokenizerRepoID: String?, modelFilePattern: String, isDraft: Bool, format: ModelFormat = .coreML, recommendation: ModelRecommendation? = nil, isEmbedding: Bool = false, embeddingDimensions: Int? = nil, embeddingPooling: EmbeddingPoolingStrategy = .mean, isCustom: Bool = false) {
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
        self.isDraft = isDraft
        self.format = format
        self.recommendation = recommendation
        self.isEmbedding = isEmbedding
        self.embeddingDimensions = embeddingDimensions
        self.embeddingPooling = Self.sanitizeEmbeddingPooling(
            embeddingPooling,
            isEmbedding: isEmbedding,
            manifestID: id
        )
        self.isCustom = isCustom
    }

    nonisolated enum CodingKeys: String, CodingKey {
        case id, name, variant, parameterCount, quantization, sizeBytes, contextLength
        case architecture, repoID, tokenizerRepoID, modelFilePattern
        case isDraft, format, recommendation, isEmbedding, embeddingDimensions, embeddingPooling, isCustom
    }

    nonisolated init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        variant = try container.decode(String.self, forKey: .variant)
        parameterCount = try container.decode(String.self, forKey: .parameterCount)
        quantization = try container.decode(String.self, forKey: .quantization)
        sizeBytes = try container.decode(Int64.self, forKey: .sizeBytes)
        contextLength = try container.decode(Int.self, forKey: .contextLength)
        architecture = try container.decode(ModelArchitecture.self, forKey: .architecture)
        repoID = try container.decode(String.self, forKey: .repoID)
        tokenizerRepoID = try container.decodeIfPresent(String.self, forKey: .tokenizerRepoID)
        modelFilePattern = try container.decode(String.self, forKey: .modelFilePattern)
        isDraft = try container.decode(Bool.self, forKey: .isDraft)
        format = try container.decode(ModelFormat.self, forKey: .format)
        recommendation = try container.decodeIfPresent(ModelRecommendation.self, forKey: .recommendation)
        isEmbedding = try container.decodeIfPresent(Bool.self, forKey: .isEmbedding) ?? false
        embeddingDimensions = try container.decodeIfPresent(Int.self, forKey: .embeddingDimensions)
        let requestedPooling: EmbeddingPoolingStrategy
        if let rawPooling = try container.decodeIfPresent(String.self, forKey: .embeddingPooling) {
            requestedPooling = EmbeddingPoolingStrategy(rawValue: rawPooling) ?? .mean
        } else {
            requestedPooling = .mean
        }
        embeddingPooling = Self.sanitizeEmbeddingPooling(
            requestedPooling,
            isEmbedding: isEmbedding,
            manifestID: id
        )
        isCustom = try container.decodeIfPresent(Bool.self, forKey: .isCustom) ?? false
    }

    private static func sanitizeEmbeddingPooling(_ pooling: EmbeddingPoolingStrategy, isEmbedding: Bool, manifestID: String) -> EmbeddingPoolingStrategy {
        guard !(!isEmbedding && pooling != .mean) else {
            assertionFailure("Non-embedding manifest '\(manifestID)' cannot use '\(pooling.rawValue)' embedding pooling. Falling back to 'mean'.")
            return .mean
        }
        return pooling
    }

    static func customGGUF(repoID: String, fileName: String, name: String, sizeBytes: Int64) -> ModelManifest {
        let id = "custom-\(repoID.replacingOccurrences(of: "/", with: "-").lowercased())-\(fileName.replacingOccurrences(of: ".gguf", with: "").lowercased())"
        let architecture = Self.inferArchitecture(from: repoID, fileName: fileName)
        return ModelManifest(
            id: id,
            name: name,
            variant: fileName.replacingOccurrences(of: ".gguf", with: ""),
            parameterCount: "?",
            quantization: Self.inferQuantization(from: fileName),
            sizeBytes: sizeBytes,
            contextLength: 4096,
            architecture: architecture,
            repoID: repoID,
            tokenizerRepoID: nil,
            modelFilePattern: fileName,
            isDraft: false,
            format: .gguf,
            isCustom: true
        )
    }

    private static func inferQuantization(from fileName: String) -> String {
        let upper = fileName.uppercased()
        let quantizations = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_XS", "IQ4_NL", "IQ4_XS"]
        for q in quantizations {
            if upper.contains(q) { return q }
        }
        return "Unknown"
    }

    private static func inferArchitecture(from repoID: String, fileName: String) -> ModelArchitecture {
        let combined = (repoID + " " + fileName).lowercased()
        if combined.contains("llama") { return .llama }
        if combined.contains("dolphin") { return .dolphin }
        if combined.contains("qwen") { return .qwen }
        if combined.contains("gemma") { return .gemma }
        if combined.contains("phi") { return .phi }
        if combined.contains("mistral") { return .mistral }
        if combined.contains("smollm") { return .smolLM }
        if combined.contains("lfm") { return .lfm2 }
        if combined.contains("bert") || combined.contains("embed") { return .bert }
        return .llama
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
        case "lfm2":
            return .lfm25
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
    case appleFoundation
}

nonisolated enum EmbeddingPoolingStrategy: String, Sendable, Codable {
    case mean
    case lastToken = "last_token"
}

nonisolated enum GGUFChatTemplateStyle: String, Sendable, Codable {
    case chatML
    case llama3
    case gemma2
    case lfm25
}

nonisolated enum ModelStatus: Sendable, Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case verifying
    case compiling
    case ready
    case failed(String)

    var blocksDownload: Bool {
        return false
    }

    var displayMessage: String {
        switch self {
        case .notDownloaded:
            return "Not downloaded"
        case .downloading:
            return "Downloading"
        case .verifying:
            return "Verifying integrity"
        case .compiling:
            return "Compiling model"
        case .ready:
            return "Ready"
        case .failed(let message):
            return message
        }
    }
}
