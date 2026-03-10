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
    let modelFilePattern: String
    let checksum: String
    let isDraft: Bool

    var sizeFormatted: String {
        let gb = Double(sizeBytes) / 1_073_741_824
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(sizeBytes) / 1_048_576
        return String(format: "%.0f MB", mb)
    }

    var downloadURL: String { "https://huggingface.co/\(repoID)" }
}

nonisolated enum ModelArchitecture: String, Sendable, Codable {
    case llama
    case phi
    case gemma
    case qwen
    case mistral
    case smolLM = "smollm"
}

nonisolated enum ModelStatus: Sendable, Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case verifying
    case compiling
    case ready
    case failed(String)
}
