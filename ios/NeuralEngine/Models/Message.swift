import Foundation

nonisolated struct Message: Identifiable, Sendable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date
    var isStreaming: Bool
    var metrics: GenerationMetrics?

    init(id: UUID = UUID(), role: MessageRole, content: String, timestamp: Date = Date(), isStreaming: Bool = false, metrics: GenerationMetrics? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.isStreaming = isStreaming
        self.metrics = metrics
    }
}

nonisolated enum MessageRole: String, Sendable, Codable {
    case system
    case user
    case assistant
}

nonisolated struct GenerationMetrics: Sendable {
    var timeToFirstToken: Double
    var prefillTokensPerSecond: Double
    var decodeTokensPerSecond: Double
    var totalTokens: Int
    var totalDuration: Double
    var acceptedSpeculativeTokens: Int
    var rejectedSpeculativeTokens: Int
}
