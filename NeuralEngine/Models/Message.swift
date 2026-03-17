import Foundation

nonisolated struct Message: Identifiable, Sendable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date
    var isStreaming: Bool
    var metrics: GenerationMetrics?
    var toolResults: [ToolResult]
    var isToolExecution: Bool

    init(id: UUID = UUID(), role: MessageRole, content: String, timestamp: Date = Date(), isStreaming: Bool = false, metrics: GenerationMetrics? = nil, toolResults: [ToolResult] = [], isToolExecution: Bool = false) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.isStreaming = isStreaming
        self.metrics = metrics
        self.toolResults = toolResults
        self.isToolExecution = isToolExecution
    }
}

nonisolated enum MessageRole: String, Sendable, Codable {
    case system
    case user
    case assistant
    case tool
}

nonisolated struct GenerationMetrics: Sendable {
    var timeToFirstToken: Double
    var prefillTokensPerSecond: Double
    var decodeTokensPerSecond: Double
    var totalTokens: Int
    var totalDuration: Double
    var acceptedSpeculativeTokens: Int
    var rejectedSpeculativeTokens: Int
    var zeroTokenProbeLatencyMS: Double = 0
    var recoveryRetryCount: Int = 0
    var fallbackMode: String = "none"
}
