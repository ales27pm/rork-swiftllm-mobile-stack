import Foundation

nonisolated struct InferenceMetrics: Sendable {
    var prefillTokensPerSecond: Double = 0
    var decodeTokensPerSecond: Double = 0
    var acceptedSpeculativeTokens: Int = 0
    var rejectedSpeculativeTokens: Int = 0
    var kvPagesInUse: Int = 0
    var activeContextLength: Int = 0
    var avgTokenLatencyMS: Double = 0
    var peakMemoryBytes: Int64 = 0
    var timeToFirstTokenMS: Double = 0
    var totalTokensGenerated: Int = 0
    var thermalState: ThermalLevel = .nominal
    var contextEvictions: Int = 0
    var zeroTokenProbeLatencyMS: Double = 0
    var recoveryRetryCount: Int = 0
    var fallbackMode: String = "none"

    var speculativeAcceptanceRate: Double {
        let total = acceptedSpeculativeTokens + rejectedSpeculativeTokens
        guard total > 0 else { return 0 }
        return Double(acceptedSpeculativeTokens) / Double(total)
    }
}

nonisolated enum ThermalLevel: String, Sendable {
    case nominal = "Nominal"
    case fair = "Fair"
    case serious = "Serious"
    case critical = "Critical"
}
