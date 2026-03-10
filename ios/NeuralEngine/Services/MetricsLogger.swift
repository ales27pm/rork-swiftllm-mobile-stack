import Foundation

@Observable
@MainActor
class MetricsLogger {
    var currentMetrics = InferenceMetrics()
    var history: [InferenceMetrics] = []
    var speedHistory: [Double] = []
    var totalEvictedTokens: Int = 0

    private var generationStart: Date?
    private var firstTokenTime: Date?
    private var tokenTimestamps: [Date] = []

    func beginGeneration() {
        generationStart = Date()
        firstTokenTime = nil
        tokenTimestamps.removeAll()
        currentMetrics = InferenceMetrics()
    }

    func recordFirstToken() {
        firstTokenTime = Date()
        if let start = generationStart {
            currentMetrics.timeToFirstTokenMS = firstTokenTime!.timeIntervalSince(start) * 1000
        }
    }

    func recordToken() {
        let now = Date()
        tokenTimestamps.append(now)
        currentMetrics.totalTokensGenerated = tokenTimestamps.count

        if tokenTimestamps.count >= 2 {
            let recentWindow = tokenTimestamps.suffix(20)
            if recentWindow.count >= 2 {
                let duration = recentWindow.last!.timeIntervalSince(recentWindow.first!)
                let tokens = recentWindow.count - 1
                let speed = Double(tokens) / max(duration, 0.001)
                currentMetrics.decodeTokensPerSecond = speed

                speedHistory.append(speed)
                if speedHistory.count > 60 {
                    speedHistory.removeFirst()
                }
            }
        }

        if let first = tokenTimestamps.first {
            let totalDuration = now.timeIntervalSince(first)
            currentMetrics.avgTokenLatencyMS = (totalDuration / Double(tokenTimestamps.count)) * 1000
        }
    }

    func recordPrefill(tokens: Int, duration: Double) {
        currentMetrics.prefillTokensPerSecond = Double(tokens) / max(duration, 0.001)
    }

    func recordSpeculative(accepted: Int, rejected: Int) {
        currentMetrics.acceptedSpeculativeTokens += accepted
        currentMetrics.rejectedSpeculativeTokens += rejected
    }

    func recordKVPages(_ count: Int) {
        currentMetrics.kvPagesInUse = count
    }

    func recordContextLength(_ length: Int) {
        currentMetrics.activeContextLength = length
    }

    func recordThermalState(_ level: ThermalLevel) {
        currentMetrics.thermalState = level
    }

    func recordMemory(_ bytes: Int64) {
        currentMetrics.peakMemoryBytes = max(currentMetrics.peakMemoryBytes, bytes)
    }

    func recordContextEviction(evictedTokens: Int) {
        totalEvictedTokens += evictedTokens
        currentMetrics.contextEvictions += 1
    }

    func endGeneration() {
        history.append(currentMetrics)
        if history.count > 100 {
            history.removeFirst()
        }
    }

    var averageDecodeSpeed: Double {
        guard !history.isEmpty else { return 0 }
        return history.map(\.decodeTokensPerSecond).reduce(0, +) / Double(history.count)
    }
}
