import Foundation

nonisolated struct DiagnosticEvent: Sendable, Identifiable {
    let id: UUID
    let code: DiagnosticCode
    let message: String
    let severity: DiagnosticSeverity
    let timestamp: Date
    let metadata: [String: String]

    init(code: DiagnosticCode, message: String, severity: DiagnosticSeverity, metadata: [String: String] = [:]) {
        self.id = UUID()
        self.code = code
        self.message = message
        self.severity = severity
        self.timestamp = Date()
        self.metadata = metadata
    }
}

nonisolated enum DiagnosticCode: String, Sendable {
    case modelEvicted = "MODEL_EVICTED"
    case memoryPressure = "MEMORY_PRESSURE"
    case thermalEscalation = "THERMAL_ESCALATION"
    case recoveryStarted = "RECOVERY_STARTED"
    case recoveryCompleted = "RECOVERY_COMPLETED"
    case recoveryFailed = "RECOVERY_FAILED"
    case computeUnitDegraded = "COMPUTE_DEGRADED"
    case contextEviction = "CONTEXT_EVICTION"
    case inferenceThrottled = "INFERENCE_THROTTLED"
    case healthCheckFailed = "HEALTH_CHECK_FAILED"
    case prefixCacheHit = "PREFIX_CACHE_HIT"
    case generationComplete = "GENERATION_COMPLETE"
    case assetIntegrityPassed = "ASSET_INTEGRITY_PASSED"
    case assetIntegrityFailed = "ASSET_INTEGRITY_FAILED"
    case assetChecksumMismatch = "ASSET_CHECKSUM_MISMATCH"
    case assetRepairStarted = "ASSET_REPAIR_STARTED"
    case assetRepairCompleted = "ASSET_REPAIR_COMPLETED"
    case assetRepairFailed = "ASSET_REPAIR_FAILED"
    case partialDownloadDetected = "PARTIAL_DOWNLOAD_DETECTED"
    case speechServiceBusy = "SPEECH_SERVICE_BUSY"
    case speechRecognizerReload = "SPEECH_RECOGNIZER_RELOAD"
    case cpuFallbackTriggered = "CPU_FALLBACK_TRIGGERED"
    case contextWindowReduced = "CONTEXT_WINDOW_REDUCED"
    case forensicDiagnosticLogged = "FORENSIC_DIAGNOSTIC_LOGGED"
}

nonisolated enum DiagnosticSeverity: Int, Sendable, Comparable {
    case info = 0
    case warning = 1
    case critical = 2

    static func < (lhs: DiagnosticSeverity, rhs: DiagnosticSeverity) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var label: String {
        switch self {
        case .info: return "Info"
        case .warning: return "Warning"
        case .critical: return "Critical"
        }
    }
}

@Observable
@MainActor
class MetricsLogger {
    var currentMetrics = InferenceMetrics()
    var history: [InferenceMetrics] = []
    var speedHistory: [Double] = []
    var totalEvictedTokens: Int = 0
    var diagnosticEvents: [DiagnosticEvent] = []
    var lastDiagnosticCode: DiagnosticCode?
    var activeComputeLabel: String = "All"
    var totalRecoveries: Int = 0
    var totalEvictions: Int = 0
    var sessionUptime: Date = Date()

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

    func recordDiagnostic(_ event: DiagnosticEvent) {
        diagnosticEvents.append(event)
        lastDiagnosticCode = event.code
        if diagnosticEvents.count > 200 {
            diagnosticEvents.removeFirst()
        }
    }

    func recordModelEviction(reason: String, computeUnits: String) {
        totalEvictions += 1
        recordDiagnostic(DiagnosticEvent(
            code: .modelEvicted,
            message: "Neural Engine eviction detected",
            severity: .critical,
            metadata: ["reason": reason, "computeUnits": computeUnits]
        ))
    }

    func recordRecovery(success: Bool, newComputeUnits: String) {
        if success {
            totalRecoveries += 1
            activeComputeLabel = newComputeUnits
            recordDiagnostic(DiagnosticEvent(
                code: .recoveryCompleted,
                message: "Model recovered on \(newComputeUnits)",
                severity: .info,
                metadata: ["computeUnits": newComputeUnits]
            ))
        } else {
            recordDiagnostic(DiagnosticEvent(
                code: .recoveryFailed,
                message: "Recovery failed",
                severity: .critical
            ))
        }
    }

    func recordComputeUnitChange(_ label: String) {
        let previous = activeComputeLabel
        activeComputeLabel = label
        if previous != label {
            recordDiagnostic(DiagnosticEvent(
                code: .computeUnitDegraded,
                message: "Compute degraded: \(previous) → \(label)",
                severity: .warning,
                metadata: ["from": previous, "to": label]
            ))
        }
    }

    func recordThrottleEvent(thermalState: String, penalty: Double) {
        recordDiagnostic(DiagnosticEvent(
            code: .inferenceThrottled,
            message: "Inference throttled at \(thermalState)",
            severity: .warning,
            metadata: ["thermalState": thermalState, "penalty": String(format: "%.1f", penalty)]
        ))
    }

    var recentCriticalEvents: [DiagnosticEvent] {
        diagnosticEvents.filter { $0.severity == .critical }.suffix(5).reversed()
    }

    var uptimeFormatted: String {
        let interval = Date().timeIntervalSince(sessionUptime)
        let minutes = Int(interval) / 60
        let seconds = Int(interval) % 60
        return String(format: "%dm %02ds", minutes, seconds)
    }

    func clearDiagnostics() {
        diagnosticEvents.removeAll()
        lastDiagnosticCode = nil
    }
}
