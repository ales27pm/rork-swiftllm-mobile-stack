import Foundation

nonisolated enum ForensicDiagnosticCode: Int, Sendable {
    case coreMLPlanFailure = -4
    case coreMLEviction = 11
    case posixENOMEM = 12
    case speechNoInput = 1110
    case speechServiceBusy = 209
    case speechRecognizerNotLoaded = 203
}

nonisolated struct ForensicDiagnostic: Sendable, Identifiable {
    let id: UUID
    let code: ForensicDiagnosticCode
    let domain: String
    let phenomenon: String
    let recoveryAction: ForensicRecoveryAction
    let timestamp: Date
    let sourceError: String?

    init(
        code: ForensicDiagnosticCode,
        domain: String,
        phenomenon: String,
        recoveryAction: ForensicRecoveryAction,
        sourceError: Error? = nil
    ) {
        self.id = UUID()
        self.code = code
        self.domain = domain
        self.phenomenon = phenomenon
        self.recoveryAction = recoveryAction
        self.timestamp = Date()
        self.sourceError = sourceError?.localizedDescription
    }
}

nonisolated enum ForensicRecoveryAction: String, Sendable {
    case switchToCPU = "Switch to CPU mode"
    case reduceContextWindow = "Reduce Context Window"
    case reloadModel = "Reload Model"
    case silentRetry = "Silent Retry"
    case linearBackoff = "Linear Backoff"
    case triggerManualReload = "Trigger Manual Reload"
}

nonisolated struct BackoffPolicy: Sendable {
    let baseDelaySeconds: Double
    let maxRetries: Int
    let multiplier: Double

    func delay(forAttempt attempt: Int) -> Double {
        min(baseDelaySeconds * pow(multiplier, Double(attempt)), 30.0)
    }

    static let linear = BackoffPolicy(baseDelaySeconds: 0.5, maxRetries: 5, multiplier: 1.0)
    static let exponential = BackoffPolicy(baseDelaySeconds: 0.5, maxRetries: 5, multiplier: 2.0)
    static let aggressive = BackoffPolicy(baseDelaySeconds: 1.0, maxRetries: 3, multiplier: 3.0)
}

enum ForensicValidator {
    private static let diagnosticTable: [Int: (String, String, ForensicRecoveryAction, BackoffPolicy)] = [
        -4: ("CoreML", "Neural Engine Plan Failure", .switchToCPU, .exponential),
        12: ("POSIX", "ENOMEM (Out of Memory)", .reduceContextWindow, .aggressive),
        11: ("CoreML", "Mid-inference Eviction", .reloadModel, .exponential),
        1110: ("Speech", "No Speech Detected", .silentRetry, .linear),
        209: ("Speech", "Recognition Service Busy", .linearBackoff, .linear),
        203: ("Speech", "Recognizer Not Loaded", .triggerManualReload, .aggressive),
    ]

    static func diagnose(_ error: Error) -> ForensicDiagnostic? {
        let nsError = error as NSError
        let code = nsError.code

        guard let entry = diagnosticTable[code] else { return nil }

        guard let forensicCode = ForensicDiagnosticCode(rawValue: code) else { return nil }

        return ForensicDiagnostic(
            code: forensicCode,
            domain: entry.0,
            phenomenon: entry.1,
            recoveryAction: entry.2,
            sourceError: error
        )
    }

    static func backoffPolicy(for code: ForensicDiagnosticCode) -> BackoffPolicy {
        let raw = code.rawValue
        return diagnosticTable[raw]?.3 ?? .linear
    }

    static func bridgeToDiagnosticEvent(_ diagnostic: ForensicDiagnostic) -> DiagnosticEvent {
        let code: DiagnosticCode
        let severity: DiagnosticSeverity

        switch diagnostic.code {
        case .coreMLPlanFailure:
            code = .computeUnitDegraded
            severity = .critical
        case .coreMLEviction:
            code = .modelEvicted
            severity = .critical
        case .posixENOMEM:
            code = .memoryPressure
            severity = .critical
        case .speechNoInput:
            code = .healthCheckFailed
            severity = .info
        case .speechServiceBusy:
            code = .inferenceThrottled
            severity = .warning
        case .speechRecognizerNotLoaded:
            code = .recoveryStarted
            severity = .warning
        }

        return DiagnosticEvent(
            code: code,
            message: "[Forensic] \(diagnostic.phenomenon): \(diagnostic.recoveryAction.rawValue)",
            severity: severity,
            metadata: [
                "forensicCode": String(diagnostic.code.rawValue),
                "domain": diagnostic.domain,
                "recovery": diagnostic.recoveryAction.rawValue
            ]
        )
    }

    static func synthesizeAndBridge(_ error: Error) -> (WrappedError, DiagnosticEvent?) {
        let wrapped = NativeErrorWrapper.synthesize(error)
        let forensic = diagnose(error)
        let event = forensic.map { bridgeToDiagnosticEvent($0) }
        return (wrapped, event)
    }

    static func recoveryLabel(for diagnostic: ForensicDiagnostic) -> String {
        let icon: String
        switch diagnostic.recoveryAction {
        case .switchToCPU: icon = "cpu"
        case .reduceContextWindow: icon = "arrow.down.circle"
        case .reloadModel: icon = "arrow.clockwise"
        case .silentRetry: icon = "arrow.triangle.2.circlepath"
        case .linearBackoff: icon = "clock.arrow.circlepath"
        case .triggerManualReload: icon = "exclamationmark.arrow.circlepath"
        }
        return "[\(icon)] \(diagnostic.domain) — \(diagnostic.phenomenon) → \(diagnostic.recoveryAction.rawValue)"
    }
}
