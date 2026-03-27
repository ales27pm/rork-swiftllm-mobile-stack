import Foundation

extension Notification.Name {
    static let forceInferenceStop = Notification.Name("com.neuralengine.forceInferenceStop")
    static let thermalStateEscalated = Notification.Name("com.neuralengine.thermalStateEscalated")
    static let memoryPressureEscalated = Notification.Name("com.neuralengine.memoryPressureEscalated")
}

@Observable
@MainActor
class ThermalGovernor {
    var currentMode: RuntimeMode = .maxPerformance
    var thermalState: ProcessInfo.ThermalState = .nominal
    var memoryPressureLevel: MemoryPressureLevel = .normal
    var isMonitoring: Bool = false
    var currentPenalty: Double = 0.0
    var inferenceThrottled: Bool = false
    var totalThrottleEvents: Int = 0
    var lastEscalationDate: Date?
    var peakThermalState: ProcessInfo.ThermalState = .nominal
    var continuousSeriousStart: Date?

    private var monitorTask: Task<Void, Never>?
    private var memorySource: DispatchSourceMemoryPressure?
    private var onCriticalMemory: (() -> Void)?
    var metricsLogger: MetricsLogger?
    private var thermalObserver: NSObjectProtocol?
    private var previousThermalState: ProcessInfo.ThermalState = .nominal
    private var lastThermalSampleDate: Date = .now

    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true

        registerThermalNotification()
        startMemoryPressureMonitor()
        startPeriodicSampler()
    }

    func stopMonitoring() {
        monitorTask?.cancel()
        monitorTask = nil
        memorySource?.cancel()
        memorySource = nil
        if let observer = thermalObserver {
            NotificationCenter.default.removeObserver(observer)
            thermalObserver = nil
        }
        isMonitoring = false
    }

    func setMemoryPressureHandler(_ handler: @escaping () -> Void) {
        onCriticalMemory = handler
    }

    var tokenDelaySeconds: Double {
        switch thermalState {
        case .nominal: return 0
        case .fair: return 0.02
        case .serious: return 0.02
        case .critical: return 0
        @unknown default: return 0
        }
    }

    var adaptiveTemperatureBoost: Float {
        switch thermalState {
        case .nominal: return 0.0
        case .fair: return 0.05
        case .serious: return 0.15
        case .critical: return 0.0
        @unknown default: return 0.0
        }
    }

    var shouldRunZeroTokenProbe: Bool {
        memoryPressureLevel != .normal || thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue || lastDiagnosticCode == .modelEvicted
    }

    private(set) var lastDiagnosticCode: DiagnosticCode?

    var shouldSuspendInference: Bool {
        currentPenalty >= 1.0 || thermalState == .critical || memoryPressureLevel == .critical
    }

    var shouldShowCoolingAdvisory: Bool {
        guard thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue else { return false }
        guard let continuousSeriousStart else { return false }
        return lastThermalSampleDate.timeIntervalSince(continuousSeriousStart) >= 90
    }

    var diagnosticSummary: String {
        let stateLabel: String
        switch thermalState {
        case .nominal: stateLabel = "Nominal"
        case .fair: stateLabel = "Fair"
        case .serious: stateLabel = "Serious"
        case .critical: stateLabel = "Critical"
        @unknown default: stateLabel = "Unknown"
        }
        return "thermal=\(stateLabel) penalty=\(String(format: "%.1f", currentPenalty)) mode=\(currentMode.rawValue) mem=\(memoryPressureLevel.rawValue) throttles=\(totalThrottleEvents)"
    }

    private func registerThermalNotification() {
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleThermalStateChange()
            }
        }
        handleThermalStateChange()
    }

    private func handleThermalStateChange() {
        let sampleDate = Date()
        let newState = ProcessInfo.processInfo.thermalState
        let oldState = previousThermalState
        previousThermalState = newState
        thermalState = newState
        lastThermalSampleDate = sampleDate

        if newState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
            continuousSeriousStart = continuousSeriousStart ?? sampleDate
        } else {
            continuousSeriousStart = nil
        }

        if newState.rawValue > peakThermalState.rawValue {
            peakThermalState = newState
        }

        updatePenalty()
        currentMode = chooseMode(thermalState: newState, memoryPressure: memoryPressureLevel)

        if newState.rawValue > oldState.rawValue {
            lastEscalationDate = sampleDate
            NotificationCenter.default.post(name: .thermalStateEscalated, object: nil, userInfo: [
                "previousState": oldState.rawValue,
                "newState": newState.rawValue,
                "penalty": currentPenalty
            ])
        }

        if newState == .critical {
            inferenceThrottled = true
            if oldState != .critical {
                totalThrottleEvents += 1
                metricsLogger?.recordThrottleEvent(thermalState: "Critical", penalty: currentPenalty)
                NotificationCenter.default.post(name: .forceInferenceStop, object: nil)
            }
        } else if newState == .serious {
            if oldState != .serious {
                metricsLogger?.recordThrottleEvent(thermalState: "Serious", penalty: currentPenalty)
            }
        } else if newState == .nominal || newState == .fair {
            inferenceThrottled = false
        }
    }

    func recordDiagnosticCode(_ code: DiagnosticCode) {
        lastDiagnosticCode = code
    }

    func clearEvictionFlag() {
        if lastDiagnosticCode == .modelEvicted {
            lastDiagnosticCode = nil
        }
    }

    private func updatePenalty() {
        switch thermalState {
        case .nominal:
            currentPenalty = 0.0
        case .fair:
            currentPenalty = 0.2
        case .serious:
            currentPenalty = 0.5
        case .critical:
            currentPenalty = 1.0
        @unknown default:
            currentPenalty = 0.0
        }
    }

    private func startPeriodicSampler() {
        monitorTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.handleThermalStateChange()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    private func startMemoryPressureMonitor() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let event = source.data
            Task { @MainActor [weak self] in
                guard let self else { return }
                let previousLevel = self.memoryPressureLevel
                if event.contains(.critical) {
                    self.memoryPressureLevel = .critical
                    self.currentMode = self.chooseMode(thermalState: self.thermalState, memoryPressure: .critical)
                    self.onCriticalMemory?()
                    self.totalThrottleEvents += 1
                    self.metricsLogger?.recordDiagnostic(DiagnosticEvent(
                        code: .memoryPressure,
                        message: "Critical memory pressure detected",
                        severity: .critical,
                        metadata: ["level": "critical"]
                    ))
                    NotificationCenter.default.post(name: .forceInferenceStop, object: nil)
                    NotificationCenter.default.post(name: .memoryPressureEscalated, object: nil, userInfo: [
                        "level": "critical",
                        "previousLevel": previousLevel.rawValue
                    ])
                } else if event.contains(.warning) {
                    self.memoryPressureLevel = .warning
                    self.currentMode = self.chooseMode(thermalState: self.thermalState, memoryPressure: .warning)
                    if previousLevel == .normal {
                        NotificationCenter.default.post(name: .memoryPressureEscalated, object: nil, userInfo: [
                            "level": "warning",
                            "previousLevel": previousLevel.rawValue
                        ])
                    }
                }
            }
        }
        source.resume()
        memorySource = source
    }

    private func chooseMode(thermalState: ProcessInfo.ThermalState, memoryPressure: MemoryPressureLevel) -> RuntimeMode {
        if memoryPressure == .critical { return .emergency }

        switch thermalState {
        case .nominal:
            return memoryPressure == .warning ? .balanced : .maxPerformance
        case .fair:
            return .balanced
        case .serious:
            return .coolDown
        case .critical:
            return .emergency
        @unknown default:
            return .balanced
        }
    }

    var thermalLevel: ThermalLevel {
        switch thermalState {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .fair
        }
    }

    var currentMemoryUsageMB: Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1_048_576
    }

    var availableMemoryMB: Double {
        let totalPhysical = Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576
        return max(0, totalPhysical - currentMemoryUsageMB)
    }

    func resetPeakTracking() {
        peakThermalState = thermalState
        totalThrottleEvents = 0
        lastEscalationDate = nil
    }

    private var recoveryBackoffAttempts: Int = 0
    private var lastRecoveryProbeDate: Date?
    private let recoveryBackoffBase: TimeInterval = 0.5
    private let maxRecoveryAttempts: Int = 5
    private(set) var isRecoveryInProgress: Bool = false

    var waitDeltaMultiplier: Double {
        switch thermalState {
        case .nominal: return 1.0
        case .fair: return 0.8
        case .serious: return 0.5
        case .critical: return 0.0
        @unknown default: return 1.0
        }
    }

    func shouldAttemptRecovery() -> Bool {
        guard !isRecoveryInProgress else { return false }
        guard memoryPressureLevel != .critical && thermalState != .critical else { return false }
        guard recoveryBackoffAttempts < maxRecoveryAttempts else { return false }

        if let lastProbe = lastRecoveryProbeDate {
            let backoff = recoveryBackoffBase * pow(2.0, Double(min(recoveryBackoffAttempts, 5)))
            guard Date().timeIntervalSince(lastProbe) >= backoff else { return false }
        }

        return true
    }

    func markRecoveryStarted() {
        isRecoveryInProgress = true
        lastRecoveryProbeDate = Date()
        recoveryBackoffAttempts += 1
        metricsLogger?.recordDiagnostic(DiagnosticEvent(
            code: .recoveryStarted,
            message: "Backoff recovery probe #\(recoveryBackoffAttempts)",
            severity: .warning,
            metadata: ["attempt": "\(recoveryBackoffAttempts)"]
        ))
    }

    func markRecoveryCompleted(success: Bool) {
        isRecoveryInProgress = false
        if success {
            recoveryBackoffAttempts = 0
            lastDiagnosticCode = nil
            metricsLogger?.recordDiagnostic(DiagnosticEvent(
                code: .recoveryCompleted,
                message: "Model recovered after backoff probe",
                severity: .info
            ))
        } else {
            metricsLogger?.recordDiagnostic(DiagnosticEvent(
                code: .recoveryFailed,
                message: "Backoff recovery probe failed (attempt \(recoveryBackoffAttempts)/\(maxRecoveryAttempts))",
                severity: .critical,
                metadata: ["attempt": "\(recoveryBackoffAttempts)"]
            ))
        }
    }

    func resetRecoveryState() {
        recoveryBackoffAttempts = 0
        lastRecoveryProbeDate = nil
        isRecoveryInProgress = false
    }

    func waitForCooldown(maxWaitSeconds: Double = 30, targetBelow: ProcessInfo.ThermalState = .serious) async -> Bool {
        let deadline = Date().addingTimeInterval(maxWaitSeconds)
        while Date() < deadline {
            handleThermalStateChange()
            if thermalState.rawValue < targetBelow.rawValue {
                return true
            }
            try? await Task.sleep(for: .seconds(2))
        }
        handleThermalStateChange()
        return thermalState.rawValue < targetBelow.rawValue
    }

    var cooldownDelaySeconds: Double {
        switch thermalState {
        case .nominal: return 0
        case .fair: return 0.5
        case .serious: return 3.0
        case .critical: return 5.0
        @unknown default: return 1.0
        }
    }

    var interCategoryCooldownSeconds: Double {
        switch thermalState {
        case .nominal: return 0
        case .fair: return 1.0
        case .serious: return 5.0
        case .critical: return 10.0
        @unknown default: return 2.0
        }
    }
}

nonisolated enum MemoryPressureLevel: String, Sendable {
    case normal = "Normal"
    case warning = "Warning"
    case critical = "Critical"
}
