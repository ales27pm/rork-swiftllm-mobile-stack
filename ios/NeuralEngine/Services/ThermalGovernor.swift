import Foundation

@Observable
@MainActor
class ThermalGovernor {
    var currentMode: RuntimeMode = .maxPerformance
    var thermalState: ProcessInfo.ThermalState = .nominal
    var memoryPressureLevel: MemoryPressureLevel = .normal
    var isMonitoring: Bool = false

    private var monitorTask: Task<Void, Never>?
    private var memorySource: DispatchSourceMemoryPressure?
    private var onCriticalMemory: (() -> Void)?

    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true

        startThermalMonitor()
        startMemoryPressureMonitor()
    }

    func stopMonitoring() {
        monitorTask?.cancel()
        monitorTask = nil
        memorySource?.cancel()
        memorySource = nil
        isMonitoring = false
    }

    func setMemoryPressureHandler(_ handler: @escaping () -> Void) {
        onCriticalMemory = handler
    }

    private func startThermalMonitor() {
        monitorTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let state = ProcessInfo.processInfo.thermalState
                self.thermalState = state
                self.currentMode = self.chooseMode(thermalState: state, memoryPressure: self.memoryPressureLevel)
                try? await Task.sleep(for: .seconds(2))
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
                if event.contains(.critical) {
                    self.memoryPressureLevel = .critical
                    self.currentMode = self.chooseMode(thermalState: self.thermalState, memoryPressure: .critical)
                    self.onCriticalMemory?()
                } else if event.contains(.warning) {
                    self.memoryPressureLevel = .warning
                    self.currentMode = self.chooseMode(thermalState: self.thermalState, memoryPressure: .warning)
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
}

nonisolated enum MemoryPressureLevel: String, Sendable {
    case normal = "Normal"
    case warning = "Warning"
    case critical = "Critical"
}
