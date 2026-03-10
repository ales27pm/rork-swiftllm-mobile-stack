import Foundation

@Observable
@MainActor
class ThermalGovernor {
    var currentMode: RuntimeMode = .maxPerformance
    var thermalState: ProcessInfo.ThermalState = .nominal
    var isMonitoring: Bool = false

    private var monitorTask: Task<Void, Never>?

    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true

        monitorTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let state = ProcessInfo.processInfo.thermalState
                self.thermalState = state
                self.currentMode = self.chooseMode(thermalState: state)
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func stopMonitoring() {
        monitorTask?.cancel()
        monitorTask = nil
        isMonitoring = false
    }

    private func chooseMode(thermalState: ProcessInfo.ThermalState) -> RuntimeMode {
        switch thermalState {
        case .nominal: return .maxPerformance
        case .fair: return .balanced
        case .serious: return .coolDown
        case .critical: return .emergency
        @unknown default: return .balanced
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
}
