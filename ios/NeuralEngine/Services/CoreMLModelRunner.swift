import Foundation
import CoreML

nonisolated enum RunnerState: String, Sendable {
    case idle
    case loading
    case ready
    case recovering
    case disposing
    case evicted
}

nonisolated final class CoreMLModelRunner: @unchecked Sendable {
    private var model: MLModel?
    private var modelURL: URL?
    private var activeComputeUnits: MLComputeUnits = .all
    private var state: RunnerState = .idle
    private var mlState: MLState?
    private let lock = NSLock()
    private var inputName: String = "input_ids"
    private var outputName: String = "logits"
    private var usesState: Bool = false

    private var consecutiveFailures: Int = 0
    private var lastSuccessfulPrediction: Date?
    private var totalRecoveries: Int = 0
    private var lastRecoveryAttempt: Date?

    private let maxConsecutiveFailures = 3
    private let recoveryBackoffBase: TimeInterval = 0.5
    private let healthCheckInterval: TimeInterval = 30

    var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return model != nil && (state == .ready || state == .recovering)
    }

    var currentState: RunnerState {
        lock.lock()
        defer { lock.unlock() }
        return state
    }

    var recoveryCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return totalRecoveries
    }

    var failureCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return consecutiveFailures
    }

    func loadModel(at url: URL, computeUnits: MLComputeUnits = .all) async throws {
        lock.lock()
        guard state != .loading else {
            lock.unlock()
            throw CoreMLRunnerError.modelNotLoaded
        }
        state = .loading
        lock.unlock()

        do {
            try await loadWithFallback(at: url, preferredUnits: computeUnits)
        } catch {
            lock.lock()
            state = .idle
            lock.unlock()
            throw error
        }
    }

    private func loadWithFallback(at url: URL, preferredUnits: MLComputeUnits) async throws {
        let fallbackChain: [MLComputeUnits]
        switch preferredUnits {
        case .all:
            fallbackChain = [.all, .cpuAndNeuralEngine, .cpuOnly]
        case .cpuAndNeuralEngine:
            fallbackChain = [.cpuAndNeuralEngine, .cpuOnly]
        case .cpuOnly:
            fallbackChain = [.cpuOnly]
        case .cpuAndGPU:
            fallbackChain = [.cpuAndGPU, .cpuOnly]
        @unknown default:
            fallbackChain = [.all, .cpuOnly]
        }

        var lastError: Error?

        for units in fallbackChain {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = units

                let loadedModel = try await MLModel.load(contentsOf: url, configuration: config)

                let spec = loadedModel.modelDescription
                let detectedInput = spec.inputDescriptionsByName.keys.first {
                    $0.contains("input_id") || $0.contains("token")
                } ?? "input_ids"
                let detectedOutput = spec.outputDescriptionsByName.keys.first {
                    $0.contains("logit") || $0.contains("token_scores")
                } ?? "logits"

                lock.lock()
                model = loadedModel
                modelURL = url
                activeComputeUnits = units
                inputName = detectedInput
                outputName = detectedOutput
                consecutiveFailures = 0
                lastSuccessfulPrediction = Date()

                let modelState = loadedModel.makeState()
                mlState = modelState
                usesState = true
                state = .ready
                lock.unlock()

                return
            } catch {
                let nsError = error as NSError
                if nsError.domain == "com.apple.CoreML" && nsError.code == -4 {
                    lastError = error
                    continue
                }
                lastError = error
                continue
            }
        }

        throw lastError ?? CoreMLRunnerError.modelNotLoaded
    }

    func predictLogits(inputIDs: [Int]) throws -> [Float] {
        lock.lock()
        guard let model else {
            lock.unlock()
            throw CoreMLRunnerError.modelNotLoaded
        }
        guard state == .ready || state == .recovering else {
            let currentState = state
            lock.unlock()
            throw CoreMLRunnerError.invalidState(currentState)
        }
        let currentState = mlState
        let currentUsesState = usesState
        let inName = inputName
        let outName = outputName
        lock.unlock()

        let seqLen = inputIDs.count
        guard seqLen > 0 else { throw CoreMLRunnerError.emptyInput }

        do {
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
            let ptr = inputArray.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)
            for i in 0..<seqLen {
                ptr[i] = Int32(inputIDs[i])
            }

            let featureDict: [String: Any] = [inName: MLFeatureValue(multiArray: inputArray)]
            let input = try MLDictionaryFeatureProvider(dictionary: featureDict)

            let output: MLFeatureProvider
            if currentUsesState, let st = currentState {
                output = try model.prediction(from: input, using: st)
            } else {
                output = try model.prediction(from: input)
            }

            guard let logitsArray = output.featureValue(for: outName)?.multiArrayValue else {
                let availableKeys = output.featureNames.joined(separator: ", ")
                throw CoreMLRunnerError.invalidOutput(availableKeys: availableKeys)
            }

            let shape = logitsArray.shape.map { $0.intValue }
            let vocabSize: Int
            let lastTokenOffset: Int

            if shape.count == 3 {
                vocabSize = shape[2]
                lastTokenOffset = (seqLen - 1) * vocabSize
            } else if shape.count == 2 {
                vocabSize = shape[1]
                lastTokenOffset = 0
            } else {
                vocabSize = shape.last ?? 0
                lastTokenOffset = 0
            }

            guard vocabSize > 0 else { throw CoreMLRunnerError.invalidOutput(availableKeys: "vocabSize=0") }

            var logits = [Float](repeating: 0, count: vocabSize)
            let floatPtr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: logitsArray.count)
            for i in 0..<vocabSize {
                logits[i] = floatPtr[lastTokenOffset + i]
            }

            lock.lock()
            consecutiveFailures = 0
            lastSuccessfulPrediction = Date()
            lock.unlock()

            return logits
        } catch {
            lock.lock()
            consecutiveFailures += 1
            let failures = consecutiveFailures
            lock.unlock()

            if isGhostModelError(error) || failures >= maxConsecutiveFailures {
                lock.lock()
                state = .evicted
                lock.unlock()
                throw CoreMLRunnerError.modelEvicted(underlyingError: error)
            }

            throw error
        }
    }

    func attemptRecovery() async throws {
        lock.lock()
        guard let url = modelURL else {
            lock.unlock()
            throw CoreMLRunnerError.modelNotLoaded
        }
        guard state == .evicted || state == .ready else {
            lock.unlock()
            return
        }

        if let lastAttempt = lastRecoveryAttempt {
            let backoff = recoveryBackoffBase * pow(2.0, Double(min(totalRecoveries, 5)))
            let elapsed = Date().timeIntervalSince(lastAttempt)
            if elapsed < backoff {
                lock.unlock()
                return
            }
        }

        state = .recovering
        lastRecoveryAttempt = Date()
        let units = activeComputeUnits
        lock.unlock()

        let degradedUnits: MLComputeUnits
        switch units {
        case .all: degradedUnits = .cpuAndNeuralEngine
        case .cpuAndNeuralEngine: degradedUnits = .cpuOnly
        default: degradedUnits = .cpuOnly
        }

        do {
            try await loadWithFallback(at: url, preferredUnits: degradedUnits)
            lock.lock()
            totalRecoveries += 1
            consecutiveFailures = 0
            lock.unlock()
        } catch {
            lock.lock()
            state = .evicted
            lock.unlock()
            throw error
        }
    }

    func healthCheck() -> HealthStatus {
        lock.lock()
        defer { lock.unlock() }

        guard model != nil else {
            return HealthStatus(isHealthy: false, state: state, failures: consecutiveFailures, recoveries: totalRecoveries, computeUnits: activeComputeUnits, staleDuration: nil)
        }

        var staleDuration: TimeInterval?
        if let lastSuccess = lastSuccessfulPrediction {
            staleDuration = Date().timeIntervalSince(lastSuccess)
        }

        let isStale = staleDuration.map { $0 > healthCheckInterval } ?? false
        let isHealthy = state == .ready && consecutiveFailures == 0 && !isStale

        return HealthStatus(
            isHealthy: isHealthy,
            state: state,
            failures: consecutiveFailures,
            recoveries: totalRecoveries,
            computeUnits: activeComputeUnits,
            staleDuration: staleDuration
        )
    }

    private func isGhostModelError(_ error: Error) -> Bool {
        let nsError = error as NSError
        if nsError.domain == "com.apple.CoreML" {
            let evictionCodes: Set<Int> = [-1, -2, -4, -6, 11]
            if evictionCodes.contains(nsError.code) { return true }
        }

        let desc = error.localizedDescription.lowercased()
        let ghostIndicators = ["evicted", "resource", "memory", "unavailable", "interrupted", "ane"]
        return ghostIndicators.contains { desc.contains($0) }
    }

    func resetState() {
        lock.lock()
        defer { lock.unlock() }
        guard let model else { return }
        if usesState {
            mlState = model.makeState()
        }
    }

    func unload() {
        lock.lock()
        state = .disposing
        model = nil
        mlState = nil
        usesState = false
        modelURL = nil
        consecutiveFailures = 0
        totalRecoveries = 0
        lastSuccessfulPrediction = nil
        lastRecoveryAttempt = nil
        state = .idle
        lock.unlock()
    }

    var vocabSizeEstimate: Int {
        lock.lock()
        defer { lock.unlock() }
        guard let model else { return 32000 }
        if let outDesc = model.modelDescription.outputDescriptionsByName[outputName],
           let constraint = outDesc.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            return shape.last ?? 32000
        }
        return 32000
    }
}

nonisolated struct HealthStatus: Sendable {
    let isHealthy: Bool
    let state: RunnerState
    let failures: Int
    let recoveries: Int
    let computeUnits: MLComputeUnits
    let staleDuration: TimeInterval?

    var diagnosticSummary: String {
        let unitsLabel: String
        switch computeUnits {
        case .all: unitsLabel = "All"
        case .cpuAndNeuralEngine: unitsLabel = "CPU+ANE"
        case .cpuOnly: unitsLabel = "CPU"
        case .cpuAndGPU: unitsLabel = "CPU+GPU"
        @unknown default: unitsLabel = "Unknown"
        }
        return "[\(state.rawValue)] units=\(unitsLabel) failures=\(failures) recoveries=\(recoveries)"
    }
}

nonisolated enum CoreMLRunnerError: Error, Sendable, LocalizedError {
    case modelNotLoaded
    case emptyInput
    case invalidOutput(availableKeys: String)
    case compilationFailed(String)
    case modelEvicted(underlyingError: Error)
    case invalidState(RunnerState)
    case recoveryFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No CoreML model loaded"
        case .emptyInput: return "Empty input token sequence"
        case .invalidOutput(let keys): return "Invalid model output. Available: \(keys)"
        case .compilationFailed(let reason): return "Model compilation failed: \(reason)"
        case .modelEvicted: return "Model was evicted from hardware. Attempting recovery..."
        case .invalidState(let state): return "Runner in invalid state: \(state.rawValue)"
        case .recoveryFailed(let reason): return "Recovery failed: \(reason)"
        }
    }
}
