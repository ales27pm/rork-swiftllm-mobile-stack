import Foundation
import UIKit
import CoreML

nonisolated enum RunnerState: String, Sendable {
    case idle
    case loading
    case ready
    case recovering
    case disposing
    case evicted
}

nonisolated protocol LogitsPredicting: Sendable {
    func predictLogits(inputIDs: [Int]) throws -> [Float]
    func predictLogitsSpan(inputIDs: [Int]) throws -> [[Float]]
}

nonisolated final class CoreMLModelRunner: LogitsPredicting, @unchecked Sendable {
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

    private var stateSerializationURL: URL?
    private var backgroundObserver: NSObjectProtocol?
    private var foregroundObserver: NSObjectProtocol?
    private var inactivityTimer: Timer?
    private var inactivityTimeoutSeconds: TimeInterval = 120
    private var isBackgrounded: Bool = false
    private var backgroundTimestamp: Date?

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

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
        let didEnterLoadingState = withLock { () -> Bool in
            guard state != .loading && state != .disposing else {
                return false
            }
            state = .loading
            return true
        }

        guard didEnterLoadingState else {
            let currentState = withLock { state }
            if currentState == .disposing {
                throw CoreMLRunnerError.invalidState(.disposing)
            }
            throw CoreMLRunnerError.modelNotLoaded
        }

        do {
            try await loadWithFallback(at: url, preferredUnits: computeUnits)
            setupLifecycleObservers()
            setupStateSerializationPath()
        } catch {
            withLock { state = .idle }
            throw error
        }
    }

    func loadModelWithTimeout(at url: URL, computeUnits: MLComputeUnits = .all, timeoutSeconds: TimeInterval = 120) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await self.loadModel(at: url, computeUnits: computeUnits)
            }

            group.addTask {
                var elapsed: TimeInterval = 0
                let checkInterval: TimeInterval = 0.5
                while elapsed < timeoutSeconds {
                    try await Task.sleep(for: .seconds(checkInterval))
                    if self.isBackgrounded {
                        elapsed -= checkInterval
                    }
                    elapsed += checkInterval
                }
                throw CoreMLRunnerError.loadTimeout(timeoutSeconds)
            }

            if let result = try await group.next() {
                group.cancelAll()
                return result
            }
            group.cancelAll()
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

                withLock {
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
                }

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
            let input = try makeInputProvider(inputIDs: inputIDs, inputName: inName)

            let output: MLFeatureProvider
            let updatedState: MLState?
            if currentUsesState, let st = currentState {
                output = try model.prediction(from: input, using: st)
                updatedState = st
            } else {
                output = try model.prediction(from: input)
                updatedState = nil
            }

            guard let logitsArray = output.featureValue(for: outName)?.multiArrayValue else {
                let availableKeys = output.featureNames.joined(separator: ", ")
                throw CoreMLRunnerError.invalidOutput(availableKeys: availableKeys)
            }

            let logits = try extractLastTokenLogits(from: logitsArray, requestedSeqLen: seqLen)
            recordPredictionSuccess(updatedState: updatedState)
            return logits
        } catch {
            try recordPredictionFailure(error)
        }
    }

    func predictLogitsSpan(inputIDs: [Int]) throws -> [[Float]] {
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
            let input = try makeInputProvider(inputIDs: inputIDs, inputName: inName)

            let output: MLFeatureProvider
            let updatedState: MLState?
            if currentUsesState, let st = currentState {
                output = try model.prediction(from: input, using: st)
                updatedState = st
            } else {
                output = try model.prediction(from: input)
                updatedState = nil
            }

            guard let logitsArray = output.featureValue(for: outName)?.multiArrayValue else {
                let availableKeys = output.featureNames.joined(separator: ", ")
                throw CoreMLRunnerError.invalidOutput(availableKeys: availableKeys)
            }

            let spanLogits = try extractSpanLogits(from: logitsArray, requestedSeqLen: seqLen)
            recordPredictionSuccess(updatedState: updatedState)
            return spanLogits
        } catch {
            try recordPredictionFailure(error)
        }
    }

    private func makeInputProvider(inputIDs: [Int], inputName: String) throws -> MLFeatureProvider {
        let seqLen = inputIDs.count
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        let ptr = inputArray.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)
        for i in 0..<seqLen {
            ptr[i] = Int32(inputIDs[i])
        }

        let featureDict: [String: Any] = [inputName: MLFeatureValue(multiArray: inputArray)]
        return try MLDictionaryFeatureProvider(dictionary: featureDict)
    }

    private func extractLastTokenLogits(from logitsArray: MLMultiArray, requestedSeqLen: Int) throws -> [Float] {
        let shape = logitsArray.shape.map { $0.intValue }
        let strides = logitsArray.strides.map { $0.intValue }
        let floatPtr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: logitsArray.count)

        switch shape.count {
        case 3:
            guard shape[0] > 0, shape[1] > 0, shape[2] > 0 else {
                throw CoreMLRunnerError.invalidOutput(availableKeys: "invalid 3D logits shape")
            }
            let targetPosition = min(max(requestedSeqLen - 1, 0), shape[1] - 1)
            let vocabSize = shape[2]
            var logits = [Float](repeating: 0, count: vocabSize)
            for vocabIndex in 0..<vocabSize {
                let offset = 0 * strides[0] + targetPosition * strides[1] + vocabIndex * strides[2]
                logits[vocabIndex] = floatPtr[offset]
            }
            return logits
        case 2:
            guard shape[0] > 0, shape[1] > 0 else {
                throw CoreMLRunnerError.invalidOutput(availableKeys: "invalid 2D logits shape")
            }
            let vocabSize = shape[1]
            var logits = [Float](repeating: 0, count: vocabSize)
            for vocabIndex in 0..<vocabSize {
                let offset = 0 * strides[0] + vocabIndex * strides[1]
                logits[vocabIndex] = floatPtr[offset]
            }
            return logits
        case 1:
            guard shape[0] > 0 else {
                throw CoreMLRunnerError.invalidOutput(availableKeys: "invalid 1D logits shape")
            }
            var logits = [Float](repeating: 0, count: shape[0])
            for vocabIndex in 0..<shape[0] {
                logits[vocabIndex] = floatPtr[vocabIndex * strides[0]]
            }
            return logits
        default:
            throw CoreMLRunnerError.invalidOutput(availableKeys: "unsupported logits shape: \(shape)")
        }
    }

    private func extractSpanLogits(from logitsArray: MLMultiArray, requestedSeqLen: Int) throws -> [[Float]] {
        let shape = logitsArray.shape.map { $0.intValue }
        let strides = logitsArray.strides.map { $0.intValue }
        let floatPtr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: logitsArray.count)

        switch shape.count {
        case 3:
            guard shape[0] > 0, shape[1] > 0, shape[2] > 0 else {
                throw CoreMLRunnerError.invalidOutput(availableKeys: "invalid 3D logits shape")
            }
            let outputSeqLen = min(requestedSeqLen, shape[1])
            let vocabSize = shape[2]
            var result: [[Float]] = []
            result.reserveCapacity(outputSeqLen)

            for position in 0..<outputSeqLen {
                var row = [Float](repeating: 0, count: vocabSize)
                for vocabIndex in 0..<vocabSize {
                    let offset = 0 * strides[0] + position * strides[1] + vocabIndex * strides[2]
                    row[vocabIndex] = floatPtr[offset]
                }
                result.append(row)
            }
            return result
        default:
            let single = try extractLastTokenLogits(from: logitsArray, requestedSeqLen: requestedSeqLen)
            return Array(repeating: single, count: requestedSeqLen)
        }
    }

    private func recordPredictionSuccess(updatedState: MLState?) {
        lock.lock()
        consecutiveFailures = 0
        lastSuccessfulPrediction = Date()
        if let updatedState {
            mlState = updatedState
        }
        lock.unlock()
    }

    private func recordPredictionFailure(_ error: Error) throws -> Never {
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

    func attemptRecovery() async throws {
        let recoveryContext: (url: URL, units: MLComputeUnits)? = withLock {
            guard let url = modelURL else {
                return nil
            }
            guard state == .evicted || state == .ready else {
                return nil
            }

            if let lastAttempt = lastRecoveryAttempt {
                let backoff = recoveryBackoffBase * pow(2.0, Double(min(totalRecoveries, 5)))
                let elapsed = Date().timeIntervalSince(lastAttempt)
                if elapsed < backoff {
                    return nil
                }
            }

            state = .recovering
            lastRecoveryAttempt = Date()
            return (url: url, units: activeComputeUnits)
        }

        guard let recoveryContext else {
            if withLock({ modelURL == nil }) {
                throw CoreMLRunnerError.modelNotLoaded
            }
            return
        }

        let degradedUnits: MLComputeUnits
        switch recoveryContext.units {
        case .all: degradedUnits = .cpuAndNeuralEngine
        case .cpuAndNeuralEngine: degradedUnits = .cpuOnly
        default: degradedUnits = .cpuOnly
        }

        do {
            try await loadWithFallback(at: recoveryContext.url, preferredUnits: degradedUnits)
            withLock {
                totalRecoveries += 1
                consecutiveFailures = 0
            }
        } catch {
            withLock { state = .evicted }
            throw error
        }
    }

    func runZeroTokenProbe() -> ZeroTokenProbeResult {
        zeroTokenProbe()
    }

    func zeroTokenProbe() -> ZeroTokenProbeResult {
        lock.lock()
        guard let model, state == .ready || state == .recovering else {
            let currentState = state
            lock.unlock()
            return ZeroTokenProbeResult(passed: false, state: currentState, latencyMS: 0)
        }
        let inName = inputName
        let outName = outputName
        let currentState = mlState
        let currentUsesState = usesState
        lock.unlock()

        let probeStart = Date()

        do {
            let inputArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            let ptr = inputArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)
            ptr[0] = 1

            let featureDict: [String: Any] = [inName: MLFeatureValue(multiArray: inputArray)]
            let input = try MLDictionaryFeatureProvider(dictionary: featureDict)

            if currentUsesState, let st = currentState {
                let _ = try model.prediction(from: input, using: st)
            } else {
                let _ = try model.prediction(from: input)
            }

            let latency = Date().timeIntervalSince(probeStart) * 1000

            lock.lock()
            lastSuccessfulPrediction = Date()
            lock.unlock()

            return ZeroTokenProbeResult(passed: true, state: .ready, latencyMS: latency)
        } catch {
            let latency = Date().timeIntervalSince(probeStart) * 1000

            if isGhostModelError(error) {
                lock.lock()
                state = .evicted
                consecutiveFailures += 1
                lock.unlock()
                return ZeroTokenProbeResult(passed: false, state: .evicted, latencyMS: latency)
            }

            lock.lock()
            consecutiveFailures += 1
            lock.unlock()
            return ZeroTokenProbeResult(passed: false, state: .ready, latencyMS: latency)
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

    func serializeStateToStorage() {
        lock.lock()
        defer { lock.unlock() }
        guard let url = stateSerializationURL else { return }
        guard state == .ready, usesState else { return }

        let metadata = StateSerializationMetadata(
            computeUnits: activeComputeUnits,
            inputName: inputName,
            outputName: outputName,
            timestamp: Date()
        )

        if let data = try? JSONEncoder().encode(metadata) {
            try? data.write(to: url)
        }
    }

    func restoreStateFromStorage() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let url = stateSerializationURL else { return false }

        guard let data = try? Data(contentsOf: url),
              let metadata = try? JSONDecoder().decode(StateSerializationMetadata.self, from: data) else {
            return false
        }

        let staleness = Date().timeIntervalSince(metadata.timestamp)
        guard staleness < 300 else {
            try? FileManager.default.removeItem(at: url)
            return false
        }

        return true
    }

    func switchToCPUOnly() async throws {
        lock.lock()
        guard let url = modelURL else {
            lock.unlock()
            throw CoreMLRunnerError.modelNotLoaded
        }
        state = .recovering
        lock.unlock()

        do {
            try await loadWithFallback(at: url, preferredUnits: .cpuOnly)
            lock.lock()
            consecutiveFailures = 0
            totalRecoveries += 1
            lock.unlock()
        } catch {
            lock.lock()
            state = .evicted
            lock.unlock()
            throw error
        }
    }

    func unload() {
        lock.lock()
        state = .disposing
        lock.unlock()

        removeLifecycleObservers()

        lock.lock()
        model = nil
        mlState = nil
        usesState = false
        modelURL = nil
        consecutiveFailures = 0
        totalRecoveries = 0
        lastSuccessfulPrediction = nil
        lastRecoveryAttempt = nil
        stateSerializationURL = nil
        state = .idle
        lock.unlock()
    }

    private func setupLifecycleObservers() {
        removeLifecycleObservers()

        backgroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleEnterBackground()
        }

        foregroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleEnterForeground()
        }
    }

    private func removeLifecycleObservers() {
        if let observer = backgroundObserver {
            NotificationCenter.default.removeObserver(observer)
            backgroundObserver = nil
        }
        if let observer = foregroundObserver {
            NotificationCenter.default.removeObserver(observer)
            foregroundObserver = nil
        }
    }

    private func handleEnterBackground() {
        lock.lock()
        isBackgrounded = true
        backgroundTimestamp = Date()
        lock.unlock()

        serializeStateToStorage()
    }

    private func handleEnterForeground() {
        lock.lock()
        isBackgrounded = false
        let bgTime = backgroundTimestamp
        backgroundTimestamp = nil
        lock.unlock()

        if let bgTime {
            let duration = Date().timeIntervalSince(bgTime)
            if duration > 60 {
                lock.lock()
                if state == .ready {
                    state = .evicted
                }
                lock.unlock()
            }
        }
    }

    private func setupStateSerializationPath() {
        let supportDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        stateSerializationURL = supportDir?.appendingPathComponent("mlstate_metadata.json")
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
    case loadTimeout(TimeInterval)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No CoreML model loaded"
        case .emptyInput: return "Empty input token sequence"
        case .invalidOutput(let keys): return "Invalid model output. Available: \(keys)"
        case .compilationFailed(let reason): return "Model compilation failed: \(reason)"
        case .modelEvicted: return "Model was evicted from hardware. Attempting recovery..."
        case .invalidState(let state): return "Runner in invalid state: \(state.rawValue)"
        case .recoveryFailed(let reason): return "Recovery failed: \(reason)"
        case .loadTimeout(let seconds): return "Model load timed out after \(Int(seconds))s"
        }
    }
}

nonisolated struct ZeroTokenProbeResult: Sendable {
    let passed: Bool
    let state: RunnerState
    let latencyMS: Double
}

nonisolated struct StateSerializationMetadata: Codable, Sendable {
    let computeUnitsRaw: Int
    let inputName: String
    let outputName: String
    let timestamp: Date

    var computeUnits: MLComputeUnits {
        MLComputeUnits(rawValue: computeUnitsRaw) ?? .all
    }

    init(computeUnits: MLComputeUnits, inputName: String, outputName: String, timestamp: Date) {
        self.computeUnitsRaw = computeUnits.rawValue
        self.inputName = inputName
        self.outputName = outputName
        self.timestamp = timestamp
    }
}
