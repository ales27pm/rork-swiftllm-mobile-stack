import Foundation
import CoreML

nonisolated final class CoreMLModelRunner: @unchecked Sendable {
    private var model: MLModel?
    private var state: MLState?
    private let lock = NSLock()
    private var inputName: String = "input_ids"
    private var outputName: String = "logits"
    private var usesState: Bool = false

    var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return model != nil
    }

    func loadModel(at url: URL, computeUnits: MLComputeUnits = .all) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let loadedModel = try await MLModel.load(contentsOf: url, configuration: config)

        let spec = loadedModel.modelDescription
        let detectedInput = spec.inputDescriptionsByName.keys.first { $0.contains("input_id") || $0.contains("token") } ?? "input_ids"
        let detectedOutput = spec.outputDescriptionsByName.keys.first { $0.contains("logit") || $0.contains("token_scores") } ?? "logits"

        lock.lock()
        model = loadedModel
        inputName = detectedInput
        outputName = detectedOutput

        let modelState = loadedModel.makeState()
        state = modelState
        usesState = true

        lock.unlock()
    }

    func predictLogits(inputIDs: [Int]) throws -> [Float] {
        lock.lock()
        guard let model else {
            lock.unlock()
            throw CoreMLRunnerError.modelNotLoaded
        }
        let currentState = state
        let currentUsesState = usesState
        let inName = inputName
        let outName = outputName
        lock.unlock()

        let seqLen = inputIDs.count
        guard seqLen > 0 else { throw CoreMLRunnerError.emptyInput }

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

        return logits
    }

    func resetState() {
        lock.lock()
        defer { lock.unlock() }
        guard let model else { return }
        if usesState {
            state = model.makeState()
        }
    }

    func unload() {
        lock.lock()
        model = nil
        state = nil
        usesState = false
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

nonisolated enum CoreMLRunnerError: Error, Sendable, LocalizedError {
    case modelNotLoaded
    case emptyInput
    case invalidOutput(availableKeys: String)
    case compilationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No CoreML model loaded"
        case .emptyInput: return "Empty input token sequence"
        case .invalidOutput(let keys): return "Invalid model output. Available: \(keys)"
        case .compilationFailed(let reason): return "Model compilation failed: \(reason)"
        }
    }
}
