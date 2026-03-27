import Foundation

nonisolated enum DiagnosticTestStatus: String, Sendable {
    case pending
    case running
    case passed
    case failed
    case skipped
    case warning
}

nonisolated enum DiagnosticCategory: String, Sendable, CaseIterable {
    case database = "Database"
    case fileSystem = "File System"
    case keyValueStore = "Key-Value Store"
    case secureStore = "Secure Store"
    case memory = "Memory Service"
    case conversation = "Conversation Service"
    case cognition = "Cognition Pipeline"
    case nlp = "NLP Processing"
    case emotion = "Emotion Analysis"
    case intent = "Intent Classification"
    case metacognition = "Metacognition"
    case thoughtTree = "Thought Tree"
    case curiosity = "Curiosity Detection"
    case contextAssembly = "Context Assembly"
    case thermal = "Thermal Governor"
    case metrics = "Metrics Logger"
    case tokenizer = "Tokenizer"
    case modelLoader = "Model Loader"
    case speech = "Speech Pipeline"
    case inference = "Inference Engine"
    case proceduralSolvers = "Procedural Solvers"
    case endToEnd = "End-to-End"
    case emotionAccuracy = "Emotion Accuracy"
    case intentAccuracy = "Intent Accuracy"
    case memoryQuality = "Memory Quality"
    case cognitionQuality = "Cognition Quality"
    case contextQuality = "Context Quality"
    case stressTest = "Stress Test"
    case inferenceDeep = "Inference Deep"
    case regressionE2E = "Regression E2E"
    case llmDiagnostic = "LLM Diagnostic"
    case vectorDatabase = "Vector Database"
    case configOptimization = "Config Optimization"
}

struct DiagnosticTestResult: Identifiable, Sendable {
    let id: UUID = UUID()
    let name: String
    let category: DiagnosticCategory
    var status: DiagnosticTestStatus
    var duration: TimeInterval
    var message: String
    var details: [String]
    let startedAt: Date

    init(name: String, category: DiagnosticCategory, status: DiagnosticTestStatus = .pending, duration: TimeInterval = 0, message: String = "", details: [String] = []) {
        self.name = name
        self.category = category
        self.status = status
        self.duration = duration
        self.message = message
        self.details = details
        self.startedAt = Date()
    }

    var statusIcon: String {
        switch status {
        case .pending: return "circle"
        case .running: return "circle.dotted"
        case .passed: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        case .skipped: return "forward.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        }
    }
}

struct DiagnosticReport: Sendable {
    let runId: String
    let deviceInfo: DeviceInfo
    let results: [DiagnosticTestResult]
    let startedAt: Date
    let completedAt: Date
    let totalTests: Int
    let passedTests: Int
    let failedTests: Int
    let warningTests: Int
    let skippedTests: Int

    var totalDuration: TimeInterval { completedAt.timeIntervalSince(startedAt) }
    var passRate: Double { totalTests > 0 ? Double(passedTests) / Double(totalTests) * 100 : 0 }
}

nonisolated struct DeviceInfo: Sendable {
    let modelName: String
    let systemVersion: String
    let processorCount: Int
    let physicalMemoryGB: Double
    let availableDiskGB: Double
    let totalDiskGB: Double
    let thermalState: String
    let batteryLevel: Float
    let batteryState: String
    let locale: String
    let timezone: String
    let appVersion: String
    let buildNumber: String
}
