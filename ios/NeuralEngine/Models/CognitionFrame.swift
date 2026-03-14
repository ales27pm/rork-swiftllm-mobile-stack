import Foundation

nonisolated struct CognitionFrame: Sendable {
    let emotion: EmotionalState
    let metacognition: MetacognitionState
    let thoughtTree: ThoughtTree
    let curiosity: CuriosityState
    let intent: IntentClassification
    let injections: [ContextInjection]
    let timestamp: Date
}

nonisolated struct MetacognitionState: Sendable {
    let complexityLevel: ComplexityLevel
    let uncertaintyLevel: Double
    let cognitiveLoad: CognitiveLoad
    let confidenceCalibration: Double
    let shouldDecompose: Bool
    let shouldSeekClarification: Bool
    let shouldSearchWeb: Bool
    let isTimeSensitive: Bool
    let ambiguityDetected: Bool
    let ambiguityReasons: [String]
    let knowledgeLimitHit: Bool
}

nonisolated enum ComplexityLevel: String, Sendable {
    case simple
    case moderate
    case complex
    case expert
}

nonisolated enum CognitiveLoad: String, Sendable {
    case low
    case medium
    case high
    case overload
}

nonisolated struct ThoughtTree: Sendable {
    let branches: [ThoughtBranch]
    let bestPath: [ThoughtBranch]
    let convergencePercent: Double
}

nonisolated struct ThoughtBranch: Sendable {
    let id: String
    let hypothesis: String
    let confidence: Double
    let evidence: [String]
    let counterpoints: [String]
    let isPruned: Bool
    let strategy: String
}

nonisolated struct CuriosityState: Sendable {
    let detectedTopics: [String]
    let knowledgeGap: Double
    let explorationPriority: Double
    let suggestedQueries: [String]
}

nonisolated enum InjectionType: String, Sendable {
    case emotion
    case thoughtTree
    case curiosity
    case metacognition
    case intent
    case priming
    case voiceMode
    case conversationSummary
}

nonisolated struct ContextInjection: Sendable {
    let type: InjectionType
    let content: String
    let priority: Double
    let estimatedTokens: Int
}
