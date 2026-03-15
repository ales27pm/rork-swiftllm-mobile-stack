import Foundation

nonisolated struct CognitionFrame: Sendable {
    let emotion: EmotionalState
    let metacognition: MetacognitionState
    let thoughtTree: ThoughtTree
    let curiosity: CuriosityState
    let intent: IntentClassification
    let injections: [ContextInjection]
    let reasoningTrace: ReasoningTrace
    let contextSignature: ContextSignature
    let timestamp: Date
}

nonisolated struct ContextSignature: Sendable {
    let intentVector: [Double]
    let topicFingerprint: [String: Double]
    let emotionalBaseline: Double
    let complexityAnchor: Double
    let signatureHash: String
}

nonisolated struct SemanticDriftResult: Sendable {
    let driftMagnitude: Double
    let driftedDimensions: [String]
    let correctionPrompt: String?
    let shouldInterrupt: Bool
}

nonisolated struct MetacognitionState: Sendable {
    let complexityLevel: ComplexityLevel
    let uncertaintyLevel: Double
    let cognitiveLoad: CognitiveLoad
    let confidenceCalibration: Double
    let convergenceScore: Double
    let shouldDecompose: Bool
    let shouldSeekClarification: Bool
    let shouldSearchWeb: Bool
    let isTimeSensitive: Bool
    let ambiguityDetected: Bool
    let ambiguityReasons: [String]
    let knowledgeLimitHit: Bool
    let selfCorrectionFlags: [SelfCorrectionFlag]
    let entropyAnalysis: EntropyAnalysis
    let ambiguityCluster: AmbiguityCluster
    let probabilityMass: ProbabilityMassResult
}

nonisolated struct SelfCorrectionFlag: Sendable {
    let domain: String
    let issue: String
    let severity: Double
}

nonisolated struct EntropyAnalysis: Sendable {
    let shannonEntropy: Double
    let semanticDensity: Double
    let tokenConceptRatio: Double
    let shouldEscalate: Bool
    let entropyPercentile: Double
}

nonisolated struct AmbiguityCluster: Sendable {
    let primaryCluster: String
    let primaryScore: Double
    let secondaryCluster: String
    let secondaryScore: Double
    let clusterDelta: Double
    let isAmbiguous: Bool
    let competingInterpretations: [String]
}

nonisolated struct ProbabilityMassResult: Sendable {
    let topCandidateMass: Double
    let remainderMass: Double
    let massRatio: Double
    let needsVerification: Bool
    let confidenceBand: ConfidenceBand
}

nonisolated enum ConfidenceBand: String, Sendable {
    case high
    case moderate
    case low
    case veryLow
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
    let prunedBranches: [ThoughtBranch]
    let convergencePercent: Double
    let iterationCount: Int
    let synthesisStrategy: SynthesisStrategy
    let maxDepthReached: Int
    let dfsExpansions: Int
    let terminalNodes: [ThoughtBranch]
}

nonisolated enum SynthesisStrategy: String, Sendable {
    case direct
    case decompose
    case multiPerspective
    case creativeExploration
    case clarifyFirst
    case hedgedResponse
}

nonisolated struct ThoughtBranch: Sendable {
    let id: String
    let hypothesis: String
    let confidence: Double
    let evidence: [String]
    let counterpoints: [String]
    let isPruned: Bool
    let pruneReason: String?
    let strategy: String
    let supportScore: Double
    let contradictionScore: Double
    let depth: Int
    let parentId: String?
    let isTerminal: Bool
}

nonisolated struct CuriosityState: Sendable {
    let detectedTopics: [String]
    let knowledgeGap: Double
    let explorationPriority: Double
    let suggestedQueries: [String]
    let valenceArousalCuriosity: Double
    let informationGapIntensity: Double
}

nonisolated struct ReasoningTrace: Sendable {
    let iterations: [ReasoningIteration]
    let finalConvergence: Double
    let dominantStrategy: SynthesisStrategy
    let totalPruned: Int
    let selfCorrections: [String]
}

nonisolated struct ReasoningIteration: Sendable {
    let index: Int
    let convergence: Double
    let activeBranches: Int
    let prunedThisRound: Int
    let insight: String
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
    case selfCorrection
    case reasoningTrace
}

nonisolated struct ContextInjection: Sendable {
    let type: InjectionType
    let content: String
    let priority: Double
    let estimatedTokens: Int
}

nonisolated enum InjectionSubtype: String, Sendable {
    case entropyEscalation
    case ambiguityResolution
    case driftCorrection
    case verificationPrompt
    case depthExpansion
}
