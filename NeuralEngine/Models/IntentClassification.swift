import Foundation

nonisolated enum IntentType: String, Sendable {
    case questionFactual = "question_factual"
    case questionOpinion = "question_opinion"
    case questionHow = "question_how"
    case questionWhy = "question_why"
    case questionComparison = "question_comparison"
    case requestAction = "request_action"
    case requestCreation = "request_creation"
    case requestAnalysis = "request_analysis"
    case requestSearch = "request_search"
    case requestMemory = "request_memory"
    case requestCalculation = "request_calculation"
    case statementOpinion = "statement_opinion"
    case statementEmotion = "statement_emotion"
    case statementInstruction = "statement_instruction"
    case statementFact = "statement_fact"
    case socialGreeting = "social_greeting"
    case socialFarewell = "social_farewell"
    case socialGratitude = "social_gratitude"
    case socialApology = "social_apology"
    case metaCorrection = "meta_correction"
    case metaClarification = "meta_clarification"
    case metaFeedback = "meta_feedback"
    case explorationBrainstorm = "exploration_brainstorm"
    case explorationDebate = "exploration_debate"
    case explorationHypothetical = "exploration_hypothetical"
}

nonisolated enum ResponseLength: String, Sendable {
    case brief
    case moderate
    case detailed
    case comprehensive
}

nonisolated struct IntentClassification: Sendable {
    let primary: IntentType
    let secondary: IntentType?
    let confidence: Double
    let requiresAction: Bool
    let requiresKnowledge: Bool
    let requiresCreativity: Bool
    let isMultiIntent: Bool
    let subIntents: [IntentType]
    let urgency: Double
    let expectedResponseLength: ResponseLength
}

nonisolated enum EmotionalValence: String, Sendable {
    case positive
    case negative
    case neutral
    case mixed
}

nonisolated enum EmotionalArousal: String, Sendable {
    case high
    case medium
    case low
}

nonisolated struct EmotionalState: Sendable {
    let valence: EmotionalValence
    let arousal: EmotionalArousal
    let dominantEmotion: String
    let style: String
    let empathyLevel: Double
    let emotionalTrajectory: String
}
