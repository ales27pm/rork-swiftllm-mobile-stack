import Foundation

@MainActor
struct CognitionEngine {
    static func process(
        userText: String,
        conversationHistory: [Message],
        memoryService: MemoryService
    ) -> CognitionFrame {
        let memoryResults = memoryService.searchMemories(query: userText, maxResults: 8)
        let associativeResults = memoryService.getAssociativeMemories(query: userText, directResults: memoryResults)
        let allMemoryResults = memoryResults + associativeResults

        let emotion = EmotionAnalyzer.analyze(text: userText, conversationHistory: conversationHistory)
        let intent = IntentClassifier.classify(text: userText, conversationHistory: conversationHistory)
        let metacognition = MetacognitionEngine.assess(
            text: userText,
            conversationHistory: conversationHistory,
            memoryResults: allMemoryResults
        )
        let curiosity = CuriosityDetector.detect(text: userText, memoryResults: allMemoryResults)
        let thoughtTree = ThoughtTreeBuilder.build(
            text: userText,
            intent: intent,
            metacognition: metacognition,
            memoryResults: allMemoryResults
        )

        var injections: [ContextInjection] = []

        let emotionInjection = EmotionAnalyzer.buildInjection(state: emotion)
        if !emotionInjection.content.isEmpty { injections.append(emotionInjection) }

        let metaInjection = MetacognitionEngine.buildInjection(state: metacognition)
        if !metaInjection.content.isEmpty { injections.append(metaInjection) }

        let thoughtInjection = ThoughtTreeBuilder.buildInjection(tree: thoughtTree)
        if !thoughtInjection.content.isEmpty { injections.append(thoughtInjection) }

        let curiosityInjection = CuriosityDetector.buildInjection(state: curiosity)
        if !curiosityInjection.content.isEmpty { injections.append(curiosityInjection) }

        let intentInjection = IntentClassifier.buildInjection(intent: intent)
        if !intentInjection.content.isEmpty { injections.append(intentInjection) }

        injections.sort { $0.priority > $1.priority }

        return CognitionFrame(
            emotion: emotion,
            metacognition: metacognition,
            thoughtTree: thoughtTree,
            curiosity: curiosity,
            intent: intent,
            injections: injections,
            timestamp: Date()
        )
    }
}
