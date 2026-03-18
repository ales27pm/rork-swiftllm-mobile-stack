import Foundation
import Testing
@testable import NeuralEngine

struct NaturalLanguagePipelineTests {

    @Test
    func intentClassifier_normalizesInflectedIntentWording() {
        let result = IntentClassifier.classify(
            text: "Could you calculate the totals after computing each item?",
            conversationHistory: []
        )

        #expect(result.primary == .requestCalculation)
        #expect(result.confidence > 0.3)
    }

    @Test @MainActor
    func memoryService_retrievesSemanticallySimilarMemoryWithDifferentPhrasing() {
        let database = DatabaseService(name: "memory-semantic-\(UUID().uuidString).sqlite3")
        defer { _ = database.deleteDatabase() }

        let service = MemoryService(database: database)
        service.clearAllMemories()
        service.addMemory(MemoryEntry(
            content: "We discussed booking flights and reserving a hotel for the Tokyo vacation.",
            keywords: ["travel", "tokyo", "flight", "hotel"],
            category: .context,
            importance: 5,
            activationLevel: 0.2
        ))
        service.addMemory(MemoryEntry(
            content: "Your preferred coffee beans are washed Ethiopian roasts.",
            keywords: ["coffee", "ethiopian", "roast"],
            category: .preference,
            importance: 3
        ))

        let results = service.searchMemories(
            query: "Find the note about organizing airfare and lodging for our trip to Japan.",
            maxResults: 3,
            minScore: 0.01,
            languageHint: "en"
        )

        #expect(results.first?.memory.content.contains("Tokyo vacation") == true)
    }

    @Test @MainActor
    func documentAnalysisLanguageHint_guidesMixedLanguageOCRDownstreamProcessing() {
        let service = DocumentAnalysisService()
        let result = DocumentAnalysisResult(
            fullText: "Factura del hotel in Berlin. Necesito booking details und Rechnung, por favor.",
            blocks: [TextBlock(text: "Factura del hotel in Berlin", confidence: 0.96, boundingBox: .zero, normalizedBox: .zero)],
            pageCount: 1,
            languageHint: "es"
        )

        let context = service.preprocessForDownstreamNLP(result)
        let intent = IntentClassifier.classify(
            text: context.fullText,
            conversationHistory: [],
            languageHint: context.languageHint
        )
        let emotion = EmotionAnalyzer.analyze(
            text: "Estoy frustrated y necesito booking details urgent ahora mismo.",
            conversationHistory: [],
            languageHint: context.languageHint
        )

        #expect(context.languageHint == "es")
        #expect(!context.sentenceSegments.isEmpty)
        #expect(context.normalizedTerms.contains(where: { $0.contains("factura") || $0.contains("hotel") }))
        #expect(intent.primary == .requestMemory || intent.primary == .requestAction || intent.primary == .questionFactual)
        #expect(emotion.valence == .negative || emotion.style == "urgent")
    }
}
