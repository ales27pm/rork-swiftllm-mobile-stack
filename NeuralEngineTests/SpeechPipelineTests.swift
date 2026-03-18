import Foundation
import Testing
@testable import NeuralEngine

struct SpeechPipelineTests {
    @Test func transcriptStabilizer_requiresConfidenceRepeatAndSilenceBeforePreview() {
        var stabilizer = TranscriptStabilizer()
        let t0 = Date()
        let first = makeUpdate(text: "open maps to san", stablePrefix: stabilizer.register(text: "open maps to san", at: t0), confidence: 0.92, emittedAt: t0)
        #expect(!stabilizer.shouldEmitPreview(update: first, now: t0.addingTimeInterval(0.7)))

        let t1 = t0.addingTimeInterval(0.1)
        let second = makeUpdate(text: "open maps to san", stablePrefix: stabilizer.register(text: "open maps to san", at: t1), confidence: 0.92, emittedAt: t1)
        #expect(stabilizer.shouldEmitPreview(update: second, now: t1.addingTimeInterval(0.7)))

        let lowConfidence = makeUpdate(text: "open maps to san", stablePrefix: second.stablePrefix, confidence: 0.2, emittedAt: t1)
        #expect(!stabilizer.shouldEmitPreview(update: lowConfidence, now: t1.addingTimeInterval(0.7)))
    }

    @Test func contextAssembler_prefersSelectedLanguageButFallsBackToDetectedRecognitionLocale() {
        #expect(ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: "fr-CA", detectedRecognitionLanguageCode: "en-US") == "fr-CA")
        #expect(ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: nil, detectedRecognitionLanguageCode: "es_MX") == "es-MX")
    }

    @Test func assembleSystemPrompt_carriesVoiceLanguageSynchronizationInstructions() {
        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: stubFrame(),
            memoryResults: [],
            conversationHistory: [Message(role: .user, content: "bonjour")],
            toolsEnabled: false,
            isVoiceMode: true,
            preferredResponseLanguageCode: "fr-CA",
            detectedRecognitionLanguageCode: "fr-FR"
        )

        #expect(prompt.contains("[Response Language]"))
        #expect(prompt.contains("Respond in fr-CA"))
        #expect(prompt.contains("Recognition input language detected as fr-FR"))
    }

    @Test func synthesisService_syncDetectedLanguageUpdatesEffectiveSpeechLanguage() {
        let service = SpeechSynthesisService()
        service.setLanguagePreferred(nil)

        let effective = service.syncDetectedLanguage("de-DE")

        #expect(effective == service.effectiveSpeechLanguageCode())
        #expect(service.effectiveSpeechLanguageCode() == "de-de" || service.effectiveSpeechLanguageCode() == "de-DE")
    }

    private func makeUpdate(text: String, stablePrefix: String, confidence: Float, emittedAt: Date) -> TranscriptUpdate {
        TranscriptUpdate(
            text: text,
            segments: [TranscriptSegment(substring: text, timestamp: 0, duration: 1, confidence: confidence)],
            isFinal: false,
            languageCode: "en-US",
            averageConfidence: confidence,
            stablePrefix: stablePrefix,
            emittedAt: emittedAt
        )
    }
}


private extension SpeechPipelineTests {
    static func stubFrame() -> CognitionFrame {
        CognitionFrame(
            emotion: EmotionalState(
                valence: .neutral,
                arousal: .low,
                dominantEmotion: "neutral",
                style: "balanced",
                empathyLevel: 0.5,
                emotionalTrajectory: "stable"
            ),
            metacognition: MetacognitionState(
                complexityLevel: .simple,
                uncertaintyLevel: 0.1,
                cognitiveLoad: .low,
                confidenceCalibration: 0.9,
                convergenceScore: 0.95,
                shouldDecompose: false,
                shouldSeekClarification: false,
                shouldSearchWeb: false,
                isTimeSensitive: false,
                ambiguityDetected: false,
                ambiguityReasons: [],
                knowledgeLimitHit: false,
                selfCorrectionFlags: [],
                entropyAnalysis: EntropyAnalysis(
                    shannonEntropy: 0.1,
                    semanticDensity: 0.1,
                    tokenConceptRatio: 1.0,
                    shouldEscalate: false,
                    entropyPercentile: 0.1
                ),
                ambiguityCluster: AmbiguityCluster(
                    primaryCluster: "none",
                    primaryScore: 1.0,
                    secondaryCluster: "none",
                    secondaryScore: 0.0,
                    clusterDelta: 1.0,
                    isAmbiguous: false,
                    competingInterpretations: []
                ),
                probabilityMass: ProbabilityMassResult(
                    topCandidateMass: 0.9,
                    remainderMass: 0.1,
                    massRatio: 9,
                    needsVerification: false,
                    confidenceBand: .high
                )
            ),
            thoughtTree: ThoughtTree(
                branches: [],
                bestPath: [],
                prunedBranches: [],
                convergencePercent: 0.95,
                iterationCount: 1,
                synthesisStrategy: .direct,
                maxDepthReached: 0,
                dfsExpansions: 0,
                terminalNodes: []
            ),
            curiosity: CuriosityState(
                detectedTopics: [],
                knowledgeGap: 0,
                explorationPriority: 0,
                suggestedQueries: [],
                valenceArousalCuriosity: 0,
                informationGapIntensity: 0
            ),
            intent: IntentClassification(
                primary: .statementFact,
                secondary: nil,
                confidence: 0.95,
                requiresAction: false,
                requiresKnowledge: false,
                requiresCreativity: false,
                isMultiIntent: false,
                subIntents: [],
                urgency: 0,
                expectedResponseLength: .brief
            ),
            injections: [],
            reasoningTrace: ReasoningTrace(
                iterations: [ReasoningIteration(index: 1, convergence: 0.95, activeBranches: 1, prunedThisRound: 0, insight: "")],
                finalConvergence: 0.95,
                dominantStrategy: .direct,
                totalPruned: 0,
                selfCorrections: []
            ),
            contextSignature: ContextSignature(
                intentVector: [],
                topicFingerprint: [:],
                emotionalBaseline: 0,
                complexityAnchor: 0,
                signatureHash: "speech-test"
            ),
            timestamp: Date()
        )
    }
}
