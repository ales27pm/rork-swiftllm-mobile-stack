import Foundation
import Testing
@testable import NeuralEngine

struct ToolingIntegrationTests {
    @Test func toolPrompt_listsEveryRegisteredToolWithDelimitedEntries() {
        let prompt = ToolExecutor.buildToolsPrompt()
        let listedTools = Set(
            prompt
                .split(separator: "\n")
                .compactMap { line -> String? in
                    let text = String(line)
                    guard text.hasPrefix("- "), let colonIndex = text.firstIndex(of: ":") else {
                        return nil
                    }

                    return String(text[text.index(text.startIndex, offsetBy: 2)..<colonIndex])
                }
        )

        #expect(listedTools == Set(DeviceToolName.allCases.map(\.rawValue)))
    }

    @Test func parser_rejectsUnknownToolCalls() {
        let payload = #"<tool_call>{\"name\":\"not_a_real_tool\",\"parameters\":{}}</tool_call>"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.isEmpty)
    }

    @Test func parser_acceptsKnownToolWithArgumentsAlias() {
        let payload = #"<tool_call>{\"name\":\"open_maps\",\"arguments\":{\"latitude\":46.006164,\"longitude\":-73.1645294}}</tool_call>"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.count == 1)
        #expect(calls[0].name == DeviceToolName.openMaps.rawValue)
        #expect((calls[0].parameters["latitude"] as? Double) == 46.006164)
        #expect((calls[0].parameters["longitude"] as? Double) == -73.1645294)
    }

    @Test func assembleSystemPrompt_includesRecentValidCoordinateOverride() {
        let history = [
            Message(role: .user, content: "Please use this: (46.0061640, -73.1645294)"),
            Message(role: .assistant, content: "Thanks")
        ]

        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: Self.stubFrame(),
            memoryResults: [],
            conversationHistory: history,
            toolsEnabled: false,
            isVoiceMode: false
        )

        #expect(prompt.contains("[User-Provided Location]"))
        #expect(prompt.contains("46.0061640"))
        #expect(prompt.contains("-73.1645294"))
    }

    @Test func assembleSystemPrompt_skipsInvalidCoordinateOverride() {
        let history = [
            Message(role: .user, content: "My location is (146.0061640, -273.1645294)")
        ]

        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: Self.stubFrame(),
            memoryResults: [],
            conversationHistory: history,
            toolsEnabled: false,
            isVoiceMode: false
        )

        #expect(!prompt.contains("[User-Provided Location]"))
        #expect(!prompt.contains("146.0061640"))
    }

    @Test func assembleSystemPrompt_ignoresStaleCoordinateOutsideRecentWindow() {
        let history = [
            Message(role: .user, content: "Use this old point: 46.0061640, -73.1645294"),
            Message(role: .assistant, content: "Acknowledged"),
            Message(role: .user, content: "Let's discuss architecture"),
            Message(role: .assistant, content: "Sure"),
            Message(role: .user, content: "No location update yet"),
            Message(role: .assistant, content: "Okay"),
            Message(role: .user, content: "Still no coordinates"),
            Message(role: .assistant, content: "Understood")
        ]

        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: Self.stubFrame(),
            memoryResults: [],
            conversationHistory: history,
            toolsEnabled: false,
            isVoiceMode: false
        )

        #expect(!prompt.contains("[User-Provided Location]"))
        #expect(!prompt.contains("46.0061640"))
    }

    @Test func assembleSystemPrompt_handlesLabeledCoordinates() {
        let history = [
            Message(role: .user, content: "latitude: 46.006164 longitude: -73.1645294")
        ]

        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: Self.stubFrame(),
            memoryResults: [],
            conversationHistory: history,
            toolsEnabled: false,
            isVoiceMode: false
        )

        #expect(prompt.contains("[User-Provided Location]"))
        #expect(prompt.contains("46.006164"))
        #expect(prompt.contains("-73.1645294"))
    }

    @Test func assembleSystemPrompt_includesLocationSafetyRequirements() {
        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: Self.stubFrame(),
            memoryResults: [],
            conversationHistory: [Message(role: .user, content: "Where am I right now?")],
            toolsEnabled: true,
            isVoiceMode: false
        )

        #expect(prompt.contains("Location response safety:"))
        #expect(prompt.contains("call get_location before answering"))
        #expect(prompt.contains("Never invent coordinates, timestamps, addresses"))
        #expect(prompt.contains("latitude, longitude, collectedAt, source"))
    }

    @Test func assembleSystemPrompt_includesUtilityConcisenessGuidance() {
        let prompt = ContextAssembler.assembleSystemPrompt(
            frame: Self.stubFrame(),
            memoryResults: [],
            conversationHistory: [Message(role: .user, content: "What's my battery level?")],
            toolsEnabled: true,
            isVoiceMode: false
        )

        #expect(prompt.contains("Utility response style guide:"))
        #expect(prompt.contains("keep responses concise (1-3 short sentences)"))
        #expect(prompt.contains("Avoid speculative internal narratives"))
    }
}

private extension ToolingIntegrationTests {
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
                signatureHash: "test"
            ),
            timestamp: Date()
        )
    }
}
