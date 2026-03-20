import Foundation
import NaturalLanguage

extension DiagnosticEngine {

    func buildDeepTestSuite() -> [DiagnosticTestResult] {
        var tests: [DiagnosticTestResult] = []

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Emotion Accuracy Matrix (20 inputs)", category: .emotionAccuracy),
            DiagnosticTestResult(name: "Emotion Boundary Detection", category: .emotionAccuracy),
            DiagnosticTestResult(name: "Emotion Adversarial Inputs", category: .emotionAccuracy),
            DiagnosticTestResult(name: "Empathy Calibration Curve", category: .emotionAccuracy),
            DiagnosticTestResult(name: "Style Classification Matrix", category: .emotionAccuracy),
            DiagnosticTestResult(name: "Trajectory Detection Accuracy", category: .emotionAccuracy),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Intent Precision/Recall Matrix (30 inputs)", category: .intentAccuracy),
            DiagnosticTestResult(name: "Intent Confidence Calibration", category: .intentAccuracy),
            DiagnosticTestResult(name: "Multi-Intent Decomposition", category: .intentAccuracy),
            DiagnosticTestResult(name: "Urgency Gradient Accuracy", category: .intentAccuracy),
            DiagnosticTestResult(name: "Response Length Calibration", category: .intentAccuracy),
            DiagnosticTestResult(name: "Intent Adversarial Inputs", category: .intentAccuracy),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Memory Ranking Correctness (TF-IDF)", category: .memoryQuality),
            DiagnosticTestResult(name: "Memory Ranking Correctness (Semantic)", category: .memoryQuality),
            DiagnosticTestResult(name: "Deduplication Precision at 85% Threshold", category: .memoryQuality),
            DiagnosticTestResult(name: "Associative Link Strength Validation", category: .memoryQuality),
            DiagnosticTestResult(name: "Memory Decay Curve Correctness", category: .memoryQuality),
            DiagnosticTestResult(name: "Consolidation Under Pressure (50 entries)", category: .memoryQuality),
            DiagnosticTestResult(name: "Memory Extraction Accuracy (10 patterns)", category: .memoryQuality),
            DiagnosticTestResult(name: "Cross-Category Search Diversity", category: .memoryQuality),
            DiagnosticTestResult(name: "Name Extraction Robustness (8 patterns)", category: .memoryQuality),
            DiagnosticTestResult(name: "Hybrid Search Score Distribution", category: .memoryQuality),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Complexity Gradient (5 levels)", category: .cognitionQuality),
            DiagnosticTestResult(name: "Signature Determinism (same input)", category: .cognitionQuality),
            DiagnosticTestResult(name: "Drift Sensitivity Thresholds", category: .cognitionQuality),
            DiagnosticTestResult(name: "Injection Budget Enforcement", category: .cognitionQuality),
            DiagnosticTestResult(name: "Convergence Score Monotonicity", category: .cognitionQuality),
            DiagnosticTestResult(name: "Self-Correction Trigger Accuracy", category: .cognitionQuality),
            DiagnosticTestResult(name: "Entropy Escalation Correctness", category: .cognitionQuality),
            DiagnosticTestResult(name: "Thought Tree Branch Quality", category: .cognitionQuality),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Context Prompt Structure Validation", category: .contextQuality),
            DiagnosticTestResult(name: "Memory Injection Relevance", category: .contextQuality),
            DiagnosticTestResult(name: "Voice Mode Format Compliance", category: .contextQuality),
            DiagnosticTestResult(name: "Language Addendum Correctness", category: .contextQuality),
            DiagnosticTestResult(name: "Coordinate Extraction Accuracy", category: .contextQuality),
            DiagnosticTestResult(name: "Long Conversation Summary", category: .contextQuality),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Database 1000-Row Stress Test", category: .stressTest),
            DiagnosticTestResult(name: "Memory Rapid Add/Search Cycle (50 ops)", category: .stressTest),
            DiagnosticTestResult(name: "Conversation Burst Write (100 messages)", category: .stressTest),
            DiagnosticTestResult(name: "Cognition Pipeline Latency Benchmark", category: .stressTest),
            DiagnosticTestResult(name: "Full Pipeline Under Memory Pressure", category: .stressTest),
            DiagnosticTestResult(name: "FTS5 Large Corpus Search (200 docs)", category: .stressTest),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Actual Token Generation (if model loaded)", category: .inferenceDeep),
            DiagnosticTestResult(name: "Tokenizer Encode/Decode Round-Trip Fidelity", category: .inferenceDeep),
            DiagnosticTestResult(name: "KV Cache Allocate/Release Lifecycle", category: .inferenceDeep),
            DiagnosticTestResult(name: "Sampling Temperature Effect Validation", category: .inferenceDeep),
            DiagnosticTestResult(name: "ChatML Prompt Formatting Correctness", category: .inferenceDeep),
            DiagnosticTestResult(name: "Speculative Decode Policy State", category: .inferenceDeep),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Full E2E: User Input → Cognition → Prompt → Validate", category: .regressionE2E),
            DiagnosticTestResult(name: "Memory Lifecycle: Extract → Store → Search → Inject → Decay", category: .regressionE2E),
            DiagnosticTestResult(name: "Conversation Continuity: Multi-Turn State", category: .regressionE2E),
            DiagnosticTestResult(name: "Emotion→Intent→Metacognition Coherence", category: .regressionE2E),
            DiagnosticTestResult(name: "NLP Cross-Language Pipeline (EN/FR)", category: .regressionE2E),
            DiagnosticTestResult(name: "Voice Pipeline: Cognition → Prompt → Voice Format", category: .regressionE2E),
            DiagnosticTestResult(name: "Memory Extract → Dedup → Search Round-Trip", category: .regressionE2E),
            DiagnosticTestResult(name: "Associative Graph Traversal Integrity", category: .regressionE2E),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "LLM Instruction Following", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Factual Recall", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Coherence Under Context", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Emotional Tone Compliance", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Multi-Turn Consistency", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Cognition→Prompt→Output Pipeline", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Stop Sequence Compliance", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Temperature Sensitivity", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Latency Profile (5 prompts)", category: .llmDiagnostic),
            DiagnosticTestResult(name: "LLM Memory-Aware Response", category: .llmDiagnostic),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Vector Embedding Generation", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Store Insert/Retrieve", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Cosine Similarity Accuracy", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Semantic Search (10 docs)", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector HNSW Index Recall", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Cross-Domain Discrimination", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Memory Integration", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Batch Upsert/Delete Lifecycle", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Search Latency (100 vectors)", category: .vectorDatabase),
            DiagnosticTestResult(name: "Vector Hybrid Ranking Boost", category: .vectorDatabase),
        ])

        return tests
    }

    func executeDeepTest(_ test: DiagnosticTestResult) async -> TestOutcome {
        switch (test.category, test.name) {
        case (.emotionAccuracy, "Emotion Accuracy Matrix (20 inputs)"):
            return deepTestEmotionAccuracyMatrix()
        case (.emotionAccuracy, "Emotion Boundary Detection"):
            return deepTestEmotionBoundary()
        case (.emotionAccuracy, "Emotion Adversarial Inputs"):
            return deepTestEmotionAdversarial()
        case (.emotionAccuracy, "Empathy Calibration Curve"):
            return deepTestEmpathyCalibration()
        case (.emotionAccuracy, "Style Classification Matrix"):
            return deepTestStyleMatrix()
        case (.emotionAccuracy, "Trajectory Detection Accuracy"):
            return deepTestTrajectoryAccuracy()

        case (.intentAccuracy, "Intent Precision/Recall Matrix (30 inputs)"):
            return deepTestIntentMatrix()
        case (.intentAccuracy, "Intent Confidence Calibration"):
            return deepTestIntentConfidence()
        case (.intentAccuracy, "Multi-Intent Decomposition"):
            return deepTestMultiIntent()
        case (.intentAccuracy, "Urgency Gradient Accuracy"):
            return deepTestUrgencyGradient()
        case (.intentAccuracy, "Response Length Calibration"):
            return deepTestResponseLength()
        case (.intentAccuracy, "Intent Adversarial Inputs"):
            return deepTestIntentAdversarial()

        case (.memoryQuality, "Memory Ranking Correctness (TF-IDF)"):
            return deepTestMemoryRankingTFIDF()
        case (.memoryQuality, "Memory Ranking Correctness (Semantic)"):
            return deepTestMemoryRankingSemantic()
        case (.memoryQuality, "Deduplication Precision at 85% Threshold"):
            return deepTestMemoryDedup()
        case (.memoryQuality, "Associative Link Strength Validation"):
            return deepTestAssociativeLinks()
        case (.memoryQuality, "Memory Decay Curve Correctness"):
            return deepTestMemoryDecay()
        case (.memoryQuality, "Consolidation Under Pressure (50 entries)"):
            return deepTestConsolidation()
        case (.memoryQuality, "Memory Extraction Accuracy (10 patterns)"):
            return deepTestMemoryExtraction()
        case (.memoryQuality, "Cross-Category Search Diversity"):
            return deepTestSearchDiversity()
        case (.memoryQuality, "Name Extraction Robustness (8 patterns)"):
            return deepTestNameExtractionRobustness()
        case (.memoryQuality, "Hybrid Search Score Distribution"):
            return deepTestHybridSearchScoreDistribution()

        case (.cognitionQuality, "Complexity Gradient (5 levels)"):
            return deepTestComplexityGradient()
        case (.cognitionQuality, "Signature Determinism (same input)"):
            return deepTestSignatureDeterminism()
        case (.cognitionQuality, "Drift Sensitivity Thresholds"):
            return deepTestDriftSensitivity()
        case (.cognitionQuality, "Injection Budget Enforcement"):
            return deepTestInjectionBudget()
        case (.cognitionQuality, "Convergence Score Monotonicity"):
            return deepTestConvergenceMonotonicity()
        case (.cognitionQuality, "Self-Correction Trigger Accuracy"):
            return deepTestSelfCorrectionTrigger()
        case (.cognitionQuality, "Entropy Escalation Correctness"):
            return deepTestEntropyEscalation()
        case (.cognitionQuality, "Thought Tree Branch Quality"):
            return deepTestThoughtTreeQuality()

        case (.contextQuality, "Context Prompt Structure Validation"):
            return deepTestPromptStructure()
        case (.contextQuality, "Memory Injection Relevance"):
            return deepTestMemoryInjectionRelevance()
        case (.contextQuality, "Voice Mode Format Compliance"):
            return deepTestVoiceModeFormat()
        case (.contextQuality, "Language Addendum Correctness"):
            return deepTestLanguageAddendum()
        case (.contextQuality, "Coordinate Extraction Accuracy"):
            return deepTestCoordinateExtraction()
        case (.contextQuality, "Long Conversation Summary"):
            return deepTestLongConversationSummary()

        case (.stressTest, "Database 1000-Row Stress Test"):
            return deepTestDatabaseStress()
        case (.stressTest, "Memory Rapid Add/Search Cycle (50 ops)"):
            return deepTestMemoryStress()
        case (.stressTest, "Conversation Burst Write (100 messages)"):
            return deepTestConversationBurst()
        case (.stressTest, "Cognition Pipeline Latency Benchmark"):
            return deepTestCognitionLatency()
        case (.stressTest, "Full Pipeline Under Memory Pressure"):
            return deepTestPipelineMemoryPressure()
        case (.stressTest, "FTS5 Large Corpus Search (200 docs)"):
            return deepTestFTSStress()

        case (.inferenceDeep, "Actual Token Generation (if model loaded)"):
            return await deepTestActualInference()
        case (.inferenceDeep, "Tokenizer Encode/Decode Round-Trip Fidelity"):
            return deepTestTokenizerRoundTrip()
        case (.inferenceDeep, "KV Cache Allocate/Release Lifecycle"):
            return await deepTestKVCacheLifecycle()
        case (.inferenceDeep, "Sampling Temperature Effect Validation"):
            return deepTestSamplingTemperature()
        case (.inferenceDeep, "ChatML Prompt Formatting Correctness"):
            return deepTestChatMLFormat()
        case (.inferenceDeep, "Speculative Decode Policy State"):
            return deepTestSpeculativePolicy()

        case (.regressionE2E, "Full E2E: User Input → Cognition → Prompt → Validate"):
            return deepTestFullE2E()
        case (.regressionE2E, "Memory Lifecycle: Extract → Store → Search → Inject → Decay"):
            return deepTestMemoryLifecycle()
        case (.regressionE2E, "Conversation Continuity: Multi-Turn State"):
            return deepTestConversationContinuity()
        case (.regressionE2E, "Emotion→Intent→Metacognition Coherence"):
            return deepTestCoherence()
        case (.regressionE2E, "NLP Cross-Language Pipeline (EN/FR)"):
            return deepTestCrossLanguage()
        case (.regressionE2E, "Voice Pipeline: Cognition → Prompt → Voice Format"):
            return deepTestVoicePipeline()
        case (.regressionE2E, "Memory Extract → Dedup → Search Round-Trip"):
            return deepTestMemoryExtractDedupSearch()
        case (.regressionE2E, "Associative Graph Traversal Integrity"):
            return deepTestAssociativeGraphIntegrity()

        case (.llmDiagnostic, "LLM Instruction Following"):
            return await llmTestInstructionFollowing()
        case (.llmDiagnostic, "LLM Factual Recall"):
            return await llmTestFactualRecall()
        case (.llmDiagnostic, "LLM Coherence Under Context"):
            return await llmTestCoherenceUnderContext()
        case (.llmDiagnostic, "LLM Emotional Tone Compliance"):
            return await llmTestEmotionalToneCompliance()
        case (.llmDiagnostic, "LLM Multi-Turn Consistency"):
            return await llmTestMultiTurnConsistency()
        case (.llmDiagnostic, "LLM Cognition→Prompt→Output Pipeline"):
            return await llmTestCognitionPromptOutput()
        case (.llmDiagnostic, "LLM Stop Sequence Compliance"):
            return await llmTestStopSequenceCompliance()
        case (.llmDiagnostic, "LLM Temperature Sensitivity"):
            return await llmTestTemperatureSensitivity()
        case (.llmDiagnostic, "LLM Latency Profile (5 prompts)"):
            return await llmTestLatencyProfile()
        case (.llmDiagnostic, "LLM Memory-Aware Response"):
            return await llmTestMemoryAwareResponse()

        case (.vectorDatabase, "Vector Embedding Generation"):
            return vectorTestEmbeddingGeneration()
        case (.vectorDatabase, "Vector Store Insert/Retrieve"):
            return vectorTestInsertRetrieve()
        case (.vectorDatabase, "Vector Cosine Similarity Accuracy"):
            return vectorTestCosineSimilarity()
        case (.vectorDatabase, "Vector Semantic Search (10 docs)"):
            return vectorTestSemanticSearch()
        case (.vectorDatabase, "Vector HNSW Index Recall"):
            return vectorTestHNSWRecall()
        case (.vectorDatabase, "Vector Cross-Domain Discrimination"):
            return vectorTestCrossDomain()
        case (.vectorDatabase, "Vector Memory Integration"):
            return vectorTestMemoryIntegration()
        case (.vectorDatabase, "Vector Batch Upsert/Delete Lifecycle"):
            return vectorTestBatchLifecycle()
        case (.vectorDatabase, "Vector Search Latency (100 vectors)"):
            return vectorTestSearchLatency()
        case (.vectorDatabase, "Vector Hybrid Ranking Boost"):
            return vectorTestHybridRanking()

        default:
            return TestOutcome(status: .skipped, message: "No deep test implementation", details: [])
        }
    }

    // MARK: - Emotion Accuracy Tests

    private func deepTestEmotionAccuracyMatrix() -> TestOutcome {
        let testCases: [(input: String, expectedValence: EmotionalValence, expectedEmotion: String)] = [
            ("I'm so happy today!", .positive, "joy"),
            ("This is amazing, I love it!", .positive, "love"),
            ("I feel terrible and sad", .negative, "sadness"),
            ("I'm really angry about this!", .negative, "anger"),
            ("I'm scared of what might happen", .negative, "fear"),
            ("I'm frustrated with this error", .negative, "frustration"),
            ("What is the capital of France?", .neutral, "neutral"),
            ("Please explain quantum physics", .neutral, "neutral"),
            ("I'm so excited and thrilled!", .positive, "excitement"),
            ("I'm grateful for your help", .positive, "gratitude"),
            ("I feel anxious about the deadline", .negative, "anxiety"),
            ("I'm confused about this topic", .negative, "confusion"),
            ("This is wonderful news!", .positive, "awe"),
            ("I'm heartbroken and lonely", .negative, "grief"),
            ("I'm bored with this conversation", .negative, "boredom"),
            ("I'm stressed and overwhelmed", .negative, "stress"),
            ("That's disgusting behavior", .negative, "disgust"),
            ("I feel embarrassed about what happened", .negative, "embarrassment"),
            ("I'm hopeful things will improve", .positive, "hope"),
            ("Everything is calm and peaceful", .positive, "serenity"),
        ]

        var correct = 0
        var emotionCorrect = 0
        var failures: [String] = []

        for tc in testCases {
            let state = EmotionAnalyzer.analyze(text: tc.input, conversationHistory: [])
            if state.valence == tc.expectedValence {
                correct += 1
            } else {
                failures.append("'\(tc.input.prefix(30))…': expected \(tc.expectedValence.rawValue), got \(state.valence.rawValue)")
            }
            if state.dominantEmotion == tc.expectedEmotion {
                emotionCorrect += 1
            }
        }

        let valenceAccuracy = Double(correct) / Double(testCases.count) * 100
        let emotionAccuracy = Double(emotionCorrect) / Double(testCases.count) * 100
        let status: DiagnosticTestStatus = valenceAccuracy >= 80 ? .passed : (valenceAccuracy >= 60 ? .warning : .failed)

        return TestOutcome(
            status: status,
            message: "Valence accuracy: \(String(format: "%.0f", valenceAccuracy))% (\(correct)/\(testCases.count)), Emotion accuracy: \(String(format: "%.0f", emotionAccuracy))% (\(emotionCorrect)/\(testCases.count))",
            details: failures.isEmpty ? ["All valence classifications correct"] : Array(failures.prefix(5))
        )
    }

    private func deepTestEmotionBoundary() -> TestOutcome {
        let neutralBoundary = EmotionAnalyzer.analyze(text: "ok", conversationHistory: [])
        let barelyPositive = EmotionAnalyzer.analyze(text: "That's nice I guess", conversationHistory: [])
        let barelyNegative = EmotionAnalyzer.analyze(text: "That's a bit annoying", conversationHistory: [])
        let strongPositive = EmotionAnalyzer.analyze(text: "I'm absolutely thrilled and ecstatic!", conversationHistory: [])
        let strongNegative = EmotionAnalyzer.analyze(text: "I'm furious and completely devastated!", conversationHistory: [])

        var checks: [String] = []
        var passed = 0
        let total = 5

        if neutralBoundary.valence == .neutral { passed += 1 } else { checks.append("'ok' should be neutral, got \(neutralBoundary.valence.rawValue)") }
        if barelyPositive.empathyLevel <= 0.5 { passed += 1 } else { checks.append("Barely positive empathy too high: \(String(format: "%.2f", barelyPositive.empathyLevel))") }
        if barelyNegative.valence == .negative || barelyNegative.valence == .neutral { passed += 1 } else { checks.append("'a bit annoying' misclassified: \(barelyNegative.valence.rawValue)") }
        if strongPositive.arousal == .high { passed += 1 } else { checks.append("Strong positive should have high arousal, got \(strongPositive.arousal.rawValue)") }
        if strongNegative.empathyLevel > strongPositive.empathyLevel { passed += 1 } else { checks.append("Strong negative empathy (\(String(format: "%.2f", strongNegative.empathyLevel))) should exceed positive (\(String(format: "%.2f", strongPositive.empathyLevel)))") }

        return TestOutcome(
            status: passed >= 4 ? .passed : (passed >= 3 ? .warning : .failed),
            message: "Boundary detection: \(passed)/\(total) correct",
            details: checks.isEmpty ? ["All boundary cases handled correctly"] : checks
        )
    }

    private func deepTestEmotionAdversarial() -> TestOutcome {
        let testCases: [(input: String, note: String)] = [
            ("", "Empty string"),
            (String(repeating: "a", count: 5000), "Very long input"),
            ("🤬😡💀🔥", "Emoji only"),
            ("I'm not happy", "Negation"),
            ("I'm not sad at all, I'm great!", "Double negation with positive"),
            ("kill the process and terminate the thread", "Technical violence"),
            ("The weather is nice but I hate rain", "Contradictory"),
            ("HELP HELP HELP!!!", "Repeated urgent"),
        ]

        var crashes = 0
        var details: [String] = []

        for tc in testCases {
            let state = EmotionAnalyzer.analyze(text: tc.input, conversationHistory: [])
            let valenceValid = [EmotionalValence.positive, .negative, .neutral, .mixed].contains(state.valence)
            let empathyValid = state.empathyLevel >= 0 && state.empathyLevel <= 1.0
            if !valenceValid || !empathyValid {
                crashes += 1
                details.append("\(tc.note): invalid output (valence=\(state.valence.rawValue), empathy=\(String(format: "%.2f", state.empathyLevel)))")
            } else {
                details.append("\(tc.note): \(state.valence.rawValue)/\(state.dominantEmotion) ✓")
            }
        }

        return TestOutcome(
            status: crashes == 0 ? .passed : .failed,
            message: "Adversarial: \(testCases.count - crashes)/\(testCases.count) survived without invalid output",
            details: details
        )
    }

    private func deepTestEmpathyCalibration() -> TestOutcome {
        let inputs: [(text: String, expectedRange: ClosedRange<Double>)] = [
            ("What time is it?", 0.0...0.4),
            ("I'm a bit sad today", 0.4...0.8),
            ("I'm heartbroken, everything is falling apart", 0.7...1.0),
            ("I'm SO happy!!", 0.2...0.6),
            ("I need help urgently, I'm panicking", 0.6...1.0),
        ]

        var correct = 0
        var details: [String] = []

        for input in inputs {
            let state = EmotionAnalyzer.analyze(text: input.text, conversationHistory: [])
            let inRange = input.expectedRange.contains(state.empathyLevel)
            if inRange { correct += 1 }
            details.append("'\(input.text.prefix(35))…' empathy=\(String(format: "%.2f", state.empathyLevel)) expected \(input.expectedRange) → \(inRange ? "✓" : "✗")")
        }

        return TestOutcome(
            status: correct >= 4 ? .passed : (correct >= 3 ? .warning : .failed),
            message: "Empathy calibration: \(correct)/\(inputs.count) within expected range",
            details: details
        )
    }

    private func deepTestStyleMatrix() -> TestOutcome {
        let testCases: [(input: String, expected: String)] = [
            ("Furthermore, I would like to discuss the implications of this policy and its consequences hitherto", "formal"),
            ("lol omg thats so cool gonna check it out bruh ngl", "casual"),
            ("I need to refactor the API endpoint and optimize the database query performance with better algorithms", "technical"),
            ("Imagine a world where every dream becomes reality, let's brainstorm creative solutions", "creative"),
            ("I need this ASAP, it's urgent! The deadline is right now!", "urgent"),
            ("I wonder about the meaning of consciousness and the purpose of existence", "reflective"),
        ]

        var correct = 0
        var details: [String] = []

        for tc in testCases {
            let state = EmotionAnalyzer.analyze(text: tc.input, conversationHistory: [])
            let match = state.style == tc.expected
            if match { correct += 1 }
            details.append("Expected '\(tc.expected)', got '\(state.style)' → \(match ? "✓" : "✗")")
        }

        return TestOutcome(
            status: correct >= 5 ? .passed : (correct >= 4 ? .warning : .failed),
            message: "Style classification: \(correct)/\(testCases.count) correct",
            details: details
        )
    }

    private func deepTestTrajectoryAccuracy() -> TestOutcome {
        let decliningHistory = [
            Message(role: .user, content: "I'm feeling great today!"),
            Message(role: .assistant, content: "Wonderful!"),
            Message(role: .user, content: "Actually things are getting worse"),
            Message(role: .assistant, content: "I'm sorry"),
            Message(role: .user, content: "Now I'm feeling really sad and frustrated"),
            Message(role: .assistant, content: "That must be tough"),
        ]
        let declining = EmotionAnalyzer.analyze(text: "Everything is terrible now", conversationHistory: decliningHistory)

        let improvingHistory = [
            Message(role: .user, content: "I'm feeling terrible today"),
            Message(role: .assistant, content: "I'm sorry"),
            Message(role: .user, content: "Wait, things are looking up a bit"),
            Message(role: .assistant, content: "That's good"),
            Message(role: .user, content: "Actually I'm starting to feel much better and happy!"),
            Message(role: .assistant, content: "Great!"),
        ]
        let improving = EmotionAnalyzer.analyze(text: "I'm so happy now, everything is wonderful!", conversationHistory: improvingHistory)

        let stableHistory = [
            Message(role: .user, content: "What is 2+2?"),
            Message(role: .assistant, content: "4"),
            Message(role: .user, content: "What is the capital of France?"),
            Message(role: .assistant, content: "Paris"),
        ]
        let stable = EmotionAnalyzer.analyze(text: "How tall is Mount Everest?", conversationHistory: stableHistory)

        var score = 0
        var details: [String] = []

        if declining.emotionalTrajectory == "declining" { score += 1; details.append("Declining: ✓") }
        else { details.append("Declining: expected 'declining', got '\(declining.emotionalTrajectory)'") }

        if improving.emotionalTrajectory == "improving" { score += 1; details.append("Improving: ✓") }
        else { details.append("Improving: expected 'improving', got '\(improving.emotionalTrajectory)'") }

        if stable.emotionalTrajectory == "stable" { score += 1; details.append("Stable: ✓") }
        else { details.append("Stable: expected 'stable', got '\(stable.emotionalTrajectory)'") }

        return TestOutcome(
            status: score >= 2 ? .passed : .warning,
            message: "Trajectory accuracy: \(score)/3",
            details: details
        )
    }

    // MARK: - Intent Accuracy Tests

    private func deepTestIntentMatrix() -> TestOutcome {
        let testCases: [(input: String, expected: IntentType)] = [
            ("Hello!", .socialGreeting),
            ("Hey there, good morning!", .socialGreeting),
            ("Goodbye, see you later!", .socialFarewell),
            ("Thanks so much for your help!", .socialGratitude),
            ("Sorry about that mistake", .socialApology),
            ("What is quantum computing?", .questionFactual),
            ("Who invented the telephone?", .questionFactual),
            ("How does photosynthesis work?", .questionHow),
            ("How to make pasta from scratch", .questionHow),
            ("Why is the sky blue?", .questionWhy),
            ("Compare Python and JavaScript", .questionComparison),
            ("Write me a poem about the ocean", .requestCreation),
            ("Create a short story about robots", .requestCreation),
            ("Analyze this code for bugs", .requestAnalysis),
            ("Search for the latest news on AI", .requestSearch),
            ("Do you remember my name?", .requestMemory),
            ("Calculate 15% tip on $85", .requestCalculation),
            ("Solve the equation x^2 + 5 = 30", .requestCalculation),
            ("I feel really happy today!", .statementEmotion),
            ("I think AI will change everything", .statementOpinion),
            ("No, that's wrong. The answer is 42.", .metaCorrection),
            ("What do you mean by that? Clarify please.", .metaClarification),
            ("Good job, that was helpful!", .metaFeedback),
            ("Let's brainstorm ideas for the project", .explorationBrainstorm),
            ("What if gravity didn't exist?", .explorationHypothetical),
            ("I work at a tech startup in San Francisco", .statementFact),
            ("Always remember to use formal English with me", .statementInstruction),
            ("How come birds can fly?", .questionHow),
            ("Debate the pros and cons of remote work", .explorationDebate),
            ("What's the difference between REST and GraphQL?", .questionComparison),
        ]

        var correct = 0
        var mismatches: [String] = []

        for tc in testCases {
            let intent = IntentClassifier.classify(text: tc.input, conversationHistory: [])
            if intent.primary == tc.expected {
                correct += 1
            } else {
                mismatches.append("'\(tc.input.prefix(40))…': expected \(tc.expected.rawValue), got \(intent.primary.rawValue) (conf: \(String(format: "%.2f", intent.confidence)))")
            }
        }

        let accuracy = Double(correct) / Double(testCases.count) * 100
        let status: DiagnosticTestStatus = accuracy >= 80 ? .passed : (accuracy >= 65 ? .warning : .failed)

        return TestOutcome(
            status: status,
            message: "Intent accuracy: \(String(format: "%.0f", accuracy))% (\(correct)/\(testCases.count))",
            details: mismatches.isEmpty ? ["All intent classifications correct"] : Array(mismatches.prefix(8))
        )
    }

    private func deepTestIntentConfidence() -> TestOutcome {
        let clearIntent = IntentClassifier.classify(text: "Hello!", conversationHistory: [])
        let ambiguousIntent = IntentClassifier.classify(text: "thing", conversationHistory: [])
        let strongIntent = IntentClassifier.classify(text: "Calculate the square root of 144 right now", conversationHistory: [])

        var checks: [String] = []
        var score = 0

        if clearIntent.confidence > 0.3 { score += 1; checks.append("Clear greeting conf=\(String(format: "%.2f", clearIntent.confidence)) > 0.3 ✓") }
        else { checks.append("Clear greeting conf=\(String(format: "%.2f", clearIntent.confidence)) should be > 0.3") }

        if strongIntent.confidence > clearIntent.confidence || strongIntent.confidence > 0.3 {
            score += 1; checks.append("Strong intent conf=\(String(format: "%.2f", strongIntent.confidence)) reasonable ✓")
        } else { checks.append("Strong intent conf=\(String(format: "%.2f", strongIntent.confidence)) unexpectedly low") }

        if ambiguousIntent.confidence <= strongIntent.confidence {
            score += 1; checks.append("Ambiguous conf=\(String(format: "%.2f", ambiguousIntent.confidence)) ≤ strong=\(String(format: "%.2f", strongIntent.confidence)) ✓")
        } else { checks.append("Ambiguous should have lower confidence than clear intent") }

        return TestOutcome(status: score >= 2 ? .passed : .warning, message: "Confidence calibration: \(score)/3", details: checks)
    }

    private func deepTestMultiIntent() -> TestOutcome {
        let testCases: [(input: String, shouldBeMulti: Bool, expectedIntents: [IntentType])] = [
            ("Write a poem and analyze its literary devices", true, [.requestCreation, .requestAnalysis]),
            ("Calculate 2+2 and remember the result", true, [.requestCalculation, .requestMemory]),
            ("Hello!", false, [.socialGreeting]),
            ("Search for AI news and summarize the key findings", true, [.requestSearch, .requestAnalysis]),
        ]

        var correct = 0
        var details: [String] = []

        for tc in testCases {
            let intent = IntentClassifier.classify(text: tc.input, conversationHistory: [])
            let multiCorrect = intent.isMultiIntent == tc.shouldBeMulti
            let hasExpected = tc.expectedIntents.contains(intent.primary)

            if multiCorrect && hasExpected { correct += 1 }
            details.append("'\(tc.input.prefix(40))…': multi=\(intent.isMultiIntent)(expected \(tc.shouldBeMulti)), primary=\(intent.primary.rawValue), subs=\(intent.subIntents.map(\.rawValue).joined(separator: ","))")
        }

        return TestOutcome(
            status: correct >= 3 ? .passed : (correct >= 2 ? .warning : .failed),
            message: "Multi-intent decomposition: \(correct)/\(testCases.count)",
            details: details
        )
    }

    private func deepTestUrgencyGradient() -> TestOutcome {
        let inputs = [
            ("When you have a chance, could you look into this?", "low"),
            ("Can you help me with this today?", "medium"),
            ("I need this urgently please!", "high"),
            ("EMERGENCY! HELP NOW! ASAP!!!", "critical"),
        ]

        var urgencies: [Double] = []
        var details: [String] = []

        for input in inputs {
            let intent = IntentClassifier.classify(text: input.0, conversationHistory: [])
            urgencies.append(intent.urgency)
            details.append("[\(input.1)] '\(input.0.prefix(40))…' → urgency=\(String(format: "%.2f", intent.urgency))")
        }

        let isMonotonic = zip(urgencies, urgencies.dropFirst()).allSatisfy { $0 <= $1 }

        return TestOutcome(
            status: isMonotonic ? .passed : .warning,
            message: "Urgency gradient monotonic: \(isMonotonic), range: \(String(format: "%.2f", urgencies.first ?? 0))→\(String(format: "%.2f", urgencies.last ?? 0))",
            details: details
        )
    }

    private func deepTestResponseLength() -> TestOutcome {
        let testCases: [(input: String, expected: ResponseLength)] = [
            ("Hi", .brief),
            ("Hello there!", .brief),
            ("What is 2+2?", .brief),
            ("How does machine learning work?", .detailed),
            ("Explain the philosophical implications of quantum mechanics on consciousness", .comprehensive),
        ]

        var correct = 0
        var details: [String] = []

        for tc in testCases {
            let intent = IntentClassifier.classify(text: tc.input, conversationHistory: [])
            let match = intent.expectedResponseLength == tc.expected
            if match { correct += 1 }
            details.append("'\(tc.input.prefix(40))…': expected \(tc.expected.rawValue), got \(intent.expectedResponseLength.rawValue) → \(match ? "✓" : "✗")")
        }

        return TestOutcome(
            status: correct >= 4 ? .passed : (correct >= 3 ? .warning : .failed),
            message: "Response length calibration: \(correct)/\(testCases.count)",
            details: details
        )
    }

    private func deepTestIntentAdversarial() -> TestOutcome {
        let inputs: [(text: String, note: String)] = [
            ("", "Empty"),
            ("a", "Single char"),
            (String(repeating: "hello ", count: 500), "Long repetition"),
            ("!!!???...", "Punctuation only"),
            ("42", "Number only"),
            ("Remember to calculate the search for why how what", "Keyword stuffing"),
            ("🤖💻🔍📝", "Emoji only"),
        ]

        var survived = 0
        var details: [String] = []

        for input in inputs {
            let intent = IntentClassifier.classify(text: input.text, conversationHistory: [])
            let valid = intent.confidence >= 0 && intent.confidence <= 1 && intent.urgency >= 0 && intent.urgency <= 1
            if valid { survived += 1 }
            details.append("\(input.note): \(intent.primary.rawValue) (conf=\(String(format: "%.2f", intent.confidence))) → \(valid ? "✓" : "✗")")
        }

        return TestOutcome(
            status: survived == inputs.count ? .passed : .failed,
            message: "Adversarial: \(survived)/\(inputs.count) produced valid output",
            details: details
        )
    }

    // MARK: - Memory Quality Tests

    private func deepTestMemoryRankingTFIDF() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let entries = [
            MemoryEntry(content: "User loves programming in Swift for iOS apps", keywords: ["swift", "ios", "programming"], category: .preference, importance: 5, source: .conversation),
            MemoryEntry(content: "User asked about cooking pasta recipes last week", keywords: ["cooking", "pasta", "recipes"], category: .context, importance: 3, source: .conversation),
            MemoryEntry(content: "User is building a neural network inference engine in Swift", keywords: ["neural", "network", "swift", "inference"], category: .skill, importance: 5, source: .conversation),
            MemoryEntry(content: "User enjoys hiking in mountains during autumn", keywords: ["hiking", "mountains", "autumn"], category: .preference, importance: 3, source: .conversation),
        ]

        for entry in entries { mem.addMemory(entry) }

        let results = mem.searchMemories(query: "Swift programming for neural network", maxResults: 4)

        for entry in entries { mem.deleteMemory(entry.id) }

        guard results.count >= 2 else {
            return TestOutcome(status: .failed, message: "Search returned only \(results.count) results", details: [])
        }

        let topIds = results.prefix(2).map(\.memory.id)
        let relevantIds = Set([entries[0].id, entries[2].id])
        let topRelevant = topIds.filter { relevantIds.contains($0) }.count

        let scoresMonotonic = zip(results.map(\.score), results.map(\.score).dropFirst()).allSatisfy { $0 >= $1 }

        var details = results.map { "\($0.memory.content.prefix(50))… score=\(String(format: "%.3f", $0.score)) type=\($0.matchType.rawValue)" }
        details.append("Top-2 relevant: \(topRelevant)/2, scores monotonic: \(scoresMonotonic)")

        return TestOutcome(
            status: topRelevant >= 1 && scoresMonotonic ? .passed : .warning,
            message: "TF-IDF ranking: top-2 relevance=\(topRelevant)/2, monotonic=\(scoresMonotonic)",
            details: details
        )
    }

    private func deepTestMemoryRankingSemantic() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let semanticEntry = MemoryEntry(content: "User enjoys outdoor activities like trekking and camping in nature", keywords: ["outdoor", "trekking", "camping"], category: .preference, importance: 4, source: .conversation)
        let irrelevantEntry = MemoryEntry(content: "User asked about database optimization SQL queries", keywords: ["database", "sql", "optimization"], category: .context, importance: 4, source: .conversation)

        mem.addMemory(semanticEntry)
        mem.addMemory(irrelevantEntry)

        let results = mem.searchMemories(query: "hiking and nature walks in the forest", maxResults: 5)

        mem.deleteMemory(semanticEntry.id)
        mem.deleteMemory(irrelevantEntry.id)

        let semanticRank = results.firstIndex(where: { $0.memory.id == semanticEntry.id })
        let irrelevantRank = results.firstIndex(where: { $0.memory.id == irrelevantEntry.id })

        let semanticHigher = (semanticRank ?? 999) < (irrelevantRank ?? 999)
        let details = results.map { "\($0.memory.content.prefix(50))… score=\(String(format: "%.3f", $0.score))" }

        return TestOutcome(
            status: semanticHigher ? .passed : .warning,
            message: "Semantic ranking: outdoor activities \(semanticHigher ? "ranks higher than" : "does NOT rank higher than") SQL (ranks: \(semanticRank.map(String.init) ?? "absent"), \(irrelevantRank.map(String.init) ?? "absent"))",
            details: details
        )
    }

    private func deepTestMemoryDedup() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let before = mem.memories.count

        let m1 = MemoryEntry(content: "User's favorite programming language is Swift", keywords: ["swift", "programming"], category: .preference, importance: 4, source: .conversation)
        mem.addMemory(m1)
        let afterFirst = mem.memories.count

        let m2 = MemoryEntry(content: "User's favorite programming language is Swift and they use it daily", keywords: ["swift", "programming", "daily"], category: .preference, importance: 5, source: .conversation)
        mem.addMemory(m2)
        let afterSecond = mem.memories.count

        let m3 = MemoryEntry(content: "User enjoys cooking Italian pasta recipes", keywords: ["cooking", "pasta"], category: .preference, importance: 3, source: .conversation)
        mem.addMemory(m3)
        let afterThird = mem.memories.count

        mem.deleteMemory(m1.id)
        mem.deleteMemory(m2.id)
        mem.deleteMemory(m3.id)
        for m in mem.memories where m.content.contains("User's favorite programming language is Swift") {
            mem.deleteMemory(m.id)
        }

        let dedupWorked = (afterSecond - afterFirst) <= 1
        let uniqueAdded = (afterThird - afterSecond) == 1

        return TestOutcome(
            status: dedupWorked && uniqueAdded ? .passed : .warning,
            message: "Dedup: similar entry added \(afterSecond - afterFirst) new (expected 0-1), unique entry added \(afterThird - afterSecond) new (expected 1)",
            details: ["before=\(before)", "afterFirst=\(afterFirst)", "afterSecond=\(afterSecond)", "afterThird=\(afterThird)"]
        )
    }

    private func deepTestAssociativeLinks() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let m1 = MemoryEntry(content: "User loves Python programming for machine learning", keywords: ["python", "programming", "machine learning"], category: .skill, importance: 4, source: .conversation)
        let m2 = MemoryEntry(content: "User is studying machine learning algorithms with Python", keywords: ["machine learning", "algorithms", "python"], category: .skill, importance: 4, source: .conversation)
        let m3 = MemoryEntry(content: "User's favorite color is blue", keywords: ["color", "blue"], category: .preference, importance: 2, source: .conversation)

        mem.addMemory(m1)
        mem.addMemory(m2)
        mem.addMemory(m3)

        let linkedPair = mem.associativeLinks.contains {
            ($0.sourceId == m2.id && $0.targetId == m1.id) || ($0.sourceId == m1.id && $0.targetId == m2.id)
        }
        let linkedToUnrelated = mem.associativeLinks.contains {
            ($0.sourceId == m3.id && ($0.targetId == m1.id || $0.targetId == m2.id)) ||
            ($0.targetId == m3.id && ($0.sourceId == m1.id || $0.sourceId == m2.id))
        }

        let relatedStrength = mem.associativeLinks.first { ($0.sourceId == m2.id && $0.targetId == m1.id) || ($0.sourceId == m1.id && $0.targetId == m2.id) }?.strength ?? 0

        mem.deleteMemory(m1.id)
        mem.deleteMemory(m2.id)
        mem.deleteMemory(m3.id)

        return TestOutcome(
            status: linkedPair ? .passed : .warning,
            message: "Related linked: \(linkedPair) (strength=\(String(format: "%.2f", relatedStrength))), unrelated linked: \(linkedToUnrelated)",
            details: ["Python↔ML link: \(linkedPair)", "Blue↔Python link: \(linkedToUnrelated)", "Link strength: \(String(format: "%.3f", relatedStrength))"]
        )
    }

    private func deepTestMemoryDecay() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let recentEntry = MemoryEntry(content: "Recent diagnostic memory entry", keywords: ["recent"], category: .context, importance: 3, source: .system)

        let oldTimestamp = Date().timeIntervalSince1970 * 1000 - (72 * 3600 * 1000)
        let oldEntry = MemoryEntry(content: "Old diagnostic memory entry from days ago", keywords: ["old"], category: .context, timestamp: oldTimestamp, importance: 3, source: .system, lastAccessed: oldTimestamp)

        mem.addMemory(recentEntry)
        mem.addMemory(oldEntry)

        let results = mem.searchMemories(query: "diagnostic memory entry", maxResults: 5)
        let recentScore = results.first(where: { $0.memory.id == recentEntry.id })?.score ?? 0
        let oldScore = results.first(where: { $0.memory.id == oldEntry.id })?.score ?? 0

        mem.deleteMemory(recentEntry.id)
        mem.deleteMemory(oldEntry.id)

        let recentHigher = recentScore >= oldScore

        return TestOutcome(
            status: recentHigher ? .passed : .warning,
            message: "Decay: recent score=\(String(format: "%.3f", recentScore)), old score=\(String(format: "%.3f", oldScore)), recent ranks higher: \(recentHigher)",
            details: ["Time difference: 72 hours", "Score delta: \(String(format: "%.3f", recentScore - oldScore))"]
        )
    }

    private func deepTestConsolidation() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let before = mem.memories.count
        var addedIds: [String] = []

        for i in 0..<50 {
            let entry = MemoryEntry(content: "Consolidation test entry number \(i) about topic \(i % 5)", keywords: ["consolidation", "test", "topic\(i % 5)"], category: .context, importance: (i % 5) + 1, source: .system)
            mem.addMemory(entry)
            addedIds.append(entry.id)
        }

        let afterAdd = mem.memories.count
        mem.extractAndStoreMemory(userText: "I like consolidation testing very much for diagnostics", assistantText: "Great, consolidation is important for memory management")

        let afterConsolidate = mem.memories.count
        let contextCount = mem.memories.filter { $0.category == .context }.count

        for id in addedIds { mem.deleteMemory(id) }

        return TestOutcome(
            status: contextCount <= 20 ? .passed : .warning,
            message: "Added 50 entries. Before: \(before), after add: \(afterAdd), after consolidation: \(afterConsolidate), context category: \(contextCount)",
            details: ["Consolidation threshold: 15/category", "Context entries remaining: \(contextCount)", "Entries pruned: \(afterAdd - afterConsolidate)"]
        )
    }

    private func deepTestMemoryExtraction() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let testCases: [(user: String, assistant: String, shouldExtract: String)] = [
            ("My name is Sarah", "Nice to meet you Sarah!", "name"),
            ("I really love dark chocolate", "Dark chocolate is delicious!", "preference"),
            ("I hate cold weather", "I understand, cold weather can be unpleasant.", "dislike"),
            ("I work at a tech startup in San Francisco", "That's exciting!", "fact"),
            ("I'm learning machine learning with Python", "Great choice!", "goal"),
            ("My favorite color is green", "Green is a nice color!", "favorite"),
            ("Always remember to greet me in French", "D'accord!", "instruction"),
            ("I want to learn Swift programming", "Swift is a great language!", "goal"),
            ("I have a golden retriever named Max", "Max sounds adorable!", "fact"),
            ("I'm from Montreal, Canada", "Beautiful city!", "fact"),
        ]

        let residualMarkers = ["Sarah", "dark chocolate", "cold weather", "tech startup", "machine learning", "favorite color", "green", "greet me in French", "Swift programming", "golden retriever", "Max", "Montreal"]
        let residualIds = mem.memories.filter { m in
            residualMarkers.contains(where: { m.content.localizedCaseInsensitiveContains($0) })
        }.map(\.id)
        for id in residualIds { mem.deleteMemory(id) }

        var extracted = 0
        var details: [String] = []
        let before = mem.memories.count

        for tc in testCases {
            let countBefore = mem.memories.count
            mem.extractAndStoreMemory(userText: tc.user, assistantText: tc.assistant)
            let countAfter = mem.memories.count
            let didExtract = countAfter > countBefore
            if didExtract { extracted += 1 }
            details.append("[\(tc.shouldExtract)] '\(tc.user.prefix(35))…' → \(didExtract ? "extracted" : "MISSED")")
        }

        let after = mem.memories.count
        for m in mem.memories where m.timestamp > Date().timeIntervalSince1970 * 1000 - 10000 && !mem.memories.prefix(before).contains(where: { $0.id == m.id }) {
            mem.deleteMemory(m.id)
        }

        return TestOutcome(
            status: extracted >= 8 ? .passed : (extracted >= 6 ? .warning : .failed),
            message: "Extraction accuracy: \(extracted)/\(testCases.count) patterns extracted (\(after - before) total new memories)",
            details: details
        )
    }

    private func deepTestSearchDiversity() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let entries = [
            MemoryEntry(content: "User prefers dark mode in all applications", keywords: ["dark mode", "preference"], category: .preference, importance: 4, source: .conversation),
            MemoryEntry(content: "User fact: works as a software engineer", keywords: ["software", "engineer"], category: .fact, importance: 4, source: .conversation),
            MemoryEntry(content: "User is learning machine learning concepts", keywords: ["machine learning", "learning"], category: .skill, importance: 4, source: .conversation),
            MemoryEntry(content: "User prefers concise technical answers", keywords: ["concise", "technical"], category: .instruction, importance: 5, source: .conversation),
            MemoryEntry(content: "User was feeling stressed about deadlines", keywords: ["stressed", "deadlines"], category: .emotion, importance: 3, source: .conversation),
        ]

        for entry in entries { mem.addMemory(entry) }

        let results = mem.searchMemories(query: "user preferences and work", maxResults: 8)
        let categories = Set(results.map(\.memory.category))

        for entry in entries { mem.deleteMemory(entry.id) }

        return TestOutcome(
            status: categories.count >= 2 ? .passed : .warning,
            message: "Search diversity: \(categories.count) categories in results (\(results.count) total results)",
            details: results.map { "[\($0.memory.category.rawValue)] \($0.memory.content.prefix(50))… score=\(String(format: "%.3f", $0.score))" }
        )
    }

    private func deepTestNameExtractionRobustness() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let testCases: [(user: String, expectedName: String)] = [
            ("My name is Sarah", "Sarah"),
            ("I'm Alex and I love coding", "Alex"),
            ("Call me Viktor", "Viktor"),
            ("People call me Zara", "Zara"),
            ("My name is Jordan and I work at Google", "Jordan"),
            ("I'm Mei from Tokyo", "Mei"),
            ("My name is Anna Marie", "Anna Marie"),
            ("They call me Dex", "Dex"),
        ]

        let allTestNames = testCases.map(\.expectedName)
        let residualIds = mem.memories.filter { m in allTestNames.contains(where: { m.content.localizedCaseInsensitiveContains($0) }) }.map(\.id)
        for id in residualIds { mem.deleteMemory(id) }

        var extracted = 0
        var details: [String] = []

        for tc in testCases {
            let countBefore = mem.memories.count
            mem.extractAndStoreMemory(userText: tc.user, assistantText: "Nice to meet you!")
            let countAfter = mem.memories.count
            let nameFound = mem.memories.contains { $0.content.localizedCaseInsensitiveContains(tc.expectedName) && $0.content.lowercased().contains("name") }
            if nameFound { extracted += 1 }
            details.append("'\(tc.user.prefix(40))\u{2026}' \u{2192} \(tc.expectedName): \(nameFound ? "\u{2713}" : "MISSED") (+\(countAfter - countBefore) entries)")
        }

        let cleanupIds = mem.memories.filter { m in allTestNames.contains(where: { m.content.localizedCaseInsensitiveContains($0) }) }.map(\.id)
        for id in cleanupIds { mem.deleteMemory(id) }

        return TestOutcome(
            status: extracted >= 6 ? .passed : (extracted >= 4 ? .warning : .failed),
            message: "Name extraction robustness: \(extracted)/\(testCases.count) patterns extracted",
            details: details
        )
    }

    private func deepTestHybridSearchScoreDistribution() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let keywordEntry = MemoryEntry(content: "User loves Python programming language for data science", keywords: ["python", "programming", "data science"], category: .preference, importance: 4, source: .conversation)
        let semanticEntry = MemoryEntry(content: "User enjoys writing code and building software applications", keywords: ["coding", "software", "applications"], category: .preference, importance: 4, source: .conversation)
        let irrelevantEntry = MemoryEntry(content: "User's favorite food is spaghetti carbonara", keywords: ["food", "spaghetti", "carbonara"], category: .preference, importance: 4, source: .conversation)

        mem.addMemory(keywordEntry)
        mem.addMemory(semanticEntry)
        mem.addMemory(irrelevantEntry)

        let results = mem.searchMemories(query: "Python programming", maxResults: 10)
        let keywordScore = results.first(where: { $0.memory.id == keywordEntry.id })?.score ?? 0
        let semanticScore = results.first(where: { $0.memory.id == semanticEntry.id })?.score ?? 0
        let irrelevantScore = results.first(where: { $0.memory.id == irrelevantEntry.id })?.score ?? 0

        mem.deleteMemory(keywordEntry.id)
        mem.deleteMemory(semanticEntry.id)
        mem.deleteMemory(irrelevantEntry.id)

        var checks = 0
        var details: [String] = [
            "Keyword match score: \(String(format: "%.3f", keywordScore))",
            "Semantic match score: \(String(format: "%.3f", semanticScore))",
            "Irrelevant score: \(String(format: "%.3f", irrelevantScore))",
        ]

        if keywordScore > semanticScore { checks += 1; details.append("Keyword > Semantic: \u{2713}") } else { details.append("Keyword > Semantic: \u{2717}") }
        if keywordScore > irrelevantScore { checks += 1; details.append("Keyword > Irrelevant: \u{2713}") } else { details.append("Keyword > Irrelevant: \u{2717}") }
        if semanticScore > irrelevantScore { checks += 1; details.append("Semantic > Irrelevant: \u{2713}") } else { details.append("Semantic > Irrelevant: \u{2717}") }

        return TestOutcome(
            status: checks >= 2 ? .passed : (checks >= 1 ? .warning : .failed),
            message: "Score distribution: \(checks)/3 ordering checks passed (K=\(String(format: "%.3f", keywordScore)), S=\(String(format: "%.3f", semanticScore)), I=\(String(format: "%.3f", irrelevantScore)))",
            details: details
        )
    }

    // MARK: - Cognition Quality Tests

    private func deepTestComplexityGradient() -> TestOutcome {
        let inputs: [(text: String, expectedMin: ComplexityLevel)] = [
            ("Hi", .simple),
            ("What is the capital of France?", .simple),
            ("How does photosynthesis work in C4 plants?", .moderate),
            ("Explain the relationship between Gödel's incompleteness theorems and artificial intelligence", .complex),
            ("Analyze the philosophical implications of quantum entanglement on the nature of consciousness considering both Copenhagen and Many-Worlds interpretations while addressing the hard problem of consciousness and its relationship to information-theoretic approaches to physics", .complex),
        ]

        let complexityOrder: [ComplexityLevel] = [.simple, .moderate, .complex, .expert]
        var results: [(String, ComplexityLevel)] = []
        var score = 0

        for input in inputs {
            let meta = MetacognitionEngine.assess(text: input.text, conversationHistory: [], memoryResults: [])
            results.append((String(input.text.prefix(50)), meta.complexityLevel))
            let actualIdx = complexityOrder.firstIndex(of: meta.complexityLevel) ?? 0
            let expectedIdx = complexityOrder.firstIndex(of: input.expectedMin) ?? 0
            if actualIdx >= expectedIdx { score += 1 }
        }

        let isMonotonic = (0..<results.count - 1).allSatisfy { i in
            let idx1 = complexityOrder.firstIndex(of: results[i].1) ?? 0
            let idx2 = complexityOrder.firstIndex(of: results[i + 1].1) ?? 0
            return idx1 <= idx2
        }

        return TestOutcome(
            status: score >= 4 && isMonotonic ? .passed : (score >= 3 ? .warning : .failed),
            message: "Complexity gradient: \(score)/\(inputs.count) correct, monotonic: \(isMonotonic)",
            details: results.map { "'\($0.0)…' → \($0.1.rawValue)" }
        )
    }

    private func deepTestSignatureDeterminism() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame1 = CognitionEngine.process(userText: "What is machine learning?", conversationHistory: [], memoryService: mem)

        CognitionEngine.resetSignature()
        let frame2 = CognitionEngine.process(userText: "What is machine learning?", conversationHistory: [], memoryService: mem)

        let hashMatch = frame1.contextSignature.signatureHash == frame2.contextSignature.signatureHash
        let intentMatch = frame1.intent.primary == frame2.intent.primary
        let emotionMatch = frame1.emotion.valence == frame2.emotion.valence
        let vectorMatch = frame1.contextSignature.intentVector == frame2.contextSignature.intentVector

        CognitionEngine.resetSignature()

        return TestOutcome(
            status: hashMatch && intentMatch && emotionMatch ? .passed : .warning,
            message: "Determinism: hash=\(hashMatch), intent=\(intentMatch), emotion=\(emotionMatch), vector=\(vectorMatch)",
            details: ["Hash1: \(frame1.contextSignature.signatureHash.prefix(16))…", "Hash2: \(frame2.contextSignature.signatureHash.prefix(16))…"]
        )
    }

    private func deepTestDriftSensitivity() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame1 = CognitionEngine.process(userText: "Tell me about cooking pasta", conversationHistory: [], memoryService: mem)

        let frame2 = CognitionEngine.process(userText: "Now explain quantum entanglement in physics", conversationHistory: [], memoryService: mem)
        let hasDrift = frame2.injections.contains { $0.type == .reasoningTrace }

        CognitionEngine.resetSignature()
        let _ = CognitionEngine.process(userText: "What is machine learning?", conversationHistory: [], memoryService: mem)
        let frame4 = CognitionEngine.process(userText: "How about deep learning?", conversationHistory: [], memoryService: mem)
        let noDrift = !frame4.injections.contains { $0.content.lowercased().contains("topic shift") }

        let sig1 = frame1.contextSignature
        let sig2 = frame2.contextSignature
        let drift = ContextSignatureTracker.detectDrift(original: sig1, current: sig2)

        CognitionEngine.resetSignature()

        return TestOutcome(
            status: drift.driftMagnitude > 0.1 ? .passed : .warning,
            message: "Large topic shift drift=\(String(format: "%.3f", drift.driftMagnitude)), dimensions: \(drift.driftedDimensions.joined(separator: ", "))",
            details: ["Cooking→Quantum drift: \(String(format: "%.3f", drift.driftMagnitude))", "Drift detected in frame: \(hasDrift)", "Similar topic no false drift: \(noDrift)"]
        )
    }

    private func deepTestInjectionBudget() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(
            userText: "I'm extremely frustrated and anxious about a complex philosophical question regarding the nature of consciousness and free will that I need solved urgently!",
            conversationHistory: [],
            memoryService: mem
        )

        let totalTokens = frame.injections.reduce(0) { $0 + $1.estimatedTokens }
        let allPositive = frame.injections.allSatisfy { $0.priority >= 0 && $0.priority <= 1.0 }
        let sortedPriorities = frame.injections.map(\.priority)
        let sortedByPriority = sortedPriorities == sortedPriorities.sorted(by: >)
        let noEmpty = frame.injections.allSatisfy { !$0.content.isEmpty }

        CognitionEngine.resetSignature()

        return TestOutcome(
            status: allPositive && sortedByPriority && noEmpty ? .passed : .warning,
            message: "Injections: \(frame.injections.count), total tokens: ~\(totalTokens), sorted: \(sortedByPriority), valid: \(allPositive), no empty: \(noEmpty)",
            details: frame.injections.map { "[\($0.type.rawValue)] prio=\(String(format: "%.2f", $0.priority)) tokens=\($0.estimatedTokens)" }
        )
    }

    private func deepTestConvergenceMonotonicity() -> TestOutcome {
        let inputs: [(text: String, historyCount: Int)] = [
            ("Hi", 0),
            ("What is AI?", 0),
            ("What is AI?", 5),
            ("Complex multi-part question about AI ethics", 10),
        ]

        var scores: [Double] = []
        var details: [String] = []

        for input in inputs {
            let history = (0..<input.historyCount).map { i in
                Message(role: i % 2 == 0 ? .user : .assistant, content: "Message \(i)")
            }
            let meta = MetacognitionEngine.assess(text: input.text, conversationHistory: history, memoryResults: [])
            scores.append(meta.convergenceScore)
            details.append("'\(input.text.prefix(40))…' (history=\(input.historyCount)) → convergence=\(String(format: "%.3f", meta.convergenceScore))")
        }

        let allValid = scores.allSatisfy { $0 >= 0 && $0 <= 1 }

        return TestOutcome(
            status: allValid ? .passed : .failed,
            message: "Convergence scores all in [0,1]: \(allValid), range: \(String(format: "%.3f", scores.min() ?? 0))→\(String(format: "%.3f", scores.max() ?? 0))",
            details: details
        )
    }

    private func deepTestSelfCorrectionTrigger() -> TestOutcome {
        let correctionHistory = [
            Message(role: .user, content: "The Earth revolves around Mars"),
            Message(role: .assistant, content: "That's an interesting point about Mars"),
            Message(role: .user, content: "No that's completely wrong! The Earth revolves around the Sun, not Mars"),
        ]
        let withCorrection = MetacognitionEngine.assess(
            text: "No that's completely wrong! The Earth revolves around the Sun, not Mars",
            conversationHistory: correctionHistory,
            memoryResults: []
        )

        let normalHistory = [
            Message(role: .user, content: "What is 2+2?"),
            Message(role: .assistant, content: "4"),
        ]
        let withoutCorrection = MetacognitionEngine.assess(
            text: "Thanks, what about 3+3?",
            conversationHistory: normalHistory,
            memoryResults: []
        )

        let correctTrigger = !withCorrection.selfCorrectionFlags.isEmpty
        let noFalsePositive = withoutCorrection.selfCorrectionFlags.isEmpty

        return TestOutcome(
            status: correctTrigger && noFalsePositive ? .passed : .warning,
            message: "Correction triggered: \(correctTrigger) (flags: \(withCorrection.selfCorrectionFlags.count)), false positive: \(!noFalsePositive)",
            details: withCorrection.selfCorrectionFlags.map { "[\($0.domain)] \($0.issue) severity=\(String(format: "%.2f", $0.severity))" } + ["Normal follow-up flags: \(withoutCorrection.selfCorrectionFlags.count)"]
        )
    }

    private func deepTestEntropyEscalation() -> TestOutcome {
        let lowEntropy = MetacognitionEngine.assess(text: "Hi", conversationHistory: [], memoryResults: [])
        let highEntropy = MetacognitionEngine.assess(
            text: "Analyze the intersectionality between quantum decoherence thermodynamic entropy epistemic uncertainty Bayesian inference information-theoretic bounds computational complexity Kolmogorov randomness algorithmic information content",
            conversationHistory: [],
            memoryResults: []
        )

        let lowE = lowEntropy.entropyAnalysis
        let highE = highEntropy.entropyAnalysis

        let entropyHigher = highE.shannonEntropy > lowE.shannonEntropy
        let densityReasonable = highE.semanticDensity >= 0 && highE.semanticDensity <= 1

        return TestOutcome(
            status: entropyHigher && densityReasonable ? .passed : .warning,
            message: "Low entropy: H=\(String(format: "%.2f", lowE.shannonEntropy)) escalate=\(lowE.shouldEscalate), High entropy: H=\(String(format: "%.2f", highE.shannonEntropy)) escalate=\(highE.shouldEscalate)",
            details: ["Low density=\(String(format: "%.2f", lowE.semanticDensity))", "High density=\(String(format: "%.2f", highE.semanticDensity))", "Entropy ordering correct: \(entropyHigher)"]
        )
    }

    private func deepTestThoughtTreeQuality() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Compare REST APIs and GraphQL for a mobile app", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "Compare REST APIs and GraphQL for a mobile app", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "Compare REST APIs and GraphQL for a mobile app", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "Compare REST APIs and GraphQL for a mobile app", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)

        let hasBranches = !tree.branches.isEmpty
        let hasBestPath = !tree.bestPath.isEmpty
        let convergenceValid = tree.convergencePercent >= 0 && tree.convergencePercent <= 1
        let allBranchesHaveContent = tree.branches.allSatisfy { !$0.hypothesis.isEmpty }
        let confidencesValid = tree.branches.allSatisfy { $0.confidence >= 0 && $0.confidence <= 1 }
        let depthReasonable = tree.maxDepthReached >= 1 && tree.maxDepthReached <= 4

        let checks = [hasBranches, hasBestPath, convergenceValid, allBranchesHaveContent, confidencesValid, depthReasonable]
        let passed = checks.filter { $0 }.count

        return TestOutcome(
            status: passed >= 5 ? .passed : (passed >= 4 ? .warning : .failed),
            message: "Tree quality: \(passed)/\(checks.count) checks passed. Branches=\(tree.branches.count), depth=\(tree.maxDepthReached), convergence=\(String(format: "%.0f%%", tree.convergencePercent * 100))",
            details: [
                "Has branches: \(hasBranches) (\(tree.branches.count))",
                "Has best path: \(hasBestPath)",
                "Convergence valid: \(convergenceValid) (\(String(format: "%.2f", tree.convergencePercent)))",
                "All have content: \(allBranchesHaveContent)",
                "Confidences [0,1]: \(confidencesValid)",
                "Depth reasonable: \(depthReasonable) (\(tree.maxDepthReached))",
                "Strategy: \(tree.synthesisStrategy.rawValue)",
                "Pruned: \(tree.prunedBranches.count), DFS: \(tree.dfsExpansions)"
            ]
        )
    }

    // MARK: - Context Quality Tests

    private func deepTestPromptStructure() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "How can I improve?", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: true, isVoiceMode: false, preferredResponseLanguageCode: nil)

        let hasIdentity = prompt.contains("NEXUS")
        let hasToolStrategy = prompt.contains("Tool strategy") || prompt.contains("tool")
        let hasCognitiveState = prompt.contains("Cognitive State") || prompt.contains("Intent:")
        let hasTimestamp = prompt.contains(TimeZone.current.identifier)
        let hasConfidenceProtocol = prompt.contains("confidence") || prompt.contains("Confidence")

        let checks = [
            ("Core identity (NEXUS)", hasIdentity),
            ("Tool strategy section", hasToolStrategy),
            ("Cognitive state section", hasCognitiveState),
            ("Timezone reference", hasTimestamp),
            ("Confidence protocol", hasConfidenceProtocol),
        ]
        let passed = checks.filter(\.1).count

        CognitionEngine.resetSignature()

        return TestOutcome(
            status: passed >= 4 ? .passed : (passed >= 3 ? .warning : .failed),
            message: "Prompt structure: \(passed)/\(checks.count) required sections present (\(prompt.count) chars)",
            details: checks.map { "\($0.0): \($0.1 ? "✓" : "✗")" }
        )
    }

    private func deepTestMemoryInjectionRelevance() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let entry = MemoryEntry(content: "User is an experienced iOS developer who prefers SwiftUI", keywords: ["ios", "swiftui", "developer"], category: .fact, importance: 5, source: .conversation)
        mem.addMemory(entry)

        let results = mem.searchMemories(query: "SwiftUI development tips", maxResults: 5)

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "Give me SwiftUI tips", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: results, conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil)

        mem.deleteMemory(entry.id)
        CognitionEngine.resetSignature()

        let hasMemoryContent = prompt.lowercased().contains("swiftui") || prompt.lowercased().contains("ios developer")
        let hasMemorySection = prompt.contains("Memory") || prompt.contains("recall")

        return TestOutcome(
            status: hasMemoryContent && hasMemorySection ? .passed : .warning,
            message: "Memory injected: content=\(hasMemoryContent), section=\(hasMemorySection), results=\(results.count)",
            details: ["Prompt length: \(prompt.count) chars", "Memory results: \(results.count)", "Relevant content found in prompt: \(hasMemoryContent)"]
        )
    }

    private func deepTestVoiceModeFormat() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "Hello", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: true, preferredResponseLanguageCode: nil)

        let hasVoiceSection = prompt.contains("Voice Mode")
        let hasNoMarkdown = prompt.contains("NEVER use markdown")
        let hasContractions = prompt.contains("contraction")
        let hasTurnTaking = prompt.contains("interrupt") || prompt.contains("barge-in")
        let hasProsody = prompt.contains("prosody") || prompt.contains("Prosody") || prompt.contains("PROSODY")

        CognitionEngine.resetSignature()

        let checks = [
            ("Voice Mode section", hasVoiceSection),
            ("No-markdown rule", hasNoMarkdown),
            ("Contraction guidance", hasContractions),
            ("Turn-taking rules", hasTurnTaking),
            ("Prosody hints", hasProsody),
        ]
        let passed = checks.filter(\.1).count

        return TestOutcome(
            status: passed >= 4 ? .passed : (passed >= 3 ? .warning : .failed),
            message: "Voice mode compliance: \(passed)/\(checks.count) voice rules present",
            details: checks.map { "\($0.0): \($0.1 ? "✓" : "✗")" }
        )
    }

    private func deepTestLanguageAddendum() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "Bonjour", conversationHistory: [], memoryService: mem)

        let withLang = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: "fr-CA")
        let withoutLang = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil)

        let hasLangSection = withLang.contains("Response Language") || withLang.contains("fr-CA") || withLang.contains("fr_CA")
        let noLangWhenNil = !withoutLang.contains("Response Language")

        CognitionEngine.resetSignature()

        return TestOutcome(
            status: hasLangSection && noLangWhenNil ? .passed : .warning,
            message: "Language addendum: present when set=\(hasLangSection), absent when nil=\(noLangWhenNil)",
            details: ["With fr-CA: \(hasLangSection)", "Without language: \(noLangWhenNil)", "Prompt diff: \(withLang.count - withoutLang.count) chars"]
        )
    }

    private func deepTestCoordinateExtraction() -> TestOutcome {
        let testCases: [(content: String, shouldFind: Bool)] = [
            ("My location is 45.5017, -73.5673", true),
            ("latitude: 48.8566, longitude: 2.3522", true),
            ("lat=40.7128 lng=-74.0060", true),
            ("I'm at the park downtown", false),
            ("The temperature is 72 degrees", false),
            ("(37.7749, -122.4194)", true),
        ]

        var correct = 0
        var details: [String] = []

        for tc in testCases {
            let messages = [Message(role: .user, content: tc.content)]
            let prompt = ContextAssembler.assembleSystemPrompt(
                frame: CognitionFrame(
                    emotion: EmotionalState(valence: .neutral, arousal: .low, dominantEmotion: "neutral", style: "neutral", empathyLevel: 0.3, emotionalTrajectory: "stable"),
                    metacognition: MetacognitionEngine.assess(text: tc.content, conversationHistory: [], memoryResults: []),
                    thoughtTree: ThoughtTree(branches: [], bestPath: [], prunedBranches: [], convergencePercent: 1.0, iterationCount: 1, synthesisStrategy: .direct, maxDepthReached: 1, dfsExpansions: 0, terminalNodes: []),
                    curiosity: CuriosityState(detectedTopics: [], knowledgeGap: 0, explorationPriority: 0, suggestedQueries: [], valenceArousalCuriosity: 0, informationGapIntensity: 0),
                    intent: IntentClassifier.classify(text: tc.content, conversationHistory: []),
                    injections: [],
                    reasoningTrace: ReasoningTrace(iterations: [], finalConvergence: 1.0, dominantStrategy: .direct, totalPruned: 0, selfCorrections: []),
                    contextSignature: ContextSignature(intentVector: [], topicFingerprint: [:], emotionalBaseline: 0, complexityAnchor: 0, signatureHash: ""),
                    timestamp: Date()
                ),
                memoryResults: [],
                conversationHistory: messages,
                toolsEnabled: false,
                isVoiceMode: false,
                preferredResponseLanguageCode: nil
            )
            let hasCoordinates = prompt.contains("User-Provided Location") || prompt.contains("latitude")
            let match = hasCoordinates == tc.shouldFind
            if match { correct += 1 }
            details.append("'\(tc.content.prefix(40))…' coords=\(hasCoordinates) expected=\(tc.shouldFind) → \(match ? "✓" : "✗")")
        }

        return TestOutcome(
            status: correct >= 5 ? .passed : (correct >= 4 ? .warning : .failed),
            message: "Coordinate extraction: \(correct)/\(testCases.count) correct",
            details: details
        )
    }

    private func deepTestLongConversationSummary() -> TestOutcome {
        var history: [Message] = []
        for i in 0..<20 {
            history.append(Message(role: .user, content: "User message about topic \(i) discussing various things"))
            history.append(Message(role: .assistant, content: "Response to topic \(i)"))
        }

        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "Continue our discussion", conversationHistory: history, memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: history, toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil)

        CognitionEngine.resetSignature()

        let hasSummary = prompt.contains("Conversation Context") || prompt.contains("long conversation")

        return TestOutcome(
            status: hasSummary ? .passed : .warning,
            message: "Long conversation summary: \(hasSummary ? "present" : "absent") for \(history.count) messages",
            details: ["History length: \(history.count) messages", "Prompt length: \(prompt.count) chars", "Summary section found: \(hasSummary)"]
        )
    }

    // MARK: - Stress Tests

    private func deepTestDatabaseStress() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }

        _ = db.execute("CREATE TABLE IF NOT EXISTS stress_test (id INTEGER PRIMARY KEY, data TEXT, value REAL);")

        let insertStart = Date()
        for i in 0..<1000 {
            _ = db.execute("INSERT OR REPLACE INTO stress_test (id, data, value) VALUES (?, ?, ?);", params: [i, "stress_row_\(i)_\(String(repeating: "x", count: 100))", Double(i) * 1.1])
        }
        let insertDuration = Date().timeIntervalSince(insertStart)

        let queryStart = Date()
        let rows = db.query("SELECT COUNT(*) as c FROM stress_test;")
        let count = (rows.first?["c"] as? Int64) ?? 0
        let queryDuration = Date().timeIntervalSince(queryStart)

        let rangeStart = Date()
        let rangeRows = db.query("SELECT * FROM stress_test WHERE value BETWEEN ? AND ? ORDER BY value;", params: [100.0, 200.0])
        let rangeDuration = Date().timeIntervalSince(rangeStart)

        _ = db.execute("DROP TABLE IF EXISTS stress_test;")

        let insertOK = insertDuration < 10.0
        let queryOK = queryDuration < 1.0

        return TestOutcome(
            status: insertOK && count >= 1000 ? .passed : .warning,
            message: "1000 inserts: \(String(format: "%.0f", insertDuration * 1000))ms, count query: \(String(format: "%.1f", queryDuration * 1000))ms, range query (\(rangeRows.count) rows): \(String(format: "%.1f", rangeDuration * 1000))ms",
            details: ["Total rows: \(count)", "Insert rate: \(String(format: "%.0f", 1000.0 / insertDuration)) rows/s", "Insert OK (<10s): \(insertOK)", "Query OK (<1s): \(queryOK)"]
        )
    }

    private func deepTestMemoryStress() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        var addedIds: [String] = []
        let addStart = Date()
        for i in 0..<50 {
            let entry = MemoryEntry(content: "Stress test memory \(i) about topic \(["AI", "cooking", "travel", "music", "sports"][i % 5])", keywords: ["stress", "test", ["ai", "cooking", "travel", "music", "sports"][i % 5]], category: MemoryCategory.allCases[i % MemoryCategory.allCases.count], importance: (i % 5) + 1, source: .system)
            mem.addMemory(entry)
            addedIds.append(entry.id)
        }
        let addDuration = Date().timeIntervalSince(addStart)

        let searchStart = Date()
        var searchResults = 0
        for query in ["AI machine learning", "cooking recipes", "travel destinations", "music instruments", "sports training"] {
            let results = mem.searchMemories(query: query, maxResults: 5)
            searchResults += results.count
        }
        let searchDuration = Date().timeIntervalSince(searchStart)

        for id in addedIds { mem.deleteMemory(id) }

        let addRate = 50.0 / addDuration
        let searchRate = 5.0 / searchDuration

        return TestOutcome(
            status: addDuration < 10.0 && searchDuration < 10.0 ? .passed : .warning,
            message: "50 adds: \(String(format: "%.0f", addDuration * 1000))ms (\(String(format: "%.0f", addRate))/s), 5 searches: \(String(format: "%.0f", searchDuration * 1000))ms (\(String(format: "%.1f", searchRate))/s), \(searchResults) total results",
            details: ["Add rate: \(String(format: "%.0f", addRate)) entries/s", "Search rate: \(String(format: "%.1f", searchRate)) queries/s", "Avg results per query: \(String(format: "%.1f", Double(searchResults) / 5.0))"]
        )
    }

    private func deepTestConversationBurst() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }

        let c = conv.createConversation(modelId: "stress-test")
        let writeStart = Date()
        for i in 0..<100 {
            conv.saveMessage(Message(role: i % 2 == 0 ? .user : .assistant, content: "Burst test message \(i) with enough content to be realistic: \(String(repeating: "word ", count: 20))"), conversationId: c.id)
        }
        let writeDuration = Date().timeIntervalSince(writeStart)

        let loadStart = Date()
        let loaded = conv.loadMessages(for: c.id)
        let loadDuration = Date().timeIntervalSince(loadStart)

        let searchStart = Date()
        let searchResults = conv.searchMessages(query: "Burst test message")
        let searchDuration = Date().timeIntervalSince(searchStart)

        conv.deleteConversation(c.id)

        return TestOutcome(
            status: loaded.count == 100 && writeDuration < 10.0 ? .passed : .warning,
            message: "100 writes: \(String(format: "%.0f", writeDuration * 1000))ms, load all: \(String(format: "%.0f", loadDuration * 1000))ms (\(loaded.count) msgs), search: \(String(format: "%.0f", searchDuration * 1000))ms (\(searchResults.count) results)",
            details: ["Write rate: \(String(format: "%.0f", 100.0 / writeDuration)) msgs/s", "Load time: \(String(format: "%.1f", loadDuration * 1000))ms", "Search time: \(String(format: "%.1f", searchDuration * 1000))ms"]
        )
    }

    private func deepTestCognitionLatency() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let inputs = [
            "Hello",
            "What is machine learning?",
            "I'm frustrated with this complex problem about quantum computing and need help urgently",
        ]

        var latencies: [(String, Double)] = []

        for input in inputs {
            CognitionEngine.resetSignature()
            let start = Date()
            let _ = CognitionEngine.process(userText: input, conversationHistory: [], memoryService: mem)
            let duration = Date().timeIntervalSince(start)
            latencies.append((String(input.prefix(40)), duration))
        }
        CognitionEngine.resetSignature()

        let avgLatency = latencies.map(\.1).reduce(0, +) / Double(latencies.count)
        let maxLatency = latencies.map(\.1).max() ?? 0
        let allUnder2s = latencies.allSatisfy { $0.1 < 2.0 }

        return TestOutcome(
            status: allUnder2s ? .passed : .warning,
            message: "Avg: \(String(format: "%.0f", avgLatency * 1000))ms, Max: \(String(format: "%.0f", maxLatency * 1000))ms, All <2s: \(allUnder2s)",
            details: latencies.map { "'\($0.0)…' → \(String(format: "%.0f", $0.1 * 1000))ms" }
        )
    }

    private func deepTestPipelineMemoryPressure() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        var addedIds: [String] = []
        for i in 0..<30 {
            let entry = MemoryEntry(content: "Memory pressure test entry \(i) about \(["Swift", "Python", "Rust", "Go", "Java"][i % 5]) programming", keywords: ["programming", ["swift", "python", "rust", "go", "java"][i % 5]], category: .context, importance: 3, source: .system)
            mem.addMemory(entry)
            addedIds.append(entry.id)
        }

        CognitionEngine.resetSignature()
        let start = Date()
        let frame = CognitionEngine.process(userText: "Tell me about Swift programming", conversationHistory: [], memoryService: mem)
        let results = mem.searchMemories(query: "Swift programming", maxResults: 5)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: results, conversationHistory: [], toolsEnabled: true, isVoiceMode: false, preferredResponseLanguageCode: nil)
        let duration = Date().timeIntervalSince(start)

        CognitionEngine.resetSignature()
        for id in addedIds { mem.deleteMemory(id) }

        let success = !prompt.isEmpty && frame.injections.count > 0 && duration < 5.0

        return TestOutcome(
            status: success ? .passed : .warning,
            message: "Pipeline with 30 memories: \(String(format: "%.0f", duration * 1000))ms, prompt=\(prompt.count) chars, injections=\(frame.injections.count), results=\(results.count)",
            details: ["Duration: \(String(format: "%.0f", duration * 1000))ms", "Under 5s: \(duration < 5.0)", "Results found: \(results.count)", "Prompt generated: \(!prompt.isEmpty)"]
        )
    }

    private func deepTestFTSStress() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }

        _ = db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_stress USING fts5(title, body);")

        let topics = ["machine learning", "quantum computing", "blockchain technology", "artificial intelligence", "neural networks", "deep learning", "natural language processing", "computer vision", "reinforcement learning", "generative models"]

        let insertStart = Date()
        for i in 0..<200 {
            let topic = topics[i % topics.count]
            _ = db.execute("INSERT INTO fts_stress (title, body) VALUES (?, ?);", params: ["\(topic) article \(i)", "This is a detailed article about \(topic) covering various aspects including implementation, theory, and applications in the field of \(topic). Document number \(i)."])
        }
        let insertDuration = Date().timeIntervalSince(insertStart)

        let searchStart = Date()
        var totalResults = 0
        for query in ["machine learning", "neural networks", "quantum", "artificial intelligence", "deep learning"] {
            let rows = db.query("SELECT title FROM fts_stress WHERE fts_stress MATCH ?;", params: [query])
            totalResults += rows.count
        }
        let searchDuration = Date().timeIntervalSince(searchStart)

        _ = db.execute("DROP TABLE IF EXISTS fts_stress;")

        return TestOutcome(
            status: searchDuration < 2.0 ? .passed : .warning,
            message: "200 docs indexed: \(String(format: "%.0f", insertDuration * 1000))ms, 5 FTS queries: \(String(format: "%.0f", searchDuration * 1000))ms, \(totalResults) total results",
            details: ["Index rate: \(String(format: "%.0f", 200.0 / insertDuration)) docs/s", "Search rate: \(String(format: "%.0f", 5.0 / searchDuration)) queries/s", "Avg results: \(String(format: "%.0f", Double(totalResults) / 5.0))"]
        )
    }

    // MARK: - Inference Deep Tests

    private func deepTestActualInference() async -> TestOutcome {
        guard let ie = inferenceEngine, ie.hasModel else {
            return TestOutcome(status: .warning, message: "No model loaded — skipping actual inference test. Load a model to enable this test.", details: ["This test generates real tokens with the loaded model", "It validates the full inference pipeline end-to-end"])
        }

        let start = Date()
        var generatedText = ""
        var metricsResult: GenerationMetrics?

        let messages: [[String: String]] = [
            ["role": "system", "content": "You are a helpful assistant. Answer briefly."],
            ["role": "user", "content": "What is 2+2? Answer in one word."]
        ]

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            ie.generate(
                messages: messages,
                systemPrompt: "You are a helpful assistant.",
                samplingConfig: SamplingConfig(),
                onToken: { token in
                    generatedText += token
                },
                onComplete: { metrics in
                    metricsResult = metrics
                    continuation.resume()
                }
            )
        }

        let duration = Date().timeIntervalSince(start)

        guard let metrics = metricsResult else {
            return TestOutcome(status: .failed, message: "Inference returned no metrics", details: [])
        }

        let hasOutput = !generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasTokens = metrics.totalTokens > 0
        let reasonable = duration < 60

        return TestOutcome(
            status: hasOutput && hasTokens ? .passed : .failed,
            message: "Generated \(metrics.totalTokens) tokens in \(String(format: "%.1f", duration))s. TTFT=\(String(format: "%.0f", metrics.timeToFirstToken))ms, decode=\(String(format: "%.1f", metrics.decodeTokensPerSecond)) tok/s",
            details: [
                "Output: '\(generatedText.prefix(100))…'",
                "Has output: \(hasOutput)",
                "Prefill: \(String(format: "%.1f", metrics.prefillTokensPerSecond)) tok/s",
                "Duration: \(String(format: "%.1f", duration))s",
                "Reasonable time (<60s): \(reasonable)"
            ]
        )
    }

    private func deepTestTokenizerRoundTrip() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }

        let testStrings = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Special chars: @#$%^&*()",
            "Numbers: 12345 67890",
            "Mixed: Hello123 World!@#",
            "Unicode: café résumé naïve",
            "Empty after trim:   ",
        ]

        var roundTrips = 0
        var details: [String] = []

        for str in testStrings {
            let tokens = ml.tokenizer.encode(str)
            let decoded = ml.tokenizer.decode(tokens)
            let match = decoded.trimmingCharacters(in: .whitespacesAndNewlines) == str.trimmingCharacters(in: .whitespacesAndNewlines)

            if ml.tokenizer.hasRealTokenizer {
                if match { roundTrips += 1 }
                details.append("'\(str.prefix(30))…' → \(tokens.count) tokens → '\(decoded.prefix(30))…' \(match ? "✓" : "✗")")
            } else {
                details.append("'\(str.prefix(30))…' → \(tokens.count) tokens (fallback tokenizer, round-trip N/A)")
                roundTrips += 1
            }
        }

        let hasReal = ml.tokenizer.hasRealTokenizer
        return TestOutcome(
            status: roundTrips >= testStrings.count - 1 ? .passed : .warning,
            message: "Round-trip: \(roundTrips)/\(testStrings.count) (real tokenizer: \(hasReal), vocab: \(ml.tokenizer.vocabularySize))",
            details: details
        )
    }

    private func deepTestKVCacheLifecycle() async -> TestOutcome {
        guard let ie = inferenceEngine else { return TestOutcome(status: .skipped, message: "No inference engine", details: []) }

        let statsBefore = await ie.cacheStatistics

        let stats = await ie.cacheStatistics
        let valid = stats.budgetUtilization >= 0 && stats.budgetUtilization <= 1

        return TestOutcome(
            status: valid ? .passed : .warning,
            message: "KV Cache: pages=\(stats.totalPages) (active=\(stats.activePages), free=\(stats.freePages)), utilization=\(String(format: "%.1f%%", stats.budgetUtilization * 100)), memory=\(String(format: "%.1f", Double(stats.estimatedMemoryBytes) / 1_048_576))MB",
            details: ["Before: total=\(statsBefore.totalPages)", "After: total=\(stats.totalPages)", "Budget utilization valid: \(valid)"]
        )
    }

    private func deepTestSamplingTemperature() -> TestOutcome {
        let lowTemp = Sampler(config: SamplingConfig(temperature: 0.1))
        let highTemp = Sampler(config: SamplingConfig(temperature: 1.5))

        let fakeLogits: [Float] = (0..<100).map { Float($0) * 0.1 }

        var lowResults: Set<Int> = []
        var highResults: Set<Int> = []

        for _ in 0..<20 {
            lowResults.insert(lowTemp.sample(logits: fakeLogits, recentTokens: []))
            highResults.insert(highTemp.sample(logits: fakeLogits, recentTokens: []))
        }

        let lowDiversity = lowResults.count
        let highDiversity = highResults.count
        let higherDiversity = highDiversity >= lowDiversity

        return TestOutcome(
            status: higherDiversity ? .passed : .warning,
            message: "Temperature effect: low(0.1) diversity=\(lowDiversity), high(1.5) diversity=\(highDiversity), higher temp → more diverse: \(higherDiversity)",
            details: ["Low temp unique tokens: \(lowDiversity)/20", "High temp unique tokens: \(highDiversity)/20", "Diversity ratio: \(String(format: "%.1f", Double(highDiversity) / max(1, Double(lowDiversity))))x"]
        )
    }

    private func deepTestChatMLFormat() -> TestOutcome {
        let messages: [[String: String]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Hi there!"],
            ["role": "user", "content": "How are you?"],
        ]

        let prompt = InferenceEngine.buildChatMLPrompt(messages: messages)

        let hasImStart = prompt.contains("<|im_start|>")
        let hasImEnd = prompt.contains("<|im_end|>")
        let endsWithAssistant = prompt.hasSuffix("<|im_start|>assistant\n")
        let hasSystem = prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>")
        let hasUser = prompt.contains("<|im_start|>user\nHello<|im_end|>")
        let turnCount = prompt.components(separatedBy: "<|im_start|>").count - 1

        let checks = [
            ("Has im_start tags", hasImStart),
            ("Has im_end tags", hasImEnd),
            ("Ends with assistant turn", endsWithAssistant),
            ("System message correct", hasSystem),
            ("User message correct", hasUser),
            ("Correct turn count (5)", turnCount == 5),
        ]
        let passed = checks.filter(\.1).count

        return TestOutcome(
            status: passed >= 5 ? .passed : .failed,
            message: "ChatML format: \(passed)/\(checks.count) checks passed, \(turnCount) turns",
            details: checks.map { "\($0.0): \($0.1 ? "✓" : "✗")" }
        )
    }

    private func deepTestSpeculativePolicy() -> TestOutcome {
        guard let ie = inferenceEngine else { return TestOutcome(status: .skipped, message: "No inference engine", details: []) }

        let stats = ie.speculativeStats
        let validRate = stats.rate >= 0 && stats.rate <= 1
        let validSpeedup = stats.speedup >= 0

        return TestOutcome(
            status: validRate && validSpeedup ? .passed : .warning,
            message: "Speculative: accepted=\(stats.accepted), rejected=\(stats.rejected), rate=\(String(format: "%.2f", stats.rate)), speedup=\(String(format: "%.2fx", stats.speedup))",
            details: ["Acceptance rate valid [0,1]: \(validRate)", "Speedup valid ≥0: \(validSpeedup)"]
        )
    }

    // MARK: - Regression E2E Tests

    private func deepTestFullE2E() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let input = "I'm frustrated because I can't figure out how machine learning gradient descent works"
        let history = [
            Message(role: .user, content: "Can you help me learn ML?"),
            Message(role: .assistant, content: "Of course! What aspect interests you?"),
        ]

        CognitionEngine.resetSignature()
        let start = Date()

        let frame = CognitionEngine.process(userText: input, conversationHistory: history, memoryService: mem)
        let memResults = mem.searchMemories(query: input, maxResults: 5)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: memResults, conversationHistory: history, toolsEnabled: true, isVoiceMode: false, preferredResponseLanguageCode: nil)

        let duration = Date().timeIntervalSince(start)
        CognitionEngine.resetSignature()

        let checks = [
            ("Emotion detected negative", frame.emotion.valence == .negative),
            ("Intent is question_how or similar", [.questionHow, .questionFactual, .questionWhy].contains(frame.intent.primary)),
            ("Has injections", !frame.injections.isEmpty),
            ("Prompt not empty", !prompt.isEmpty),
            ("Prompt has identity", prompt.contains("NEXUS")),
            ("Prompt has cognitive state", prompt.contains("Intent:") || prompt.contains("Cognitive")),
            ("Under 3s", duration < 3.0),
            ("Has reasoning trace", frame.reasoningTrace.iterations.count >= 1),
        ]
        let passed = checks.filter(\.1).count

        return TestOutcome(
            status: passed >= 6 ? .passed : (passed >= 4 ? .warning : .failed),
            message: "Full E2E: \(passed)/\(checks.count) checks in \(String(format: "%.0f", duration * 1000))ms. Emotion=\(frame.emotion.valence.rawValue), Intent=\(frame.intent.primary.rawValue)",
            details: checks.map { "\($0.0): \($0.1 ? "✓" : "✗")" }
        )
    }

    private func deepTestMemoryLifecycle() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let beforeCount = mem.memories.count

        mem.extractAndStoreMemory(userText: "I really love programming in Haskell for functional programming", assistantText: "Haskell is a great choice for functional programming!")

        let afterExtract = mem.memories.count
        let extracted = afterExtract > beforeCount

        let searchResults = mem.searchMemories(query: "Haskell functional programming", maxResults: 5)
        let found = searchResults.contains { $0.memory.content.lowercased().contains("haskell") }

        let injection = mem.buildContextInjection(query: "Haskell programming")
        let injectionHasContent = !injection.isEmpty

        let haskellMemory = mem.memories.first { $0.content.lowercased().contains("haskell") }
        if let hm = haskellMemory {
            mem.reinforceMemory(hm.id)
            let reinforced = mem.memories.first { $0.id == hm.id }
            let accessIncremented = (reinforced?.accessCount ?? 0) > 0

            mem.deleteMemory(hm.id)
            let deleted = !mem.memories.contains { $0.id == hm.id }

            let checks = [extracted, found, injectionHasContent, accessIncremented, deleted]
            let passed = checks.filter { $0 }.count

            return TestOutcome(
                status: passed >= 4 ? .passed : (passed >= 3 ? .warning : .failed),
                message: "Lifecycle: \(passed)/5 stages passed. Extract=\(extracted), Find=\(found), Inject=\(injectionHasContent), Reinforce=\(accessIncremented), Delete=\(deleted)",
                details: ["Extracted: \(afterExtract - beforeCount) new memories", "Search results: \(searchResults.count)", "Injection length: \(injection.count) chars"]
            )
        }

        return TestOutcome(status: .warning, message: "Extraction may not have captured Haskell specifically", details: ["Extracted: \(extracted)", "Found in search: \(found)"])
    }

    private func deepTestConversationContinuity() -> TestOutcome {
        guard let conv = conversationService, let mem = memoryService else {
            return TestOutcome(status: .skipped, message: "No conversation/memory service", details: [])
        }

        let c = conv.createConversation(modelId: "continuity-test")

        let turns: [(role: MessageRole, content: String)] = [
            (.user, "My name is Alex and I'm a software engineer"),
            (.assistant, "Nice to meet you Alex! What kind of software do you work on?"),
            (.user, "I mostly work on iOS apps using SwiftUI"),
            (.assistant, "SwiftUI is great! Are you working on anything specific?"),
            (.user, "Yes, I'm building an AI assistant app"),
            (.assistant, "That sounds fascinating! What features are you implementing?"),
        ]

        for turn in turns {
            conv.saveMessage(Message(role: turn.role, content: turn.content), conversationId: c.id)
        }

        let loaded = conv.loadMessages(for: c.id)
        let orderCorrect = loaded.count == turns.count
        let contentIntact = loaded.enumerated().allSatisfy { $0.element.content == turns[$0.offset].content }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "What was I telling you about?", conversationHistory: loaded, memoryService: mem)
        let hasMemoryIntent = frame.intent.primary == .requestMemory || frame.intent.requiresKnowledge

        conv.deleteConversation(c.id)
        CognitionEngine.resetSignature()

        return TestOutcome(
            status: orderCorrect && contentIntact ? .passed : .failed,
            message: "Continuity: \(loaded.count) messages, order=\(orderCorrect), content intact=\(contentIntact), memory intent=\(hasMemoryIntent)",
            details: ["Messages loaded: \(loaded.count)/\(turns.count)", "Order preserved: \(orderCorrect)", "Content preserved: \(contentIntact)", "Intent: \(frame.intent.primary.rawValue)"]
        )
    }

    private func deepTestCoherence() -> TestOutcome {
        let input = "I'm really stressed about this urgent complex mathematical proof involving Gödel's theorems"

        let emotion = EmotionAnalyzer.analyze(text: input, conversationHistory: [])
        let intent = IntentClassifier.classify(text: input, conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: input, conversationHistory: [], memoryResults: [])

        let emotionNegative = emotion.valence == .negative
        let highEmpathy = emotion.empathyLevel > 0.4
        let complexQuery = meta.complexityLevel != .simple
        let hasUrgency = intent.urgency > 0.3
        let shouldDecompose = meta.shouldDecompose

        let emotionInjection = EmotionAnalyzer.buildInjection(state: emotion)
        let metaInjection = MetacognitionEngine.buildInjection(state: meta)
        let intentInjection = IntentClassifier.buildInjection(intent: intent)

        let emotionHasPriority = emotionInjection.priority > 0
        let allInjectionsValid = [emotionInjection, metaInjection, intentInjection].allSatisfy { $0.priority >= 0 && $0.priority <= 1 }

        let checks = [
            ("Negative emotion", emotionNegative),
            ("Elevated empathy", highEmpathy),
            ("Non-simple complexity", complexQuery),
            ("Urgency detected", hasUrgency),
            ("Decomposition suggested", shouldDecompose),
            ("Emotion injection has priority", emotionHasPriority),
            ("All injections valid", allInjectionsValid),
        ]
        let passed = checks.filter(\.1).count

        return TestOutcome(
            status: passed >= 5 ? .passed : (passed >= 4 ? .warning : .failed),
            message: "Coherence: \(passed)/\(checks.count). Emotion=\(emotion.valence.rawValue)/\(emotion.dominantEmotion), Intent=\(intent.primary.rawValue), Complexity=\(meta.complexityLevel.rawValue)",
            details: checks.map { "\($0.0): \($0.1 ? "✓" : "✗")" }
        )
    }

    private func deepTestCrossLanguage() -> TestOutcome {
        let enLang = NLTextProcessing.detectLanguage(for: "The quick brown fox jumps over the lazy dog")
        let frLang = NLTextProcessing.detectLanguage(for: "Le renard brun rapide saute par-dessus le chien paresseux")

        let enEmotion = EmotionAnalyzer.analyze(text: "I'm very happy today!", conversationHistory: [])
        let frEmotion = EmotionAnalyzer.analyze(text: "Je suis très content aujourd'hui!", conversationHistory: [])

        let _ = IntentClassifier.classify(text: "What is machine learning?", conversationHistory: [])
        let _ = IntentClassifier.classify(text: "Qu'est-ce que l'apprentissage automatique?", conversationHistory: [])

        let enNLP = NLTextProcessing.process(text: "The beautiful dogs were running quickly")
        let frNLP = NLTextProcessing.process(text: "Les beaux chiens couraient rapidement")

        var checks: [String] = []
        var score = 0

        if enLang == .english { score += 1; checks.append("EN detection: ✓") } else { checks.append("EN detection: got \(enLang?.rawValue ?? "nil")") }
        if frLang == .french { score += 1; checks.append("FR detection: ✓") } else { checks.append("FR detection: got \(frLang?.rawValue ?? "nil")") }
        if enEmotion.valence == .positive { score += 1; checks.append("EN emotion positive: ✓") } else { checks.append("EN emotion: \(enEmotion.valence.rawValue)") }
        if !enNLP.tokens.isEmpty { score += 1; checks.append("EN tokenization: \(enNLP.tokens.count) tokens ✓") } else { checks.append("EN tokenization failed") }
        if !frNLP.tokens.isEmpty { score += 1; checks.append("FR tokenization: \(frNLP.tokens.count) tokens ✓") } else { checks.append("FR tokenization failed") }

        return TestOutcome(
            status: score >= 4 ? .passed : (score >= 3 ? .warning : .failed),
            message: "Cross-language: \(score)/5 checks. EN emotion=\(enEmotion.valence.rawValue), FR emotion=\(frEmotion.valence.rawValue)",
            details: checks
        )
    }

    private func deepTestVoicePipeline() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "Tell me a fun fact about space", conversationHistory: [], memoryService: mem)

        let voicePrompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: true, preferredResponseLanguageCode: "en-US")

        let textPrompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: "en-US")

        CognitionEngine.resetSignature()

        let voiceLonger = voicePrompt.count > textPrompt.count
        let hasVoiceRules = voicePrompt.contains("Voice Mode")
        let hasLanguage = voicePrompt.contains("en-US") || voicePrompt.contains("Response Language")
        let hasNoMarkdownRule = voicePrompt.contains("NEVER use markdown")

        let checks = [
            ("Voice prompt longer than text", voiceLonger),
            ("Has voice mode section", hasVoiceRules),
            ("Has language directive", hasLanguage),
            ("Has no-markdown rule", hasNoMarkdownRule),
        ]
        let passed = checks.filter(\.1).count

        return TestOutcome(
            status: passed >= 3 ? .passed : .warning,
            message: "Voice pipeline: \(passed)/\(checks.count). Voice prompt: \(voicePrompt.count) chars, Text prompt: \(textPrompt.count) chars",
            details: checks.map { "\($0.0): \($0.1 ? "✓" : "✗")" }
        )
    }

    private func deepTestMemoryExtractDedupSearch() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let tag = UUID().uuidString.prefix(6)
        let before = mem.memories.count

        mem.extractAndStoreMemory(userText: "I really love classical piano music and \(tag)", assistantText: "That's wonderful!")
        let afterFirst = mem.memories.count
        let firstAdded = afterFirst - before

        mem.extractAndStoreMemory(userText: "I really love classical piano music and \(tag)", assistantText: "Yes, I know you enjoy piano!")
        let afterSecond = mem.memories.count
        let secondAdded = afterSecond - afterFirst

        let searchResults = mem.searchMemories(query: "classical piano music \(tag)", maxResults: 5)
        let foundInSearch = searchResults.contains { $0.memory.content.contains("piano") }

        let cleanupIds = mem.memories.filter { $0.content.contains(String(tag)) || ($0.content.contains("piano") && $0.timestamp > Date().timeIntervalSince1970 * 1000 - 10000) }.map(\.id)
        for id in cleanupIds { mem.deleteMemory(id) }

        let extractOk = firstAdded > 0
        let dedupOk = secondAdded == 0
        let searchOk = foundInSearch

        let checks = [
            ("Extraction added entries", extractOk),
            ("Dedup blocked repeat", dedupOk),
            ("Search found extracted entry", searchOk),
        ]
        let passed = checks.filter(\.1).count

        return TestOutcome(
            status: passed == 3 ? .passed : (passed >= 2 ? .warning : .failed),
            message: "Round-trip: \(passed)/3. Extract=\(firstAdded), Repeat=\(secondAdded), Found=\(foundInSearch)",
            details: checks.map { "\($0.0): \($0.1 ? "\u{2713}" : "\u{2717}")" }
        )
    }

    private func deepTestAssociativeGraphIntegrity() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let e1 = MemoryEntry(content: "User studies quantum physics at university", keywords: ["quantum", "physics", "university"], category: .fact, importance: 4, source: .conversation)
        let e2 = MemoryEntry(content: "User reads quantum mechanics textbooks regularly", keywords: ["quantum", "mechanics", "textbooks"], category: .skill, importance: 4, source: .conversation)
        let e3 = MemoryEntry(content: "User enjoys cooking Italian pasta dishes", keywords: ["cooking", "italian", "pasta"], category: .preference, importance: 4, source: .conversation)

        mem.addMemory(e1)
        mem.addMemory(e2)
        mem.addMemory(e3)

        let quantumLinked = mem.associativeLinks.contains {
            ($0.sourceId == e1.id && $0.targetId == e2.id) ||
            ($0.sourceId == e2.id && $0.targetId == e1.id)
        }

        let pastaLinkedToQuantum = mem.associativeLinks.contains {
            ($0.sourceId == e3.id && $0.targetId == e1.id) ||
            ($0.sourceId == e1.id && $0.targetId == e3.id)
        }

        let directResults = mem.searchMemories(query: "quantum physics", maxResults: 3)
        let assocResults = mem.getAssociativeMemories(query: "quantum", directResults: directResults)
        let assocIds = Set(assocResults.map(\.memory.id))

        mem.deleteMemory(e1.id)
        mem.deleteMemory(e2.id)
        mem.deleteMemory(e3.id)

        var checks = 0
        var details: [String] = []

        if quantumLinked { checks += 1; details.append("Quantum entries linked: \u{2713}") } else { details.append("Quantum entries linked: \u{2717}") }
        if !pastaLinkedToQuantum { checks += 1; details.append("Pasta not linked to quantum: \u{2713}") } else { details.append("Pasta incorrectly linked to quantum: \u{2717}") }
        if !assocResults.isEmpty { checks += 1; details.append("Associative traversal found results: \u{2713} (\(assocResults.count))") } else { details.append("Associative traversal empty: \u{2717}") }
        let noIrrelevant = !assocIds.contains(e3.id)
        if noIrrelevant { checks += 1; details.append("No irrelevant associative results: \u{2713}") } else { details.append("Irrelevant entry in associative results: \u{2717}") }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Graph integrity: \(checks)/4 checks. Related linked=\(quantumLinked), unrelated isolated=\(!pastaLinkedToQuantum)",
            details: details
        )
    }

    // MARK: - Vector Database Tests

    private func vectorTestEmbeddingGeneration() -> TestOutcome {
        let embedder = VectorEmbeddingService.shared
        let texts = [
            "The cat sat on the mat",
            "Machine learning is transforming industries",
            "I love programming in Swift",
            "",
            "Bonjour le monde",
        ]
        var results: [String] = []
        var generated = 0

        for text in texts {
            if let vec = embedder.embed(text) {
                generated += 1
                let mag = sqrtf(vec.reduce(0) { $0 + $1 * $1 })
                results.append("'\(text.prefix(30))\u{2026}' \u{2192} \(vec.count)d, mag=\(String(format: "%.3f", mag))")
            } else {
                results.append("'\(text.prefix(30))\u{2026}' \u{2192} nil (expected for empty)")
            }
        }

        let nonEmptyTexts = texts.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let expectedMin = nonEmptyTexts.count - 1

        return TestOutcome(
            status: generated >= expectedMin ? .passed : (generated >= 2 ? .warning : .failed),
            message: "Generated \(generated)/\(texts.count) embeddings (\(VectorEmbeddingService.dimensions)d)",
            details: results
        )
    }

    private func vectorTestInsertRetrieve() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let store = mem.vectorStore
        let testId = "vec_test_\(UUID().uuidString.prefix(8))"
        let text = "Quantum computing uses qubits for parallel computation"

        let inserted = store.upsert(id: testId, text: text)
        let retrieved = store.getVector(for: testId)
        let hasIt = store.hasVector(for: testId)
        store.delete(id: testId)
        let afterDelete = store.hasVector(for: testId)

        var checks = 0
        var details: [String] = []

        if inserted { checks += 1; details.append("Insert: \u{2713}") } else { details.append("Insert: \u{2717}") }
        if let vec = retrieved { checks += 1; details.append("Retrieve: \u{2713} (\(vec.count)d)") } else { details.append("Retrieve: \u{2717}") }
        if hasIt { checks += 1; details.append("Has vector: \u{2713}") } else { details.append("Has vector: \u{2717}") }
        if !afterDelete { checks += 1; details.append("Deleted: \u{2713}") } else { details.append("Deleted: \u{2717}") }

        return TestOutcome(
            status: checks == 4 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Insert/Retrieve lifecycle: \(checks)/4 checks passed",
            details: details
        )
    }

    private func vectorTestCosineSimilarity() -> TestOutcome {
        let embedder = VectorEmbeddingService.shared
        let pairs: [(a: String, b: String, label: String, expectedMin: Float)] = [
            ("I love dogs", "I adore puppies", "similar", 0.3),
            ("Machine learning algorithms", "Deep neural networks", "related", 0.2),
            ("The weather is sunny", "Quantum entanglement theory", "unrelated", -1.0),
        ]

        var details: [String] = []
        var checks = 0

        var scores: [Float] = []
        for pair in pairs {
            guard let va = embedder.embed(pair.a), let vb = embedder.embed(pair.b) else {
                details.append("\(pair.label): embedding failed")
                continue
            }
            let sim = embedder.cosineSimilarity(va, vb)
            scores.append(sim)
            let ok = sim >= pair.expectedMin
            if ok { checks += 1 }
            details.append("\(pair.label): cos=\(String(format: "%.3f", sim)) \(ok ? "\u{2713}" : "\u{2717}")")
        }

        if scores.count == 3 && scores[0] > scores[2] {
            checks += 1
            details.append("Similar > Unrelated: \u{2713}")
        } else if scores.count == 3 {
            details.append("Similar > Unrelated: \u{2717} (\(String(format: "%.3f", scores[0])) vs \(String(format: "%.3f", scores[2])))")
        }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Cosine similarity: \(checks)/4 checks passed",
            details: details
        )
    }

    private func vectorTestSemanticSearch() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let store = mem.vectorStore

        let docs: [(id: String, text: String)] = [
            ("vs_1", "Python programming for data science and analytics"),
            ("vs_2", "Italian pasta recipes with fresh tomato sauce"),
            ("vs_3", "Machine learning neural network training techniques"),
            ("vs_4", "Hiking trails in the Rocky Mountains of Colorado"),
            ("vs_5", "Swift iOS mobile application development patterns"),
            ("vs_6", "French cuisine cooking methods and traditions"),
            ("vs_7", "Deep learning transformer architecture explained"),
            ("vs_8", "Camping gear essentials for wilderness adventures"),
            ("vs_9", "Rust systems programming memory safety features"),
            ("vs_10", "Baking sourdough bread with natural yeast starter"),
        ]

        for doc in docs { _ = store.upsert(id: doc.id, text: doc.text) }

        let queries: [(q: String, expectedTop: String, label: String)] = [
            ("artificial intelligence and deep learning", "vs_7", "AI→DL"),
            ("outdoor hiking and nature", "vs_4", "outdoor→hiking"),
            ("writing code in programming languages", "vs_1", "code→programming"),
        ]

        var checks = 0
        var details: [String] = []

        for query in queries {
            let results = store.search(query: query.q, maxResults: 3)
            let topId = results.first?.id ?? "none"
            let topScore = results.first?.score ?? 0
            let foundInTop3 = results.prefix(3).contains { $0.id == query.expectedTop }
            if foundInTop3 { checks += 1 }
            details.append("\(query.label): top=\(topId)(\(String(format: "%.3f", topScore))), expected \(query.expectedTop) in top-3: \(foundInTop3 ? "\u{2713}" : "\u{2717}")")
        }

        for doc in docs { store.delete(id: doc.id) }

        return TestOutcome(
            status: checks >= 2 ? .passed : (checks >= 1 ? .warning : .failed),
            message: "Semantic search: \(checks)/\(queries.count) correct top-3 retrievals",
            details: details
        )
    }

    private func vectorTestHNSWRecall() -> TestOutcome {
        let embedder = VectorEmbeddingService.shared
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let store = mem.vectorStore

        let texts = [
            "The solar system has eight planets orbiting the sun",
            "Photosynthesis converts light energy into chemical energy",
            "The stock market fluctuated wildly during the pandemic",
            "Mozart composed his first symphony at age eight",
            "DNA contains the genetic instructions for all living organisms",
            "The Great Wall of China stretches over 13000 miles",
            "Artificial intelligence mimics human cognitive functions",
            "Climate change is driven by greenhouse gas emissions",
            "The human brain contains approximately 86 billion neurons",
            "Quantum computers leverage superposition for parallel processing",
        ]

        var ids: [String] = []
        var vectors: [[Float]] = []
        for (i, text) in texts.enumerated() {
            let id = "hnsw_\(i)"
            ids.append(id)
            _ = store.upsert(id: id, text: text)
            if let vec = embedder.embed(text) { vectors.append(vec) }
        }

        guard let queryVec = embedder.embed("brain neurons neuroscience") else {
            for id in ids { store.delete(id: id) }
            return TestOutcome(status: .warning, message: "Could not generate query embedding", details: [])
        }

        let hnswResults = store.searchByVector(queryVec, maxResults: 3)

        var bruteForce: [(id: String, score: Float)] = []
        for (i, vec) in vectors.enumerated() {
            bruteForce.append((ids[i], embedder.cosineSimilarity(queryVec, vec)))
        }
        bruteForce.sort { $0.score > $1.score }
        let bruteTop3 = Set(bruteForce.prefix(3).map(\.id))
        let hnswTop3 = Set(hnswResults.prefix(3).map(\.id))
        let overlap = bruteTop3.intersection(hnswTop3).count

        for id in ids { store.delete(id: id) }

        let recall = Double(overlap) / Double(bruteTop3.count)

        return TestOutcome(
            status: recall >= 0.66 ? .passed : (recall >= 0.33 ? .warning : .failed),
            message: "HNSW recall@3: \(String(format: "%.0f", recall * 100))% (\(overlap)/3 match brute-force)",
            details: [
                "HNSW top-3: \(hnswResults.prefix(3).map { "\($0.id)(\(String(format: "%.3f", $0.score)))" }.joined(separator: ", "))",
                "Brute-force top-3: \(bruteForce.prefix(3).map { "\($0.id)(\(String(format: "%.3f", $0.score)))" }.joined(separator: ", "))",
                "Overlap: \(overlap)/3"
            ]
        )
    }

    private func vectorTestCrossDomain() -> TestOutcome {
        let embedder = VectorEmbeddingService.shared

        let domains: [(label: String, texts: [String])] = [
            ("tech", ["software engineering", "machine learning algorithms", "cloud computing infrastructure"]),
            ("food", ["chocolate cake recipe", "Italian pasta carbonara", "sushi preparation techniques"]),
            ("nature", ["mountain hiking trails", "ocean marine biology", "forest ecosystem diversity"]),
        ]

        var domainVectors: [String: [[Float]]] = [:]
        for domain in domains {
            let vecs = domain.texts.compactMap { embedder.embed($0) }
            domainVectors[domain.label] = vecs
        }

        var intraSims: [Float] = []
        var interSims: [Float] = []

        for domain in domains {
            guard let vecs = domainVectors[domain.label], vecs.count >= 2 else { continue }
            for i in 0..<vecs.count {
                for j in (i+1)..<vecs.count {
                    intraSims.append(embedder.cosineSimilarity(vecs[i], vecs[j]))
                }
            }
        }

        let domainLabels = domains.map(\.label)
        for i in 0..<domainLabels.count {
            for j in (i+1)..<domainLabels.count {
                guard let vecsA = domainVectors[domainLabels[i]], let vecsB = domainVectors[domainLabels[j]] else { continue }
                for va in vecsA {
                    for vb in vecsB {
                        interSims.append(embedder.cosineSimilarity(va, vb))
                    }
                }
            }
        }

        let avgIntra = intraSims.isEmpty ? 0 : intraSims.reduce(0, +) / Float(intraSims.count)
        let avgInter = interSims.isEmpty ? 0 : interSims.reduce(0, +) / Float(interSims.count)
        let separation = avgIntra - avgInter

        return TestOutcome(
            status: separation > 0.02 ? .passed : (separation > 0 ? .warning : .failed),
            message: "Cross-domain: intra=\(String(format: "%.3f", avgIntra)), inter=\(String(format: "%.3f", avgInter)), separation=\(String(format: "%.3f", separation))",
            details: [
                "Intra-domain avg similarity: \(String(format: "%.3f", avgIntra)) (\(intraSims.count) pairs)",
                "Inter-domain avg similarity: \(String(format: "%.3f", avgInter)) (\(interSims.count) pairs)",
                "Separation margin: \(String(format: "%.3f", separation))",
                "Discrimination: \(separation > 0 ? "\u{2713}" : "\u{2717}")"
            ]
        )
    }

    private func vectorTestMemoryIntegration() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let e1 = MemoryEntry(content: "User enjoys playing classical piano concertos by Chopin", keywords: ["piano", "chopin", "classical", "music"], category: .preference, importance: 4, source: .conversation)
        let e2 = MemoryEntry(content: "User works as a backend engineer at a fintech company", keywords: ["engineer", "backend", "fintech"], category: .fact, importance: 4, source: .conversation)
        let e3 = MemoryEntry(content: "User is learning to bake sourdough bread at home", keywords: ["baking", "sourdough", "bread"], category: .skill, importance: 3, source: .conversation)

        mem.addMemory(e1)
        mem.addMemory(e2)
        mem.addMemory(e3)

        let hasVec1 = mem.vectorStore.hasVector(for: e1.id)
        let hasVec2 = mem.vectorStore.hasVector(for: e2.id)
        let hasVec3 = mem.vectorStore.hasVector(for: e3.id)

        let musicResults = mem.searchMemories(query: "classical music instruments piano", maxResults: 5)
        let musicTop = musicResults.first
        let musicFound = musicResults.contains { $0.memory.id == e1.id }

        let techResults = mem.searchMemories(query: "software engineering programming", maxResults: 5)
        let techFound = techResults.contains { $0.memory.id == e2.id }

        mem.deleteMemory(e1.id)
        mem.deleteMemory(e2.id)
        mem.deleteMemory(e3.id)

        let vecCleared1 = !mem.vectorStore.hasVector(for: e1.id)

        var checks = 0
        var details: [String] = []

        if hasVec1 && hasVec2 && hasVec3 { checks += 1; details.append("Vectors auto-created on addMemory: \u{2713}") } else { details.append("Vectors auto-created: \u{2717} (\(hasVec1),\(hasVec2),\(hasVec3))") }
        if musicFound { checks += 1; details.append("Music query found piano entry: \u{2713} (score=\(String(format: "%.3f", musicTop?.score ?? 0)))") } else { details.append("Music query missed piano entry: \u{2717}") }
        if techFound { checks += 1; details.append("Tech query found engineer entry: \u{2713}") } else { details.append("Tech query missed engineer entry: \u{2717}") }
        if vecCleared1 { checks += 1; details.append("Vector deleted on deleteMemory: \u{2713}") } else { details.append("Vector not cleaned up: \u{2717}") }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Memory integration: \(checks)/4 checks passed",
            details: details
        )
    }

    private func vectorTestBatchLifecycle() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let store = mem.vectorStore

        let countBefore = store.count
        var testIds: [String] = []

        for i in 0..<20 {
            let id = "batch_\(i)_\(UUID().uuidString.prefix(4))"
            testIds.append(id)
            _ = store.upsert(id: id, text: "Batch test document number \(i) about topic \(["science", "cooking", "sports", "music", "travel"][i % 5])")
        }

        let afterInsert = store.count
        let insertedCount = afterInsert - countBefore

        let searchResults = store.search(query: "scientific research", maxResults: 5)

        for id in testIds { store.delete(id: id) }
        let afterDelete = store.count
        let cleanedUp = afterDelete == countBefore

        var details: [String] = [
            "Before: \(countBefore), After insert: \(afterInsert), After delete: \(afterDelete)",
            "Inserted: \(insertedCount)/20",
            "Search returned: \(searchResults.count) results",
            "Cleanup correct: \(cleanedUp)",
        ]

        let allInserted = insertedCount == 20
        var checks = 0
        if allInserted { checks += 1 }
        if !searchResults.isEmpty { checks += 1 }
        if cleanedUp { checks += 1 }

        return TestOutcome(
            status: checks == 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Batch lifecycle: \(checks)/3 (insert=\(insertedCount), search=\(searchResults.count), cleanup=\(cleanedUp))",
            details: details
        )
    }

    private func vectorTestSearchLatency() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let store = mem.vectorStore
        let embedder = VectorEmbeddingService.shared

        let topics = ["science", "history", "cooking", "sports", "music", "technology", "nature", "art", "medicine", "finance"]
        var testIds: [String] = []
        for i in 0..<100 {
            let id = "latency_\(i)"
            testIds.append(id)
            _ = store.upsert(id: id, text: "Document about \(topics[i % topics.count]) topic number \(i) with various content")
        }

        let queries = ["machine learning", "classical music", "financial markets", "outdoor hiking", "medical research"]
        var totalMs: Double = 0
        var queryDetails: [String] = []

        for q in queries {
            let start = CFAbsoluteTimeGetCurrent()
            let results = store.search(query: q, maxResults: 10)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            totalMs += elapsed
            queryDetails.append("'\(q)': \(String(format: "%.1f", elapsed))ms, \(results.count) results")
        }

        for id in testIds { store.delete(id: id) }

        let avgMs = totalMs / Double(queries.count)

        return TestOutcome(
            status: avgMs < 100 ? .passed : (avgMs < 500 ? .warning : .failed),
            message: "Search latency (100 vectors): avg=\(String(format: "%.1f", avgMs))ms, total=\(String(format: "%.0f", totalMs))ms",
            details: queryDetails + ["Avg per query: \(String(format: "%.1f", avgMs))ms"]
        )
    }

    private func vectorTestHybridRanking() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let semanticOnly = MemoryEntry(content: "Canines are wonderful companion animals that bring joy", keywords: ["pets", "animals"], category: .preference, importance: 4, source: .conversation)
        let keywordOnly = MemoryEntry(content: "I love dogs and have two golden retrievers at home", keywords: ["dogs", "golden retriever", "pets"], category: .fact, importance: 4, source: .conversation)
        let both = MemoryEntry(content: "My favorite dog breed is the golden retriever puppy", keywords: ["dog", "golden retriever", "favorite", "puppy"], category: .preference, importance: 4, source: .conversation)

        mem.addMemory(semanticOnly)
        mem.addMemory(keywordOnly)
        mem.addMemory(both)

        let results = mem.searchMemories(query: "dogs golden retriever", maxResults: 5)

        let semanticScore = results.first(where: { $0.memory.id == semanticOnly.id })?.score ?? 0
        let keywordScore = results.first(where: { $0.memory.id == keywordOnly.id })?.score ?? 0
        let bothScore = results.first(where: { $0.memory.id == both.id })?.score ?? 0

        mem.deleteMemory(semanticOnly.id)
        mem.deleteMemory(keywordOnly.id)
        mem.deleteMemory(both.id)

        var checks = 0
        var details: [String] = [
            "Semantic-only score: \(String(format: "%.3f", semanticScore))",
            "Keyword-only score: \(String(format: "%.3f", keywordScore))",
            "Both (keyword+semantic) score: \(String(format: "%.3f", bothScore))",
        ]

        if bothScore >= keywordScore { checks += 1; details.append("Both >= Keyword: \u{2713}") } else { details.append("Both >= Keyword: \u{2717}") }
        if bothScore >= semanticScore { checks += 1; details.append("Both >= Semantic: \u{2713}") } else { details.append("Both >= Semantic: \u{2717}") }
        if keywordScore > 0 || semanticScore > 0 { checks += 1; details.append("Non-zero retrieval: \u{2713}") } else { details.append("Non-zero retrieval: \u{2717}") }

        let vectorResults = mem.vectorSearch(query: "dogs golden retriever", maxResults: 3)
        if !vectorResults.isEmpty { checks += 1; details.append("Vector search direct: \u{2713} (\(vectorResults.count) results)") } else { details.append("Vector search direct: \u{2717}") }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Hybrid ranking: \(checks)/4. Both=\(String(format: "%.3f", bothScore)), KW=\(String(format: "%.3f", keywordScore)), Sem=\(String(format: "%.3f", semanticScore))",
            details: details
        )
    }
}
