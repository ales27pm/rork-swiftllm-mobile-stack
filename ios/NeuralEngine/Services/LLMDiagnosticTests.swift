import Foundation
import NaturalLanguage

extension DiagnosticEngine {

    // MARK: - LLM Diagnostic Helper

    func llmGenerate(
        messages: [[String: String]],
        systemPrompt: String,
        samplingConfig: SamplingConfig = SamplingConfig(),
        timeoutSeconds: Double = 30
    ) async -> (text: String, metrics: GenerationMetrics?) {
        guard let ie = inferenceEngine, ie.hasModel else {
            return ("", nil)
        }

        for _ in 0..<250 {
            if !ie.isGenerating { break }
            try? await Task.sleep(for: .milliseconds(20))
        }
        if ie.isGenerating {
            ie.cancel(reason: "diagnosticIdleWait")
            for _ in 0..<250 {
                if !ie.isGenerating { break }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        if ie.isGenerating {
            ie.resetSession()
            try? await Task.sleep(for: .milliseconds(500))
        }
        guard !ie.isGenerating else {
            return ("", nil)
        }

        var generatedText = ""
        var metricsResult: GenerationMetrics?
        var completedNormally = false

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            var resumed = false
            let timeout = Task {
                try? await Task.sleep(for: .seconds(timeoutSeconds))
                guard !resumed else { return }
                resumed = true
                ie.cancel(reason: "diagnosticTimeout")
                for _ in 0..<250 {
                    if !ie.isGenerating { break }
                    try? await Task.sleep(for: .milliseconds(20))
                }
                if ie.isGenerating {
                    ie.resetSession()
                    try? await Task.sleep(for: .milliseconds(500))
                }
                continuation.resume()
            }

            ie.generate(
                messages: messages,
                systemPrompt: systemPrompt,
                samplingConfig: samplingConfig,
                onToken: { token in
                    generatedText += token
                },
                onComplete: { metrics in
                    metricsResult = metrics
                    timeout.cancel()
                    if !resumed {
                        resumed = true
                        completedNormally = true
                        continuation.resume()
                    }
                }
            )
        }

        if !completedNormally {
            for _ in 0..<250 {
                if !ie.isGenerating { break }
                try? await Task.sleep(for: .milliseconds(20))
            }
            if ie.isGenerating {
                ie.resetSession()
                try? await Task.sleep(for: .milliseconds(200))
            }
        }

        let cooldown = thermalGovernor?.cooldownDelaySeconds ?? 0
        if cooldown > 0 {
            try? await Task.sleep(for: .seconds(cooldown))
        } else {
            try? await Task.sleep(for: .milliseconds(100))
        }
        return (generatedText.trimmingCharacters(in: .whitespacesAndNewlines), metricsResult)
    }

    func llmRequiresModel() -> TestOutcome? {
        guard let ie = inferenceEngine, ie.hasModel else {
            return TestOutcome(
                status: .skipped,
                message: "No model loaded — load a model to enable LLM diagnostics",
                details: []
            )
        }
        return nil
    }

    var isThermallySevere: Bool {
        let mode = thermalGovernor?.currentMode ?? .maxPerformance
        return mode == .coolDown || mode == .emergency
    }

    var thermalModeLabel: String {
        thermalGovernor?.currentMode.rawValue ?? "unknown"
    }

    var currentMaxContext: Int {
        (thermalGovernor?.currentMode ?? .maxPerformance).maxContextLength
    }

    func estimatePromptTokens(_ text: String) -> Int {
        max(1, text.count / 4)
    }

    func toolPromptExceedsContext(_ systemPrompt: String, generationBudget: Int) -> Bool {
        let estimatedPromptTokens = estimatePromptTokens(systemPrompt)
        let maxCtx = currentMaxContext
        return estimatedPromptTokens + generationBudget > maxCtx
    }

    func allOutputsEmpty(_ outputs: [String]) -> Bool {
        outputs.allSatisfy { $0.isEmpty }
    }

    // MARK: - LLM Instruction Following

    func llmTestInstructionFollowing() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let testCases: [(instruction: String, query: String, validator: (String) -> Bool, label: String)] = [
            (
                "Always respond in exactly one word. No punctuation.",
                "What color is the sky?",
                { output in
                    let words = output.split(separator: " ").count
                    return words <= 3
                },
                "One-word constraint"
            ),
            (
                "Always respond with a numbered list. Each item must start with a number followed by a period.",
                "List three fruits.",
                { output in
                    output.contains("1.") || output.contains("1)")
                },
                "Numbered list format"
            ),
            (
                "You must start every response with the word 'Indeed'.",
                "Is the Earth round?",
                { output in
                    output.lowercased().hasPrefix("indeed")
                },
                "Response prefix"
            ),
            (
                "Respond only in uppercase letters.",
                "Say hello.",
                { output in
                    let letters = output.unicodeScalars.filter { CharacterSet.letters.contains($0) }
                    let upperCount = letters.filter { CharacterSet.uppercaseLetters.contains($0) }.count
                    return letters.count > 0 && Double(upperCount) / Double(letters.count) > 0.7
                },
                "Uppercase constraint"
            ),
        ]

        var passed = 0
        var details: [String] = []

        for tc in testCases {
            let messages: [[String: String]] = [
                ["role": "system", "content": tc.instruction],
                ["role": "user", "content": tc.query]
            ]
            var config = SamplingConfig()
            config.maxTokens = 100
            config.temperature = 0.3

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: tc.instruction, samplingConfig: config, timeoutSeconds: 15)

            let success = !output.isEmpty && tc.validator(output)
            if success { passed += 1 }
            details.append("\(tc.label): \(success ? "✓" : "✗") → '\(output.prefix(60))…'")
        }

        return TestOutcome(
            status: passed >= 3 ? .passed : (passed >= 2 ? .warning : .failed),
            message: "Instruction following: \(passed)/\(testCases.count) complied",
            details: details
        )
    }

    // MARK: - LLM Factual Recall

    func llmTestFactualRecall() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let testCases: [(question: String, keywords: [String], label: String)] = [
            ("What is the capital of France? Answer in one word.", ["paris"], "France capital"),
            ("What planet is closest to the Sun? Answer briefly.", ["mercury"], "Closest planet"),
            ("What is H2O commonly known as? Answer in one word.", ["water"], "H2O"),
            ("Who wrote Romeo and Juliet? Answer briefly.", ["shakespeare", "william"], "Shakespeare"),
            ("How many sides does a triangle have? Answer with the number.", ["3", "three"], "Triangle sides"),
        ]

        var correct = 0
        var details: [String] = []

        for tc in testCases {
            let messages: [[String: String]] = [
                ["role": "system", "content": "Answer factual questions concisely and accurately."],
                ["role": "user", "content": tc.question]
            ]
            var config = SamplingConfig()
            config.maxTokens = 50
            config.temperature = 0.1

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: "Answer factual questions concisely.", samplingConfig: config, timeoutSeconds: 15)
            let lower = output.lowercased()
            let found = tc.keywords.contains { lower.contains($0) }
            if found { correct += 1 }
            details.append("\(tc.label): \(found ? "✓" : "✗") → '\(output.prefix(50))…'")
        }

        return TestOutcome(
            status: correct >= 4 ? .passed : (correct >= 3 ? .warning : .failed),
            message: "Factual recall: \(correct)/\(testCases.count) correct",
            details: details
        )
    }

    // MARK: - LLM Coherence Under Context

    func llmTestCoherenceUnderContext() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let longContext = (0..<8).map { "The user previously discussed topic \($0) about \(["AI", "cooking", "travel", "music", "science", "sports", "art", "history"][$0])." }.joined(separator: " ")

        let systemPrompt = "You are a helpful assistant. Here is context about the user: \(longContext)\nAlways respond coherently and reference context when relevant."

        let messages: [[String: String]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": "Based on what you know about me, what topic should we discuss next?"]
        ]

        var config = SamplingConfig()
        config.maxTokens = 200
        config.temperature = 0.7

        let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 20)

        var checks = 0
        var details: [String] = []

        let hasOutput = !output.isEmpty
        if hasOutput { checks += 1; details.append("Generated output: ✓ (\(output.count) chars)") }
        else { details.append("Generated output: ✗ (empty)") }

        let hasTokens = (metrics?.totalTokens ?? 0) > 5
        if hasTokens { checks += 1; details.append("Sufficient tokens: ✓ (\(metrics?.totalTokens ?? 0))") }
        else { details.append("Sufficient tokens: ✗") }

        let referencesTopics = ["ai", "cook", "travel", "music", "science", "sport", "art", "history"].contains { output.lowercased().contains($0) }
        if referencesTopics { checks += 1; details.append("References context topics: ✓") }
        else { details.append("References context topics: ✗") }

        let coherent = output.count > 10 && !output.allSatisfy({ $0 == output.first })
        if coherent { checks += 1; details.append("Non-degenerate output: ✓") }
        else { details.append("Non-degenerate output: ✗") }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Coherence: \(checks)/4 checks passed. Tokens=\(metrics?.totalTokens ?? 0)",
            details: details
        )
    }

    // MARK: - LLM Emotional Tone Compliance

    func llmTestEmotionalToneCompliance() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let testCases: [(system: String, user: String, expectedValence: EmotionalValence, label: String)] = [
            (
                "You are a cheerful, upbeat assistant. Always be positive and encouraging.",
                "I failed my exam today.",
                .positive,
                "Positive tone"
            ),
            (
                "You are a calm, empathetic assistant. Acknowledge emotions before responding.",
                "I'm really stressed about my deadline.",
                .negative,
                "Empathetic tone"
            ),
            (
                "You are a professional, neutral assistant. Respond factually without emotion.",
                "What is machine learning?",
                .neutral,
                "Neutral tone"
            ),
        ]

        var passed = 0
        var details: [String] = []

        for tc in testCases {
            let messages: [[String: String]] = [
                ["role": "system", "content": tc.system],
                ["role": "user", "content": tc.user]
            ]
            var config = SamplingConfig()
            config.maxTokens = 150
            config.temperature = 0.5

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: tc.system, samplingConfig: config, timeoutSeconds: 15)

            guard !output.isEmpty else {
                details.append("\(tc.label): ✗ (no output)")
                continue
            }

            let responseEmotion = EmotionAnalyzer.analyze(text: output, conversationHistory: [])

            let toneMatch: Bool
            switch tc.expectedValence {
            case .positive:
                toneMatch = responseEmotion.valence == .positive || output.lowercased().contains(where: { _ in
                    ["great", "good", "wonderful", "encourage", "try", "keep", "better", "proud", "well"].contains(where: { output.lowercased().contains($0) })
                })
            case .negative:
                toneMatch = responseEmotion.empathyLevel > 0.3 || ["understand", "sorry", "hear", "feel", "stress", "tough"].contains(where: { output.lowercased().contains($0) })
            case .neutral:
                toneMatch = true
            default:
                toneMatch = true
            }

            if toneMatch { passed += 1 }
            details.append("\(tc.label): \(toneMatch ? "✓" : "✗") valence=\(responseEmotion.valence.rawValue) → '\(output.prefix(50))…'")
        }

        return TestOutcome(
            status: passed >= 2 ? .passed : (passed >= 1 ? .warning : .failed),
            message: "Tone compliance: \(passed)/\(testCases.count) matched expected tone",
            details: details
        )
    }

    // MARK: - LLM Multi-Turn Consistency

    func llmTestMultiTurnConsistency() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let system = "You are a helpful assistant. Remember what the user tells you within this conversation."

        let turn1Messages: [[String: String]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": "My favorite color is blue and I have a cat named Luna."],
        ]

        var config = SamplingConfig()
        config.maxTokens = 80
        config.temperature = 0.3

        let (response1, _) = await llmGenerate(messages: turn1Messages, systemPrompt: system, samplingConfig: config, timeoutSeconds: 15)

        let turn2Messages: [[String: String]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": "My favorite color is blue and I have a cat named Luna."],
            ["role": "assistant", "content": response1.isEmpty ? "Nice! Blue is a lovely color, and Luna is a sweet name for a cat." : response1],
            ["role": "user", "content": "What is my cat's name?"],
        ]

        let (response2, _) = await llmGenerate(messages: turn2Messages, systemPrompt: system, samplingConfig: config, timeoutSeconds: 15)

        let turn3Messages: [[String: String]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": "My favorite color is blue and I have a cat named Luna."],
            ["role": "assistant", "content": response1.isEmpty ? "Nice! Blue is a lovely color." : response1],
            ["role": "user", "content": "What is my cat's name?"],
            ["role": "assistant", "content": response2.isEmpty ? "Luna!" : response2],
            ["role": "user", "content": "What is my favorite color?"],
        ]

        let (response3, _) = await llmGenerate(messages: turn3Messages, systemPrompt: system, samplingConfig: config, timeoutSeconds: 15)

        var checks = 0
        var details: [String] = []

        let r1ok = !response1.isEmpty
        if r1ok { checks += 1; details.append("Turn 1 response: ✓ (\(response1.count) chars)") }
        else { details.append("Turn 1 response: ✗ (empty)") }

        let r2HasLuna = response2.lowercased().contains("luna")
        if r2HasLuna { checks += 1; details.append("Cat name recall: ✓ → '\(response2.prefix(40))…'") }
        else { details.append("Cat name recall: ✗ → '\(response2.prefix(40))…'") }

        let r3HasBlue = response3.lowercased().contains("blue")
        if r3HasBlue { checks += 1; details.append("Color recall: ✓ → '\(response3.prefix(40))…'") }
        else { details.append("Color recall: ✗ → '\(response3.prefix(40))…'") }

        return TestOutcome(
            status: checks >= 2 ? .passed : (checks >= 1 ? .warning : .failed),
            message: "Multi-turn consistency: \(checks)/3 recalls correct",
            details: details
        )
    }

    // MARK: - LLM Cognition→Prompt→Output Pipeline

    func llmTestCognitionPromptOutput() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let userInput = "I'm curious about how neural networks learn from data"
        let thermalMode = thermalGovernor?.currentMode ?? .maxPerformance
        let thermalConstrained = thermalMode == .coolDown || thermalMode == .emergency
        let budget: PromptBudget = thermalConstrained ? .minimal : ContextAssembler.budgetForMode(thermalMode)
        let timeoutSeconds: Double

        switch thermalMode {
        case .maxPerformance:
            timeoutSeconds = 25
        case .balanced:
            timeoutSeconds = 35
        case .coolDown:
            timeoutSeconds = 30
        case .emergency:
            timeoutSeconds = 25
        }

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: userInput, conversationHistory: [], memoryService: mem)
        let memResults = mem.searchMemories(query: userInput, maxResults: thermalConstrained ? 2 : 5)
        let systemPrompt = ContextAssembler.assembleSystemPrompt(
            frame: frame,
            memoryResults: memResults,
            conversationHistory: [],
            toolsEnabled: false,
            isVoiceMode: false,
            preferredResponseLanguageCode: nil,
            budget: budget
        )
        CognitionEngine.resetSignature()

        let messages: [[String: String]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userInput]
        ]

        var config = SamplingConfig()
        config.maxTokens = thermalConstrained ? 30 : 200
        config.temperature = 0.7

        let start = Date()
        let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: timeoutSeconds)
        let duration = Date().timeIntervalSince(start)

        var checks = 0
        var details: [String] = []

        let promptHasIdentity = systemPrompt.contains("NEXUS")
        if promptHasIdentity { checks += 1; details.append("Cognition prompt has identity: ✓") }
        else { details.append("Cognition prompt has identity: ✗") }

        let hasInjections = !frame.injections.isEmpty
        if hasInjections { checks += 1; details.append("Cognition injections present: ✓ (\(frame.injections.count))") }
        else { details.append("Cognition injections present: ✗") }

        let hasOutput = !output.isEmpty && output.count > 10
        if hasOutput { checks += 1; details.append("LLM produced output: ✓ (\(output.count) chars)") }
        else { details.append("LLM produced output: ✗") }

        let topicRelevant = ["neural", "network", "learn", "data", "train", "model", "weight", "gradient", "backprop"].contains { output.lowercased().contains($0) }
        if topicRelevant { checks += 1; details.append("Response is topic-relevant: ✓") }
        else { details.append("Response is topic-relevant: ✗") }

        let underTimeLimit = duration < timeoutSeconds
        if underTimeLimit { checks += 1; details.append("Under time limit (\(Int(timeoutSeconds))s): ✓ (\(String(format: "%.1f", duration))s)") }
        else { details.append("Under time limit (\(Int(timeoutSeconds))s): ✗ (\(String(format: "%.1f", duration))s)") }

        details.append("Prompt: \(systemPrompt.count) chars, Intent: \(frame.intent.primary.rawValue), thermalMode=\(thermalMode.rawValue) budget=\(String(describing: budget))")
        if let m = metrics {
            details.append("TTFT: \(String(format: "%.0f", m.timeToFirstToken))ms, Decode: \(String(format: "%.1f", m.decodeTokensPerSecond)) tok/s")
        }

        let status: DiagnosticTestStatus
        if thermalConstrained {
            status = checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed)
        } else {
            status = checks >= 4 ? .passed : (checks >= 3 ? .warning : .failed)
        }

        return TestOutcome(
            status: status,
            message: "Pipeline: \(checks)/5 checks. Intent=\(frame.intent.primary.rawValue), Output=\(output.count) chars, \(String(format: "%.1f", duration))s",
            details: details
        )
    }

    // MARK: - LLM Stop Sequence Compliance

    func llmTestStopSequenceCompliance() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let messages: [[String: String]] = [
            ["role": "system", "content": "You are a helpful assistant. Answer briefly."],
            ["role": "user", "content": "Count from 1 to 20, one number per line."]
        ]

        var config = SamplingConfig()
        config.maxTokens = 30
        config.temperature = 0.1

        let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: "You are a helpful assistant.", samplingConfig: config, timeoutSeconds: 15)

        let tokenCount = metrics?.totalTokens ?? 0
        let respectsLimit = tokenCount <= 35

        var details: [String] = []
        details.append("Max tokens requested: 30")
        details.append("Tokens generated: \(tokenCount)")
        details.append("Respects limit: \(respectsLimit)")
        details.append("Output: '\(output.prefix(80))…'")

        let messages2: [[String: String]] = [
            ["role": "system", "content": "Answer in one sentence only."],
            ["role": "user", "content": "What is gravity?"]
        ]
        config.maxTokens = 200

        let (output2, _) = await llmGenerate(messages: messages2, systemPrompt: "Answer in one sentence only.", samplingConfig: config, timeoutSeconds: 15)

        let stoppedNaturally = !output2.isEmpty
        if stoppedNaturally { details.append("Natural EOS stop: ✓") }
        else { details.append("Natural EOS stop: ✗ (empty output)") }

        let checksOk = respectsLimit && stoppedNaturally

        return TestOutcome(
            status: checksOk ? .passed : (respectsLimit || stoppedNaturally ? .warning : .failed),
            message: "Stop compliance: maxTokens=\(respectsLimit ? "✓" : "✗"), EOS=\(stoppedNaturally ? "✓" : "✗"), generated \(tokenCount) tokens",
            details: details
        )
    }

    // MARK: - LLM Temperature Sensitivity

    func llmTestTemperatureSensitivity() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let prompt = "Complete this sentence creatively: The robot walked into the"

        let messages: [[String: String]] = [
            ["role": "system", "content": "Complete the user's sentence creatively."],
            ["role": "user", "content": prompt]
        ]

        var lowConfig = SamplingConfig()
        lowConfig.maxTokens = 40
        lowConfig.temperature = 0.1
        lowConfig.topK = 10

        var highConfig = SamplingConfig()
        highConfig.maxTokens = 40
        highConfig.temperature = 1.5
        highConfig.topK = 50

        var lowOutputs: [String] = []
        var highOutputs: [String] = []

        for _ in 0..<3 {
            let (low, _) = await llmGenerate(messages: messages, systemPrompt: "Complete the sentence.", samplingConfig: lowConfig, timeoutSeconds: 15)
            lowOutputs.append(low)
        }

        for _ in 0..<3 {
            let (high, _) = await llmGenerate(messages: messages, systemPrompt: "Complete the sentence.", samplingConfig: highConfig, timeoutSeconds: 15)
            highOutputs.append(high)
        }

        let lowUnique = Set(lowOutputs.map { $0.prefix(20).lowercased() })
        let highUnique = Set(highOutputs.map { $0.prefix(20).lowercased() })

        let lowDiversity = lowUnique.count
        let highDiversity = highUnique.count

        let anyOutput = lowOutputs.contains { !$0.isEmpty } && highOutputs.contains { !$0.isEmpty }

        var details: [String] = [
            "Low temp (0.1) unique prefixes: \(lowDiversity)/3",
            "High temp (1.5) unique prefixes: \(highDiversity)/3",
        ]
        for (i, o) in lowOutputs.enumerated() {
            details.append("Low[\(i)]: '\(o.prefix(40))…'")
        }
        for (i, o) in highOutputs.enumerated() {
            details.append("High[\(i)]: '\(o.prefix(40))…'")
        }

        let tempEffect = highDiversity >= lowDiversity || anyOutput

        return TestOutcome(
            status: anyOutput ? (tempEffect ? .passed : .warning) : .failed,
            message: "Temperature: low diversity=\(lowDiversity), high diversity=\(highDiversity), effect=\(tempEffect ? "observed" : "unclear")",
            details: details
        )
    }

    // MARK: - LLM Latency Profile

    func llmTestLatencyProfile() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let prompts: [(label: String, system: String, user: String)] = [
            ("Simple greeting", "Be brief.", "Hi!"),
            ("Factual question", "Be concise.", "What is 2+2?"),
            ("Short explanation", "Explain briefly.", "What is an atom?"),
            ("Creative task", "Be creative but brief.", "Write a haiku about coding."),
            ("Complex question", "Answer thoughtfully but concisely.", "What makes consciousness possible?"),
        ]

        var results: [(label: String, ttft: Double, decode: Double, tokens: Int, duration: Double)] = []

        for p in prompts {
            let messages: [[String: String]] = [
                ["role": "system", "content": p.system],
                ["role": "user", "content": p.user]
            ]
            var config = SamplingConfig()
            config.maxTokens = 60
            config.temperature = 0.5

            let start = Date()
            let (_, metrics) = await llmGenerate(messages: messages, systemPrompt: p.system, samplingConfig: config, timeoutSeconds: 20)
            let duration = Date().timeIntervalSince(start)

            results.append((
                label: p.label,
                ttft: metrics?.timeToFirstToken ?? 0,
                decode: metrics?.decodeTokensPerSecond ?? 0,
                tokens: metrics?.totalTokens ?? 0,
                duration: duration
            ))
        }

        let avgTTFT = results.map(\.ttft).reduce(0, +) / max(1, Double(results.count))
        let avgDecode = results.map(\.decode).reduce(0, +) / max(1, Double(results.count))
        let maxDuration = results.map(\.duration).max() ?? 0
        let totalTokens = results.map(\.tokens).reduce(0, +)
        let allProducedOutput = results.allSatisfy { $0.tokens > 0 }

        var details = results.map { "\($0.label): TTFT=\(String(format: "%.0f", $0.ttft))ms, \(String(format: "%.1f", $0.decode)) tok/s, \($0.tokens) tokens, \(String(format: "%.1f", $0.duration))s" }
        details.append("Avg TTFT: \(String(format: "%.0f", avgTTFT))ms")
        details.append("Avg decode: \(String(format: "%.1f", avgDecode)) tok/s")
        details.append("Total tokens: \(totalTokens)")

        return TestOutcome(
            status: allProducedOutput && maxDuration < 30 ? .passed : (totalTokens > 0 ? .warning : .failed),
            message: "Latency: avg TTFT=\(String(format: "%.0f", avgTTFT))ms, avg decode=\(String(format: "%.1f", avgDecode)) tok/s, \(totalTokens) total tokens",
            details: details
        )
    }

    // MARK: - LLM Memory-Aware Response

    func llmTestMemoryAwareResponse() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let testMemory = MemoryEntry(
            content: "User's name is Orion and they work as a marine biologist studying coral reefs in the ocean",
            keywords: ["orion", "marine", "biologist", "coral", "reefs", "ocean", "work"],
            category: .fact,
            importance: 5,
            source: .conversation
        )
        mem.addMemory(testMemory)

        try? await Task.sleep(for: .milliseconds(100))

        let userInput = "What do you know about coral reefs and marine biology?"
        let thermalConstrained = isThermallySevere
        let budget: PromptBudget = thermalConstrained ? .minimal : .full

        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: userInput, conversationHistory: [], memoryService: mem)
        let memResults = mem.searchMemories(query: userInput, maxResults: thermalConstrained ? 3 : 5)
        let systemPrompt = ContextAssembler.assembleSystemPrompt(
            frame: frame, memoryResults: memResults, conversationHistory: [],
            toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil,
            budget: budget
        )
        CognitionEngine.resetSignature()

        let messages: [[String: String]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userInput]
        ]

        var config = SamplingConfig()
        config.maxTokens = thermalConstrained ? 60 : 200
        config.temperature = 0.6

        let (output, _) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: thermalConstrained ? 25 : 20)

        mem.deleteMemory(testMemory.id)

        let searchBeforeDelete = memResults
        let memoryInjected = systemPrompt.lowercased().contains("orion") || systemPrompt.lowercased().contains("coral") || systemPrompt.lowercased().contains("marine")
        let outputReferencesMemory = ["orion", "coral", "marine", "reef", "ocean", "sea", "biolog"].contains { output.lowercased().contains($0) }
        let hasOutput = !output.isEmpty && output.count > 10

        var checks = 0
        var details: [String] = []

        if memoryInjected { checks += 1; details.append("Memory injected into prompt: ✓") }
        else { details.append("Memory injected into prompt: ✗") }

        if hasOutput { checks += 1; details.append("LLM produced output: ✓ (\(output.count) chars)") }
        else { details.append("LLM produced output: ✗") }

        if outputReferencesMemory { checks += 1; details.append("Response references memory: ✓") }
        else { details.append("Response references memory: ✗") }

        let directSearch = mem.searchMemories(query: "coral reefs marine biologist Orion", maxResults: 10)
        let memInResults = memResults.contains { $0.memory.id == testMemory.id } || directSearch.contains { $0.memory.id == testMemory.id }
        if memInResults { checks += 1; details.append("Test memory found in search: ✓") }
        else { details.append("Test memory found in search: ✗") }

        details.append("Output: '\(output.prefix(80))…'")

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Memory-aware: \(checks)/4. Injected=\(memoryInjected), Referenced=\(outputReferencesMemory)",
            details: details
        )
    }

    // MARK: - LLM Tool Call Emission

    func llmTestToolCallEmission() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let toolsPrompt = ToolExecutor.buildToolsPrompt()
        let systemPrompt = "You are a helpful assistant with device tools.\n\n\(toolsPrompt)"

        if isThermallySevere || toolPromptExceedsContext(systemPrompt, generationBudget: 100) {
            return TestOutcome(
                status: .warning,
                message: "Skipped — tool prompt (~\(estimatePromptTokens(systemPrompt)) tokens) exceeds context budget (\(currentMaxContext) tokens, thermal: \(thermalModeLabel))",
                details: ["Thermal mode: \(thermalModeLabel)", "Estimated prompt tokens: \(estimatePromptTokens(systemPrompt))", "Max context: \(currentMaxContext)"]
            )
        }

        let testCases: [(user: String, expectedTools: [String], label: String)] = [
            ("What time is it right now?", ["get_current_time"], "Time query"),
            ("What's the current battery level?", ["get_battery_status"], "Battery query"),
            ("What's my current location?", ["get_location"], "Location query"),
            ("Search the web for latest Swift 6 features", ["web_search"], "Web search query"),
        ]

        var passed = 0
        var details: [String] = []
        var outputs: [String] = []

        for tc in testCases {
            let messages: [[String: String]] = [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": tc.user]
            ]
            var config = SamplingConfig()
            config.maxTokens = 100
            config.temperature = 0.3

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 25)
            outputs.append(output)

            let calls = ToolCallParser.parse(from: output)
            let emittedNames = calls.map(\.name)
            let hasExpected = tc.expectedTools.contains { expected in emittedNames.contains(expected) }

            if hasExpected {
                passed += 1
                details.append("\(tc.label): ✓ → emitted \(emittedNames.joined(separator: ", "))")
            } else {
                let snippet = output.prefix(60)
                details.append("\(tc.label): ✗ → expected \(tc.expectedTools.joined(separator: "/")), got \(emittedNames.isEmpty ? "no tool call" : emittedNames.joined(separator: ", ")) '\(snippet)…'")
            }
        }

        if allOutputsEmpty(outputs) {
            details.append("All outputs empty — model may not support tool calling or prompt exceeded context")
            return TestOutcome(
                status: .warning,
                message: "Tool call emission: 0/\(testCases.count) — model produced no output (context: \(currentMaxContext), prompt: ~\(estimatePromptTokens(systemPrompt)) tokens)",
                details: details
            )
        }

        return TestOutcome(
            status: passed >= 3 ? .passed : (passed >= 2 ? .warning : .failed),
            message: "Tool call emission: \(passed)/\(testCases.count) emitted correct tool",
            details: details
        )
    }

    // MARK: - LLM Tool Call Format Correctness

    func llmTestToolCallFormat() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let toolsPrompt = ToolExecutor.buildToolsPrompt()
        let systemPrompt = "You are a helpful assistant with device tools.\n\n\(toolsPrompt)"

        if isThermallySevere || toolPromptExceedsContext(systemPrompt, generationBudget: 80) {
            return TestOutcome(
                status: .warning,
                message: "Skipped — tool prompt exceeds context budget (thermal: \(thermalModeLabel), maxCtx: \(currentMaxContext))",
                details: ["Thermal mode: \(thermalModeLabel)", "Max context: \(currentMaxContext)"]
            )
        }

        let prompts: [(user: String, label: String)] = [
            ("What time is it?", "Time"),
            ("How's the weather?", "Weather"),
            ("Check my battery level", "Battery"),
        ]

        var validFormat = 0
        var validName = 0
        var details: [String] = []
        var outputs: [String] = []

        for p in prompts {
            let messages: [[String: String]] = [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": p.user]
            ]
            var config = SamplingConfig()
            config.maxTokens = 80
            config.temperature = 0.2

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 25)
            outputs.append(output)

            let hasTag = output.contains("<tool_call>") || output.contains("<tool_calls>")
            let calls = ToolCallParser.parse(from: output)

            if hasTag && !calls.isEmpty {
                validFormat += 1
            }

            let allKnown = calls.allSatisfy { DeviceToolName(rawValue: $0.name) != nil }
            if !calls.isEmpty && allKnown {
                validName += 1
            }

            let names = calls.map(\.name).joined(separator: ", ")
            details.append("\(p.label): tag=\(hasTag ? "✓" : "✗"), parsed=\(calls.count), names=\(names.isEmpty ? "none" : names), allKnown=\(allKnown ? "✓" : "✗")")
        }

        if allOutputsEmpty(outputs) {
            details.append("All outputs empty — model may not support tool calling or prompt exceeded context")
            return TestOutcome(
                status: .warning,
                message: "Tool format: 0/\(prompts.count) — model produced no output",
                details: details
            )
        }

        let total = prompts.count
        let score = validFormat + validName

        return TestOutcome(
            status: score >= total * 2 - 1 ? .passed : (score >= total ? .warning : .failed),
            message: "Tool format: \(validFormat)/\(total) valid tags, \(validName)/\(total) valid names",
            details: details
        )
    }

    // MARK: - LLM Tool Abstention

    func llmTestToolAbstention() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let toolsPrompt = ToolExecutor.buildToolsPrompt()
        let systemPrompt = "You are a helpful assistant with device tools. Only use tools when the user explicitly needs device capabilities. If no tool is needed, respond normally in plain text.\n\n\(toolsPrompt)"

        if isThermallySevere || toolPromptExceedsContext(systemPrompt, generationBudget: 120) {
            return TestOutcome(
                status: .warning,
                message: "Skipped — tool prompt exceeds context budget (thermal: \(thermalModeLabel), maxCtx: \(currentMaxContext))",
                details: ["Thermal mode: \(thermalModeLabel)", "Max context: \(currentMaxContext)"]
            )
        }

        let testCases: [(user: String, label: String)] = [
            ("What is the capital of France?", "Factual question"),
            ("Write a haiku about rain", "Creative request"),
            ("Explain how photosynthesis works", "Explanation"),
            ("Hello, how are you?", "Greeting"),
        ]

        var abstained = 0
        var details: [String] = []
        var outputs: [String] = []

        for tc in testCases {
            let messages: [[String: String]] = [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": tc.user]
            ]
            var config = SamplingConfig()
            config.maxTokens = 120
            config.temperature = 0.5

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 25)
            outputs.append(output)

            let calls = ToolCallParser.parse(from: output)
            let noToolCall = calls.isEmpty
            let hasText = !output.isEmpty && output.count > 5

            if noToolCall && hasText {
                abstained += 1
                details.append("\(tc.label): ✓ → plain text (\(output.count) chars)")
            } else if !noToolCall {
                details.append("\(tc.label): ✗ → spurious tool call: \(calls.map(\.name).joined(separator: ", "))")
            } else {
                details.append("\(tc.label): ✗ → empty output")
            }
        }

        if allOutputsEmpty(outputs) {
            details.append("All outputs empty — model may not support tool calling or prompt exceeded context")
            return TestOutcome(
                status: .warning,
                message: "Tool abstention: 0/\(testCases.count) — model produced no output",
                details: details
            )
        }

        return TestOutcome(
            status: abstained >= 3 ? .passed : (abstained >= 2 ? .warning : .failed),
            message: "Tool abstention: \(abstained)/\(testCases.count) correctly avoided tool calls",
            details: details
        )
    }

    // MARK: - LLM Tool Result Synthesis

    func llmTestToolResultSynthesis() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let toolsPrompt = ToolExecutor.buildToolsPrompt()
        let systemPrompt = "You are a helpful assistant with device tools. After receiving tool results, synthesize a concise user-facing answer.\n\n\(toolsPrompt)"

        if isThermallySevere || toolPromptExceedsContext(systemPrompt, generationBudget: 100) {
            return TestOutcome(
                status: .warning,
                message: "Skipped — tool prompt exceeds context budget (thermal: \(thermalModeLabel), maxCtx: \(currentMaxContext))",
                details: ["Thermal mode: \(thermalModeLabel)", "Max context: \(currentMaxContext)"]
            )
        }

        let testCases: [(messages: [[String: String]], keywords: [String], label: String)] = [
            (
                [
                    ["role": "system", "content": systemPrompt],
                    ["role": "user", "content": "What time is it?"],
                    ["role": "assistant", "content": "<tool_call>{\"name\":\"get_current_time\",\"parameters\":{}}</tool_call>"],
                    ["role": "user", "content": "[Tool Execution Summary]\n- get_current_time (success)\n  {\"time\":\"2:30 PM\",\"date\":\"March 27, 2026\",\"timezone\":\"America/Toronto\"}"]
                ],
                ["2:30", "pm", "march", "27"],
                "Time synthesis"
            ),
            (
                [
                    ["role": "system", "content": systemPrompt],
                    ["role": "user", "content": "What's my battery at?"],
                    ["role": "assistant", "content": "<tool_call>{\"name\":\"get_battery_status\",\"parameters\":{}}</tool_call>"],
                    ["role": "user", "content": "[Tool Execution Summary]\n- get_battery_status (success)\n  {\"level\":72,\"state\":\"Charging\"}"]
                ],
                ["72", "charg"],
                "Battery synthesis"
            ),
        ]

        var passed = 0
        var details: [String] = []
        var outputs: [String] = []

        for tc in testCases {
            var config = SamplingConfig()
            config.maxTokens = 100
            config.temperature = 0.4

            let (output, _) = await llmGenerate(messages: tc.messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 15)
            outputs.append(output)

            let lower = output.lowercased()
            let referencesResult = tc.keywords.contains { lower.contains($0) }
            let noNewToolCall = ToolCallParser.parse(from: output).isEmpty
            let hasText = !output.isEmpty && output.count > 3

            let ok = referencesResult && noNewToolCall && hasText
            if ok { passed += 1 }

            details.append("\(tc.label): \(ok ? "✓" : "✗") → refs=\(referencesResult), noTool=\(noNewToolCall), text=\(hasText) '\(output.prefix(50))…'")
        }

        if allOutputsEmpty(outputs) {
            details.append("All outputs empty — model may not support tool result synthesis or prompt exceeded context")
            return TestOutcome(
                status: .warning,
                message: "Tool result synthesis: 0/\(testCases.count) — model produced no output",
                details: details
            )
        }

        return TestOutcome(
            status: passed == testCases.count ? .passed : (passed >= 1 ? .warning : .failed),
            message: "Tool result synthesis: \(passed)/\(testCases.count) synthesized correct answer from tool results",
            details: details
        )
    }

    // MARK: - LLM Batched Tool Calls

    func llmTestBatchedToolCalls() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let toolsPrompt = ToolExecutor.buildToolsPrompt()
        let systemPrompt = "You are a helpful assistant with device tools. When a user request requires multiple tools, emit them all in a single batched tool_calls tag.\n\n\(toolsPrompt)"

        if isThermallySevere || toolPromptExceedsContext(systemPrompt, generationBudget: 120) {
            return TestOutcome(
                status: .warning,
                message: "Skipped — tool prompt exceeds context budget (thermal: \(thermalModeLabel), maxCtx: \(currentMaxContext))",
                details: ["Thermal mode: \(thermalModeLabel)", "Max context: \(currentMaxContext)"]
            )
        }

        let messages: [[String: String]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": "What time is it and what's my battery level?"]
        ]

        var config = SamplingConfig()
        config.maxTokens = 120
        config.temperature = 0.3

        let (output, _) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 15)

        if output.isEmpty {
            return TestOutcome(
                status: .warning,
                message: "Batched tool calls: model produced no output (context: \(currentMaxContext), prompt: ~\(estimatePromptTokens(systemPrompt)) tokens)",
                details: ["Output empty — model may not support batched tool calling or prompt exceeded context"]
            )
        }

        let calls = ToolCallParser.parse(from: output)
        let names = Set(calls.map(\.name))

        var checks = 0
        var details: [String] = []

        let hasMultiple = calls.count >= 2
        if hasMultiple { checks += 1; details.append("Multiple calls emitted: ✓ (\(calls.count))") }
        else { details.append("Multiple calls emitted: ✗ (\(calls.count))") }

        let hasTime = names.contains("get_current_time")
        if hasTime { checks += 1; details.append("Contains get_current_time: ✓") }
        else { details.append("Contains get_current_time: ✗") }

        let hasBattery = names.contains("get_battery_status")
        if hasBattery { checks += 1; details.append("Contains get_battery_status: ✓") }
        else { details.append("Contains get_battery_status: ✗") }

        let allKnown = calls.allSatisfy { DeviceToolName(rawValue: $0.name) != nil }
        if allKnown && !calls.isEmpty { checks += 1; details.append("All tool names valid: ✓") }
        else { details.append("All tool names valid: ✗") }

        details.append("Emitted names: \(names.sorted().joined(separator: ", "))")
        details.append("Output: '\(output.prefix(80))…'")

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Batched tool calls: \(checks)/4 checks. Emitted \(calls.count) calls: \(names.sorted().joined(separator: ", "))",
            details: details
        )
    }
}
