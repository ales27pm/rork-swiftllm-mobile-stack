import Foundation

extension DiagnosticEngine {

    // MARK: - Temperature vs Quality Trade-off

    func configTestTemperatureQuality() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let prompt = "What is the capital of Japan? Answer in one sentence."
        let temperatures: [Float] = [0.0, 0.3, 0.7, 1.0, 1.5]

        struct TempResult {
            let temp: Float
            let output: String
            let ttft: Double
            let decode: Double
            let tokens: Int
            let coherent: Bool
        }

        var results: [TempResult] = []

        for temp in temperatures {
            var config = SamplingConfig()
            config.temperature = temp
            config.maxTokens = 60
            config.topK = 40

            let messages: [[String: String]] = [
                ["role": "system", "content": "Answer factual questions concisely."],
                ["role": "user", "content": prompt]
            ]

            let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: "Answer factual questions concisely.", samplingConfig: config, timeoutSeconds: 15)

            let lower = output.lowercased()
            let coherent = lower.contains("tokyo") || (output.count > 5 && !output.allSatisfy { $0 == output.first })

            results.append(TempResult(
                temp: temp,
                output: output,
                ttft: metrics?.timeToFirstToken ?? 0,
                decode: metrics?.decodeTokensPerSecond ?? 0,
                tokens: metrics?.totalTokens ?? 0,
                coherent: coherent
            ))
        }

        let anyOutput = results.contains { $0.tokens > 0 }
        let lowTempAccurate = results.filter({ $0.temp <= 0.3 }).contains { $0.output.lowercased().contains("tokyo") }
        let highTempDiverse = results.filter({ $0.temp >= 1.0 }).contains { !$0.output.isEmpty }
        let coherentCount = results.filter(\.coherent).count

        var details = results.map { r in
            "temp=\(String(format: "%.1f", r.temp)): \(r.tokens) tok, TTFT=\(String(format: "%.0f", r.ttft))ms, \(String(format: "%.1f", r.decode)) tok/s, coherent=\(r.coherent ? "✓" : "✗") → '\(r.output.prefix(40))…'"
        }
        details.append("Low temp (≤0.3) accurate: \(lowTempAccurate ? "✓" : "✗")")
        details.append("High temp (≥1.0) produces output: \(highTempDiverse ? "✓" : "✗")")
        details.append("Coherent outputs: \(coherentCount)/\(results.count)")

        var checks = 0
        if anyOutput { checks += 1 }
        if lowTempAccurate { checks += 1 }
        if highTempDiverse { checks += 1 }
        if coherentCount >= 3 { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Temperature sweep: \(checks)/4 checks. Coherent=\(coherentCount)/\(results.count), lowAccurate=\(lowTempAccurate), highProduces=\(highTempDiverse)",
            details: details
        )
    }

    // MARK: - TopK Sweep

    func configTestTopKSweep() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let topKValues: [Int] = [1, 5, 20, 40, 100]
        let prompt = "Complete this sentence creatively: The stars above the mountain"

        struct TopKResult {
            let topK: Int
            let output: String
            let tokens: Int
            let uniqueWords: Int
        }

        var results: [TopKResult] = []

        for k in topKValues {
            var config = SamplingConfig()
            config.topK = k
            config.temperature = 0.8
            config.maxTokens = 40

            let messages: [[String: String]] = [
                ["role": "system", "content": "Complete the sentence creatively."],
                ["role": "user", "content": prompt]
            ]

            let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: "Complete the sentence.", samplingConfig: config, timeoutSeconds: 15)

            let words = Set(output.lowercased().split { $0.isWhitespace || $0.isPunctuation }.map(String.init))

            results.append(TopKResult(
                topK: k,
                output: output,
                tokens: metrics?.totalTokens ?? 0,
                uniqueWords: words.count
            ))
        }

        let anyOutput = results.contains { $0.tokens > 0 }
        let narrowK = results.first(where: { $0.topK == 1 })
        let wideK = results.first(where: { $0.topK == 100 })
        let diversityIncreases = (wideK?.uniqueWords ?? 0) >= (narrowK?.uniqueWords ?? 0)

        var details = results.map { r in
            "topK=\(r.topK): \(r.tokens) tok, \(r.uniqueWords) unique words → '\(r.output.prefix(40))…'"
        }
        details.append("Diversity increases with topK: \(diversityIncreases ? "✓" : "✗")")

        var checks = 0
        if anyOutput { checks += 1 }
        if (narrowK?.tokens ?? 0) > 0 { checks += 1 }
        if diversityIncreases { checks += 1 }
        if results.filter({ $0.tokens > 0 }).count >= 3 { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "TopK sweep: \(checks)/4. narrow(\(narrowK?.uniqueWords ?? 0) words) vs wide(\(wideK?.uniqueWords ?? 0) words), diversityUp=\(diversityIncreases)",
            details: details
        )
    }

    // MARK: - TopP Nucleus Sampling Comparison

    func configTestTopPComparison() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let topPValues: [Float] = [0.1, 0.5, 0.9, 1.0]
        let prompt = "Describe the ocean in one sentence."

        struct TopPResult {
            let topP: Float
            let output: String
            let tokens: Int
            let length: Int
        }

        var results: [TopPResult] = []

        for p in topPValues {
            var config = SamplingConfig()
            config.topP = p
            config.temperature = 0.7
            config.maxTokens = 50
            config.topK = 40

            let messages: [[String: String]] = [
                ["role": "system", "content": "Respond concisely."],
                ["role": "user", "content": prompt]
            ]

            let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: "Respond concisely.", samplingConfig: config, timeoutSeconds: 15)

            results.append(TopPResult(
                topP: p,
                output: output,
                tokens: metrics?.totalTokens ?? 0,
                length: output.count
            ))
        }

        let anyOutput = results.contains { $0.tokens > 0 }
        let narrowP = results.first(where: { $0.topP == 0.1 })
        let wideP = results.first(where: { $0.topP == 1.0 })
        let allCoherent = results.filter({ $0.tokens > 0 }).allSatisfy { $0.output.count > 5 }

        var details = results.map { r in
            "topP=\(String(format: "%.1f", r.topP)): \(r.tokens) tok, \(r.length) chars → '\(r.output.prefix(50))…'"
        }
        details.append("Narrow (0.1) output: \(narrowP?.tokens ?? 0) tok")
        details.append("Wide (1.0) output: \(wideP?.tokens ?? 0) tok")
        details.append("All coherent: \(allCoherent ? "✓" : "✗")")

        var checks = 0
        if anyOutput { checks += 1 }
        if (narrowP?.tokens ?? 0) > 0 { checks += 1 }
        if (wideP?.tokens ?? 0) > 0 { checks += 1 }
        if allCoherent { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "TopP comparison: \(checks)/4. narrow=\(narrowP?.tokens ?? 0)tok, wide=\(wideP?.tokens ?? 0)tok, allCoherent=\(allCoherent)",
            details: details
        )
    }

    // MARK: - Repetition Penalty Effect

    func configTestRepetitionPenalty() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let penalties: [Float] = [1.0, 1.1, 1.3, 1.5]
        let prompt = "Tell me about dogs. Dogs are great pets. Dogs"

        struct RepResult {
            let penalty: Float
            let output: String
            let tokens: Int
            let repetitionRatio: Double
        }

        var results: [RepResult] = []

        for pen in penalties {
            var config = SamplingConfig()
            config.repetitionPenalty = pen
            config.temperature = 0.7
            config.maxTokens = 80
            config.topK = 40

            let messages: [[String: String]] = [
                ["role": "system", "content": "Continue the user's text naturally."],
                ["role": "user", "content": prompt]
            ]

            let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: "Continue naturally.", samplingConfig: config, timeoutSeconds: 15)

            let words = output.lowercased().split { $0.isWhitespace || $0.isPunctuation }.map(String.init)
            let totalWords = max(1, words.count)
            let uniqueWords = Set(words).count
            let repRatio = 1.0 - (Double(uniqueWords) / Double(totalWords))

            results.append(RepResult(
                penalty: pen,
                output: output,
                tokens: metrics?.totalTokens ?? 0,
                repetitionRatio: repRatio
            ))
        }

        let anyOutput = results.contains { $0.tokens > 0 }
        let noPenalty = results.first(where: { $0.penalty == 1.0 })
        let highPenalty = results.first(where: { $0.penalty == 1.5 })
        let penaltyReducesRepetition = (highPenalty?.repetitionRatio ?? 1.0) <= (noPenalty?.repetitionRatio ?? 0.0) + 0.15

        var details = results.map { r in
            "repPen=\(String(format: "%.1f", r.penalty)): \(r.tokens) tok, repRatio=\(String(format: "%.2f", r.repetitionRatio)) → '\(r.output.prefix(40))…'"
        }
        details.append("Penalty reduces repetition: \(penaltyReducesRepetition ? "✓" : "✗")")
        details.append("No penalty rep ratio: \(String(format: "%.2f", noPenalty?.repetitionRatio ?? 0))")
        details.append("High penalty rep ratio: \(String(format: "%.2f", highPenalty?.repetitionRatio ?? 0))")

        var checks = 0
        if anyOutput { checks += 1 }
        if (noPenalty?.tokens ?? 0) > 0 { checks += 1 }
        if (highPenalty?.tokens ?? 0) > 0 { checks += 1 }
        if penaltyReducesRepetition { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Repetition penalty: \(checks)/4. noPen=\(String(format: "%.2f", noPenalty?.repetitionRatio ?? 0)), highPen=\(String(format: "%.2f", highPenalty?.repetitionRatio ?? 0)), reduces=\(penaltyReducesRepetition)",
            details: details
        )
    }

    // MARK: - Max Tokens Budget vs Latency

    func configTestMaxTokensBudget() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let budgets: [Int] = [10, 30, 80, 200]
        let prompt = "Explain how computers work."

        struct BudgetResult {
            let maxTokens: Int
            let output: String
            let tokens: Int
            let duration: Double
            let tokPerSec: Double
        }

        var results: [BudgetResult] = []

        for budget in budgets {
            var config = SamplingConfig()
            config.maxTokens = budget
            config.temperature = 0.5
            config.topK = 40

            let messages: [[String: String]] = [
                ["role": "system", "content": "Explain clearly."],
                ["role": "user", "content": prompt]
            ]

            let start = Date()
            let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: "Explain clearly.", samplingConfig: config, timeoutSeconds: 25)
            let duration = Date().timeIntervalSince(start)

            results.append(BudgetResult(
                maxTokens: budget,
                output: output,
                tokens: metrics?.totalTokens ?? 0,
                duration: duration,
                tokPerSec: metrics?.decodeTokensPerSecond ?? 0
            ))
        }

        let anyOutput = results.contains { $0.tokens > 0 }
        let respectsLimits = results.allSatisfy { $0.tokens <= $0.maxTokens + 5 || $0.tokens == 0 }
        let smallerBudgetFaster = (results.first?.duration ?? 999) <= (results.last?.duration ?? 0) + 1.0
        let largerBudgetMoreTokens = (results.last?.tokens ?? 0) >= (results.first?.tokens ?? 999)

        var details = results.map { r in
            "maxTok=\(r.maxTokens): \(r.tokens) tok, \(String(format: "%.1f", r.duration))s, \(String(format: "%.1f", r.tokPerSec)) tok/s → '\(r.output.prefix(35))…'"
        }
        details.append("Respects token limits: \(respectsLimits ? "✓" : "✗")")
        details.append("Smaller budget faster: \(smallerBudgetFaster ? "✓" : "✗")")
        details.append("Larger budget more tokens: \(largerBudgetMoreTokens ? "✓" : "✗")")

        var checks = 0
        if anyOutput { checks += 1 }
        if respectsLimits { checks += 1 }
        if smallerBudgetFaster { checks += 1 }
        if largerBudgetMoreTokens { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Token budget: \(checks)/4. limits=\(respectsLimits), fasterSmall=\(smallerBudgetFaster), moreTokensLarge=\(largerBudgetMoreTokens)",
            details: details
        )
    }

    // MARK: - Prompt Budget Quality Comparison

    func configTestPromptBudget() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }

        let userInput = "Tell me about Swift programming"
        let budgets: [(PromptBudget, String)] = [(.full, "full"), (.compact, "compact"), (.minimal, "minimal")]

        struct BudgetResult {
            let label: String
            let promptLength: Int
            let injectionCount: Int
            let output: String
            let tokens: Int
            let duration: Double
        }

        var results: [BudgetResult] = []

        for (budget, label) in budgets {
            CognitionEngine.resetSignature()
            let frame = CognitionEngine.process(userText: userInput, conversationHistory: [], memoryService: mem)
            let memResults = mem.searchMemories(query: userInput, maxResults: 5)
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
            config.maxTokens = 80
            config.temperature = 0.5

            let start = Date()
            let (output, metrics) = await llmGenerate(messages: messages, systemPrompt: systemPrompt, samplingConfig: config, timeoutSeconds: 20)
            let duration = Date().timeIntervalSince(start)

            results.append(BudgetResult(
                label: label,
                promptLength: systemPrompt.count,
                injectionCount: frame.injections.count,
                output: output,
                tokens: metrics?.totalTokens ?? 0,
                duration: duration
            ))
        }

        let fullResult = results.first(where: { $0.label == "full" })
        let minResult = results.first(where: { $0.label == "minimal" })
        let anyOutput = results.contains { $0.tokens > 0 }
        let fullLongerPrompt = (fullResult?.promptLength ?? 0) > (minResult?.promptLength ?? 0)
        let minimalFaster = (minResult?.duration ?? 999) <= (fullResult?.duration ?? 0) + 2.0

        var details = results.map { r in
            "\(r.label): prompt=\(r.promptLength) chars, injections=\(r.injectionCount), \(r.tokens) tok, \(String(format: "%.1f", r.duration))s → '\(r.output.prefix(35))…'"
        }
        details.append("Full has longer prompt: \(fullLongerPrompt ? "✓" : "✗")")
        details.append("Minimal faster/comparable: \(minimalFaster ? "✓" : "✗")")

        var checks = 0
        if anyOutput { checks += 1 }
        if fullLongerPrompt { checks += 1 }
        if minimalFaster { checks += 1 }
        if results.filter({ $0.tokens > 0 }).count >= 2 { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Prompt budget: \(checks)/4. full=\(fullResult?.promptLength ?? 0)ch/\(fullResult?.tokens ?? 0)tok, minimal=\(minResult?.promptLength ?? 0)ch/\(minResult?.tokens ?? 0)tok",
            details: details
        )
    }

    // MARK: - Sampling Seed Determinism

    func configTestSeedDeterminism() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        let prompt = "Count from 1 to 5."
        let seed: UInt64 = 12345

        var seededOutputs: [String] = []
        var unseededOutputs: [String] = []

        for _ in 0..<3 {
            var config = SamplingConfig()
            config.temperature = 0.3
            config.maxTokens = 30
            config.samplerSeed = seed

            let messages: [[String: String]] = [
                ["role": "system", "content": "Answer concisely."],
                ["role": "user", "content": prompt]
            ]

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: "Answer concisely.", samplingConfig: config, timeoutSeconds: 15)
            seededOutputs.append(output)
        }

        for _ in 0..<3 {
            var config = SamplingConfig()
            config.temperature = 0.3
            config.maxTokens = 30
            config.samplerSeed = nil

            let messages: [[String: String]] = [
                ["role": "system", "content": "Answer concisely."],
                ["role": "user", "content": prompt]
            ]

            let (output, _) = await llmGenerate(messages: messages, systemPrompt: "Answer concisely.", samplingConfig: config, timeoutSeconds: 15)
            unseededOutputs.append(output)
        }

        let seededUnique = Set(seededOutputs.map { $0.prefix(30).lowercased() })
        let unseededUnique = Set(unseededOutputs.map { $0.prefix(30).lowercased() })
        let seededDeterministic = seededUnique.count == 1
        let anySeeded = seededOutputs.contains { !$0.isEmpty }
        let anyUnseeded = unseededOutputs.contains { !$0.isEmpty }

        var details: [String] = [
            "Seeded unique prefixes: \(seededUnique.count)/3 (deterministic=\(seededDeterministic ? "✓" : "✗"))",
            "Unseeded unique prefixes: \(unseededUnique.count)/3",
        ]
        for (i, o) in seededOutputs.enumerated() {
            details.append("Seed[\(i)]: '\(o.prefix(40))…'")
        }
        for (i, o) in unseededOutputs.enumerated() {
            details.append("NoSeed[\(i)]: '\(o.prefix(40))…'")
        }

        var checks = 0
        if anySeeded { checks += 1 }
        if anyUnseeded { checks += 1 }
        if seededDeterministic { checks += 1 }
        if seededUnique.count <= unseededUnique.count || seededDeterministic { checks += 1 }

        return TestOutcome(
            status: checks >= 3 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Seed determinism: \(checks)/4. seededUnique=\(seededUnique.count), unseededUnique=\(unseededUnique.count), deterministic=\(seededDeterministic)",
            details: details
        )
    }

    // MARK: - Combined Config Profile Benchmark

    func configTestCombinedProfiles() async -> TestOutcome {
        if let skip = llmRequiresModel() { return skip }

        struct Profile {
            let name: String
            let config: SamplingConfig
        }

        var preciseConfig = SamplingConfig()
        preciseConfig.temperature = 0.1
        preciseConfig.topK = 5
        preciseConfig.topP = 0.5
        preciseConfig.repetitionPenalty = 1.2
        preciseConfig.maxTokens = 60

        var balancedConfig = SamplingConfig()
        balancedConfig.temperature = 0.7
        balancedConfig.topK = 40
        balancedConfig.topP = 0.9
        balancedConfig.repetitionPenalty = 1.1
        balancedConfig.maxTokens = 100

        var creativeConfig = SamplingConfig()
        creativeConfig.temperature = 1.2
        creativeConfig.topK = 80
        creativeConfig.topP = 0.95
        creativeConfig.repetitionPenalty = 1.0
        creativeConfig.maxTokens = 120

        var conservativeConfig = SamplingConfig()
        conservativeConfig.temperature = 0.3
        conservativeConfig.topK = 10
        conservativeConfig.topP = 0.7
        conservativeConfig.repetitionPenalty = 1.3
        conservativeConfig.maxTokens = 40

        let profiles: [Profile] = [
            Profile(name: "Precise", config: preciseConfig),
            Profile(name: "Balanced", config: balancedConfig),
            Profile(name: "Creative", config: creativeConfig),
            Profile(name: "Conservative", config: conservativeConfig),
        ]

        struct ProfileResult {
            let name: String
            let tokens: Int
            let ttft: Double
            let decode: Double
            let duration: Double
            let output: String
            let coherent: Bool
        }

        let factualPrompt: [[String: String]] = [
            ["role": "system", "content": "Answer factual questions accurately."],
            ["role": "user", "content": "What is the largest planet in our solar system?"]
        ]

        let creativePrompt: [[String: String]] = [
            ["role": "system", "content": "Write creatively."],
            ["role": "user", "content": "Describe a sunset over the ocean in two sentences."]
        ]

        var factualResults: [ProfileResult] = []
        var creativeResults: [ProfileResult] = []

        for profile in profiles {
            let start1 = Date()
            let (fOut, fMet) = await llmGenerate(messages: factualPrompt, systemPrompt: "Answer accurately.", samplingConfig: profile.config, timeoutSeconds: 20)
            let dur1 = Date().timeIntervalSince(start1)
            let fCoherent = fOut.lowercased().contains("jupiter") || (fOut.count > 10 && !fOut.allSatisfy { $0 == fOut.first })

            factualResults.append(ProfileResult(
                name: profile.name,
                tokens: fMet?.totalTokens ?? 0,
                ttft: fMet?.timeToFirstToken ?? 0,
                decode: fMet?.decodeTokensPerSecond ?? 0,
                duration: dur1,
                output: fOut,
                coherent: fCoherent
            ))

            let start2 = Date()
            let (cOut, cMet) = await llmGenerate(messages: creativePrompt, systemPrompt: "Write creatively.", samplingConfig: profile.config, timeoutSeconds: 20)
            let dur2 = Date().timeIntervalSince(start2)
            let cCoherent = cOut.count > 15 && !cOut.allSatisfy { $0 == cOut.first }

            creativeResults.append(ProfileResult(
                name: profile.name,
                tokens: cMet?.totalTokens ?? 0,
                ttft: cMet?.timeToFirstToken ?? 0,
                decode: cMet?.decodeTokensPerSecond ?? 0,
                duration: dur2,
                output: cOut,
                coherent: cCoherent
            ))
        }

        let anyFactual = factualResults.contains { $0.tokens > 0 }
        let anyCreative = creativeResults.contains { $0.tokens > 0 }
        let preciseFactualAccurate = factualResults.first(where: { $0.name == "Precise" })?.output.lowercased().contains("jupiter") ?? false
        let totalCoherent = factualResults.filter(\.coherent).count + creativeResults.filter(\.coherent).count

        var details: [String] = ["— Factual Task —"]
        for r in factualResults {
            details.append("[\(r.name)] \(r.tokens) tok, TTFT=\(String(format: "%.0f", r.ttft))ms, \(String(format: "%.1f", r.decode)) tok/s, \(String(format: "%.1f", r.duration))s, coherent=\(r.coherent ? "✓" : "✗") → '\(r.output.prefix(35))…'")
        }
        details.append("— Creative Task —")
        for r in creativeResults {
            details.append("[\(r.name)] \(r.tokens) tok, TTFT=\(String(format: "%.0f", r.ttft))ms, \(String(format: "%.1f", r.decode)) tok/s, \(String(format: "%.1f", r.duration))s, coherent=\(r.coherent ? "✓" : "✗") → '\(r.output.prefix(35))…'")
        }
        details.append("Precise factual accurate: \(preciseFactualAccurate ? "✓" : "✗")")
        details.append("Total coherent: \(totalCoherent)/\(factualResults.count + creativeResults.count)")

        var checks = 0
        if anyFactual { checks += 1 }
        if anyCreative { checks += 1 }
        if preciseFactualAccurate { checks += 1 }
        if totalCoherent >= 4 { checks += 1 }
        if totalCoherent >= 6 { checks += 1 }

        return TestOutcome(
            status: checks >= 4 ? .passed : (checks >= 2 ? .warning : .failed),
            message: "Combined profiles: \(checks)/5. coherent=\(totalCoherent)/\(factualResults.count + creativeResults.count), preciseAccurate=\(preciseFactualAccurate)",
            details: details
        )
    }
}
