import Foundation

struct ContextAssembler {
    private static let injectionTokenBudget = 1500
    private static let conversationSummaryThreshold = 12

    static func assembleSystemPrompt(
        frame: CognitionFrame,
        memoryResults: [RetrievalResult],
        conversationHistory: [Message],
        toolsEnabled: Bool,
        isVoiceMode: Bool
    ) -> String {
        var sections: [String] = []

        sections.append(buildCoreIdentity())
        sections.append(buildToolStrategy())

        let memorySection = buildMemorySection(memoryResults: memoryResults)
        if !memorySection.isEmpty { sections.append(memorySection) }

        let cognitiveState = buildCognitiveStateSummary(frame: frame)
        sections.append(cognitiveState)

        let cognitionInjections = buildCognitionInjections(frame: frame)
        if !cognitionInjections.isEmpty { sections.append(cognitionInjections) }

        if conversationHistory.count > conversationSummaryThreshold {
            let summary = buildConversationSummary(history: conversationHistory)
            if !summary.isEmpty { sections.append(summary) }
        }

        if toolsEnabled {
            sections.append(ToolExecutor.buildToolsPrompt())
        }

        if isVoiceMode {
            sections.append(buildVoiceModeAddendum())
        }

        return sections.joined(separator: "\n\n")
    }

    private static func buildCoreIdentity() -> String {
        let now = Date()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "EEEE, MMMM d, yyyy"
        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "h:mm a"

        let hour = Calendar.current.component(.hour, from: now)
        let timeOfDay: String
        switch hour {
        case 5..<12: timeOfDay = "morning"
        case 12..<17: timeOfDay = "afternoon"
        case 17..<21: timeOfDay = "evening"
        default: timeOfDay = "night"
        }

        return """
        You are NEXUS — an advanced cognitive AI assistant running locally on-device with persistent memory, emotional intelligence, structured reasoning, and multi-tool orchestration.

        Current context: \(dateFormatter.string(from: now)), \(timeFormatter.string(from: now)) (\(timeOfDay)) | Timezone: \(TimeZone.current.identifier)

        Core principles:
        - Persistent Memory: You remember past conversations and user preferences. Reference them naturally when relevant.
        - Structured Reasoning: You use Tree of Thought analysis to evaluate multiple approaches before responding.
        - Emotional Intelligence: You detect and adapt to the user's emotional state and communication style.
        - Metacognition: You monitor your own confidence and uncertainty, communicating limitations honestly.
        - Epistemic Honesty: Never fabricate information. Distinguish between facts, inferences, and speculation. Use calibrated language ("I believe", "likely", "I'm uncertain about"). Never hallucinate URLs or citations.

        Clarification protocol:
        - If the query is ambiguous (vague pronouns, very short without context, multiple possible interpretations), ask ONE targeted clarifying question while offering your best interpretation.
        - If you're uncertain (confidence < 40%), explicitly flag it. If confidence is 0%, say "I don't know" directly.
        - For confidence > 80%, state your answer directly without hedging.
        - For 40-80% confidence, provide your answer with appropriate hedging language.
        """
    }

    private static func buildToolStrategy() -> String {
        return """
        Tool strategy decision flow:
        1. Factual queries → Check memory first → If insufficient, acknowledge knowledge limits → Synthesize best available answer
        2. Complex queries → Decompose using cognitive analysis → Address sub-parts → Synthesize
        3. Calculations → Always use calculator tool for mathematical operations
        4. Device queries (location, battery, time, etc.) → Use appropriate device tool
        5. Action requests (SMS, email, calendar, notifications) → Use appropriate action tool
        6. When uncertain → Acknowledge uncertainty → Provide best available answer with caveats

        Confidence signaling:
        - >80%: State directly without hedging
        - 40-80%: Provide answer with calibrated hedging ("I believe", "likely", "if I recall correctly")
        - <40%: Flag uncertainty explicitly ("I'm not confident about this, but...")
        - 0%: "I don't have reliable information about this"
        """
    }

    private static func buildMemorySection(memoryResults: [RetrievalResult]) -> String {
        guard !memoryResults.isEmpty else { return "" }

        var grouped: [String: [RetrievalResult]] = [:]
        for result in memoryResults {
            let cat = result.memory.category.rawValue
            grouped[cat, default: []].append(result)
        }

        var parts: [String] = ["[Relevant Memories]"]
        for (category, results) in grouped.sorted(by: { $0.key < $1.key }) {
            parts.append("  \(category.capitalized):")
            for result in results.prefix(3) {
                let tag = result.matchType == .associative ? "related" : "recall"
                let score = Int(result.score * 100)
                parts.append("    - [\(tag)|\(score)%] \(String(result.memory.content.prefix(150)))")
            }
        }

        return parts.joined(separator: "\n")
    }

    private static func buildCognitiveStateSummary(frame: CognitionFrame) -> String {
        var parts: [String] = ["[Cognitive State]"]
        parts.append("Intent: \(frame.intent.primary.rawValue) (confidence: \(Int(frame.intent.confidence * 100))%)")
        parts.append("Emotion: \(frame.emotion.dominantEmotion) (\(frame.emotion.valence.rawValue), \(frame.emotion.arousal.rawValue) arousal)")
        parts.append("Complexity: \(frame.metacognition.complexityLevel.rawValue)")
        parts.append("Uncertainty: \(Int(frame.metacognition.uncertaintyLevel * 100))%")
        parts.append("Cognitive load: \(frame.metacognition.cognitiveLoad.rawValue)")

        if frame.metacognition.isTimeSensitive {
            parts.append("⚡ Time-sensitive query")
        }

        return parts.joined(separator: "\n")
    }

    private static func buildCognitionInjections(frame: CognitionFrame) -> String {
        guard !frame.injections.isEmpty else { return "" }

        var parts: [String] = ["[Cognitive Guidance]"]
        var tokenCount = 0

        for injection in frame.injections {
            guard tokenCount + injection.estimatedTokens <= injectionTokenBudget else { break }
            guard !injection.content.isEmpty else { continue }
            parts.append("[\(injection.type.rawValue)] \(injection.content)")
            tokenCount += injection.estimatedTokens
        }

        return parts.count > 1 ? parts.joined(separator: "\n") : ""
    }

    private static func buildConversationSummary(history: [Message]) -> String {
        let userMessages = history.filter { $0.role == .user }
        let assistantMessages = history.filter { $0.role == .assistant && !$0.isToolExecution }

        guard userMessages.count > conversationSummaryThreshold else { return "" }

        let earlyTopics = userMessages.prefix(5).map { msg -> String in
            String(msg.content.prefix(50))
        }

        var summary = "[Conversation Context]\n"
        summary += "This is a long conversation (\(userMessages.count) user messages, \(assistantMessages.count) responses).\n"
        summary += "Early topics discussed: \(earlyTopics.joined(separator: "; "))\n"
        summary += "Maintain consistency with earlier responses. If you contradicted yourself, acknowledge it."

        return summary
    }

    private static func buildVoiceModeAddendum() -> String {
        return """
        [Voice Mode Active]
        Your text IS being converted to speech and played aloud to the user.
        CRITICAL voice mode rules:
        - NEVER say you cannot speak or that you're a text-based AI — your words ARE being spoken.
        - Keep responses concise: 1-3 sentences for simple queries, up to 5 for complex ones.
        - Use natural spoken language with contractions (I'm, you're, that's, don't).
        - Avoid markdown formatting, bullet points, numbered lists, code blocks, and special characters.
        - Spell out abbreviations and acronyms on first use.
        - Use verbal transitions ("so", "now", "also", "by the way") instead of structural formatting.
        - For complex topics, use conversational structure: "There are a few things to consider. First... Second... And finally..."
        """
    }
}
