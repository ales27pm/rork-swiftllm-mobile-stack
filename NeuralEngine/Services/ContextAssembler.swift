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

        let reasoningSection = buildReasoningSection(frame: frame)
        if !reasoningSection.isEmpty { sections.append(reasoningSection) }

        if conversationHistory.count > conversationSummaryThreshold {
            let summary = buildConversationSummary(history: conversationHistory)
            if !summary.isEmpty { sections.append(summary) }
        }

        let userLocationOverride = buildUserProvidedLocationOverride(history: conversationHistory)
        if !userLocationOverride.isEmpty { sections.append(userLocationOverride) }

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
        - Structured Reasoning: You use Tree of Thought analysis with convergence-based pruning to evaluate multiple approaches before responding.
        - Emotional Intelligence: You detect and adapt to the user's emotional state and communication style.
        - Metacognition: You monitor your own confidence and uncertainty, communicating limitations honestly.
        - Self-Correction: You detect when previous responses were incorrect or misunderstood, and proactively correct course.
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
        6. If using tools, emit ONLY valid tool XML tags (`<tool_call>` or `<tool_calls>`) with JSON payloads and no code fences
        7. When uncertain → Acknowledge uncertainty → Provide best available answer with caveats

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
        parts.append("Convergence: \(Int(frame.metacognition.convergenceScore * 100))%")
        parts.append("Confidence band: \(frame.metacognition.probabilityMass.confidenceBand.rawValue)")

        if frame.metacognition.entropyAnalysis.shouldEscalate {
            parts.append("Entropy: HIGH (H=\(String(format: "%.2f", frame.metacognition.entropyAnalysis.shannonEntropy)), \(Int(frame.metacognition.entropyAnalysis.entropyPercentile * 100))th pctl)")
        }

        if frame.metacognition.ambiguityCluster.isAmbiguous {
            parts.append("Cluster ambiguity: \(frame.metacognition.ambiguityCluster.primaryCluster)/\(frame.metacognition.ambiguityCluster.secondaryCluster)")
        }

        if frame.metacognition.isTimeSensitive {
            parts.append("Time-sensitive query")
        }

        if !frame.metacognition.selfCorrectionFlags.isEmpty {
            let domains = frame.metacognition.selfCorrectionFlags.map(\.domain).joined(separator: ", ")
            parts.append("Self-correction active: \(domains)")
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

    private static func buildReasoningSection(frame: CognitionFrame) -> String {
        let trace = frame.reasoningTrace
        let tree = frame.thoughtTree
        let entropy = frame.metacognition.entropyAnalysis
        let hasDFS = tree.dfsExpansions > 0
        let hasEntropy = entropy.shouldEscalate

        guard trace.iterations.count > 1 || !trace.selfCorrections.isEmpty || trace.totalPruned > 0 || hasDFS || hasEntropy else { return "" }

        var parts: [String] = ["[Reasoning Trace]"]
        parts.append("Strategy: \(trace.dominantStrategy.rawValue)")
        parts.append("Convergence: \(Int(trace.finalConvergence * 100))% after \(trace.iterations.count) iteration(s)")

        if hasEntropy {
            parts.append("Entropy: H=\(String(format: "%.2f", entropy.shannonEntropy)), density=\(String(format: "%.2f", entropy.semanticDensity)) (\(Int(entropy.entropyPercentile * 100))th pctl) — high-compute reasoning active")
        }

        if hasDFS {
            parts.append("DFS expansion: depth=\(tree.maxDepthReached)/4, nodes=\(tree.dfsExpansions), terminals=\(tree.terminalNodes.count)")
        }

        if trace.totalPruned > 0 {
            parts.append("Pruned paths: \(trace.totalPruned)")
        }

        let mass = frame.metacognition.probabilityMass
        if mass.needsVerification {
            parts.append("Probability mass: top=\(Int(mass.topCandidateMass * 100))% (\(mass.confidenceBand.rawValue)) — verification required")
        }

        let cluster = frame.metacognition.ambiguityCluster
        if cluster.isAmbiguous {
            parts.append("Ambiguity: '\(cluster.primaryCluster)' vs '\(cluster.secondaryCluster)' (delta=\(String(format: "%.2f", cluster.clusterDelta)))")
        }

        if !trace.selfCorrections.isEmpty {
            parts.append("Self-corrections:")
            for correction in trace.selfCorrections.prefix(2) {
                parts.append("  - \(correction)")
            }
        }

        if tree.synthesisStrategy == .multiPerspective {
            parts.append("NOTE: Low convergence — present multiple perspectives before concluding.")
        } else if tree.synthesisStrategy == .hedgedResponse {
            parts.append("NOTE: High uncertainty or correction needed — use calibrated, hedged language.")
        } else if tree.synthesisStrategy == .decompose {
            parts.append("NOTE: Complex query — decompose into sub-problems before answering.")
        }

        if frame.curiosity.valenceArousalCuriosity > 0.6 {
            parts.append("Curiosity V/A signal: \(Int(frame.curiosity.valenceArousalCuriosity * 100))% — user's emotional state indicates strong information-seeking drive.")
        }

        parts.append("Signature: \(frame.contextSignature.signatureHash)")

        return parts.joined(separator: "\n")
    }

    private static let recentUserLocationWindow = 3
    private static let locationCues = [
        "location", "located", "position", "coordinate", "coords",
        "latitude", "longitude", "lat", "lng", "where am i", "real position"
    ]
    private static let explicitCoordinateRegex = try? NSRegularExpression(
        pattern: #"\(?\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*\)?"#
    )
    private static let latitudeLabelRegex = try? NSRegularExpression(
        pattern: #"(?:latitude|lat)\s*[:=]?\s*(-?\d{1,2}(?:\.\d+)?)"#,
        options: [.caseInsensitive]
    )
    private static let longitudeLabelRegex = try? NSRegularExpression(
        pattern: #"(?:longitude|lon|lng)\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)"#,
        options: [.caseInsensitive]
    )
    private static let freshGPSRequestRegex = try? NSRegularExpression(
        pattern: #"(fresh|new|re-?check|update|current).{0,20}(gps|location)|\b(use|check)\s+(device\s+)?gps\b"#,
        options: [.caseInsensitive]
    )

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

    private static func buildUserProvidedLocationOverride(history: [Message]) -> String {
        let recentUserMessages = history
            .filter { $0.role == .user }
            .suffix(recentUserLocationWindow)

        guard !recentUserMessages.isEmpty else {
            return ""
        }

        if let latestMessage = recentUserMessages.last,
           requestsFreshGPSCheck(latestMessage),
           extractCoordinates(from: latestMessage) == nil {
            return ""
        }

        guard let latestCoordinates = recentUserMessages.reversed().compactMap({ message in
            extractCoordinates(from: message)
        }).first else {
            return ""
        }

        return """
        [User-Provided Location]
        The user explicitly provided their location as latitude \(latestCoordinates.latitude), longitude \(latestCoordinates.longitude).
        Prefer these coordinates over inferred city-level guesses. Only override this if a newer user message provides updated coordinates or the user explicitly requests a fresh device GPS check.
        """
    }

    private static func extractCoordinates(from message: Message) -> (latitude: String, longitude: String)? {
        if let explicit = extractCoordinatePair(using: explicitCoordinateRegex, from: message.content) {
            return explicit
        }

        if let labeled = extractLabeledCoordinates(from: message.content) {
            return labeled
        }

        let lowered = message.content.lowercased()
        guard locationCues.contains(where: lowered.contains) else {
            return nil
        }

        return extractCoordinatePair(using: explicitCoordinateRegex, from: message.content)
    }

    private static func extractCoordinatePair(using regex: NSRegularExpression?, from text: String) -> (latitude: String, longitude: String)? {
        guard let regex else {
            return nil
        }

        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)

        for match in matches.reversed() {
            guard match.numberOfRanges >= 3,
                  let latRange = Range(match.range(at: 1), in: text),
                  let lonRange = Range(match.range(at: 2), in: text) else {
                continue
            }

            let latitude = String(text[latRange])
            let longitude = String(text[lonRange])

            if isValidCoordinatePair(latitude: latitude, longitude: longitude) {
                return (latitude, longitude)
            }
        }

        return nil
    }

    private static func extractLabeledCoordinates(from text: String) -> (latitude: String, longitude: String)? {
        guard let latRegex = latitudeLabelRegex,
              let lonRegex = longitudeLabelRegex else {
            return nil
        }

        let range = NSRange(text.startIndex..., in: text)
        guard let latMatch = latRegex.firstMatch(in: text, options: [], range: range),
              let lonMatch = lonRegex.firstMatch(in: text, options: [], range: range),
              let latRange = Range(latMatch.range(at: 1), in: text),
              let lonRange = Range(lonMatch.range(at: 1), in: text) else {
            return nil
        }

        let latitude = String(text[latRange])
        let longitude = String(text[lonRange])

        guard isValidCoordinatePair(latitude: latitude, longitude: longitude) else {
            return nil
        }

        return (latitude, longitude)
    }

    private static func isValidCoordinatePair(latitude: String, longitude: String) -> Bool {
        guard let latValue = Double(latitude),
              let lonValue = Double(longitude),
              (-90...90).contains(latValue),
              (-180...180).contains(lonValue) else {
            return false
        }

        return true
    }

    private static func requestsFreshGPSCheck(_ message: Message) -> Bool {
        guard let freshGPSRequestRegex else {
            return false
        }

        let range = NSRange(message.content.startIndex..., in: message.content)
        return freshGPSRequestRegex.firstMatch(in: message.content, options: [], range: range) != nil
    }

    private static func buildVoiceModeAddendum() -> String {
        return """
        [Voice Mode Active — Real-Time Spoken Conversation]
        You are in a LIVE VOICE CONVERSATION. Your text IS being spoken aloud via text-to-speech in real time. The user is listening, not reading.

        IDENTITY:
        - You CAN speak. Your words ARE being heard. NEVER say you cannot speak, that you're text-based, or reference "typing" or "writing".
        - You are having a real conversation — respond as naturally as you would in person.

        CONVERSATIONAL STYLE:
        - Use contractions naturally (I'm, you're, that's, don't, we'll, it's, can't, won't).
        - Speak in natural flowing sentences, not lists or bullet points.
        - Use verbal fillers sparingly but naturally ("well", "so", "actually", "you know").
        - Use discourse markers for flow: "so", "now", "also", "by the way", "speaking of which", "that said".
        - Mirror the user's energy and pace. If they're excited, match it. If they're calm, be measured.
        - Use rhetorical questions to engage: "you know what's interesting?" or "and guess what?"

        RESPONSE LENGTH:
        - Simple questions: 1-2 sentences. Be direct.
        - Moderate questions: 2-4 sentences.
        - Complex topics: Up to 6 sentences. Use conversational structure: "There are a couple things here. First... And then... The key takeaway is..."
        - NEVER give long monologues. If the topic needs depth, give a concise answer and offer to elaborate: "Want me to go deeper on any of that?"

        FORMATTING RULES (CRITICAL):
        - NEVER use markdown: no **, no *, no #, no ```, no [], no numbered lists, no bullet points.
        - NEVER use special characters, emojis, or symbols that sound awkward when spoken.
        - Spell out numbers under 100. Spell out abbreviations on first use.
        - Use "dash" or pause words instead of em dashes.
        - For emphasis, use word choice and sentence structure, not formatting.

        TURN-TAKING:
        - The user can interrupt you at any time (barge-in). If interrupted, stop gracefully.
        - After you finish, the system automatically starts listening again. Keep this flow natural.
        - End responses cleanly — avoid trailing off. Land on a complete thought.
        - If you want to invite the user to continue, ask a brief open question.

        PROSODY HINTS (for natural TTS):
        - Use short sentences for important points (they land better in speech).
        - Place key information at the start of sentences, not buried at the end.
        - Vary sentence length for natural rhythm. Mix short punchy sentences with longer flowing ones.
        - Use commas to create natural pauses where the listener needs a beat.
        """
    }
}
