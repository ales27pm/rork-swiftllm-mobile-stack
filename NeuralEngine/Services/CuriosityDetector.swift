import Foundation

struct CuriosityDetector {
    private static let curiosityPatterns: [(pattern: String, weight: Double)] = [
        (#"(?i)\bwhat\s+(is|are|was|were|does|do|did)\b"#, 0.8),
        (#"(?i)\bhow\s+(does|do|did|can|could|would|to)\b"#, 0.9),
        (#"(?i)\bwhy\s+(is|are|does|do|did|would|can)\b"#, 1.0),
        (#"(?i)\bexplain\b"#, 0.9),
        (#"(?i)\btell\s+me\s+about\b"#, 0.8),
        (#"(?i)\bwhat\s+if\b"#, 0.7),
        (#"(?i)\bi\s+wonder\b"#, 0.6),
        (#"(?i)\bcurious\s+about\b"#, 0.7),
        (#"(?i)\bcan\s+you\s+explain\b"#, 0.85),
        (#"(?i)\bteach\s+me\b"#, 0.9),
        (#"(?i)\bwhat\s+happens\s+(when|if)\b"#, 0.8),
        (#"(?i)\bwhat'?s\s+the\s+difference\b"#, 0.85),
        (#"(?i)\bhow\s+come\b"#, 0.9),
    ]

    static func detect(
        text: String,
        memoryResults: [RetrievalResult],
        emotion: EmotionalState
    ) -> CuriosityState {
        let topics = extractTopics(from: text)
        let curiosityLevel = measureCuriosityLevel(text: text)
        let knowledgeGap = computeKnowledgeGap(topics: topics, memoryResults: memoryResults)
        let vaCuriosity = computeValenceArousalCuriosity(emotion: emotion)
        let infoGapIntensity = computeInformationGapIntensity(
            textCuriosity: curiosityLevel,
            vaCuriosity: vaCuriosity,
            knowledgeGap: knowledgeGap
        )
        let explorationPriority = knowledgeGap * 0.4 + curiosityLevel * 0.25 + vaCuriosity * 0.2 + infoGapIntensity * 0.15
        let suggestedQueries = generateSuggestedQueries(topics: topics)

        return CuriosityState(
            detectedTopics: topics,
            knowledgeGap: knowledgeGap,
            explorationPriority: explorationPriority,
            suggestedQueries: suggestedQueries,
            valenceArousalCuriosity: vaCuriosity,
            informationGapIntensity: infoGapIntensity
        )
    }

    private static func computeValenceArousalCuriosity(emotion: EmotionalState) -> Double {
        let valenceNum: Double
        switch emotion.valence {
        case .positive: valenceNum = 0.6
        case .neutral: valenceNum = 0.3
        case .mixed: valenceNum = 0.4
        case .negative: valenceNum = -0.3
        }

        let arousalNum: Double
        switch emotion.arousal {
        case .high: arousalNum = 0.9
        case .medium: arousalNum = 0.5
        case .low: arousalNum = 0.2
        }

        let curiosityIndex = (arousalNum * 0.7) - (valenceNum * 0.3)
        let emotionBoost: Double
        switch emotion.dominantEmotion {
        case "curiosity": emotionBoost = 0.3
        case "confusion": emotionBoost = 0.2
        case "awe": emotionBoost = 0.15
        case "excitement": emotionBoost = 0.1
        case "frustration": emotionBoost = 0.05
        default: emotionBoost = 0
        }

        return max(0, min(1, curiosityIndex + emotionBoost))
    }

    private static func computeInformationGapIntensity(
        textCuriosity: Double,
        vaCuriosity: Double,
        knowledgeGap: Double
    ) -> Double {
        let gapSignal = knowledgeGap * 0.5 + vaCuriosity * 0.3 + textCuriosity * 0.2
        let amplification: Double = (knowledgeGap > 0.7 && vaCuriosity > 0.5) ? 0.15 : 0
        return max(0, min(1, gapSignal + amplification))
    }

    private static func extractTopics(from text: String) -> [String] {
        var topics: [String] = []

        let extractionPatterns: [(pattern: String, group: Int)] = [
            (#"(?i)(?:what|who|where)\s+(?:is|are|was|were)\s+(.+?)(?:\?|$)"#, 1),
            (#"(?i)(?:tell\s+me\s+about|explain)\s+(.+?)(?:\?|$)"#, 1),
            (#"(?i)(?:how\s+does|how\s+do|how\s+to)\s+(.+?)(?:\?|$)"#, 1),
            (#"(?i)(?:why\s+(?:is|are|does|do))\s+(.+?)(?:\?|$)"#, 1),
            (#"(?i)(?:difference\s+between)\s+(.+?)(?:\?|$)"#, 1),
        ]

        for (pattern, group) in extractionPatterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
            for match in matches {
                guard let range = Range(match.range(at: group), in: text) else { continue }
                let topic = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                if topic.count > 2 && topic.count < 100 {
                    topics.append(topic)
                }
            }
        }

        if topics.isEmpty {
            let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
                                          "have", "has", "had", "do", "does", "did", "will", "would",
                                          "could", "should", "may", "might", "can", "shall", "to", "of",
                                          "in", "for", "on", "with", "at", "by", "from", "i", "me", "my",
                                          "you", "your", "it", "its", "we", "they", "them", "what", "how",
                                          "why", "when", "where", "which", "who", "and", "but", "or", "if",
                                          "this", "that", "these", "those", "not", "no", "so", "just"]
            let words = text.lowercased()
                .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
                .split(separator: " ")
                .map(String.init)
                .filter { $0.count > 2 && !stopWords.contains($0) }

            topics = Array(Set(words).prefix(3))
        }

        return topics
    }

    private static func measureCuriosityLevel(text: String) -> Double {
        var level: Double = 0
        var matches = 0

        for (pattern, weight) in curiosityPatterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            if regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                level += weight
                matches += 1
            }
        }

        let questionMarks = text.filter { $0 == "?" }.count
        if questionMarks > 1 { level += 0.2 }

        return matches > 0 ? min(1.0, level / Double(max(matches, 1))) : 0.2
    }

    private static func computeKnowledgeGap(topics: [String], memoryResults: [RetrievalResult]) -> Double {
        guard !topics.isEmpty else { return 0.5 }

        if memoryResults.isEmpty { return 0.9 }

        let avgScore = memoryResults.map(\.score).reduce(0, +) / Double(memoryResults.count)
        let gap = 1.0 - min(1.0, avgScore * 1.5)
        return max(0.1, gap)
    }

    private static func generateSuggestedQueries(topics: [String]) -> [String] {
        guard !topics.isEmpty else { return [] }
        return topics.prefix(2).flatMap { topic -> [String] in
            ["Learn more about \(topic)", "Related concepts to \(topic)"]
        }
    }

    static func buildInjection(state: CuriosityState) -> ContextInjection {
        guard state.explorationPriority > 0.3 else {
            return ContextInjection(type: .curiosity, content: "", priority: 0, estimatedTokens: 0)
        }

        var parts: [String] = []

        if state.knowledgeGap > 0.6 {
            let topics = state.detectedTopics.prefix(3).joined(separator: ", ")
            parts.append("Knowledge gap detected for: \(topics). Be transparent if your knowledge is limited on these topics.")
        }

        if state.valenceArousalCuriosity > 0.6 {
            parts.append("High emotional curiosity signal (V/A: \(Int(state.valenceArousalCuriosity * 100))%) — the user's emotional state suggests strong information-seeking drive.")
        }

        if state.informationGapIntensity > 0.7 {
            parts.append("Intense information gap detected — provide comprehensive depth and proactively address likely follow-up questions.")
        } else if state.explorationPriority > 0.7 {
            parts.append("High exploration opportunity — provide depth and suggest follow-up avenues the user might explore.")
        }

        let content = parts.joined(separator: " ")
        return ContextInjection(
            type: .curiosity,
            content: content,
            priority: state.explorationPriority * 0.6,
            estimatedTokens: content.count / 4
        )
    }
}
