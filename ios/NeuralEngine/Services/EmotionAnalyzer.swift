import Foundation

struct EmotionAnalyzer {
    private static let emotionLexicon: [(pattern: String, valence: Double, arousal: Double, label: String)] = {
        guard let entries = ResourceLoader.load([EmotionLexiconEntry].self, from: "emotion_lexicon") else { return [] }
        return entries.map { ($0.pattern, $0.valence, $0.arousal, $0.label) }
    }()

    private static let stylePatterns: [(pattern: String, style: String)] = {
        guard let entries = ResourceLoader.load([StylePatternEntry].self, from: "style_patterns") else { return [] }
        return entries.map { ($0.pattern, $0.style) }
    }()

    static func analyze(text: String, conversationHistory: [Message], languageHint: String? = nil) -> EmotionalState {
        let processed = NLTextProcessing.process(text: text, languageHint: languageHint)
        let normalized = processed.normalizedText
        let searchableTerms = Set(processed.searchableTerms)

        var totalValence: Double = 0
        var totalArousal: Double = 0
        var matchCount: Double = 0
        var dominantLabel = "neutral"
        var maxWeight: Double = 0

        for entry in emotionLexicon {
            let normalizedPattern = NLTextProcessing.normalizeForMatching(entry.pattern, languageHint: languageHint)
            let matched = normalized.contains(normalizedPattern) || searchableTerms.contains(normalizedPattern)
            if matched {
                totalValence += entry.valence
                totalArousal += entry.arousal
                matchCount += 1
                let weight = abs(entry.valence) + entry.arousal
                if weight > maxWeight {
                    maxWeight = weight
                    dominantLabel = entry.label
                }
            }
        }

        let avgValence = matchCount > 0 ? totalValence / matchCount : 0
        let avgArousal = matchCount > 0 ? totalArousal / matchCount : 0.3

        let valence: EmotionalValence
        if matchCount == 0 {
            valence = .neutral
        } else if avgValence > 0.2 {
            valence = .positive
        } else if avgValence < -0.2 {
            valence = .negative
        } else {
            let hasPos = emotionLexicon.contains { $0.valence > 0.2 && normalized.contains(NLTextProcessing.normalizeForMatching($0.pattern, languageHint: languageHint)) }
            let hasNeg = emotionLexicon.contains { $0.valence < -0.2 && normalized.contains(NLTextProcessing.normalizeForMatching($0.pattern, languageHint: languageHint)) }
            valence = (hasPos && hasNeg) ? .mixed : .neutral
        }

        let arousal: EmotionalArousal
        if avgArousal > 0.7 { arousal = .high }
        else if avgArousal > 0.4 { arousal = .medium }
        else { arousal = .low }

        let style = detectStyle(text: text, normalizedText: normalized)
        let trajectory = detectTrajectory(currentValence: avgValence, history: conversationHistory, languageHint: languageHint)
        let intensityMod = detectIntensityModifier(text)
        let empathyLevel = computeEmpathyLevel(valence: avgValence, arousal: avgArousal, intensityModifier: intensityMod)

        return EmotionalState(
            valence: valence,
            arousal: arousal,
            dominantEmotion: dominantLabel,
            style: style,
            empathyLevel: empathyLevel,
            emotionalTrajectory: trajectory
        )
    }

    private static func detectStyle(text: String, normalizedText: String) -> String {
        var styleCounts: [String: Int] = [:]
        for (pattern, style) in stylePatterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let count = regex.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text)) + regex.numberOfMatches(in: normalizedText, range: NSRange(normalizedText.startIndex..., in: normalizedText))
            if count > 0 {
                styleCounts[style, default: 0] += count
            }
        }
        return styleCounts.max(by: { $0.value < $1.value })?.key ?? "neutral"
    }

    private static func detectTrajectory(currentValence: Double, history: [Message], languageHint: String?) -> String {
        let recentUserMessages = history.suffix(6).filter { $0.role == .user }
        guard recentUserMessages.count >= 2 else { return "stable" }

        var previousValences: [Double] = []
        for msg in recentUserMessages {
            let processed = NLTextProcessing.process(text: msg.content, languageHint: languageHint)
            let normalized = processed.normalizedText
            var v: Double = 0
            var c: Double = 0
            for entry in emotionLexicon {
                let normalizedPattern = NLTextProcessing.normalizeForMatching(entry.pattern, languageHint: languageHint)
                if normalized.contains(normalizedPattern) {
                    v += entry.valence
                    c += 1
                }
            }
            previousValences.append(c > 0 ? v / c : 0)
        }

        guard let first = previousValences.first else { return "stable" }
        let trend = currentValence - first
        if trend > 0.3 { return "improving" }
        if trend < -0.3 { return "declining" }
        if abs(currentValence) > 0.5 && previousValences.allSatisfy({ abs($0) > 0.3 }) { return "sustained_intense" }
        return "stable"
    }

    private static func computeEmpathyLevel(valence: Double, arousal: Double, intensityModifier: Double = 1.0) -> Double {
        if valence < -0.15 {
            let severityBoost = arousal > 0.7 ? 0.15 : 0
            let base = 0.45 + abs(valence) * 0.4 + arousal * 0.25 + severityBoost
            return min(1.0, base * intensityModifier)
        }
        if valence > 0.5 && arousal > 0.6 {
            return 0.5
        }
        return 0.3
    }

    private static func detectIntensityModifier(_ text: String) -> Double {
        let lower = text.lowercased()
        let hedges = ["a bit", "a little", "slightly", "somewhat", "kind of", "kinda", "sort of", "sorta", "maybe", "mildly"]
        let amplifiers = ["very", "really", "extremely", "so ", "incredibly", "absolutely", "completely", "totally"]
        for hedge in hedges where lower.contains(hedge) { return 0.7 }
        for amp in amplifiers where lower.contains(amp) { return 1.15 }
        return 1.0
    }

    static func buildInjection(state: EmotionalState) -> ContextInjection {
        var content = ""

        switch state.valence {
        case .negative:
            switch state.dominantEmotion {
            case "frustration", "anger", "irritation":
                content = "User appears frustrated. Respond with calm empathy. Acknowledge their difficulty before offering solutions. Avoid dismissive language."
            case "sadness", "grief", "loneliness":
                content = "User expresses sadness. Lead with warmth and validation. Avoid toxic positivity. Let them feel heard before redirecting."
            case "anxiety", "fear", "stress":
                content = "User shows anxiety. Use grounding, reassuring language. Break complex topics into manageable steps. Provide structure."
            case "confusion":
                content = "User seems confused. Offer clear, structured explanations. Use analogies. Check understanding incrementally."
            default:
                content = "User's tone suggests distress. Respond with empathy and patience."
            }
        case .positive:
            if state.arousal == .high {
                content = "User is enthusiastic. Match their energy while staying grounded. Build on their momentum."
            } else {
                content = "User has a positive tone. Maintain warmth and engagement."
            }
        case .mixed:
            content = "User shows mixed emotions. Acknowledge the complexity of their feelings. Be balanced in your response."
        case .neutral:
            content = ""
        }

        if state.style == "formal" {
            content += content.isEmpty ? "" : " "
            content += "User communicates formally. Match their register — avoid colloquialisms."
        } else if state.style == "casual" {
            content += content.isEmpty ? "" : " "
            content += "User is casual. You can be conversational, but stay substantive."
        } else if state.style == "technical" {
            content += content.isEmpty ? "" : " "
            content += "User is technical. Use precise terminology. Skip basic explanations unless asked."
        } else if state.style == "urgent" {
            content += content.isEmpty ? "" : " "
            content += "User needs urgency. Prioritize actionable answers. Be concise and direct."
        }

        if state.emotionalTrajectory == "declining" {
            content += " Emotional trajectory is declining — be especially supportive."
        } else if state.emotionalTrajectory == "sustained_intense" {
            content += " Sustained emotional intensity detected — consider gently checking in."
        }

        let priority: Double = state.valence == .negative ? 0.9 : (state.valence == .positive && state.arousal == .high ? 0.5 : 0.3)

        return ContextInjection(
            type: .emotion,
            content: content,
            priority: content.isEmpty ? 0 : priority,
            estimatedTokens: max(0, content.count / 4)
        )
    }
}
