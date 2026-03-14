import Foundation

struct MetacognitionEngine {
    static func assess(
        text: String,
        conversationHistory: [Message],
        memoryResults: [RetrievalResult]
    ) -> MetacognitionState {
        let complexity = assessComplexity(text: text)
        let ambiguity = detectAmbiguity(text: text, history: conversationHistory)
        let uncertainty = computeUncertainty(text: text, memoryCount: memoryResults.count)
        let cognitiveLoad = assessCognitiveLoad(complexity: complexity, historyCount: conversationHistory.count)
        let confidence = computeConfidence(uncertainty: uncertainty, memoryCount: memoryResults.count)
        let timeSensitive = detectTimeSensitivity(text: text)
        let knowledgeLimit = memoryResults.isEmpty && containsKnowledgeQuery(text)

        let shouldDecompose = complexity == .complex || complexity == .expert
        let shouldClarify = ambiguity.detected && confidence < 0.5
        let shouldSearch = knowledgeLimit || uncertainty > 0.6

        return MetacognitionState(
            complexityLevel: complexity,
            uncertaintyLevel: uncertainty,
            cognitiveLoad: cognitiveLoad,
            confidenceCalibration: confidence,
            shouldDecompose: shouldDecompose,
            shouldSeekClarification: shouldClarify,
            shouldSearchWeb: shouldSearch,
            isTimeSensitive: timeSensitive,
            ambiguityDetected: ambiguity.detected,
            ambiguityReasons: ambiguity.reasons,
            knowledgeLimitHit: knowledgeLimit
        )
    }

    private static func assessComplexity(text: String) -> ComplexityLevel {
        var score: Double = 0
        let words = text.split(separator: " ")
        let wordCount = words.count

        if wordCount > 50 { score += 2 }
        else if wordCount > 25 { score += 1 }

        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        if sentences.count > 3 { score += 1 }

        let complexPatterns = [
            #"(?i)\b(compare|contrast|analyze|evaluate|synthesize|assess)\b"#,
            #"(?i)\b(relationship|correlation|causation|implication|consequence)\b"#,
            #"(?i)\b(however|although|nevertheless|on the other hand|conversely)\b"#,
            #"(?i)\b(step.by.step|first.*then.*finally|multiple|several)\b"#,
            #"(?i)\b(trade.?off|pros.and.cons|advantages.and.disadvantages)\b"#,
            #"(?i)\b(design|architect|implement|build|create.*system)\b"#,
        ]

        for pattern in complexPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                score += 1
            }
        }

        let expertPatterns = [
            #"(?i)\b(prove|derive|theorem|axiom|lemma)\b"#,
            #"(?i)\b(differential|integral|eigenvalue|topology)\b"#,
            #"(?i)\b(quantum|relativity|entropy|thermodynamic)\b"#,
            #"(?i)\b(constitutional|jurisprudence|precedent)\b"#,
        ]

        for pattern in expertPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                score += 2
            }
        }

        if text.contains("?") && text.filter({ $0 == "?" }).count > 1 { score += 1 }

        if score >= 6 { return .expert }
        if score >= 4 { return .complex }
        if score >= 2 { return .moderate }
        return .simple
    }

    private static func detectAmbiguity(text: String, history: [Message]) -> (detected: Bool, reasons: [String]) {
        var reasons: [String] = []
        let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = lower.split(separator: " ")

        let pronounStarters = ["it", "this", "that", "they", "them", "those", "these", "he", "she"]
        if let firstWord = words.first, pronounStarters.contains(String(firstWord)) {
            let hasRecentContext = history.suffix(4).contains { $0.role == .user || $0.role == .assistant }
            if !hasRecentContext {
                reasons.append("Starts with pronoun '\(firstWord)' without clear referent")
            }
        }

        if words.count <= 3 && !lower.contains("hi") && !lower.contains("hello") && !lower.contains("hey") {
            reasons.append("Very short query — may need more context")
        }

        if lower.hasPrefix("do ") || lower.hasPrefix("can ") || lower.hasPrefix("should ") {
            if !lower.contains("?") {
                reasons.append("Imperative without explicit question — intent unclear")
            }
        }

        let vaguePatterns = [
            #"(?i)^(help|fix|change|update|make)\s"#,
            #"(?i)\b(something|stuff|things|whatever)\b"#,
            #"(?i)\b(the thing|the one|the other)\b"#,
        ]

        for pattern in vaguePatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                reasons.append("Contains vague reference that could have multiple interpretations")
                break
            }
        }

        return (detected: !reasons.isEmpty, reasons: reasons)
    }

    private static func computeUncertainty(text: String, memoryCount: Int) -> Double {
        var uncertainty: Double = 0.3

        if memoryCount == 0 { uncertainty += 0.2 }
        else if memoryCount < 3 { uncertainty += 0.1 }

        let uncertainPatterns = [
            #"(?i)\b(latest|newest|current|recent|today|this week|this month|2024|2025|2026)\b"#,
            #"(?i)\b(price|cost|stock|weather|score|result)\b"#,
            #"(?i)\b(breaking|news|update|announcement)\b"#,
        ]

        for pattern in uncertainPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                uncertainty += 0.15
            }
        }

        let certainPatterns = [
            #"(?i)\b(define|definition|what is|explain|how does)\b"#,
            #"(?i)\b(math|calculate|compute|convert)\b"#,
        ]

        for pattern in certainPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                uncertainty -= 0.1
            }
        }

        return max(0, min(1, uncertainty))
    }

    private static func assessCognitiveLoad(complexity: ComplexityLevel, historyCount: Int) -> CognitiveLoad {
        var load: Int = 0
        switch complexity {
        case .simple: load += 1
        case .moderate: load += 2
        case .complex: load += 3
        case .expert: load += 4
        }
        if historyCount > 20 { load += 2 }
        else if historyCount > 10 { load += 1 }

        if load >= 5 { return .overload }
        if load >= 4 { return .high }
        if load >= 2 { return .medium }
        return .low
    }

    private static func computeConfidence(uncertainty: Double, memoryCount: Int) -> Double {
        var confidence = 1.0 - uncertainty
        if memoryCount > 3 { confidence += 0.1 }
        return max(0, min(1, confidence))
    }

    private static func detectTimeSensitivity(text: String) -> Bool {
        let patterns = [
            #"(?i)\b(now|right now|immediately|asap|urgent|today|tonight|this morning|this afternoon)\b"#,
            #"(?i)\b(deadline|due|expires|expiring|hurry|quick|fast)\b"#,
        ]
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                return true
            }
        }
        return false
    }

    private static func containsKnowledgeQuery(_ text: String) -> Bool {
        let patterns = [
            #"(?i)\b(who is|what is|where is|when did|how does|why does|tell me about)\b"#,
            #"(?i)\b(explain|describe|define|what are|how do)\b"#,
        ]
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                return true
            }
        }
        return false
    }

    static func buildInjection(state: MetacognitionState) -> ContextInjection {
        var parts: [String] = []

        if state.uncertaintyLevel > 0.6 {
            let pct = Int(state.uncertaintyLevel * 100)
            parts.append("Uncertainty elevated (\(pct)%). Hedge appropriately — use calibrated language ('I believe', 'likely', 'I'm not certain').")
        }

        if state.shouldDecompose {
            parts.append("This is a \(state.complexityLevel.rawValue)-level query. Decompose into sub-problems before answering.")
        }

        if state.shouldSeekClarification {
            let reasons = state.ambiguityReasons.prefix(2).joined(separator: "; ")
            parts.append("Ambiguity detected: \(reasons). Consider asking a targeted clarifying question.")
        }

        if state.knowledgeLimitHit {
            parts.append("No stored knowledge matches this query. Be transparent about limitations. Suggest the user verify critical facts.")
        }

        if state.isTimeSensitive {
            parts.append("Time-sensitive query detected. Prioritize actionable, concise responses.")
        }

        if state.cognitiveLoad == .overload || state.cognitiveLoad == .high {
            parts.append("High cognitive load context. Structure your response with clear sections or numbered steps.")
        }

        let content = parts.joined(separator: " ")
        let priority: Double = state.uncertaintyLevel > 0.6 ? 0.85 : (state.shouldDecompose ? 0.7 : 0.4)

        return ContextInjection(
            type: .metacognition,
            content: content,
            priority: content.isEmpty ? 0 : priority,
            estimatedTokens: content.count / 4
        )
    }
}
