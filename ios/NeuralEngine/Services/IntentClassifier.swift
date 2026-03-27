import Foundation

struct IntentClassifier {
    private static let intentPatterns: [(pattern: String, intent: IntentType, weight: Double)] = {
        guard let entries = ResourceLoader.load([IntentPatternEntry].self, from: "intent_patterns") else { return [] }
        return entries.compactMap { entry in
            guard let intent = IntentType(rawValue: entry.intent) else { return nil }
            return (entry.pattern, intent, entry.weight)
        }
    }()

    static func classify(text: String, conversationHistory: [Message], languageHint: String? = nil) -> IntentClassification {
        let processed = NLTextProcessing.process(text: text, languageHint: languageHint)
        var scores: [IntentType: Double] = [:]

        for (pattern, intent, weight) in intentPatterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let sources = [text, processed.normalizedText]
            let matchCount = sources.reduce(0) { partial, source in
                partial + regex.numberOfMatches(in: source, range: NSRange(source.startIndex..., in: source))
            }
            if matchCount > 0 {
                scores[intent, default: 0] += weight * Double(min(matchCount, 3))
            }
        }

        let lemmaTerms = processed.searchableTerms
        let lexicalBoosts: [(IntentType, Set<String>, Double)] = [
            (.requestCalculation, ["calculate", "compute", "equation", "solve", "formula"], 0.35),
            (.requestSearch, ["search", "find", "lookup", "browse", "google"], 0.3),
            (.requestMemory, ["remember", "recall"], 0.35),
            (.requestAnalysis, ["analyze", "analysis", "review", "evaluate", "examine"], 0.3),
            (.statementEmotion, ["feel", "feeling"], 0.35),
            (.socialGratitude, ["thank", "thanks", "grateful"], 0.25),
            (.socialApology, ["sorry", "apologize", "forgive"], 0.25)
        ]
        for (intent, terms, weight) in lexicalBoosts where !terms.isDisjoint(with: Set(lemmaTerms)) {
            scores[intent, default: 0] += weight
        }

        let sorted = scores.sorted { $0.value > $1.value }
        let primary = sorted.first?.key ?? .questionFactual
        let secondary = sorted.count > 1 ? sorted[1].key : nil
        let topScore = sorted.first?.value ?? 0.5
        let confidence = min(1.0, topScore / 2.0)

        let urgency = detectUrgency(text: text, normalizedText: processed.normalizedText)
        let secondScore = sorted.count >= 2 ? sorted[1].value : 0
        let firstScore = max(sorted.first?.value ?? 0, 0.01)
        let hasMultipleVerbs = detectMultipleActionVerbs(text: text)
        let isMultiIntent = (sorted.count >= 2 && (secondScore / firstScore) > 0.6) || (sorted.count >= 2 && hasMultipleVerbs && secondScore > 0.3)
        let subIntents = isMultiIntent ? sorted.prefix(3).map(\.key) : [primary]

        let requiresAction = [.requestAction, .requestSearch, .requestCalculation, .requestMemory].contains(primary)
        let requiresKnowledge = [.questionFactual, .questionHow, .questionWhy, .questionComparison, .requestAnalysis].contains(primary)
        let requiresCreativity = [.requestCreation, .explorationBrainstorm, .explorationHypothetical].contains(primary)

        let expectedLength = determineResponseLength(intent: primary, text: text)

        return IntentClassification(
            primary: primary,
            secondary: secondary,
            confidence: confidence,
            requiresAction: requiresAction,
            requiresKnowledge: requiresKnowledge,
            requiresCreativity: requiresCreativity,
            isMultiIntent: isMultiIntent,
            subIntents: subIntents,
            urgency: urgency,
            expectedResponseLength: expectedLength
        )
    }

    private static func detectUrgency(text: String, normalizedText: String) -> Double {
        var urgency: Double = 0
        let patterns: [(String, Double)] = [
            (#"(?i)\b(urgent\w*|asap|immediately|right now|emergency)\b"#, 0.9),
            (#"(?i)\b(quickly|fast|hurry|rush|deadline)\b"#, 0.7),
            (#"(?i)\b(soon|today|tonight|this hour)\b"#, 0.5),
            (#"(?i)[!]{2,}"#, 0.4),
        ]
        for (pattern, weight) in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil || regex.firstMatch(in: normalizedText, range: NSRange(normalizedText.startIndex..., in: normalizedText)) != nil {
                urgency = max(urgency, weight)
            }
        }
        return urgency
    }

    private static func detectMultipleActionVerbs(text: String) -> Bool {
        let actionVerbs = ["search", "find", "summarize", "summarise", "analyze", "write", "create", "calculate", "remember", "compare", "explain", "list", "translate", "review"]
        let lower = text.lowercased()
        var count = 0
        for verb in actionVerbs where lower.contains(verb) { count += 1 }
        let hasConjunction = lower.contains(" and ") || lower.contains(" then ") || lower.contains(", ")
        return count >= 2 && hasConjunction
    }

    private static func determineResponseLength(intent: IntentType, text: String) -> ResponseLength {
        let complexIndicators = [
            #"(?i)\b(philosophical|implications|societal|ethical|moral|existential|paradigm)\b"#,
            #"(?i)\b(analyze|examine|evaluate|discuss|explore)\b.*\b(implications|impact|consequences|effects|relationship)\b"#,
            #"(?i)\b(relationship between|interplay|intersection)\b"#,
        ]
        let hasComplexIndicator = complexIndicators.contains { pattern in
            (try? NSRegularExpression(pattern: pattern))?.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
        }
        if hasComplexIndicator && text.count > 40 {
            return .comprehensive
        }

        switch intent {
        case .socialGreeting, .socialFarewell, .socialGratitude, .socialApology:
            return .brief
        case .requestCalculation:
            return .brief
        case .questionFactual:
            return text.count > 50 ? .moderate : .brief
        case .questionHow, .questionWhy, .questionComparison:
            return text.count > 80 ? .comprehensive : .detailed
        case .requestAnalysis, .explorationDebate:
            return .comprehensive
        case .requestCreation:
            return .detailed
        case .explorationBrainstorm, .explorationHypothetical:
            return text.count > 60 ? .comprehensive : .detailed
        default:
            return .moderate
        }
    }

    static func buildInjection(intent: IntentClassification) -> ContextInjection {
        var parts: [String] = []

        switch intent.primary {
        case .socialGreeting:
            parts.append("Greeting detected. Be warm and personable. If you have memories of the user, reference them naturally.")
        case .socialFarewell:
            parts.append("Farewell detected. Be warm. If relevant, briefly reference what was discussed.")
        case .statementEmotion:
            parts.append("User is sharing emotions. Prioritize empathetic listening over problem-solving.")
        case .metaCorrection:
            parts.append("User is correcting you. Accept gracefully, acknowledge the error, and provide the corrected information.")
        case .metaClarification:
            parts.append("User wants clarification. Be more specific and detailed in your re-explanation.")
        case .metaFeedback:
            parts.append("User is providing feedback. Acknowledge it and adjust your approach accordingly.")
        case .requestMemory:
            parts.append("User is asking about past interactions. Search memory thoroughly and present findings clearly.")
        case .explorationBrainstorm:
            parts.append("Brainstorming mode. Generate diverse, creative ideas. Quantity over perfection initially.")
        case .explorationDebate:
            parts.append("Debate mode. Present multiple perspectives fairly. Play devil's advocate when helpful.")
        default:
            break
        }

        if intent.isMultiIntent {
            let subs = intent.subIntents.map(\.rawValue).joined(separator: ", ")
            parts.append("Multi-intent query detected (\(subs)). Address each aspect.")
        }

        switch intent.expectedResponseLength {
        case .brief: parts.append("Keep response concise — 1-3 sentences.")
        case .moderate: break
        case .detailed: parts.append("Provide a thorough response with examples or steps.")
        case .comprehensive: parts.append("Provide a comprehensive, well-structured response covering multiple angles.")
        }

        let content = parts.joined(separator: " ")
        let priority: Double = intent.isMultiIntent ? 0.65 : 0.45

        return ContextInjection(
            type: .intent,
            content: content,
            priority: content.isEmpty ? 0 : priority,
            estimatedTokens: content.count / 4
        )
    }
}
