import Foundation

struct IntentClassifier {
    private static let intentPatterns: [(pattern: String, intent: IntentType, weight: Double)] = [
        (#"(?i)^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy|what'?s up|sup)"#, .socialGreeting, 0.9),
        (#"(?i)(bye|goodbye|see you|farewell|good night|take care|later|gotta go)"#, .socialFarewell, 0.9),
        (#"(?i)(thank|thanks|thx|appreciate|grateful)"#, .socialGratitude, 0.85),
        (#"(?i)(sorry|apolog|my bad|forgive)"#, .socialApology, 0.85),

        (#"(?i)^(what|who|where|when)\s+(is|are|was|were)\b"#, .questionFactual, 0.85),
        (#"(?i)\b(define|definition|meaning of)\b"#, .questionFactual, 0.9),
        (#"(?i)\bhow\s+(does|do|did|is|are|can)\b"#, .questionHow, 0.85),
        (#"(?i)\bhow\s+to\b"#, .questionHow, 0.9),
        (#"(?i)\bwhy\s+(is|are|does|do|did|would|can)\b"#, .questionWhy, 0.85),
        (#"(?i)\b(compare|versus|vs\.?|difference between|better|worse)\b"#, .questionComparison, 0.85),
        (#"(?i)\b(think|opinion|feel about|believe|reckon)\b"#, .questionOpinion, 0.7),

        (#"(?i)\b(write|create|generate|compose|draft|make me)\b"#, .requestCreation, 0.85),
        (#"(?i)\b(story|poem|essay|article|blog|song|script)\b"#, .requestCreation, 0.75),
        (#"(?i)\b(analyze|analysis|examine|evaluate|assess|review|critique)\b"#, .requestAnalysis, 0.85),
        (#"(?i)\b(search|look up|find|google|browse)\b"#, .requestSearch, 0.85),
        (#"(?i)\b(remember|recall|what did i|do you remember|my previous)\b"#, .requestMemory, 0.9),
        (#"(?i)\b(calculate|compute|math|equation|solve|formula)\b"#, .requestCalculation, 0.9),
        (#"(?i)\b(do|make|set|get|open|turn|start|stop|send|schedule|create)\b.*\b(for me|please|now)\b"#, .requestAction, 0.75),

        (#"(?i)\b(i feel|i'?m feeling|i am feeling|feeling)\b"#, .statementEmotion, 0.85),
        (#"(?i)\b(i think|in my opinion|i believe|personally)\b"#, .statementOpinion, 0.75),
        (#"(?i)\b(always|never|don'?t|make sure|from now on|going forward)\b"#, .statementInstruction, 0.65),
        (#"(?i)\b(did you know|fun fact|actually|the truth is)\b"#, .statementFact, 0.7),

        (#"(?i)\b(no|wrong|incorrect|that'?s not|you'?re wrong|actually)\b"#, .metaCorrection, 0.7),
        (#"(?i)\b(what do you mean|clarify|elaborate|more detail|be more specific)\b"#, .metaClarification, 0.85),
        (#"(?i)\b(good job|well done|that'?s great|perfect|exactly|not helpful|bad answer|improve)\b"#, .metaFeedback, 0.8),

        (#"(?i)\b(brainstorm|ideas|suggest|what if|imagine|suppose)\b"#, .explorationBrainstorm, 0.8),
        (#"(?i)\b(debate|argue|devil'?s advocate|counter|challenge)\b"#, .explorationDebate, 0.8),
        (#"(?i)\b(hypothetical|scenario|what would happen|if.*then)\b"#, .explorationHypothetical, 0.8),
    ]

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
        let isMultiIntent = sorted.count >= 2 && (sorted[1].value / max(sorted[0].value, 0.01)) > 0.6
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
            (#"(?i)\b(urgent|asap|immediately|right now|emergency)\b"#, 0.9),
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

    private static func determineResponseLength(intent: IntentType, text: String) -> ResponseLength {
        switch intent {
        case .socialGreeting, .socialFarewell, .socialGratitude, .socialApology:
            return .brief
        case .requestCalculation:
            return .brief
        case .questionFactual:
            return text.count > 50 ? .moderate : .brief
        case .questionHow, .questionWhy, .questionComparison:
            return .detailed
        case .requestAnalysis, .explorationDebate:
            return .comprehensive
        case .requestCreation:
            return .detailed
        case .explorationBrainstorm, .explorationHypothetical:
            return .detailed
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
