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

        let entropyAnalysis = computeShannonEntropy(text: text, history: conversationHistory)
        let ambiguityCluster = computeAmbiguityCluster(text: text, history: conversationHistory)
        let probabilityMass = computeProbabilityMass(
            text: text,
            uncertainty: uncertainty,
            memoryResults: memoryResults,
            entropyAnalysis: entropyAnalysis
        )

        var adjustedUncertainty = uncertainty
        if probabilityMass.needsVerification {
            adjustedUncertainty = min(1.0, uncertainty + 0.15)
        }
        if entropyAnalysis.shouldEscalate {
            adjustedUncertainty = min(1.0, adjustedUncertainty + 0.1)
        }

        let convergence = computeConvergenceScore(
            text: text,
            complexity: complexity,
            uncertainty: adjustedUncertainty,
            memoryCount: memoryResults.count,
            historyCount: conversationHistory.count
        )
        let selfCorrections = detectSelfCorrectionNeeds(
            text: text,
            history: conversationHistory,
            uncertainty: adjustedUncertainty
        )

        let shouldDecompose = complexity == .complex || complexity == .expert || entropyAnalysis.shouldEscalate
        let shouldClarify = (ambiguity.detected && confidence < 0.5) || ambiguityCluster.isAmbiguous
        let shouldSearch = knowledgeLimit || adjustedUncertainty > 0.6

        return MetacognitionState(
            complexityLevel: complexity,
            uncertaintyLevel: adjustedUncertainty,
            cognitiveLoad: cognitiveLoad,
            confidenceCalibration: confidence,
            convergenceScore: convergence,
            shouldDecompose: shouldDecompose,
            shouldSeekClarification: shouldClarify,
            shouldSearchWeb: shouldSearch,
            isTimeSensitive: timeSensitive,
            ambiguityDetected: ambiguity.detected || ambiguityCluster.isAmbiguous,
            ambiguityReasons: ambiguity.reasons + ambiguityCluster.competingInterpretations,
            knowledgeLimitHit: knowledgeLimit,
            selfCorrectionFlags: selfCorrections,
            entropyAnalysis: entropyAnalysis,
            ambiguityCluster: ambiguityCluster,
            probabilityMass: probabilityMass
        )
    }

    private static func computeConvergenceScore(
        text: String,
        complexity: ComplexityLevel,
        uncertainty: Double,
        memoryCount: Int,
        historyCount: Int
    ) -> Double {
        let complexityWeight: Double
        switch complexity {
        case .simple: complexityWeight = 0.9
        case .moderate: complexityWeight = 0.7
        case .complex: complexityWeight = 0.4
        case .expert: complexityWeight = 0.2
        }

        let memoryBoost = memoryCount > 0 ? min(0.2, Double(memoryCount) * 0.04) : 0
        let historyPenalty = min(0.15, Double(historyCount) * 0.01)

        let sigmoid = 1.0 / (1.0 + exp(-Double(historyCount) + 3.0))
        let raw = complexityWeight * (1.0 - uncertainty) + memoryBoost - historyPenalty + sigmoid * 0.1
        return max(0, min(1, raw))
    }

    private static func detectSelfCorrectionNeeds(
        text: String,
        history: [Message],
        uncertainty: Double
    ) -> [SelfCorrectionFlag] {
        var flags: [SelfCorrectionFlag] = []

        let correctionPatterns: [(pattern: String, domain: String, issue: String)] = [
            (#"(?i)\b(no|wrong|incorrect|that'?s not right|you'?re wrong|actually)\b"#, "factual", "User indicates previous response was incorrect"),
            (#"(?i)\b(i said|i meant|i was asking|not what i meant|misunderstood)\b"#, "comprehension", "User clarifying intent — previous interpretation may be wrong"),
            (#"(?i)\b(again|repeat|already told you|i just said)\b"#, "attention", "User repeating themselves — previous response missed key information"),
            (#"(?i)\b(but earlier|you said|contradicting|inconsistent)\b"#, "consistency", "Potential contradiction with earlier response detected"),
        ]

        for (pattern, domain, issue) in correctionPatterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            if regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                let severity = uncertainty > 0.5 ? 0.9 : 0.7
                flags.append(SelfCorrectionFlag(domain: domain, issue: issue, severity: severity))
            }
        }

        let recentAssistant = history.suffix(4).filter { $0.role == .assistant }
        let recentUser = history.suffix(4).filter { $0.role == .user }
        if recentAssistant.count >= 2 && recentUser.count >= 2 {
            let lastAssistant = recentAssistant.last?.content.lowercased() ?? ""
            let currentLower = text.lowercased()
            if currentLower.contains("no") || currentLower.contains("wrong") || currentLower.contains("not") {
                if lastAssistant.count > 20 {
                    flags.append(SelfCorrectionFlag(
                        domain: "dialogue",
                        issue: "Negative feedback loop detected — user may be dissatisfied with response direction",
                        severity: 0.6
                    ))
                }
            }
        }

        return flags
    }

    private static func assessComplexity(text: String) -> ComplexityLevel {
        var score: Double = 0
        let words = text.split(separator: " ")
        let wordCount = words.count

        if wordCount > 50 { score += 2 }
        else if wordCount > 25 { score += 1.5 }
        else if wordCount > 12 { score += 0.5 }

        let emotionalComplexityPatterns = [
            #"(?i)\b(stressed|frustrated|confused|struggling|can'?t figure|overwhelmed|stuck on)\b"#,
            #"(?i)\b(complex|complicated|difficult|hard|challenging|tricky)\b"#,
        ]
        for pattern in emotionalComplexityPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                score += 0.75
            }
        }

        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        if sentences.count > 3 { score += 1 }

        let moderatePatterns = [
            #"(?i)\b(how does|how do|how is|how are|how can)\b"#,
            #"(?i)\b(explain|describe|elaborate)\b"#,
            #"(?i)\b(why does|why do|why is|why are)\b"#,
            #"(?i)\b(difference|between|versus)\b"#,
        ]

        for pattern in moderatePatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                score += 0.5
            }
        }

        let complexPatterns = [
            #"(?i)\b(compare|contrast|analyze|evaluate|synthesize|assess)\b"#,
            #"(?i)\b(relationship|correlation|causation|implication|consequence)\b"#,
            #"(?i)\b(however|although|nevertheless|on the other hand|conversely)\b"#,
            #"(?i)\b(step.by.step|first.*then.*finally|multiple|several)\b"#,
            #"(?i)\b(trade.?off|pros.and.cons|advantages.and.disadvantages)\b"#,
            #"(?i)\b(design|architect|implement|build|create.*system)\b"#,
            #"(?i)\b(philosophical|implications|consciousness|intelligence)\b"#,
            #"(?i)\b(considering|addressing|while|both.*and)\b"#,
        ]

        for pattern in complexPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                score += 1
            }
        }

        let domainSpecificPatterns = [
            #"(?i)\b(photosynthesis|mitochondria|genome|protein|enzyme|metabolism)\b"#,
            #"(?i)\b(algorithm|neural network|machine learning|compiler|runtime)\b"#,
            #"(?i)\b(incompleteness|godel|turing|decidability|computability)\b"#,
            #"(?i)\b(epistemology|ontology|metaphysics|phenomenology)\b"#,
            #"(?i)\b(macroeconomic|geopolitical|socioeconomic)\b"#,
        ]

        for pattern in domainSpecificPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                score += 1.5
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

        if score >= 5.5 { return .expert }
        if score >= 3.0 { return .complex }
        if score >= 1.2 { return .moderate }
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

        if state.convergenceScore < 0.3 {
            parts.append("Low reasoning convergence — multiple valid interpretations exist. Present alternatives explicitly.")
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

    static func buildSelfCorrectionInjection(flags: [SelfCorrectionFlag]) -> ContextInjection {
        guard !flags.isEmpty else {
            return ContextInjection(type: .selfCorrection, content: "", priority: 0, estimatedTokens: 0)
        }

        var parts: [String] = ["[Self-Correction Active]"]
        for flag in flags.sorted(by: { $0.severity > $1.severity }).prefix(3) {
            parts.append("- [\(flag.domain)] \(flag.issue)")
        }

        let highSeverity = flags.contains { $0.severity > 0.7 }
        if highSeverity {
            parts.append("IMPORTANT: Acknowledge the correction explicitly. Do not repeat the same error. Re-evaluate your assumptions.")
        }

        let content = parts.joined(separator: "\n")
        let maxSeverity = flags.map(\.severity).max() ?? 0
        return ContextInjection(
            type: .selfCorrection,
            content: content,
            priority: min(0.95, maxSeverity),
            estimatedTokens: content.count / 4
        )
    }

    static func buildEntropyInjection(state: MetacognitionState) -> ContextInjection {
        let entropy = state.entropyAnalysis
        let cluster = state.ambiguityCluster
        let mass = state.probabilityMass

        var parts: [String] = []

        if entropy.shouldEscalate {
            parts.append("[Entropy Escalation] Shannon entropy=\(String(format: "%.2f", entropy.shannonEntropy)), semantic density=\(String(format: "%.2f", entropy.semanticDensity)) (\(Int(entropy.entropyPercentile * 100))th percentile). High-compute reasoning iteration required — decompose systematically.")
        }

        if cluster.isAmbiguous {
            let delta = String(format: "%.2f", cluster.clusterDelta)
            parts.append("[Ambiguity Cluster] Competing: '\(cluster.primaryCluster)' (\(Int(cluster.primaryScore * 100))%) vs '\(cluster.secondaryCluster)' (\(Int(cluster.secondaryScore * 100))%), delta=\(delta). Ask a targeted clarifying question with both interpretations.")
        }

        if mass.needsVerification {
            parts.append("[Verification Required] Top candidate mass=\(Int(mass.topCandidateMass * 100))%, band=\(mass.confidenceBand.rawValue). Verify knowledge before final decoding — inject constraint review.")
        }

        guard !parts.isEmpty else {
            return ContextInjection(type: .metacognition, content: "", priority: 0, estimatedTokens: 0)
        }

        let content = parts.joined(separator: "\n")
        let priority: Double = entropy.shouldEscalate ? 0.88 : (cluster.isAmbiguous ? 0.82 : 0.7)
        return ContextInjection(
            type: .metacognition,
            content: content,
            priority: priority,
            estimatedTokens: content.count / 4
        )
    }

    private static func computeShannonEntropy(text: String, history: [Message]) -> EntropyAnalysis {
        let words = text.lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
            .split(separator: " ")
            .map(String.init)
            .filter { $0.count > 1 }

        let totalCount = Double(words.count)
        guard totalCount > 0 else {
            return EntropyAnalysis(shannonEntropy: 0, semanticDensity: 0, tokenConceptRatio: 0, shouldEscalate: false, entropyPercentile: 0)
        }

        var freq: [String: Int] = [:]
        for w in words { freq[w, default: 0] += 1 }

        var entropy: Double = 0
        for (_, count) in freq {
            let p = Double(count) / totalCount
            if p > 0 { entropy -= p * log2(p) }
        }

        let uniqueCount = Double(freq.count)
        let maxEntropy = totalCount > 1 ? log2(totalCount) : 1.0
        let normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0

        let conceptMarkers = [
            #"(?i)\b(because|therefore|however|although|since|while|despite|unless|whereas)\b"#,
            #"(?i)\b(if|then|when|where|which|whose|whom)\b"#,
            #"(?i)\b(analyze|evaluate|compare|contrast|synthesize|implement|design)\b"#,
        ]

        var conceptCount: Double = 0
        for pattern in conceptMarkers {
            if let regex = try? NSRegularExpression(pattern: pattern) {
                conceptCount += Double(regex.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text)))
            }
        }

        let tokenConceptRatio = totalCount > 0 ? conceptCount / totalCount : 0
        let semanticDensity = (uniqueCount / totalCount) * (1.0 + tokenConceptRatio)

        let entropyPercentile: Double
        if normalizedEntropy > 0.85 { entropyPercentile = 0.95 }
        else if normalizedEntropy > 0.7 { entropyPercentile = 0.8 }
        else if normalizedEntropy > 0.5 { entropyPercentile = 0.6 }
        else { entropyPercentile = 0.3 }

        let shouldEscalate = normalizedEntropy > 0.7 || semanticDensity > 0.8

        return EntropyAnalysis(
            shannonEntropy: entropy,
            semanticDensity: semanticDensity,
            tokenConceptRatio: tokenConceptRatio,
            shouldEscalate: shouldEscalate,
            entropyPercentile: entropyPercentile
        )
    }

    private static func computeAmbiguityCluster(text: String, history: [Message]) -> AmbiguityCluster {
        let intentClusters: [(name: String, keywords: [String])] = [
            ("factual", ["what", "who", "where", "when", "define", "meaning", "fact", "true", "false", "is it"]),
            ("creative", ["write", "create", "imagine", "story", "poem", "design", "invent", "brainstorm", "idea"]),
            ("instructional", ["how", "steps", "tutorial", "guide", "teach", "learn", "explain", "show me", "help me"]),
            ("analytical", ["why", "analyze", "compare", "evaluate", "assess", "reason", "cause", "effect", "impact"]),
            ("social", ["hello", "hi", "thanks", "sorry", "bye", "how are you", "good morning", "hey"]),
            ("action", ["do", "make", "set", "send", "open", "calculate", "find", "search", "schedule", "remind"]),
            ("reflective", ["think", "feel", "believe", "wonder", "ponder", "opinion", "perspective", "meaning of life"]),
        ]

        let lower = text.lowercased()
        let words = Set(lower.split(separator: " ").map(String.init))

        var scores: [(name: String, score: Double)] = []
        for cluster in intentClusters {
            var score: Double = 0
            for keyword in cluster.keywords {
                if keyword.contains(" ") {
                    if lower.contains(keyword) { score += 1.2 }
                } else if words.contains(keyword) {
                    score += 1.0
                }
            }
            let normalizedScore = cluster.keywords.isEmpty ? 0 : score / Double(cluster.keywords.count)
            scores.append((cluster.name, normalizedScore))
        }

        scores.sort { $0.1 > $1.1 }

        let primary = scores.first ?? ("unknown", 0)
        let secondary = scores.count > 1 ? scores[1] : ("unknown", 0)
        let delta = primary.1 - secondary.1
        let isAmbiguous = delta < 0.15 && primary.1 > 0.05

        var competing: [String] = []
        if isAmbiguous {
            competing.append("Query maps to both '\(primary.0)' and '\(secondary.0)' clusters with low separation (delta=\(String(format: "%.2f", delta)))")
        }

        return AmbiguityCluster(
            primaryCluster: primary.0,
            primaryScore: primary.1,
            secondaryCluster: secondary.0,
            secondaryScore: secondary.1,
            clusterDelta: delta,
            isAmbiguous: isAmbiguous,
            competingInterpretations: competing
        )
    }

    private static func computeProbabilityMass(
        text: String,
        uncertainty: Double,
        memoryResults: [RetrievalResult],
        entropyAnalysis: EntropyAnalysis
    ) -> ProbabilityMassResult {
        let memoryStrength = memoryResults.isEmpty ? 0.0 : memoryResults.map(\.score).reduce(0, +) / Double(memoryResults.count)
        let topCandidateMass = (1.0 - uncertainty) * (0.5 + memoryStrength * 0.3) * (1.0 - entropyAnalysis.shannonEntropy * 0.1)
        let clampedMass = max(0.05, min(0.95, topCandidateMass))
        let remainderMass = 1.0 - clampedMass
        let massRatio = clampedMass / max(0.01, remainderMass)
        let needsVerification = clampedMass < 0.4

        let band: ConfidenceBand
        if clampedMass >= 0.7 { band = .high }
        else if clampedMass >= 0.5 { band = .moderate }
        else if clampedMass >= 0.3 { band = .low }
        else { band = .veryLow }

        return ProbabilityMassResult(
            topCandidateMass: clampedMass,
            remainderMass: remainderMass,
            massRatio: massRatio,
            needsVerification: needsVerification,
            confidenceBand: band
        )
    }
}
