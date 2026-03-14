import Foundation

struct ThoughtTreeBuilder {
    static func build(
        text: String,
        intent: IntentClassification,
        metacognition: MetacognitionState,
        memoryResults: [RetrievalResult]
    ) -> ThoughtTree {
        var branches: [ThoughtBranch] = []

        let directBranch = ThoughtBranch(
            id: "direct",
            hypothesis: "Direct response based on available knowledge",
            confidence: computeDirectConfidence(metacognition: metacognition, memoryResults: memoryResults),
            evidence: buildDirectEvidence(intent: intent, memoryResults: memoryResults),
            counterpoints: buildDirectCounterpoints(metacognition: metacognition),
            isPruned: false,
            strategy: "Respond directly using existing knowledge and context"
        )
        branches.append(directBranch)

        if metacognition.shouldDecompose {
            let decomposeBranch = ThoughtBranch(
                id: "decompose",
                hypothesis: "Decompose into sub-problems for systematic resolution",
                confidence: metacognition.complexityLevel == .expert ? 0.85 : 0.7,
                evidence: ["Query complexity: \(metacognition.complexityLevel.rawValue)", "Multiple aspects detected"],
                counterpoints: ["May over-complicate a simpler question", "Increases response length"],
                isPruned: false,
                strategy: "Break down into 2-4 sub-questions, address each, then synthesize"
            )
            branches.append(decomposeBranch)
        }

        if !memoryResults.isEmpty {
            let memoryBranch = ThoughtBranch(
                id: "memory",
                hypothesis: "Leverage stored knowledge for personalized response",
                confidence: computeMemoryConfidence(memoryResults: memoryResults),
                evidence: memoryResults.prefix(3).map { "Memory[\($0.matchType.rawValue)]: \(String($0.memory.content.prefix(60)))" },
                counterpoints: memoryResults.allSatisfy({ $0.score < 0.3 }) ? ["Memory matches are weak — may not be relevant"] : [],
                isPruned: false,
                strategy: "Integrate retrieved memories to provide contextual, personalized answers"
            )
            branches.append(memoryBranch)
        }

        if metacognition.knowledgeLimitHit || metacognition.uncertaintyLevel > 0.6 {
            let searchBranch = ThoughtBranch(
                id: "search",
                hypothesis: "Acknowledge knowledge gap and suggest verification",
                confidence: 0.5,
                evidence: ["Uncertainty: \(Int(metacognition.uncertaintyLevel * 100))%", "No strong memory matches"],
                counterpoints: ["May have sufficient general knowledge", "User may not need precise data"],
                isPruned: false,
                strategy: "Be transparent about limitations, provide best available answer with caveats"
            )
            branches.append(searchBranch)
        }

        if intent.requiresCreativity {
            let creativeBranch = ThoughtBranch(
                id: "creative",
                hypothesis: "Apply creative divergent thinking",
                confidence: 0.65,
                evidence: ["Intent requires creativity", "Open-ended query structure"],
                counterpoints: ["User may prefer practical over creative"],
                isPruned: false,
                strategy: "Generate multiple creative options, then let the best emerge"
            )
            branches.append(creativeBranch)
        }

        if metacognition.shouldSeekClarification {
            let clarifyBranch = ThoughtBranch(
                id: "clarify",
                hypothesis: "Seek clarification before responding",
                confidence: metacognition.ambiguityReasons.count > 1 ? 0.75 : 0.5,
                evidence: metacognition.ambiguityReasons,
                counterpoints: ["May frustrate user who wants a quick answer", "Could make a reasonable assumption instead"],
                isPruned: false,
                strategy: "Ask a targeted clarifying question while offering preliminary thoughts"
            )
            branches.append(clarifyBranch)
        }

        let sorted = branches.sorted { $0.confidence > $1.confidence }
        let bestPath = Array(sorted.filter { !$0.isPruned }.prefix(3))

        let topConfidences = bestPath.map(\.confidence)
        let convergence: Double
        if topConfidences.count >= 2 {
            let spread = (topConfidences.first ?? 0) - (topConfidences.last ?? 0)
            convergence = max(0, min(1, 1.0 - spread))
        } else {
            convergence = topConfidences.first ?? 0.5
        }

        return ThoughtTree(
            branches: sorted,
            bestPath: bestPath,
            convergencePercent: convergence
        )
    }

    private static func computeDirectConfidence(metacognition: MetacognitionState, memoryResults: [RetrievalResult]) -> Double {
        var conf = 1.0 - metacognition.uncertaintyLevel
        if !memoryResults.isEmpty { conf += 0.1 }
        if metacognition.complexityLevel == .simple { conf += 0.1 }
        if metacognition.complexityLevel == .expert { conf -= 0.2 }
        return max(0.1, min(0.95, conf))
    }

    private static func computeMemoryConfidence(memoryResults: [RetrievalResult]) -> Double {
        guard !memoryResults.isEmpty else { return 0 }
        let avgScore = memoryResults.map(\.score).reduce(0, +) / Double(memoryResults.count)
        return min(0.9, avgScore + 0.3)
    }

    private static func buildDirectEvidence(intent: IntentClassification, memoryResults: [RetrievalResult]) -> [String] {
        var evidence: [String] = []
        evidence.append("Intent: \(intent.primary.rawValue) (confidence: \(Int(intent.confidence * 100))%)")
        if !memoryResults.isEmpty {
            evidence.append("\(memoryResults.count) relevant memories found")
        }
        if intent.requiresKnowledge {
            evidence.append("Query requires factual knowledge")
        }
        return evidence
    }

    private static func buildDirectCounterpoints(metacognition: MetacognitionState) -> [String] {
        var counterpoints: [String] = []
        if metacognition.uncertaintyLevel > 0.5 {
            counterpoints.append("Moderate uncertainty — answer may be incomplete")
        }
        if metacognition.ambiguityDetected {
            counterpoints.append("Query has ambiguous elements")
        }
        return counterpoints
    }

    static func buildInjection(tree: ThoughtTree) -> ContextInjection {
        guard !tree.bestPath.isEmpty else {
            return ContextInjection(type: .thoughtTree, content: "", priority: 0, estimatedTokens: 0)
        }

        var parts: [String] = []
        parts.append("Reasoning strategy (convergence: \(Int(tree.convergencePercent * 100))%):")

        for (i, branch) in tree.bestPath.prefix(3).enumerated() {
            parts.append("  \(i + 1). [\(branch.id)] \(branch.strategy) (conf: \(Int(branch.confidence * 100))%)")
        }

        if tree.convergencePercent < 0.4 {
            parts.append("Low convergence — consider multiple angles in your response.")
        }

        let content = parts.joined(separator: "\n")
        let priority: Double = tree.convergencePercent < 0.5 ? 0.75 : 0.5

        return ContextInjection(
            type: .thoughtTree,
            content: content,
            priority: priority,
            estimatedTokens: content.count / 4
        )
    }
}
