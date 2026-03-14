import Foundation

struct ThoughtTreeBuilder {
    private static let maxIterations = 3
    private static let convergenceThreshold = 0.75
    private static let pruneConfidenceThreshold = 0.25

    static func build(
        text: String,
        intent: IntentClassification,
        metacognition: MetacognitionState,
        memoryResults: [RetrievalResult],
        emotion: EmotionalState
    ) -> ThoughtTree {
        var branches = generateInitialBranches(
            text: text,
            intent: intent,
            metacognition: metacognition,
            memoryResults: memoryResults
        )

        var iterations: [ReasoningIteration] = []
        var totalPruned = 0

        for iteration in 0..<maxIterations {
            let (refined, pruned, insight) = refineAndPrune(
                branches: branches,
                metacognition: metacognition,
                emotion: emotion,
                iteration: iteration
            )

            let convergence = computeConvergence(branches: refined)
            totalPruned += pruned.count

            iterations.append(ReasoningIteration(
                index: iteration,
                convergence: convergence,
                activeBranches: refined.filter { !$0.isPruned }.count,
                prunedThisRound: pruned.count,
                insight: insight
            ))

            branches = refined

            if convergence >= convergenceThreshold {
                break
            }
        }

        let active = branches.filter { !$0.isPruned }.sorted { $0.confidence > $1.confidence }
        let pruned = branches.filter { $0.isPruned }
        let bestPath = Array(active.prefix(3))
        let convergence = computeConvergence(branches: branches)
        let strategy = determineSynthesisStrategy(
            bestPath: bestPath,
            metacognition: metacognition,
            convergence: convergence
        )

        return ThoughtTree(
            branches: branches.sorted { $0.confidence > $1.confidence },
            bestPath: bestPath,
            prunedBranches: pruned,
            convergencePercent: convergence,
            iterationCount: iterations.count,
            synthesisStrategy: strategy
        )
    }

    static func buildReasoningTrace(
        tree: ThoughtTree,
        metacognition: MetacognitionState,
        iterations: [ReasoningIteration]
    ) -> ReasoningTrace {
        var selfCorrections: [String] = []
        for flag in metacognition.selfCorrectionFlags {
            selfCorrections.append("[\(flag.domain)] \(flag.issue)")
        }

        return ReasoningTrace(
            iterations: iterations,
            finalConvergence: tree.convergencePercent,
            dominantStrategy: tree.synthesisStrategy,
            totalPruned: tree.prunedBranches.count,
            selfCorrections: selfCorrections
        )
    }

    private static func generateInitialBranches(
        text: String,
        intent: IntentClassification,
        metacognition: MetacognitionState,
        memoryResults: [RetrievalResult]
    ) -> [ThoughtBranch] {
        var branches: [ThoughtBranch] = []

        let directConf = computeDirectConfidence(metacognition: metacognition, memoryResults: memoryResults)
        let directEvidence = buildDirectEvidence(intent: intent, memoryResults: memoryResults)
        let directCounter = buildDirectCounterpoints(metacognition: metacognition)
        branches.append(ThoughtBranch(
            id: "direct",
            hypothesis: "Direct response based on available knowledge",
            confidence: directConf,
            evidence: directEvidence,
            counterpoints: directCounter,
            isPruned: false,
            pruneReason: nil,
            strategy: "Respond directly using existing knowledge and context",
            supportScore: directConf * Double(directEvidence.count),
            contradictionScore: Double(directCounter.count) * 0.2
        ))

        if metacognition.shouldDecompose {
            let conf: Double = metacognition.complexityLevel == .expert ? 0.85 : 0.7
            branches.append(ThoughtBranch(
                id: "decompose",
                hypothesis: "Decompose into sub-problems for systematic resolution",
                confidence: conf,
                evidence: ["Query complexity: \(metacognition.complexityLevel.rawValue)", "Multiple aspects detected"],
                counterpoints: ["May over-complicate a simpler question", "Increases response length"],
                isPruned: false,
                pruneReason: nil,
                strategy: "Break down into 2-4 sub-questions, address each, then synthesize",
                supportScore: conf * 2.0,
                contradictionScore: 0.4
            ))
        }

        if !memoryResults.isEmpty {
            let memConf = computeMemoryConfidence(memoryResults: memoryResults)
            branches.append(ThoughtBranch(
                id: "memory",
                hypothesis: "Leverage stored knowledge for personalized response",
                confidence: memConf,
                evidence: memoryResults.prefix(3).map { "Memory[\($0.matchType.rawValue)]: \(String($0.memory.content.prefix(60)))" },
                counterpoints: memoryResults.allSatisfy({ $0.score < 0.3 }) ? ["Memory matches are weak — may not be relevant"] : [],
                isPruned: false,
                pruneReason: nil,
                strategy: "Integrate retrieved memories to provide contextual, personalized answers",
                supportScore: memConf * Double(memoryResults.count),
                contradictionScore: memoryResults.allSatisfy({ $0.score < 0.3 }) ? 0.5 : 0.1
            ))
        }

        if metacognition.knowledgeLimitHit || metacognition.uncertaintyLevel > 0.6 {
            branches.append(ThoughtBranch(
                id: "search",
                hypothesis: "Acknowledge knowledge gap and suggest verification",
                confidence: 0.5,
                evidence: ["Uncertainty: \(Int(metacognition.uncertaintyLevel * 100))%", "No strong memory matches"],
                counterpoints: ["May have sufficient general knowledge", "User may not need precise data"],
                isPruned: false,
                pruneReason: nil,
                strategy: "Be transparent about limitations, provide best available answer with caveats",
                supportScore: 1.0,
                contradictionScore: 0.4
            ))
        }

        if intent.requiresCreativity {
            branches.append(ThoughtBranch(
                id: "creative",
                hypothesis: "Apply creative divergent thinking",
                confidence: 0.65,
                evidence: ["Intent requires creativity", "Open-ended query structure"],
                counterpoints: ["User may prefer practical over creative"],
                isPruned: false,
                pruneReason: nil,
                strategy: "Generate multiple creative options, then let the best emerge",
                supportScore: 1.3,
                contradictionScore: 0.2
            ))
        }

        if metacognition.shouldSeekClarification {
            let conf: Double = metacognition.ambiguityReasons.count > 1 ? 0.75 : 0.5
            branches.append(ThoughtBranch(
                id: "clarify",
                hypothesis: "Seek clarification before responding",
                confidence: conf,
                evidence: metacognition.ambiguityReasons,
                counterpoints: ["May frustrate user who wants a quick answer", "Could make a reasonable assumption instead"],
                isPruned: false,
                pruneReason: nil,
                strategy: "Ask a targeted clarifying question while offering preliminary thoughts",
                supportScore: conf * Double(metacognition.ambiguityReasons.count),
                contradictionScore: 0.3
            ))
        }

        if !metacognition.selfCorrectionFlags.isEmpty {
            let maxSeverity = metacognition.selfCorrectionFlags.map(\.severity).max() ?? 0.5
            branches.append(ThoughtBranch(
                id: "self_correct",
                hypothesis: "Apply self-correction based on detected issues",
                confidence: maxSeverity,
                evidence: metacognition.selfCorrectionFlags.map { "[\($0.domain)] \($0.issue)" },
                counterpoints: ["Correction may be over-reactive to ambiguous signals"],
                isPruned: false,
                pruneReason: nil,
                strategy: "Acknowledge previous errors, re-evaluate assumptions, provide corrected response",
                supportScore: maxSeverity * Double(metacognition.selfCorrectionFlags.count),
                contradictionScore: 0.15
            ))
        }

        return branches
    }

    private static func refineAndPrune(
        branches: [ThoughtBranch],
        metacognition: MetacognitionState,
        emotion: EmotionalState,
        iteration: Int
    ) -> (refined: [ThoughtBranch], pruned: [ThoughtBranch], insight: String) {
        var refined: [ThoughtBranch] = []
        var newlyPruned: [ThoughtBranch] = []
        var insight = ""

        let emotionBoost: [String: Double] = {
            switch emotion.valence {
            case .negative:
                return ["clarify": 0.1, "self_correct": 0.15, "creative": -0.1]
            case .positive:
                return ["creative": 0.1, "direct": 0.05]
            case .mixed:
                return ["clarify": 0.05, "decompose": 0.05]
            case .neutral:
                return [:]
            }
        }()

        for branch in branches {
            guard !branch.isPruned else {
                refined.append(branch)
                continue
            }

            let boost = emotionBoost[branch.id] ?? 0
            let netSupport = branch.supportScore - branch.contradictionScore
            var adjustedConfidence = branch.confidence + boost + (netSupport * 0.05 * Double(iteration + 1))
            adjustedConfidence = max(0, min(1, adjustedConfidence))

            if adjustedConfidence < pruneConfidenceThreshold && iteration > 0 {
                let pruned = ThoughtBranch(
                    id: branch.id,
                    hypothesis: branch.hypothesis,
                    confidence: adjustedConfidence,
                    evidence: branch.evidence,
                    counterpoints: branch.counterpoints,
                    isPruned: true,
                    pruneReason: "Confidence \(Int(adjustedConfidence * 100))% below threshold after iteration \(iteration + 1)",
                    strategy: branch.strategy,
                    supportScore: branch.supportScore,
                    contradictionScore: branch.contradictionScore
                )
                refined.append(pruned)
                newlyPruned.append(pruned)
            } else {
                refined.append(ThoughtBranch(
                    id: branch.id,
                    hypothesis: branch.hypothesis,
                    confidence: adjustedConfidence,
                    evidence: branch.evidence,
                    counterpoints: branch.counterpoints,
                    isPruned: false,
                    pruneReason: nil,
                    strategy: branch.strategy,
                    supportScore: branch.supportScore,
                    contradictionScore: branch.contradictionScore
                ))
            }
        }

        let activeCount = refined.filter { !$0.isPruned }.count
        if !newlyPruned.isEmpty {
            insight = "Pruned \(newlyPruned.count) low-confidence branch(es); \(activeCount) active"
        } else if iteration == 0 {
            insight = "Initial evaluation: \(activeCount) branches generated"
        } else {
            insight = "Stable iteration: all \(activeCount) branches retained"
        }

        return (refined, newlyPruned, insight)
    }

    private static func computeConvergence(branches: [ThoughtBranch]) -> Double {
        let active = branches.filter { !$0.isPruned }
        guard active.count >= 2 else { return active.first?.confidence ?? 0.5 }

        let sorted = active.sorted { $0.confidence > $1.confidence }
        let topConf = sorted[0].confidence
        let secondConf = sorted[1].confidence
        let spread = topConf - secondConf

        let dominanceFactor = topConf > 0.7 ? 0.15 : 0
        let fewBranchesFactor = active.count <= 2 ? 0.1 : 0
        let convergence = max(0, min(1, (1.0 - spread * 2.0) + dominanceFactor + fewBranchesFactor))

        return convergence
    }

    private static func determineSynthesisStrategy(
        bestPath: [ThoughtBranch],
        metacognition: MetacognitionState,
        convergence: Double
    ) -> SynthesisStrategy {
        guard let top = bestPath.first else { return .direct }

        if metacognition.selfCorrectionFlags.contains(where: { $0.severity > 0.7 }) {
            return .hedgedResponse
        }

        switch top.id {
        case "decompose": return .decompose
        case "creative": return .creativeExploration
        case "clarify": return .clarifyFirst
        case "search" where metacognition.uncertaintyLevel > 0.7: return .hedgedResponse
        default: break
        }

        if convergence < 0.4 && bestPath.count >= 2 {
            return .multiPerspective
        }

        return .direct
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
        parts.append("Reasoning strategy: \(tree.synthesisStrategy.rawValue) (convergence: \(Int(tree.convergencePercent * 100))%, iterations: \(tree.iterationCount)):")

        for (i, branch) in tree.bestPath.prefix(3).enumerated() {
            parts.append("  \(i + 1). [\(branch.id)] \(branch.strategy) (conf: \(Int(branch.confidence * 100))%)")
        }

        if !tree.prunedBranches.isEmpty {
            let prunedIds = tree.prunedBranches.map(\.id).joined(separator: ", ")
            parts.append("  Pruned: \(prunedIds)")
        }

        if tree.convergencePercent < 0.4 {
            parts.append("Low convergence — consider multiple angles in your response.")
        }

        if tree.synthesisStrategy == .hedgedResponse {
            parts.append("Use calibrated language. Acknowledge what you're uncertain about.")
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
