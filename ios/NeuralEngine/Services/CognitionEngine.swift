import Foundation

@MainActor
struct CognitionEngine {
    static func process(
        userText: String,
        conversationHistory: [Message],
        memoryService: MemoryService
    ) -> CognitionFrame {
        let memoryResults = memoryService.searchMemories(query: userText, maxResults: 8)
        let associativeResults = memoryService.getAssociativeMemories(query: userText, directResults: memoryResults)
        let allMemoryResults = memoryResults + associativeResults

        let emotion = EmotionAnalyzer.analyze(text: userText, conversationHistory: conversationHistory)
        let intent = IntentClassifier.classify(text: userText, conversationHistory: conversationHistory)
        let metacognition = MetacognitionEngine.assess(
            text: userText,
            conversationHistory: conversationHistory,
            memoryResults: allMemoryResults
        )
        let curiosity = CuriosityDetector.detect(
            text: userText,
            memoryResults: allMemoryResults,
            emotion: emotion
        )
        let thoughtTree = ThoughtTreeBuilder.build(
            text: userText,
            intent: intent,
            metacognition: metacognition,
            memoryResults: allMemoryResults,
            emotion: emotion
        )

        let reasoningTrace = buildReasoningTrace(
            tree: thoughtTree,
            metacognition: metacognition
        )

        var injections: [ContextInjection] = []

        let emotionInjection = EmotionAnalyzer.buildInjection(state: emotion)
        if !emotionInjection.content.isEmpty { injections.append(emotionInjection) }

        let metaInjection = MetacognitionEngine.buildInjection(state: metacognition)
        if !metaInjection.content.isEmpty { injections.append(metaInjection) }

        let thoughtInjection = ThoughtTreeBuilder.buildInjection(tree: thoughtTree)
        if !thoughtInjection.content.isEmpty { injections.append(thoughtInjection) }

        let curiosityInjection = CuriosityDetector.buildInjection(state: curiosity)
        if !curiosityInjection.content.isEmpty { injections.append(curiosityInjection) }

        let intentInjection = IntentClassifier.buildInjection(intent: intent)
        if !intentInjection.content.isEmpty { injections.append(intentInjection) }

        if !metacognition.selfCorrectionFlags.isEmpty {
            let correctionInjection = MetacognitionEngine.buildSelfCorrectionInjection(flags: metacognition.selfCorrectionFlags)
            if !correctionInjection.content.isEmpty { injections.append(correctionInjection) }
        }

        let traceInjection = buildReasoningTraceInjection(trace: reasoningTrace)
        if !traceInjection.content.isEmpty { injections.append(traceInjection) }

        injections.sort { $0.priority > $1.priority }

        return CognitionFrame(
            emotion: emotion,
            metacognition: metacognition,
            thoughtTree: thoughtTree,
            curiosity: curiosity,
            intent: intent,
            injections: injections,
            reasoningTrace: reasoningTrace,
            timestamp: Date()
        )
    }

    private static func buildReasoningTrace(
        tree: ThoughtTree,
        metacognition: MetacognitionState
    ) -> ReasoningTrace {
        var selfCorrections: [String] = []
        for flag in metacognition.selfCorrectionFlags {
            selfCorrections.append("[\(flag.domain)] \(flag.issue)")
        }

        var iterations: [ReasoningIteration] = []
        for i in 0..<tree.iterationCount {
            let activeAtStep = tree.branches.filter { !$0.isPruned || (i == tree.iterationCount - 1) }.count
            iterations.append(ReasoningIteration(
                index: i,
                convergence: i == tree.iterationCount - 1 ? tree.convergencePercent : tree.convergencePercent * Double(i + 1) / Double(tree.iterationCount),
                activeBranches: activeAtStep,
                prunedThisRound: i == tree.iterationCount - 1 ? tree.prunedBranches.count : 0,
                insight: i == 0 ? "Initial branch generation" : "Refinement pass \(i + 1)"
            ))
        }

        return ReasoningTrace(
            iterations: iterations,
            finalConvergence: tree.convergencePercent,
            dominantStrategy: tree.synthesisStrategy,
            totalPruned: tree.prunedBranches.count,
            selfCorrections: selfCorrections
        )
    }

    private static func buildReasoningTraceInjection(trace: ReasoningTrace) -> ContextInjection {
        guard trace.iterations.count > 1 || !trace.selfCorrections.isEmpty else {
            return ContextInjection(type: .reasoningTrace, content: "", priority: 0, estimatedTokens: 0)
        }

        var parts: [String] = []

        if trace.iterations.count > 1 {
            parts.append("Reasoning refined over \(trace.iterations.count) iterations (convergence: \(Int(trace.finalConvergence * 100))%).")
        }

        if trace.totalPruned > 0 {
            parts.append("\(trace.totalPruned) low-confidence reasoning path(s) pruned.")
        }

        if !trace.selfCorrections.isEmpty {
            parts.append("Self-correction signals: \(trace.selfCorrections.prefix(2).joined(separator: "; "))")
        }

        parts.append("Dominant strategy: \(trace.dominantStrategy.rawValue)")

        let content = parts.joined(separator: " ")
        let priority: Double = trace.selfCorrections.isEmpty ? 0.4 : 0.8
        return ContextInjection(
            type: .reasoningTrace,
            content: content,
            priority: priority,
            estimatedTokens: content.count / 4
        )
    }
}
