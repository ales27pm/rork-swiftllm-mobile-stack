import Foundation
import CryptoKit

struct ContextSignatureTracker {
    private static let intentDimensions = [
        "factual", "creative", "instructional", "analytical",
        "social", "action", "reflective", "corrective"
    ]

    static func computeSignature(
        text: String,
        intent: IntentClassification,
        emotion: EmotionalState,
        metacognition: MetacognitionState
    ) -> ContextSignature {
        let intentVector = buildIntentVector(intent: intent)
        let topicFingerprint = buildTopicFingerprint(text: text)
        let emotionalBaseline = computeEmotionalBaseline(emotion: emotion)
        let complexityAnchor = computeComplexityAnchor(metacognition: metacognition)
        let hash = computeSignatureHash(
            intentVector: intentVector,
            topicFingerprint: topicFingerprint,
            emotionalBaseline: emotionalBaseline,
            complexityAnchor: complexityAnchor
        )

        return ContextSignature(
            intentVector: intentVector,
            topicFingerprint: topicFingerprint,
            emotionalBaseline: emotionalBaseline,
            complexityAnchor: complexityAnchor,
            signatureHash: hash
        )
    }

    static func detectDrift(
        original: ContextSignature,
        current: ContextSignature
    ) -> SemanticDriftResult {
        let intentDrift = computeVectorDistance(original.intentVector, current.intentVector)

        var topicDrift: Double = 0
        let allKeys = Set(original.topicFingerprint.keys).union(current.topicFingerprint.keys)
        if !allKeys.isEmpty {
            var totalDelta: Double = 0
            for key in allKeys {
                let origVal = original.topicFingerprint[key] ?? 0
                let currVal = current.topicFingerprint[key] ?? 0
                totalDelta += abs(origVal - currVal)
            }
            topicDrift = totalDelta / Double(allKeys.count)
        }

        let emotionDrift = abs(original.emotionalBaseline - current.emotionalBaseline)
        let complexityDrift = abs(original.complexityAnchor - current.complexityAnchor)

        let weights: [Double] = [0.4, 0.3, 0.15, 0.15]
        let driftMagnitude = min(1.0,
            intentDrift * weights[0] +
            topicDrift * weights[1] +
            emotionDrift * weights[2] +
            complexityDrift * weights[3]
        )

        var driftedDimensions: [String] = []
        if intentDrift > 0.3 { driftedDimensions.append("intent (delta=\(String(format: "%.2f", intentDrift)))") }
        if topicDrift > 0.4 { driftedDimensions.append("topic (delta=\(String(format: "%.2f", topicDrift)))") }
        if emotionDrift > 0.5 { driftedDimensions.append("emotion (delta=\(String(format: "%.2f", emotionDrift)))") }
        if complexityDrift > 0.4 { driftedDimensions.append("complexity (delta=\(String(format: "%.2f", complexityDrift)))") }

        let shouldInterrupt = driftMagnitude > 0.6
        let correctionPrompt = buildCorrectionPrompt(
            driftMagnitude: driftMagnitude,
            driftedDimensions: driftedDimensions,
            shouldInterrupt: shouldInterrupt
        )

        return SemanticDriftResult(
            driftMagnitude: driftMagnitude,
            driftedDimensions: driftedDimensions,
            correctionPrompt: correctionPrompt,
            shouldInterrupt: shouldInterrupt
        )
    }

    static func buildDriftInjection(drift: SemanticDriftResult) -> ContextInjection {
        guard drift.driftMagnitude > 0.3, let prompt = drift.correctionPrompt else {
            return ContextInjection(type: .selfCorrection, content: "", priority: 0, estimatedTokens: 0)
        }

        var parts: [String] = ["[Semantic Drift Detection]"]
        parts.append("Drift magnitude: \(Int(drift.driftMagnitude * 100))%")

        if !drift.driftedDimensions.isEmpty {
            parts.append("Drifted dimensions: \(drift.driftedDimensions.joined(separator: ", "))")
        }

        parts.append(prompt)

        if drift.shouldInterrupt {
            parts.append("CRITICAL: Response is deviating significantly from the user's original intent. Re-anchor to the original query constraints before continuing.")
        }

        let content = parts.joined(separator: "\n")
        let priority = min(0.95, drift.driftMagnitude + 0.2)

        return ContextInjection(
            type: .selfCorrection,
            content: content,
            priority: priority,
            estimatedTokens: content.count / 4
        )
    }

    private static func buildIntentVector(intent: IntentClassification) -> [Double] {
        var vector = Array(repeating: 0.0, count: intentDimensions.count)

        let intentMapping: [IntentType: [(index: Int, weight: Double)]] = [
            .questionFactual: [(0, 0.9)],
            .questionHow: [(2, 0.8), (0, 0.2)],
            .questionWhy: [(3, 0.8), (0, 0.2)],
            .questionComparison: [(3, 0.7), (0, 0.3)],
            .questionOpinion: [(6, 0.7), (0, 0.3)],
            .requestCreation: [(1, 0.9)],
            .requestAnalysis: [(3, 0.9)],
            .requestSearch: [(0, 0.6), (5, 0.4)],
            .requestMemory: [(0, 0.5), (5, 0.5)],
            .requestCalculation: [(5, 0.9)],
            .requestAction: [(5, 0.9)],
            .statementEmotion: [(6, 0.8), (4, 0.2)],
            .statementOpinion: [(6, 0.7), (3, 0.3)],
            .statementInstruction: [(2, 0.7), (5, 0.3)],
            .statementFact: [(0, 0.8), (3, 0.2)],
            .socialGreeting: [(4, 0.9)],
            .socialFarewell: [(4, 0.9)],
            .socialGratitude: [(4, 0.9)],
            .socialApology: [(4, 0.9)],
            .metaCorrection: [(7, 0.9)],
            .metaClarification: [(7, 0.5), (2, 0.5)],
            .metaFeedback: [(7, 0.7), (4, 0.3)],
            .explorationBrainstorm: [(1, 0.6), (3, 0.4)],
            .explorationDebate: [(3, 0.7), (6, 0.3)],
            .explorationHypothetical: [(1, 0.4), (3, 0.4), (6, 0.2)],
        ]

        if let mappings = intentMapping[intent.primary] {
            for mapping in mappings {
                if mapping.index < vector.count {
                    vector[mapping.index] += mapping.weight * intent.confidence
                }
            }
        }

        if let secondary = intent.secondary, let mappings = intentMapping[secondary] {
            for mapping in mappings {
                if mapping.index < vector.count {
                    vector[mapping.index] += mapping.weight * 0.3
                }
            }
        }

        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        if magnitude > 0 {
            vector = vector.map { $0 / magnitude }
        }

        return vector
    }

    private static func buildTopicFingerprint(text: String) -> [String: Double] {
        let stopWords: Set<String> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "i", "me", "my",
            "you", "your", "it", "its", "we", "they", "them", "what", "how",
            "why", "when", "where", "which", "who", "and", "but", "or", "if",
            "this", "that", "these", "those", "not", "no", "so", "just"
        ]

        let words = text.lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
            .split(separator: " ")
            .map(String.init)
            .filter { $0.count > 2 && !stopWords.contains($0) }

        let totalCount = Double(words.count)
        guard totalCount > 0 else { return [:] }

        var freq: [String: Int] = [:]
        for w in words { freq[w, default: 0] += 1 }

        var fingerprint: [String: Double] = [:]
        for (word, count) in freq.sorted(by: { $0.value > $1.value }).prefix(10) {
            fingerprint[word] = Double(count) / totalCount
        }

        return fingerprint
    }

    private static func computeEmotionalBaseline(emotion: EmotionalState) -> Double {
        let valenceNum: Double
        switch emotion.valence {
        case .positive: valenceNum = 0.7
        case .neutral: valenceNum = 0.0
        case .mixed: valenceNum = 0.2
        case .negative: valenceNum = -0.7
        }

        let arousalNum: Double
        switch emotion.arousal {
        case .high: arousalNum = 0.9
        case .medium: arousalNum = 0.5
        case .low: arousalNum = 0.2
        }

        return (valenceNum + arousalNum) / 2.0
    }

    private static func computeComplexityAnchor(metacognition: MetacognitionState) -> Double {
        let base: Double
        switch metacognition.complexityLevel {
        case .simple: base = 0.2
        case .moderate: base = 0.45
        case .complex: base = 0.7
        case .expert: base = 0.95
        }

        let entropyBoost = metacognition.entropyAnalysis.shouldEscalate ? 0.1 : 0
        return min(1.0, base + entropyBoost)
    }

    private static func computeVectorDistance(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 1.0 }

        let dotProduct = zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
        let magA = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let magB = sqrt(b.reduce(0) { $0 + $1 * $1 })

        guard magA > 0, magB > 0 else { return 1.0 }

        let cosineSimilarity = dotProduct / (magA * magB)
        return 1.0 - max(0, min(1, cosineSimilarity))
    }

    private static func computeSignatureHash(
        intentVector: [Double],
        topicFingerprint: [String: Double],
        emotionalBaseline: Double,
        complexityAnchor: Double
    ) -> String {
        var data = intentVector.map { String(format: "%.4f", $0) }.joined(separator: ",")
        data += "|"
        data += topicFingerprint.sorted(by: { $0.key < $1.key }).map { "\($0.key):\(String(format: "%.4f", $0.value))" }.joined(separator: ",")
        data += "|\(String(format: "%.4f", emotionalBaseline))|\(String(format: "%.4f", complexityAnchor))"

        let digest = SHA256.hash(data: Data(data.utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    private static func buildCorrectionPrompt(
        driftMagnitude: Double,
        driftedDimensions: [String],
        shouldInterrupt: Bool
    ) -> String? {
        guard driftMagnitude > 0.3 else { return nil }

        if shouldInterrupt {
            return "Review the original user query constraints. Your reasoning has drifted across: \(driftedDimensions.joined(separator: ", ")). Re-align your response to the user's stated intent before continuing."
        }

        if driftedDimensions.contains(where: { $0.hasPrefix("intent") }) {
            return "Minor intent drift detected. Ensure your response addresses the user's primary question directly."
        }

        if driftedDimensions.contains(where: { $0.hasPrefix("topic") }) {
            return "Topic drift detected. Stay focused on the user's original subject matter."
        }

        return "Slight reasoning drift detected (\(Int(driftMagnitude * 100))%). Maintain focus on the original query."
    }
}
