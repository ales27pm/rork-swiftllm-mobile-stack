import Foundation

struct SpeculationPolicy: Sendable {
    var k: Int = 4
    var minK: Int = 1
    var maxK: Int = 8
    private var acceptanceHistory: [Double] = []
    private var latencyHistory: [Double] = []
    private var verificationCostHistory: [Double] = []
    private var acceptedLatencyHistory: [Double] = []
    private var consecutiveFullAccepts: Int = 0
    private var consecutivePartialRejects: Int = 0
    var totalDraftTokens: Int = 0
    var totalAcceptedTokens: Int = 0
    var totalRejectedTokens: Int = 0
    var totalVerifications: Int = 0
    var totalCorrections: Int = 0

    mutating func update(acceptanceRate: Double, latencyEfficiency: Double) {
        acceptanceHistory.append(acceptanceRate)
        if acceptanceHistory.count > 20 {
            acceptanceHistory.removeFirst()
        }

        if acceptanceRate >= 1.0 {
            consecutiveFullAccepts += 1
            consecutivePartialRejects = 0
        } else if acceptanceRate < 0.5 {
            consecutivePartialRejects += 1
            consecutiveFullAccepts = 0
        } else {
            consecutiveFullAccepts = 0
            consecutivePartialRejects = 0
        }

        let avgRate = acceptanceHistory.reduce(0, +) / Double(acceptanceHistory.count)

        if consecutiveFullAccepts >= 2 && latencyEfficiency >= 1.0 {
            k = min(k + 2, maxK)
        } else if avgRate > 0.85 && latencyEfficiency >= 0.9 {
            k = min(k + 1, maxK)
        } else if consecutivePartialRejects >= 2 || latencyEfficiency < 0.6 {
            k = max(k - 2, minK)
        } else if avgRate < 0.50 {
            k = max(k - 1, minK)
        }
    }

    mutating func recordVerification(
        draftCount: Int,
        acceptedCount: Int,
        rejectedCount: Int,
        correctionCount: Int,
        draftLatencyMS: Double,
        verifyLatencyMS: Double,
        committedCount: Int,
        mismatchIndex: Int?
    ) {
        totalDraftTokens += draftCount
        totalAcceptedTokens += acceptedCount
        totalRejectedTokens += rejectedCount
        totalCorrections += correctionCount
        totalVerifications += 1

        latencyHistory.append(draftLatencyMS)
        if latencyHistory.count > 20 {
            latencyHistory.removeFirst()
        }

        verificationCostHistory.append(verifyLatencyMS)
        if verificationCostHistory.count > 20 {
            verificationCostHistory.removeFirst()
        }

        acceptedLatencyHistory.append((draftLatencyMS + verifyLatencyMS) / Double(max(committedCount, 1)))
        if acceptedLatencyHistory.count > 20 {
            acceptedLatencyHistory.removeFirst()
        }

        let rate = draftCount > 0 ? Double(acceptedCount) / Double(draftCount) : 0
        let baselinePerToken = verifyLatencyMS / Double(max(draftCount, 1))
        let actualPerCommitted = (draftLatencyMS + verifyLatencyMS) / Double(max(committedCount, 1))
        let mismatchPenalty = mismatchIndex.map { 1.0 - (Double($0) / Double(max(draftCount, 1))) } ?? 0
        let latencyEfficiency = baselinePerToken > 0
            ? (baselinePerToken / max(actualPerCommitted, 0.001)) * (1.0 - 0.25 * mismatchPenalty)
            : 1.0
        update(acceptanceRate: rate, latencyEfficiency: latencyEfficiency)
    }

    var shouldUseSpeculation: Bool {
        guard totalVerifications >= 3 else { return true }

        let avgAcceptance = currentAcceptanceRate
        if avgAcceptance < 0.25 { return false }

        if let avgDraft = averageDraftLatencyMS, let avgVerify = averageVerificationLatencyMS {
            let specCost = avgDraft + avgVerify
            let estimatedNormalCost = avgVerify * Double(k)
            if specCost > estimatedNormalCost * 1.5 { return false }
        }

        if let acceptedLatency = averageAcceptedLatencyMS,
           let verifyLatency = averageVerificationLatencyMS,
           acceptedLatency > verifyLatency * 1.15 {
            return false
        }

        return true
    }

    var currentAcceptanceRate: Double {
        guard !acceptanceHistory.isEmpty else { return 0 }
        return acceptanceHistory.reduce(0, +) / Double(acceptanceHistory.count)
    }

    var lifetimeAcceptanceRate: Double {
        guard totalDraftTokens > 0 else { return 0 }
        return Double(totalAcceptedTokens) / Double(totalDraftTokens)
    }

    var averageDraftLatencyMS: Double? {
        guard !latencyHistory.isEmpty else { return nil }
        return latencyHistory.reduce(0, +) / Double(latencyHistory.count)
    }

    var averageVerificationLatencyMS: Double? {
        guard !verificationCostHistory.isEmpty else { return nil }
        return verificationCostHistory.reduce(0, +) / Double(verificationCostHistory.count)
    }

    var averageAcceptedLatencyMS: Double? {
        guard !acceptedLatencyHistory.isEmpty else { return nil }
        return acceptedLatencyHistory.reduce(0, +) / Double(acceptedLatencyHistory.count)
    }

    var effectiveSpeedup: Double {
        guard totalVerifications > 0, totalAcceptedTokens > 0 else { return 1.0 }
        return Double(totalAcceptedTokens + totalVerifications) / Double(totalVerifications)
    }

    mutating func reset() {
        k = 4
        acceptanceHistory.removeAll()
        latencyHistory.removeAll()
        verificationCostHistory.removeAll()
        acceptedLatencyHistory.removeAll()
        consecutiveFullAccepts = 0
        consecutivePartialRejects = 0
        totalDraftTokens = 0
        totalAcceptedTokens = 0
        totalRejectedTokens = 0
        totalVerifications = 0
        totalCorrections = 0
    }
}
