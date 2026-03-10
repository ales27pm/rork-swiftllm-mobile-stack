import Foundation

struct SpeculationPolicy: Sendable {
    var k: Int = 4
    var minK: Int = 1
    var maxK: Int = 8
    private var acceptanceHistory: [Double] = []

    mutating func update(acceptanceRate: Double) {
        acceptanceHistory.append(acceptanceRate)
        if acceptanceHistory.count > 10 {
            acceptanceHistory.removeFirst()
        }

        let avgRate = acceptanceHistory.reduce(0, +) / Double(acceptanceHistory.count)

        if avgRate > 0.85 {
            k = min(k + 1, maxK)
        } else if avgRate < 0.50 {
            k = max(k - 1, minK)
        }
    }

    var currentAcceptanceRate: Double {
        guard !acceptanceHistory.isEmpty else { return 0 }
        return acceptanceHistory.reduce(0, +) / Double(acceptanceHistory.count)
    }

    mutating func reset() {
        k = 4
        acceptanceHistory.removeAll()
    }
}
