import Foundation

nonisolated enum RuntimeMode: String, Sendable, CaseIterable {
    case maxPerformance = "Max Performance"
    case balanced = "Balanced"
    case coolDown = "Cool Down"
    case emergency = "Emergency"

    var icon: String {
        switch self {
        case .maxPerformance: return "bolt.fill"
        case .balanced: return "gauge.with.dots.needle.50percent"
        case .coolDown: return "snowflake"
        case .emergency: return "exclamationmark.triangle.fill"
        }
    }

    var speculativeEnabled: Bool {
        switch self {
        case .maxPerformance, .balanced: return true
        case .coolDown, .emergency: return false
        }
    }

    var maxDraftTokens: Int {
        switch self {
        case .maxPerformance: return 8
        case .balanced: return 4
        case .coolDown: return 2
        case .emergency: return 0
        }
    }

    var maxContextLength: Int {
        switch self {
        case .maxPerformance: return 4096
        case .balanced: return 2048
        case .coolDown: return 1024
        case .emergency: return 512
        }
    }
}
