import SwiftUI

struct AgentActionBar: View {
    let agent: AssistantAgent
    let onCapabilitySelected: (AgentCapability) -> Void

    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 10) {
                ForEach(AgentCapability.allCases) { capability in
                    actionChip(capability)
                }
            }
        }
        .contentMargins(.horizontal, 14)
        .scrollIndicators(.hidden)
    }

    private func actionChip(_ capability: AgentCapability) -> some View {
        Button {
            onCapabilitySelected(capability)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: capability.icon)
                    .font(.system(size: 12, weight: .semibold))

                Text(capability.rawValue)
                    .font(.caption.weight(.medium))

                if let badge = badgeText(for: capability) {
                    Text(badge)
                        .font(.system(size: 9).weight(.bold).monospacedDigit())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(badgeColor(for: capability))
                        .clipShape(Capsule())
                }
            }
            .foregroundStyle(chipColor(for: capability))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(chipColor(for: capability).opacity(0.1))
            .clipShape(Capsule())
        }
    }

    private func chipColor(for capability: AgentCapability) -> Color {
        switch capability {
        case .history: return .blue
        case .memory: return .purple
        case .browse: return .cyan
        case .map: return .green
        case .scan: return .orange
        case .models: return .indigo
        case .metrics: return .teal
        case .reasoning: return .pink
        case .personas: return .mint
        case .context: return .yellow
        }
    }

    private func badgeText(for capability: AgentCapability) -> String? {
        switch capability {
        case .memory:
            let count = agent.memoryCount
            return count > 0 ? "\(count)" : nil
        case .history:
            let count = agent.conversationCount
            return count > 0 ? "\(count)" : nil
        case .metrics:
            let status = agent.systemHealthStatus
            return status != .optimal ? "!" : nil
        default:
            return nil
        }
    }

    private func badgeColor(for capability: AgentCapability) -> Color {
        switch capability {
        case .metrics:
            return agent.systemHealthStatus == .critical ? .red : .orange
        default:
            return chipColor(for: capability)
        }
    }
}

struct AgentStatusStrip: View {
    let agent: AssistantAgent

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 5) {
                Circle()
                    .fill(agent.hasActiveModel ? .green : Color(.tertiaryLabel))
                    .frame(width: 6, height: 6)

                Text(agent.activeModelName)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            HStack(spacing: 8) {
                if agent.memoryCount > 0 {
                    Label("\(agent.memoryCount)", systemImage: "brain")
                        .font(.caption2)
                        .foregroundStyle(.purple)
                }

                healthBadge
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(Color(.secondarySystemBackground).opacity(0.6))
    }

    private var healthBadge: some View {
        HStack(spacing: 3) {
            Circle()
                .fill(healthColor)
                .frame(width: 5, height: 5)

            Text(agent.systemHealthStatus.rawValue)
                .font(.system(size: 9).weight(.semibold))
                .foregroundStyle(healthColor)
        }
    }

    private var healthColor: Color {
        switch agent.systemHealthStatus {
        case .optimal: return .green
        case .degraded: return .orange
        case .critical: return .red
        }
    }
}
