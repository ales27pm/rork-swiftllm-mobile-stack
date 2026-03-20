import SwiftUI

struct ReasoningReplayView: View {
    let chatViewModel: ChatViewModel
    @State private var selectedEntry: ReasoningReplayEntry?
    @State private var animatedIndex: Int = 0
    @State private var isAnimating: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if let frame = chatViewModel.lastCognitionFrame {
                        liveStateCard(frame)
                    }

                    if !chatViewModel.reasoningReplayLog.isEmpty {
                        timelineSection
                    } else {
                        emptyState
                    }
                }
                .padding(.vertical, 16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Reasoning Replay")
            .navigationBarTitleDisplayMode(.large)
        }
    }

    private func liveStateCard(_ frame: CognitionFrame) -> some View {
        VStack(spacing: 16) {
            HStack {
                Label("Live Cognitive State", systemImage: "brain.head.profile.fill")
                    .font(.subheadline.bold())
                    .foregroundStyle(.purple)
                Spacer()
                Text(frame.reasoningTrace.dominantStrategy.rawValue)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.purple.opacity(0.12))
                    .foregroundStyle(.purple)
                    .clipShape(Capsule())
            }

            convergenceGauge(value: frame.reasoningTrace.finalConvergence)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                metricTile("Branches", value: "\(frame.thoughtTree.branches.count - frame.thoughtTree.prunedBranches.count)", icon: "arrow.triangle.branch", color: .blue)
                metricTile("Pruned", value: "\(frame.thoughtTree.prunedBranches.count)", icon: "scissors", color: .orange)
                metricTile("Depth", value: "\(frame.thoughtTree.maxDepthReached)", icon: "arrow.down.right", color: .green)
                metricTile("Iterations", value: "\(frame.reasoningTrace.iterations.count)", icon: "repeat", color: .teal)
            }

            if !frame.thoughtTree.bestPath.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Best Path")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)

                    ForEach(Array(frame.thoughtTree.bestPath.prefix(3).enumerated()), id: \.offset) { idx, branch in
                        branchRow(branch, index: idx)
                    }
                }
            }

            if frame.thoughtTree.dfsExpansions > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.triangle.swap")
                        .font(.caption)
                        .foregroundStyle(.indigo)
                    Text("DFS: depth \(frame.thoughtTree.maxDepthReached)/4, \(frame.thoughtTree.dfsExpansions) expansions, \(frame.thoughtTree.terminalNodes.count) terminals")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.indigo.opacity(0.06))
                .clipShape(.rect(cornerRadius: 8))
            }

            if !frame.metacognition.selfCorrectionFlags.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Self-Corrections", systemImage: "exclamationmark.arrow.circlepath")
                        .font(.caption.bold())
                        .foregroundStyle(.orange)
                    ForEach(frame.metacognition.selfCorrectionFlags, id: \.domain) { flag in
                        Text("[\(flag.domain)] \(flag.issue)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
        .padding(.horizontal, 16)
    }

    private func convergenceGauge(value: Double) -> some View {
        VStack(spacing: 6) {
            HStack {
                Text("Convergence")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(value * 100))%")
                    .font(.caption.bold().monospacedDigit())
                    .foregroundStyle(convergenceColor(value))
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.quaternarySystemFill))
                        .frame(height: 6)

                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [convergenceColor(value).opacity(0.7), convergenceColor(value)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geo.size.width * value, height: 6)
                        .animation(.spring(duration: 0.6), value: value)
                }
            }
            .frame(height: 6)
        }
    }

    private func convergenceColor(_ value: Double) -> Color {
        if value > 0.7 { return .green }
        if value > 0.4 { return .orange }
        return .red
    }

    private func metricTile(_ label: String, value: String, icon: String, color: Color) -> some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundStyle(color)
                Text(value)
                    .font(.title3.bold().monospacedDigit())
            }
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(color.opacity(0.06))
        .clipShape(.rect(cornerRadius: 10))
    }

    private func branchRow(_ branch: ThoughtBranch, index: Int) -> some View {
        HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(branch.isPruned ? Color.red.opacity(0.12) : Color.blue.opacity(0.12))
                    .frame(width: 28, height: 28)
                Text("\(index + 1)")
                    .font(.caption2.bold().monospacedDigit())
                    .foregroundStyle(branch.isPruned ? .red : .blue)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(branch.id)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                Text(branch.strategy)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer(minLength: 0)

            VStack(alignment: .trailing, spacing: 2) {
                Text("\(Int(branch.confidence * 100))%")
                    .font(.caption.bold().monospacedDigit())
                    .foregroundStyle(branch.confidence > 0.7 ? .green : branch.confidence > 0.4 ? .orange : .red)
                if branch.depth > 0 {
                    Text("d\(branch.depth)")
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(8)
        .background(Color(.tertiarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 8))
    }

    private var timelineSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Session Timeline", systemImage: "clock.arrow.circlepath")
                    .font(.subheadline.bold())
                Spacer()
                Text("\(chatViewModel.reasoningReplayLog.count) entries")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)

            LazyVStack(spacing: 8) {
                ForEach(chatViewModel.reasoningReplayLog.reversed()) { entry in
                    timelineEntry(entry)
                }
            }
            .padding(.horizontal, 16)
        }
    }

    private func timelineEntry(_ entry: ReasoningReplayEntry) -> some View {
        HStack(spacing: 12) {
            VStack(spacing: 2) {
                Circle()
                    .fill(convergenceColor(entry.convergence))
                    .frame(width: 8, height: 8)
                Rectangle()
                    .fill(Color(.quaternarySystemFill))
                    .frame(width: 2)
            }
            .frame(width: 8)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(entry.strategy)
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Text(entry.timestamp.formatted(date: .omitted, time: .shortened))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                HStack(spacing: 12) {
                    Label("\(Int(entry.convergence * 100))%", systemImage: "target")
                    Label("\(entry.activeBranches) active", systemImage: "arrow.triangle.branch")
                    if entry.prunedBranches > 0 {
                        Label("\(entry.prunedBranches) pruned", systemImage: "scissors")
                            .foregroundStyle(.orange)
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    Text(entry.complexityLevel)
                        .font(.system(size: 9).weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color(.quaternarySystemFill))
                        .clipShape(Capsule())

                    if entry.selfCorrectionCount > 0 {
                        Text("\(entry.selfCorrectionCount) corrections")
                            .font(.system(size: 9).weight(.medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.orange.opacity(0.1))
                            .foregroundStyle(.orange)
                            .clipShape(Capsule())
                    }
                }
            }
        }
        .padding(10)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 10))
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer(minLength: 40)
            Image(systemName: "brain.head.profile")
                .font(.system(size: 48))
                .foregroundStyle(.quaternary)
            Text("No Reasoning Data")
                .font(.title3.bold())
            Text("Send a message to see the thought tree\nanalysis and convergence replay")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Spacer(minLength: 40)
        }
    }
}
