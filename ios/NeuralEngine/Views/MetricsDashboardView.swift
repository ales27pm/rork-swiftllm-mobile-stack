import SwiftUI

struct MetricsDashboardView: View {
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let inferenceEngine: InferenceEngine

    @State private var selectedSection: DashboardSection = .overview
    @State private var pulseEviction: Bool = false

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                sectionPicker
                
                switch selectedSection {
                case .overview:
                    diagnosticBanner
                    speedSparklineSection
                    liveMetricsSection
                    thermalSection
                case .hardware:
                    hardwareStatusSection
                    cacheSection
                    memorySection
                case .diagnostics:
                    diagnosticSummarySection
                    diagnosticEventLogSection
                case .history:
                    historySection
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Metrics")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button("Reset Diagnostics", systemImage: "arrow.counterclockwise") {
                        metricsLogger.clearDiagnostics()
                        thermalGovernor.resetPeakTracking()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var sectionPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(DashboardSection.allCases, id: \.self) { section in
                    Button {
                        withAnimation(.spring(duration: 0.3)) {
                            selectedSection = section
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: section.icon)
                                .font(.caption2)
                            Text(section.title)
                                .font(.caption.weight(.semibold))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(selectedSection == section ? Color.blue : Color(.tertiarySystemGroupedBackground))
                        .foregroundStyle(selectedSection == section ? .white : .primary)
                        .clipShape(Capsule())
                    }
                }
            }
        }
        .contentMargins(.horizontal, 0)
    }

    @ViewBuilder
    private var diagnosticBanner: some View {
        if let lastCode = metricsLogger.lastDiagnosticCode, lastCode == .modelEvicted || lastCode == .recoveryFailed {
            HStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title3)
                    .foregroundStyle(.white)
                    .symbolEffect(.pulse, options: .repeating)

                VStack(alignment: .leading, spacing: 2) {
                    Text(lastCode == .modelEvicted ? "Neural Engine Eviction Detected" : "Recovery Failed")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)

                    Text(lastCode == .modelEvicted ? "Hardware resources reclaimed. Recovering..." : "Model could not be restored. Restart recommended.")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.85))
                }

                Spacer(minLength: 0)
            }
            .padding(14)
            .background(
                LinearGradient(
                    colors: [.red, .orange],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .clipShape(.rect(cornerRadius: 14))
            .transition(.asymmetric(
                insertion: .move(edge: .top).combined(with: .opacity),
                removal: .opacity
            ))
        }

        if thermalGovernor.memoryPressureLevel != .normal {
            HStack(spacing: 10) {
                Image(systemName: "memorychip.fill")
                    .foregroundStyle(thermalGovernor.memoryPressureLevel == .critical ? .red : .yellow)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Memory Pressure: \(thermalGovernor.memoryPressureLevel.rawValue)")
                        .font(.subheadline.bold())
                    Text("Performance may be reduced to conserve memory")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }
            .padding(12)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(
                        thermalGovernor.memoryPressureLevel == .critical ? Color.red.opacity(0.4) : Color.yellow.opacity(0.4),
                        lineWidth: 1
                    )
            )
        }
    }

    private var speedSparklineSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Decode Speed", systemImage: "waveform.path.ecg")
                    .font(.headline)

                Spacer()

                Text("\(metricsLogger.currentMetrics.decodeTokensPerSecond, specifier: "%.1f") tok/s")
                    .font(.title2.bold().monospacedDigit())
                    .foregroundStyle(.blue)
                    .contentTransition(.numericText(value: metricsLogger.currentMetrics.decodeTokensPerSecond))
                    .animation(.spring(duration: 0.3), value: metricsLogger.currentMetrics.decodeTokensPerSecond)
            }

            SparklineView(data: metricsLogger.speedHistory, color: .blue)
                .frame(height: 60)
                .clipShape(.rect(cornerRadius: 8))

            HStack {
                Text("Last 60 samples")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Spacer()
                if let peak = metricsLogger.speedHistory.max() {
                    Text("Peak: \(peak, specifier: "%.1f") tok/s")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var liveMetricsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Live Inference", systemImage: "bolt.horizontal.fill")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                MetricCardView(
                    title: "Prefill Speed",
                    value: String(format: "%.0f", metricsLogger.currentMetrics.prefillTokensPerSecond),
                    unit: "tok/s",
                    icon: "bolt.horizontal.fill",
                    color: .purple
                )

                MetricCardView(
                    title: "Time to First Token",
                    value: String(format: "%.0f", metricsLogger.currentMetrics.timeToFirstTokenMS),
                    unit: "ms",
                    icon: "timer",
                    color: .orange
                )

                MetricCardView(
                    title: "Token Latency",
                    value: String(format: "%.1f", metricsLogger.currentMetrics.avgTokenLatencyMS),
                    unit: "ms/tok",
                    icon: "clock.arrow.circlepath",
                    color: .green
                )

                MetricCardView(
                    title: "Context Length",
                    value: "\(metricsLogger.currentMetrics.activeContextLength)",
                    unit: "tokens",
                    icon: "text.line.last.and.arrowtriangle.forward",
                    color: .teal
                )

                MetricCardView(
                    title: "Total Generated",
                    value: "\(metricsLogger.currentMetrics.totalTokensGenerated)",
                    unit: "tokens",
                    icon: "number",
                    color: .indigo
                )

                MetricCardView(
                    title: "Avg Decode Speed",
                    value: String(format: "%.1f", metricsLogger.averageDecodeSpeed),
                    unit: "tok/s",
                    icon: "chart.line.uptrend.xyaxis",
                    color: .blue
                )
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var thermalSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Thermal & Runtime", systemImage: "thermometer.medium")
                .font(.headline)

            HStack(spacing: 0) {
                thermalGaugeItem(
                    title: "Thermal",
                    value: thermalGovernor.thermalLevel.rawValue,
                    color: thermalColor,
                    icon: nil
                )

                Divider()
                    .frame(height: 40)

                thermalGaugeItem(
                    title: "Mode",
                    value: thermalGovernor.currentMode.rawValue,
                    color: runtimeModeColor,
                    icon: thermalGovernor.currentMode.icon
                )

                Divider()
                    .frame(height: 40)

                thermalGaugeItem(
                    title: "Speculative",
                    value: thermalGovernor.currentMode.speculativeEnabled ? "ON" : "OFF",
                    color: thermalGovernor.currentMode.speculativeEnabled ? .green : .red,
                    icon: nil
                )
            }

            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Max Draft")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(thermalGovernor.currentMode.maxDraftTokens)")
                        .font(.headline.monospacedDigit())
                }

                Spacer()

                VStack(alignment: .leading, spacing: 4) {
                    Text("Max Context")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(thermalGovernor.currentMode.maxContextLength)")
                        .font(.headline.monospacedDigit())
                }

                Spacer()

                VStack(alignment: .leading, spacing: 4) {
                    Text("Throttles")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(thermalGovernor.totalThrottleEvents)")
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(thermalGovernor.totalThrottleEvents > 0 ? .orange : .primary)
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var hardwareStatusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Hardware Status", systemImage: "cpu")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                hardwareCard(
                    title: "Compute Units",
                    value: metricsLogger.activeComputeLabel,
                    icon: "cpu",
                    color: computeUnitColor
                )

                hardwareCard(
                    title: "Runner State",
                    value: runnerStateLabel,
                    icon: runnerStateIcon,
                    color: runnerStateColor
                )

                hardwareCard(
                    title: "Recoveries",
                    value: "\(metricsLogger.totalRecoveries)",
                    icon: "arrow.triangle.2.circlepath",
                    color: metricsLogger.totalRecoveries > 0 ? .orange : .green
                )

                hardwareCard(
                    title: "Evictions",
                    value: "\(metricsLogger.totalEvictions)",
                    icon: "xmark.octagon",
                    color: metricsLogger.totalEvictions > 0 ? .red : .green
                )
            }

            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Memory Usage")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(thermalGovernor.currentMemoryUsageMB, specifier: "%.0f") MB")
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(.orange)
                }

                Spacer()

                VStack(alignment: .leading, spacing: 4) {
                    Text("Available")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(thermalGovernor.availableMemoryMB, specifier: "%.0f") MB")
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(.green)
                }

                Spacer()

                VStack(alignment: .leading, spacing: 4) {
                    Text("Session Uptime")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(metricsLogger.uptimeFormatted)
                        .font(.headline.monospacedDigit())
                }
            }

            if thermalGovernor.memoryPressureLevel != .normal {
                memoryPressureGauge
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var memoryPressureGauge: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Memory Pressure")
                    .font(.caption.bold())
                Spacer()
                Text(thermalGovernor.memoryPressureLevel.rawValue)
                    .font(.caption.bold())
                    .foregroundStyle(memoryPressureColor)
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.tertiarySystemGroupedBackground))

                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [.green, memoryPressureColor],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geo.size.width * memoryPressureFraction)
                        .animation(.spring(duration: 0.5), value: memoryPressureFraction)
                }
            }
            .frame(height: 8)
        }
        .padding(10)
        .background(Color(.tertiarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 10))
    }

    private func hardwareCard(title: String, value: String, icon: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundStyle(color)
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(value)
                .font(.subheadline.bold())
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(.tertiarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 10))
    }

    private var diagnosticSummarySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Diagnostic Summary", systemImage: "stethoscope")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                diagnosticStat(
                    label: "Events",
                    value: "\(metricsLogger.diagnosticEvents.count)",
                    color: .blue
                )
                diagnosticStat(
                    label: "Warnings",
                    value: "\(metricsLogger.diagnosticEvents.filter { $0.severity == .warning }.count)",
                    color: .yellow
                )
                diagnosticStat(
                    label: "Critical",
                    value: "\(metricsLogger.diagnosticEvents.filter { $0.severity == .critical }.count)",
                    color: .red
                )
            }

            if let health = inferenceEngine.lastHealthStatus {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(health.isHealthy ? .green : .red)
                            .frame(width: 8, height: 8)
                        Text("Health: \(health.isHealthy ? "Healthy" : "Degraded")")
                            .font(.subheadline.bold())
                    }

                    Text(health.diagnosticSummary)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                .padding(10)
                .background(Color(.tertiarySystemGroupedBackground))
                .clipShape(.rect(cornerRadius: 10))
            }

            Text(thermalGovernor.diagnosticSummary)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(.tertiarySystemGroupedBackground))
                .clipShape(.rect(cornerRadius: 10))
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private func diagnosticStat(label: String, value: String, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title2.bold().monospacedDigit())
                .foregroundStyle(color)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color(.tertiarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 10))
    }

    private var diagnosticEventLogSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Event Log", systemImage: "list.bullet.rectangle")
                    .font(.headline)
                Spacer()
                Text("\(metricsLogger.diagnosticEvents.count) events")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            if metricsLogger.diagnosticEvents.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.shield")
                        .font(.title2)
                        .foregroundStyle(.green)
                    Text("No diagnostic events")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Text("System is running normally")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else {
                ForEach(Array(metricsLogger.diagnosticEvents.suffix(20).reversed().enumerated()), id: \.element.id) { _, event in
                    diagnosticEventRow(event)
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private func diagnosticEventRow(_ event: DiagnosticEvent) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(severityColor(event.severity))
                .frame(width: 8, height: 8)
                .padding(.top, 5)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(event.code.rawValue)
                        .font(.caption.bold().monospaced())
                        .foregroundStyle(severityColor(event.severity))

                    Spacer()

                    Text(event.timestamp, style: .time)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                Text(event.message)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if !event.metadata.isEmpty {
                    HStack(spacing: 6) {
                        ForEach(Array(event.metadata.keys.sorted().prefix(3)), id: \.self) { key in
                            Text("\(key): \(event.metadata[key] ?? "")")
                                .font(.system(size: 9).monospaced())
                                .foregroundStyle(.tertiary)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(Color(.quaternarySystemFill))
                                .clipShape(Capsule())
                        }
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func thermalGaugeItem(title: String, value: String, color: Color, icon: String?) -> some View {
        VStack(spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.caption)
                        .foregroundStyle(color)
                } else {
                    Circle()
                        .fill(color)
                        .frame(width: 8, height: 8)
                }

                Text(value)
                    .font(.subheadline.bold())
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var cacheSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("KV Cache & Speculation", systemImage: "memorychip")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                MetricCardView(
                    title: "KV Pages",
                    value: "\(metricsLogger.currentMetrics.kvPagesInUse)",
                    unit: "active",
                    icon: "square.grid.3x3.topleft.filled",
                    color: .mint
                )

                MetricCardView(
                    title: "Spec. Accepted",
                    value: "\(metricsLogger.currentMetrics.acceptedSpeculativeTokens)",
                    unit: "tokens",
                    icon: "checkmark.circle.fill",
                    color: .green
                )

                MetricCardView(
                    title: "Spec. Rejected",
                    value: "\(metricsLogger.currentMetrics.rejectedSpeculativeTokens)",
                    unit: "tokens",
                    icon: "xmark.circle.fill",
                    color: .red
                )

                MetricCardView(
                    title: "Accept Rate",
                    value: String(format: "%.0f", metricsLogger.currentMetrics.speculativeAcceptanceRate * 100),
                    unit: "%",
                    icon: "chart.bar.fill",
                    color: .cyan
                )
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var memorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Memory", systemImage: "memorychip.fill")
                .font(.headline)

            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Peak KV Memory")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    let mb = Double(metricsLogger.currentMetrics.peakMemoryBytes) / 1_048_576
                    Text("\(mb, specifier: "%.1f") MB")
                        .font(.title3.bold().monospacedDigit())
                        .foregroundStyle(.orange)
                }

                Spacer()

                VStack(alignment: .leading, spacing: 4) {
                    Text("Context Evictions")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("\(metricsLogger.currentMetrics.contextEvictions)")
                        .font(.title3.bold().monospacedDigit())
                        .foregroundStyle(metricsLogger.currentMetrics.contextEvictions > 0 ? .yellow : .primary)
                }

                Spacer()

                VStack(alignment: .leading, spacing: 4) {
                    Text("Total Evicted")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("\(metricsLogger.totalEvictedTokens)")
                        .font(.title3.bold().monospacedDigit())
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var historySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Generation History", systemImage: "clock.arrow.circlepath")
                .font(.headline)

            if metricsLogger.history.isEmpty {
                historyEmptyState
            } else {
                historyList
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
    }

    private var historyEmptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.line.text.clipboard")
                .font(.title2)
                .foregroundStyle(.tertiary)
            Text("No generations yet")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
    }

    private var historyList: some View {
        let recentHistory = Array(metricsLogger.history.suffix(10))
        let totalCount = metricsLogger.history.count
        return ForEach(Array(recentHistory.enumerated()), id: \.offset) { index, entry in
            historyRow(index: index, entry: entry, totalCount: totalCount, recentCount: recentHistory.count)
            if index < recentHistory.count - 1 {
                Divider()
            }
        }
    }

    private func historyRow(index: Int, entry: InferenceMetrics, totalCount: Int, recentCount: Int) -> some View {
        let rowNumber = totalCount - (recentCount - 1 - index)
        let speedText = String(format: "%.1f tok/s", entry.decodeTokensPerSecond)
        let detailText = "\(entry.totalTokensGenerated) tokens \u{00B7} \(String(format: "%.0f", entry.timeToFirstTokenMS))ms TTFT"

        return HStack(spacing: 12) {
            Text("#\(rowNumber)")
                .font(.caption.monospacedDigit().bold())
                .foregroundStyle(.secondary)
                .frame(width: 28, alignment: .leading)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(speedText)
                        .font(.subheadline.bold().monospacedDigit())

                    if entry.speculativeAcceptanceRate > 0 {
                        speculativeBadge(rate: entry.speculativeAcceptanceRate)
                    }
                }

                Text(detailText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(entry.thermalState.rawValue)
                .font(.caption2)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(thermalHistoryColor(entry.thermalState).opacity(0.15))
                .foregroundStyle(thermalHistoryColor(entry.thermalState))
                .clipShape(Capsule())
        }
    }

    private func speculativeBadge(rate: Double) -> some View {
        HStack(spacing: 2) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 8))
            Text(String(format: "%.0f%%", rate * 100))
        }
        .font(.caption2.bold())
        .foregroundStyle(.blue)
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        .background(.blue.opacity(0.1))
        .clipShape(Capsule())
    }

    private var thermalColor: Color {
        switch thermalGovernor.thermalLevel {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        }
    }

    private var runtimeModeColor: Color {
        switch thermalGovernor.currentMode {
        case .maxPerformance: return .blue
        case .balanced: return .green
        case .coolDown: return .orange
        case .emergency: return .red
        }
    }

    private var computeUnitColor: Color {
        switch metricsLogger.activeComputeLabel {
        case "All": return .green
        case "CPU+ANE": return .blue
        case "CPU+GPU": return .purple
        case "CPU": return .orange
        default: return .secondary
        }
    }

    private var runnerStateLabel: String {
        inferenceEngine.lastHealthStatus?.state.rawValue.capitalized ?? "Unknown"
    }

    private var runnerStateIcon: String {
        guard let health = inferenceEngine.lastHealthStatus else { return "questionmark.circle" }
        switch health.state {
        case .idle: return "moon"
        case .loading: return "arrow.down.circle"
        case .ready: return "checkmark.circle.fill"
        case .recovering: return "arrow.triangle.2.circlepath"
        case .disposing: return "trash"
        case .evicted: return "exclamationmark.triangle.fill"
        }
    }

    private var runnerStateColor: Color {
        guard let health = inferenceEngine.lastHealthStatus else { return .secondary }
        switch health.state {
        case .ready: return .green
        case .recovering: return .orange
        case .evicted: return .red
        default: return .secondary
        }
    }

    private var memoryPressureColor: Color {
        switch thermalGovernor.memoryPressureLevel {
        case .normal: return .green
        case .warning: return .yellow
        case .critical: return .red
        }
    }

    private var memoryPressureFraction: CGFloat {
        switch thermalGovernor.memoryPressureLevel {
        case .normal: return 0.3
        case .warning: return 0.65
        case .critical: return 0.95
        }
    }

    private func severityColor(_ severity: DiagnosticSeverity) -> Color {
        switch severity {
        case .info: return .blue
        case .warning: return .yellow
        case .critical: return .red
        }
    }

    private func thermalHistoryColor(_ level: ThermalLevel) -> Color {
        switch level {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        }
    }
}

enum DashboardSection: String, CaseIterable {
    case overview
    case hardware
    case diagnostics
    case history

    var title: String {
        switch self {
        case .overview: return "Overview"
        case .hardware: return "Hardware"
        case .diagnostics: return "Diagnostics"
        case .history: return "History"
        }
    }

    var icon: String {
        switch self {
        case .overview: return "gauge.with.dots.needle.67percent"
        case .hardware: return "cpu"
        case .diagnostics: return "stethoscope"
        case .history: return "clock.arrow.circlepath"
        }
    }
}

struct SparklineView: View {
    let data: [Double]
    let color: Color

    var body: some View {
        GeometryReader { geo in
            if data.count >= 2 {
                let minVal = (data.min() ?? 0) * 0.9
                let maxVal = max((data.max() ?? 1) * 1.1, minVal + 1)
                let range = maxVal - minVal

                ZStack {
                    color.opacity(0.05)

                    Path { path in
                        let stepX = geo.size.width / CGFloat(max(data.count - 1, 1))
                        for (i, val) in data.enumerated() {
                            let x = CGFloat(i) * stepX
                            let y = geo.size.height * (1 - CGFloat((val - minVal) / range))
                            if i == 0 {
                                path.move(to: CGPoint(x: x, y: y))
                            } else {
                                path.addLine(to: CGPoint(x: x, y: y))
                            }
                        }
                    }
                    .stroke(color, lineWidth: 1.5)

                    Path { path in
                        let stepX = geo.size.width / CGFloat(max(data.count - 1, 1))
                        for (i, val) in data.enumerated() {
                            let x = CGFloat(i) * stepX
                            let y = geo.size.height * (1 - CGFloat((val - minVal) / range))
                            if i == 0 {
                                path.move(to: CGPoint(x: x, y: y))
                            } else {
                                path.addLine(to: CGPoint(x: x, y: y))
                            }
                        }
                        path.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height))
                        path.addLine(to: CGPoint(x: 0, y: geo.size.height))
                        path.closeSubpath()
                    }
                    .fill(
                        LinearGradient(
                            colors: [color.opacity(0.3), color.opacity(0.0)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
            } else {
                color.opacity(0.05)
                    .overlay {
                        Text("Waiting for data...")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
            }
        }
    }
}

struct MetricCardView: View {
    let title: String
    let value: String
    let unit: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundStyle(color)

                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.title3.bold().monospacedDigit())

                Text(unit)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(.tertiarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 10))
    }
}
