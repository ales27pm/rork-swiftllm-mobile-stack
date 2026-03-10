import SwiftUI

struct MetricsDashboardView: View {
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let inferenceEngine: InferenceEngine

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                speedSparklineSection
                liveMetricsSection
                thermalSection
                cacheSection
                memorySection
                historySection
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Metrics")
        .navigationBarTitleDisplayMode(.large)
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
                    Text("Evictions")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(metricsLogger.totalEvictedTokens)")
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(metricsLogger.totalEvictedTokens > 0 ? .orange : .primary)
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
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
            } else {
                ForEach(Array(metricsLogger.history.suffix(5).enumerated()), id: \.offset) { index, entry in
                    HStack(spacing: 12) {
                        Text("#\(metricsLogger.history.count - (4 - index))")
                            .font(.caption.monospacedDigit().bold())
                            .foregroundStyle(.secondary)
                            .frame(width: 28, alignment: .leading)

                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text("\(entry.decodeTokensPerSecond, specifier: "%.1f") tok/s")
                                    .font(.subheadline.bold().monospacedDigit())

                                if entry.speculativeAcceptanceRate > 0 {
                                    HStack(spacing: 2) {
                                        Image(systemName: "bolt.fill")
                                            .font(.system(size: 8))
                                        Text("\(entry.speculativeAcceptanceRate * 100, specifier: "%.0f")%")
                                    }
                                    .font(.caption2.bold())
                                    .foregroundStyle(.blue)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 2)
                                    .background(.blue.opacity(0.1))
                                    .clipShape(Capsule())
                                }
                            }

                            Text("\(entry.totalTokensGenerated) tokens · \(entry.timeToFirstTokenMS, specifier: "%.0f")ms TTFT")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        Text(entry.thermalState.rawValue)
                            .font(.caption2)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color(.tertiarySystemBackground))
                            .clipShape(Capsule())
                    }

                    if index < metricsLogger.history.suffix(5).count - 1 {
                        Divider()
                    }
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
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
