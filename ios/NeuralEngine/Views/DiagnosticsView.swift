import SwiftUI

struct DiagnosticsView: View {
    let database: DatabaseService
    let keyValueStore: KeyValueStore
    let secureStore: SecureStore
    let fileSystem: FileSystemService
    let memoryService: MemoryService?
    let conversationService: ConversationService?
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let inferenceEngine: InferenceEngine?

    @State private var engine = DiagnosticEngine()
    @State private var selectedCategory: DiagnosticCategory?
    @State private var showShareSheet: Bool = false
    @State private var scrollToBottom: Bool = false

    var body: some View {
        ScrollViewReader { proxy in
            List {
                headerSection
                if engine.isRunning || !engine.results.isEmpty {
                    progressSection
                }
                if !engine.results.isEmpty {
                    resultsSection
                }
                if let url = engine.reportURL, !engine.isRunning {
                    reportSection(url: url)
                }
                Color.clear.frame(height: 1).id("bottom")
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Diagnostics")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if engine.isRunning {
                        Button("Cancel") {
                            engine.cancel()
                        }
                        .foregroundStyle(.red)
                    }
                }
            }
            .onChange(of: engine.currentTestIndex) { _, _ in
                withAnimation {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .sheet(isPresented: $showShareSheet) {
                if let url = engine.reportURL {
                    ShareSheet(items: [url])
                }
            }
        }
        .task {
            engine.configure(
                database: database,
                keyValueStore: keyValueStore,
                secureStore: secureStore,
                fileSystem: fileSystem,
                memoryService: memoryService,
                conversationService: conversationService,
                metricsLogger: metricsLogger,
                thermalGovernor: thermalGovernor,
                modelLoader: modelLoader,
                inferenceEngine: inferenceEngine
            )
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        Section {
            VStack(spacing: 16) {
                Image(systemName: "stethoscope")
                    .font(.system(size: 44))
                    .foregroundStyle(.tint)
                    .symbolEffect(.pulse, options: .repeating, isActive: engine.isRunning)

                VStack(spacing: 4) {
                    Text("System Diagnostics")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Run exhaustive tests on all subsystems with real data — no mocks or stubs.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                if !engine.isRunning && engine.results.isEmpty {
                    Button {
                        Task { await engine.runAllTests() }
                    } label: {
                        Label("Run All Tests", systemImage: "play.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                } else if !engine.isRunning && !engine.results.isEmpty {
                    Button {
                        Task { await engine.runAllTests() }
                    } label: {
                        Label("Re-Run All Tests", systemImage: "arrow.clockwise")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }
            }
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Progress

    private var progressSection: some View {
        Section {
            VStack(spacing: 12) {
                if engine.isRunning {
                    HStack(spacing: 12) {
                        ProgressView()
                            .controlSize(.small)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(engine.currentCategory?.rawValue ?? "Preparing…")
                                .font(.subheadline)
                                .fontWeight(.medium)

                            Text("\(engine.currentTestIndex + 1) of \(engine.totalTests)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        Text(String(format: "%.0f%%", engine.progress * 100))
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                }

                ProgressView(value: engine.isRunning ? engine.progress : 1.0)
                    .tint(progressTint)

                HStack(spacing: 16) {
                    StatBadge(count: engine.passedCount, label: "Passed", color: .green, icon: "checkmark.circle.fill")
                    StatBadge(count: engine.failedCount, label: "Failed", color: .red, icon: "xmark.circle.fill")
                    StatBadge(count: engine.warningCount, label: "Warn", color: .orange, icon: "exclamationmark.triangle.fill")
                    StatBadge(count: engine.skippedCount, label: "Skip", color: .secondary, icon: "forward.circle.fill")
                }

                if let start = engine.startTime {
                    let elapsed = (engine.completionTime ?? Date()).timeIntervalSince(start)
                    Text("Elapsed: \(String(format: "%.1f", elapsed))s")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.vertical, 4)
        } header: {
            Text("Progress")
        }
    }

    private var progressTint: Color {
        if engine.failedCount > 0 { return .red }
        if engine.warningCount > 0 { return .orange }
        return .green
    }

    // MARK: - Results

    private var resultsSection: some View {
        ForEach(DiagnosticCategory.allCases, id: \.rawValue) { category in
            let categoryResults = engine.results.filter { $0.category == category }
            if !categoryResults.isEmpty {
                Section {
                    ForEach(categoryResults) { result in
                        TestResultRow(result: result)
                    }
                } header: {
                    HStack {
                        Text(category.rawValue)
                        Spacer()
                        let passed = categoryResults.filter { $0.status == .passed }.count
                        let total = categoryResults.count
                        Text("\(passed)/\(total)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Report

    private func reportSection(url: URL) -> some View {
        Section {
            VStack(spacing: 12) {
                HStack {
                    Image(systemName: "doc.text.fill")
                        .font(.title3)
                        .foregroundStyle(.tint)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Diagnostic Report Ready")
                            .font(.subheadline)
                            .fontWeight(.medium)

                        Text(url.lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()
                }

                HStack(spacing: 12) {
                    ShareLink(item: url) {
                        Label("Share Log", systemImage: "square.and.arrow.up")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)

                    Button {
                        showShareSheet = true
                    } label: {
                        Label("Export", systemImage: "arrow.down.doc")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                }
            }
            .padding(.vertical, 4)
        } header: {
            Text("Report")
        }
    }
}

// MARK: - Subviews

private struct StatBadge: View {
    let count: Int
    let label: String
    let color: Color
    let icon: String

    var body: some View {
        VStack(spacing: 2) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                Text("\(count)")
                    .fontWeight(.bold)
                    .monospacedDigit()
            }
            .foregroundStyle(color)

            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct TestResultRow: View {
    let result: DiagnosticTestResult

    @State private var isExpanded: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: result.statusIcon)
                    .foregroundStyle(statusColor)
                    .font(.subheadline)

                VStack(alignment: .leading, spacing: 2) {
                    Text(result.name)
                        .font(.subheadline)
                        .fontWeight(.medium)

                    if !result.message.isEmpty {
                        Text(result.message)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(isExpanded ? nil : 1)
                    }
                }

                Spacer()

                if result.status != .pending && result.status != .running {
                    Text(String(format: "%.0fms", result.duration * 1000))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .monospacedDigit()
                }

                if result.status == .running {
                    ProgressView()
                        .controlSize(.mini)
                }
            }

            if isExpanded && !result.details.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(result.details.enumerated()), id: \.offset) { _, detail in
                        Text("→ \(detail)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.leading, 28)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if !result.details.isEmpty {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            }
        }
    }

    private var statusColor: Color {
        switch result.status {
        case .passed: return .green
        case .failed: return .red
        case .warning: return .orange
        case .skipped: return .secondary
        case .running: return .blue
        case .pending: return .secondary.opacity(0.4)
        }
    }
}


