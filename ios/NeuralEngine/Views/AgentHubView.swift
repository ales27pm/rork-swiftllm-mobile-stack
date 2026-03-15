import SwiftUI

struct AgentHubView: View {
    @Bindable var chatViewModel: ChatViewModel
    @Bindable var speechViewModel: SpeechViewModel
    let agent: AssistantAgent
    let historyViewModel: HistoryViewModel?
    let memoryViewModel: MemoryViewModel?
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let inferenceEngine: InferenceEngine?
    let onLoadConversation: (String) -> Void

    @State private var activeSheet: AgentCapability?
    @State private var showSpeechMode: Bool = false
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if let error = chatViewModel.lastError {
                    diagnosticBanner(error)
                }

                AgentStatusStrip(agent: agent)

                if chatViewModel.messages.isEmpty {
                    emptyState
                } else {
                    messageList
                }

                if chatViewModel.isGenerating, let frame = chatViewModel.lastCognitionFrame {
                    reasoningStatusBar(frame)
                }

                actionBarSection

                inputBar
            }
            .background(Color(.systemBackground))
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $chatViewModel.toolExecutor.showShareSheet) {
                if !chatViewModel.toolExecutor.shareItems.isEmpty {
                    ShareSheet(items: chatViewModel.toolExecutor.shareItems)
                }
            }
            .sheet(item: $activeSheet) { capability in
                capabilitySheet(capability)
            }
            .toolbar {
                ToolbarItem(placement: .principal) {
                    headerTitle
                }
                ToolbarItem(placement: .topBarLeading) {
                    if chatViewModel.isGenerating || chatViewModel.isExecutingTools {
                        liveSpeedBadge
                    } else {
                        Button {
                            chatViewModel.newConversation()
                        } label: {
                            Image(systemName: "square.and.pencil")
                                .foregroundStyle(.blue)
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        if !chatViewModel.messages.isEmpty {
                            Button("Copy Conversation", systemImage: "doc.on.doc") {
                                copyConversation()
                            }
                        }

                        Button("History", systemImage: "clock.arrow.circlepath") {
                            activeSheet = .history
                        }

                        Button("Memory", systemImage: "brain") {
                            activeSheet = .memory
                        }

                        Button("Metrics", systemImage: "gauge.with.dots.needle.67percent") {
                            activeSheet = .metrics
                        }

                        Divider()

                        Toggle(isOn: Binding(
                            get: { chatViewModel.toolsEnabled },
                            set: { chatViewModel.toolsEnabled = $0; chatViewModel.saveSettings() }
                        )) {
                            Label("Device Tools", systemImage: "wrench.and.screwdriver")
                        }

                        if !chatViewModel.messages.isEmpty {
                            Divider()
                            Button("Clear Chat", systemImage: "trash", role: .destructive) {
                                chatViewModel.clearChat()
                            }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var headerTitle: some View {
        VStack(spacing: 1) {
            HStack(spacing: 4) {
                Text("Nexus")
                    .font(.subheadline.bold())

                if chatViewModel.toolsEnabled {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.orange)
                }
            }

            HStack(spacing: 4) {
                Circle()
                    .fill(chatViewModel.hasActiveModel ? .green : Color(.tertiaryLabel))
                    .frame(width: 5, height: 5)

                Text(chatViewModel.activeModelName)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var liveSpeedBadge: some View {
        HStack(spacing: 4) {
            ProgressView()
                .controlSize(.mini)

            if chatViewModel.isExecutingTools {
                Text("Tools")
                    .font(.caption2.monospacedDigit().bold())
                    .foregroundStyle(.orange)
            } else {
                Text("\(chatViewModel.metricsLogger.currentMetrics.decodeTokensPerSecond, specifier: "%.1f") t/s")
                    .font(.caption2.monospacedDigit().bold())
                    .foregroundStyle(.blue)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(chatViewModel.isExecutingTools ? .orange.opacity(0.1) : .blue.opacity(0.1))
        .clipShape(Capsule())
    }

    private var emptyState: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer(minLength: 20)

                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [.blue.opacity(0.15), .purple.opacity(0.15)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 100, height: 100)

                    Image(systemName: "brain.filled.head.profile")
                        .font(.system(size: 44))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.blue, .purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .symbolEffect(.breathe, options: .repeating)
                }

                VStack(spacing: 8) {
                    Text("Nexus Agent")
                        .font(.title2.bold())

                    Text("Your unified on-device AI assistant\nChat · Search · Scan · Navigate · Remember")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                if !chatViewModel.hasActiveModel {
                    Label("Load a model from the Models tab", systemImage: "arrow.down.circle")
                        .font(.callout.weight(.medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(.blue.gradient)
                        .clipShape(Capsule())
                }

                capabilityGrid

                VStack(spacing: 8) {
                    promptSuggestion("What's my battery level?", icon: "battery.100percent")
                    promptSuggestion("Search the web for latest AI news", icon: "globe")
                    promptSuggestion("What's on my calendar this week?", icon: "calendar")
                    promptSuggestion("Where am I right now?", icon: "location.fill")
                }
                .padding(.top, 4)

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 24)
        }
    }

    private var capabilityGrid: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible())
        ], spacing: 12) {
            capabilityTile("History", icon: "clock.arrow.circlepath", color: .blue, badge: agent.conversationCount > 0 ? "\(agent.conversationCount)" : nil) {
                activeSheet = .history
            }
            capabilityTile("Memory", icon: "brain", color: .purple, badge: agent.memoryCount > 0 ? "\(agent.memoryCount)" : nil) {
                activeSheet = .memory
            }
            capabilityTile("Browse", icon: "globe", color: .cyan, badge: nil) {
                activeSheet = .browse
            }
            capabilityTile("Map", icon: "map", color: .green, badge: nil) {
                activeSheet = .map
            }
            capabilityTile("Scan", icon: "doc.text.viewfinder", color: .orange, badge: nil) {
                activeSheet = .scan
            }
            capabilityTile("Metrics", icon: "gauge.with.dots.needle.67percent", color: .teal, badge: agent.systemHealthStatus != .optimal ? "!" : nil) {
                activeSheet = .metrics
            }
        }
        .padding(.horizontal, 8)
    }

    private func capabilityTile(_ title: String, icon: String, color: Color, badge: String?, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 6) {
                ZStack(alignment: .topTrailing) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(color.opacity(0.12))
                            .frame(width: 44, height: 44)

                        Image(systemName: icon)
                            .font(.system(size: 18))
                            .foregroundStyle(color)
                    }

                    if let badge {
                        Text(badge)
                            .font(.system(size: 8).weight(.bold).monospacedDigit())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(color)
                            .clipShape(Capsule())
                            .offset(x: 4, y: -4)
                    }
                }

                Text(title)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.primary)
            }
        }
    }

    private func promptSuggestion(_ text: String, icon: String) -> some View {
        Button {
            chatViewModel.inputText = text
            chatViewModel.sendMessage()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundStyle(.blue)
                    .frame(width: 24)

                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.primary)

                Spacer()

                Image(systemName: "arrow.up.circle.fill")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.secondarySystemBackground))
            .clipShape(.rect(cornerRadius: 12))
        }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(chatViewModel.messages) { message in
                        if message.isToolExecution {
                            ToolResultBubbleView(message: message)
                                .id(message.id)
                        } else if message.role != .tool {
                            MessageBubbleView(message: message)
                                .id(message.id)
                                .contextMenu {
                                    Button("Copy", systemImage: "doc.on.doc") {
                                        UIPasteboard.general.string = message.content
                                    }
                                    if message.role == .assistant, let metrics = message.metrics {
                                        Button("Copy Metrics", systemImage: "chart.bar") {
                                            let text = "\(metrics.decodeTokensPerSecond.formatted(.number.precision(.fractionLength(1)))) tok/s · \(metrics.totalTokens) tokens · \(metrics.totalDuration.formatted(.number.precision(.fractionLength(2))))s"
                                            UIPasteboard.general.string = text
                                        }
                                    }
                                }
                        }
                    }
                }
                .padding(.vertical, 12)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: chatViewModel.messages.count) { _, _ in
                if let last = chatViewModel.messages.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: chatViewModel.messages.last?.content) { _, _ in
                if let last = chatViewModel.messages.last, last.isStreaming {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private var actionBarSection: some View {
        AgentActionBar(agent: agent) { capability in
            activeSheet = capability
        }
        .padding(.vertical, 4)
        .background(Color(.secondarySystemBackground).opacity(0.5))
    }

    private var inputBar: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(alignment: .bottom, spacing: 10) {
                TextField("Message", text: $chatViewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .focused($isInputFocused)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(.tertiarySystemBackground))
                    .clipShape(.rect(cornerRadius: 20))
                    .onSubmit {
                        if !chatViewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            chatViewModel.sendMessage()
                        }
                    }

                if chatViewModel.isGenerating || chatViewModel.isExecutingTools {
                    VStack(spacing: 4) {
                        if let length = chatViewModel.expectedResponseLength {
                            Text(length.rawValue.capitalized)
                                .font(.system(size: 9).weight(.semibold))
                                .foregroundStyle(.secondary)
                                .transition(.opacity)
                        }
                        Button {
                            chatViewModel.stopGeneration()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.red)
                                .symbolEffect(.pulse, options: .repeating)
                        }
                    }
                    .sensoryFeedback(.impact(weight: .medium), trigger: chatViewModel.isGenerating)
                } else {
                    HStack(spacing: 8) {
                        Button {
                            showSpeechMode = true
                        } label: {
                            Image(systemName: "waveform.circle.fill")
                                .font(.title2)
                                .foregroundStyle(
                                    LinearGradient(
                                        colors: [.blue, .purple],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        }

                        Button {
                            chatViewModel.sendIntent()
                        } label: {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.title2)
                                .foregroundStyle(
                                    chatViewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                        ? Color(.tertiaryLabel)
                                        : .blue
                                )
                        }
                        .disabled(chatViewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .sensoryFeedback(.impact(weight: .light), trigger: chatViewModel.messages.count)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(.secondarySystemBackground))
        }
        .fullScreenCover(isPresented: $showSpeechMode) {
            SpeechModeView(viewModel: speechViewModel)
        }
    }

    @ViewBuilder
    private func capabilitySheet(_ capability: AgentCapability) -> some View {
        switch capability {
        case .history:
            if let historyVM = historyViewModel {
                HistoryView(viewModel: historyVM) { conversationId in
                    onLoadConversation(conversationId)
                    activeSheet = nil
                }
            }
        case .memory:
            if let memoryVM = memoryViewModel {
                MemoryView(viewModel: memoryVM)
            }
        case .browse:
            WebSearchView()
        case .map:
            NavigationStack {
                MapView()
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") { activeSheet = nil }
                        }
                    }
            }
        case .scan:
            NavigationStack {
                DocumentAnalysisView()
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") { activeSheet = nil }
                        }
                    }
            }
        case .models:
            NavigationStack {
                if let modelVM = ModelManagerViewModel(modelLoader: agent.modelLoader) as ModelManagerViewModel? {
                    ModelManagerView(viewModel: modelVM)
                        .toolbar {
                            ToolbarItem(placement: .topBarLeading) {
                                Button("Done") { activeSheet = nil }
                            }
                        }
                }
            }
        case .metrics:
            NavigationStack {
                if let engine = inferenceEngine {
                    MetricsDashboardView(
                        metricsLogger: metricsLogger,
                        thermalGovernor: thermalGovernor,
                        inferenceEngine: engine,
                        chatViewModel: chatViewModel
                    )
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") { activeSheet = nil }
                        }
                    }
                }
            }
        }
    }

    private func diagnosticBanner(_ error: WrappedError) -> some View {
        HStack(spacing: 10) {
            Image(systemName: error.severity == .critical ? "exclamationmark.octagon.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(error.severity == .critical ? .red : .orange)

            VStack(alignment: .leading, spacing: 2) {
                Text(error.userMessage)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)

                HStack(spacing: 4) {
                    Text(error.domain.rawValue)
                        .font(.system(size: 9).monospaced())
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color(.quaternarySystemFill))
                        .clipShape(Capsule())

                    if error.recoveryAction != .none {
                        Text(error.recoveryAction.rawValue)
                            .font(.system(size: 9).monospaced())
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Color.blue.opacity(0.1))
                            .clipShape(Capsule())
                    }
                }
            }

            Spacer(minLength: 0)

            Button {
                chatViewModel.dismissError()
            } label: {
                Image(systemName: "xmark")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 24, height: 24)
                    .background(Color(.quaternarySystemFill))
                    .clipShape(Circle())
            }
        }
        .padding(10)
        .background(error.severity == .critical ? Color.red.opacity(0.08) : Color.orange.opacity(0.08))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundStyle(error.severity == .critical ? Color.red.opacity(0.2) : Color.orange.opacity(0.2)),
            alignment: .bottom
        )
        .transition(.move(edge: .top).combined(with: .opacity))
        .animation(.spring(duration: 0.3), value: chatViewModel.lastError?.id)
    }

    private func reasoningStatusBar(_ frame: CognitionFrame) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "brain.head.profile.fill")
                .font(.caption2)
                .foregroundStyle(.purple)
                .symbolEffect(.pulse, options: .repeating)

            Text(frame.reasoningTrace.dominantStrategy.rawValue)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.purple)

            Spacer()

            HStack(spacing: 4) {
                Text("\(Int(frame.reasoningTrace.finalConvergence * 100))%")
                    .font(.caption2.monospacedDigit().bold())
                    .foregroundStyle(frame.reasoningTrace.finalConvergence > 0.7 ? .green : .orange)

                Text("converged")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }

            if frame.thoughtTree.prunedBranches.count > 0 {
                Text("\(frame.thoughtTree.prunedBranches.count) pruned")
                    .font(.system(size: 9).monospaced())
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Color.orange.opacity(0.1))
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(.secondarySystemBackground))
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .animation(.spring(duration: 0.3), value: chatViewModel.isGenerating)
    }

    private func copyConversation() {
        let text = chatViewModel.messages.filter { !$0.isToolExecution }.map { msg in
            let role = msg.role == .user ? "You" : "Nexus"
            return "\(role): \(msg.content)"
        }.joined(separator: "\n\n")
        UIPasteboard.general.string = text
    }
}
