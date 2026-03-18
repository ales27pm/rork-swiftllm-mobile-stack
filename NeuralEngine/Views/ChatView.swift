import SwiftUI

struct ChatView: View {
    @Bindable var viewModel: ChatViewModel
    @Bindable var speechViewModel: SpeechViewModel
    @FocusState private var isInputFocused: Bool
    @State private var showSpeechMode: Bool = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if let error = viewModel.lastError {
                    diagnosticBanner(error)
                }

                if viewModel.messages.isEmpty {
                    emptyState
                } else {
                    messageList
                }

                if viewModel.isGenerating, let frame = viewModel.lastCognitionFrame {
                    reasoningStatusBar(frame)
                }

                inputBar
            }
            .background(Color(.systemBackground))
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $viewModel.toolExecutor.showShareSheet) {
                if !viewModel.toolExecutor.shareItems.isEmpty {
                    ShareSheet(items: viewModel.toolExecutor.shareItems)
                }
            }
            .sheet(isPresented: $viewModel.toolExecutor.showSMSComposer) {
                MessageComposerView(toolExecutor: viewModel.toolExecutor)
            }
            .sheet(isPresented: $viewModel.toolExecutor.showEmailComposer) {
                MailComposerView(toolExecutor: viewModel.toolExecutor)
            }
            .toolbar {
                ToolbarItem(placement: .principal) {
                    headerTitle
                }
                ToolbarItem(placement: .topBarLeading) {
                    if viewModel.isGenerating || viewModel.isExecutingTools {
                        liveSpeedBadge
                    } else {
                        Button {
                            viewModel.newConversation()
                        } label: {
                            Image(systemName: "square.and.pencil")
                                .foregroundStyle(.blue)
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        if !viewModel.messages.isEmpty {
                            Button("Copy Conversation", systemImage: "doc.on.doc") {
                                copyConversation()
                            }
                        }
                        if !viewModel.messages.isEmpty {
                            Button("Clear Chat", systemImage: "trash", role: .destructive) {
                                viewModel.clearChat()
                            }
                        }
                        Divider()
                        Toggle(isOn: Binding(
                            get: { viewModel.toolsEnabled },
                            set: { viewModel.toolsEnabled = $0; viewModel.saveSettings() }
                        )) {
                            Label("Device Tools", systemImage: "wrench.and.screwdriver")
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

                if viewModel.toolsEnabled {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.orange)
                }
            }

            HStack(spacing: 4) {
                Circle()
                    .fill(viewModel.hasActiveModel ? .green : Color(.tertiaryLabel))
                    .frame(width: 5, height: 5)

                Text(viewModel.activeModelName)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var liveSpeedBadge: some View {
        HStack(spacing: 4) {
            ProgressView()
                .controlSize(.mini)

            if viewModel.isExecutingTools {
                Text("Tools")
                    .font(.caption2.monospacedDigit().bold())
                    .foregroundStyle(.orange)
            } else {
                Text("\(viewModel.metricsLogger.currentMetrics.decodeTokensPerSecond, specifier: "%.1f") t/s")
                    .font(.caption2.monospacedDigit().bold())
                    .foregroundStyle(.blue)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(viewModel.isExecutingTools ? .orange.opacity(0.1) : .blue.opacity(0.1))
        .clipShape(Capsule())
    }

    private var emptyState: some View {
        VStack(spacing: 24) {
            Spacer()

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
                Text("Nexus AI")
                    .font(.title2.bold())

                Text("On-device intelligence with tools\nPrivate · Fast · Contextual")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            if !viewModel.hasActiveModel {
                Label("Load a model from the Models tab", systemImage: "arrow.down.circle")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(.blue.gradient)
                    .clipShape(Capsule())
            }

            VStack(spacing: 8) {
                promptSuggestion("What's my battery level?", icon: "battery.100percent")
                promptSuggestion("Where am I right now?", icon: "location.fill")
                promptSuggestion("What's on my calendar this week?", icon: "calendar")
                promptSuggestion("What time is it?", icon: "clock.fill")
            }
            .padding(.top, 4)

            Spacer()
        }
        .padding(.horizontal, 32)
    }

    private func promptSuggestion(_ text: String, icon: String) -> some View {
        Button {
            viewModel.inputText = text
            viewModel.sendMessage()
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
                    ForEach(viewModel.messages) { message in
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
            .onChange(of: viewModel.messages.count) { _, _ in
                if let last = viewModel.messages.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: viewModel.messages.last?.content) { _, _ in
                if let last = viewModel.messages.last, last.isStreaming {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private var inputBar: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(alignment: .bottom, spacing: 10) {
                TextField("Message", text: $viewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .focused($isInputFocused)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(.tertiarySystemBackground))
                    .clipShape(.rect(cornerRadius: 20))
                    .onSubmit {
                        if !viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            viewModel.sendMessage()
                        }
                    }

                if viewModel.isGenerating || viewModel.isExecutingTools {
                    VStack(spacing: 4) {
                        if let length = viewModel.expectedResponseLength {
                            Text(length.rawValue.capitalized)
                                .font(.system(size: 9).weight(.semibold))
                                .foregroundStyle(.secondary)
                                .transition(.opacity)
                        }
                        Button {
                            viewModel.stopGeneration()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.red)
                                .symbolEffect(.pulse, options: .repeating)
                        }
                    }
                    .sensoryFeedback(.impact(weight: .medium), trigger: viewModel.isGenerating)
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
                            viewModel.sendIntent()
                        } label: {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.title2)
                                .foregroundStyle(
                                    viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                        ? Color(.tertiaryLabel)
                                        : .blue
                                )
                        }
                        .disabled(viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .sensoryFeedback(.impact(weight: .light), trigger: viewModel.messages.count)
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
                viewModel.dismissError()
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
        .animation(.spring(duration: 0.3), value: viewModel.lastError?.id)
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
        .animation(.spring(duration: 0.3), value: viewModel.isGenerating)
    }

    private func copyConversation() {
        let text = viewModel.messages.filter { !$0.isToolExecution }.map { msg in
            let role = msg.role == .user ? "You" : "Nexus"
            return "\(role): \(msg.content)"
        }.joined(separator: "\n\n")
        UIPasteboard.general.string = text
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
