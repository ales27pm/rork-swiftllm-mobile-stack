import SwiftUI

struct ChatView: View {
    @Bindable var viewModel: ChatViewModel
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if viewModel.messages.isEmpty {
                    emptyState
                } else {
                    messageList
                }

                inputBar
            }
            .background(Color(.systemBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    headerTitle
                }
                ToolbarItem(placement: .topBarLeading) {
                    if viewModel.isGenerating {
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
            Text("Nexus")
                .font(.subheadline.bold())

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

            Text("\(viewModel.metricsLogger.currentMetrics.decodeTokensPerSecond, specifier: "%.1f") t/s")
                .font(.caption2.monospacedDigit().bold())
                .foregroundStyle(.blue)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.blue.opacity(0.1))
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

                Text("On-device intelligence with memory\nPrivate · Fast · Contextual")
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
                promptSuggestion("What can you remember about me?", icon: "brain")
                promptSuggestion("Explain how on-device inference works", icon: "cpu")
                promptSuggestion("Help me brainstorm project ideas", icon: "lightbulb.fill")
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

                if viewModel.isGenerating {
                    Button {
                        viewModel.stopGeneration()
                    } label: {
                        Image(systemName: "stop.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.red)
                            .symbolEffect(.pulse, options: .repeating)
                    }
                    .sensoryFeedback(.impact(weight: .medium), trigger: viewModel.isGenerating)
                } else {
                    Button {
                        viewModel.sendMessage()
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
                    .sensoryFeedback(.impact(weight: .light), trigger: viewModel.messages.count)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(.secondarySystemBackground))
        }
    }

    private func copyConversation() {
        let text = viewModel.messages.map { msg in
            let role = msg.role == .user ? "You" : "Nexus"
            return "\(role): \(msg.content)"
        }.joined(separator: "\n\n")
        UIPasteboard.general.string = text
    }
}
