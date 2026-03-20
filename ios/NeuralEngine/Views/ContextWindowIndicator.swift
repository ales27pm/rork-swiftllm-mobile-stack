import SwiftUI

struct ContextWindowIndicator: View {
    let chatViewModel: ChatViewModel
    let inferenceEngine: InferenceEngine?

    private var contextWindowSize: Int { 2048 }

    private var estimatedTokensUsed: Int {
        let systemPromptTokens = chatViewModel.systemPrompt.count / 4
        let messageTokens = chatViewModel.messages.reduce(0) { sum, msg in
            sum + (msg.content.count / 4)
        }
        let cognitiveOverhead = chatViewModel.lastCognitionFrame != nil ? 400 : 0
        return systemPromptTokens + messageTokens + cognitiveOverhead
    }

    private var usage: Double {
        min(1.0, Double(estimatedTokensUsed) / Double(max(contextWindowSize, 1)))
    }

    private var maxTokensSetting: Int {
        chatViewModel.samplingConfig.maxTokens
    }

    private var remainingTokens: Int {
        max(0, maxTokensSetting - estimatedTokensUsed)
    }

    var body: some View {
        VStack(spacing: 14) {
            ringGauge

            VStack(spacing: 10) {
                tokenRow("System Prompt", tokens: chatViewModel.systemPrompt.count / 4, color: .purple)
                tokenRow("Conversation", tokens: chatViewModel.messages.reduce(0) { $0 + ($1.content.count / 4) }, color: .blue)
                if chatViewModel.lastCognitionFrame != nil {
                    tokenRow("Cognitive Context", tokens: 400, color: .orange)
                }
                Divider()
                HStack {
                    Text("Remaining for Response")
                        .font(.caption.weight(.medium))
                    Spacer()
                    Text("~\(remainingTokens) tokens")
                        .font(.caption.bold().monospacedDigit())
                        .foregroundStyle(remainingTokens > 500 ? .green : remainingTokens > 200 ? .orange : .red)
                }
            }

            if usage > 0.8 {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("Context window is \(Int(usage * 100))% full. Consider starting a new conversation for best results.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
                .background(Color.orange.opacity(0.06))
                .clipShape(.rect(cornerRadius: 8))
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
        .padding(.horizontal, 16)
    }

    private var ringGauge: some View {
        ZStack {
            Circle()
                .stroke(Color(.quaternarySystemFill), lineWidth: 10)
                .frame(width: 100, height: 100)

            Circle()
                .trim(from: 0, to: usage)
                .stroke(
                    AngularGradient(
                        colors: [gaugeColor.opacity(0.5), gaugeColor],
                        center: .center
                    ),
                    style: StrokeStyle(lineWidth: 10, lineCap: .round)
                )
                .frame(width: 100, height: 100)
                .rotationEffect(.degrees(-90))
                .animation(.spring(duration: 0.6), value: usage)

            VStack(spacing: 2) {
                Text("\(Int(usage * 100))%")
                    .font(.title3.bold().monospacedDigit())
                Text("used")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var gaugeColor: Color {
        if usage > 0.85 { return .red }
        if usage > 0.65 { return .orange }
        return .blue
    }

    private func tokenRow(_ label: String, tokens: Int, color: Color) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text("~\(tokens)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }
}
