import SwiftUI

struct MessageBubbleView: View {
    let message: Message

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .user {
                Spacer(minLength: 48)
            }

            if message.role == .assistant {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [.purple, .blue],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 28, height: 28)

                    Image(systemName: "brain")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .padding(.top, 2)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                if message.isStreaming && message.content.isEmpty {
                    typingIndicator
                } else {
                    textBubble
                }

                if let metrics = message.metrics {
                    metricsTag(metrics)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }

            if message.role == .assistant {
                Spacer(minLength: 48)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }

    private var textBubble: some View {
        Text(formattedContent)
            .font(.body)
            .foregroundStyle(message.role == .user ? .white : .primary)
            .textSelection(.enabled)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(bubbleBackground)
            .clipShape(.rect(cornerRadius: 18, style: .continuous))
            .overlay {
                if message.isStreaming {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [.blue.opacity(0.4), .purple.opacity(0.4), .blue.opacity(0.4)],
                                startPoint: .leading,
                                endPoint: .trailing
                            ),
                            lineWidth: 1.5
                        )
                }
            }
    }

    private var bubbleBackground: AnyShapeStyle {
        if message.role == .user {
            AnyShapeStyle(.blue.gradient)
        } else {
            AnyShapeStyle(Color(.secondarySystemBackground))
        }
    }

    private var formattedContent: AttributedString {
        var result = AttributedString(message.content)

        if let codeRanges = try? AttributedString(
            markdown: message.content,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            result = codeRanges
        }

        return result
    }

    private var typingIndicator: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color(.tertiaryLabel))
                    .frame(width: 7, height: 7)
                    .scaleEffect(1.0)
                    .animation(
                        .easeInOut(duration: 0.5)
                            .repeatForever(autoreverses: true)
                            .delay(Double(i) * 0.15),
                        value: message.isStreaming
                    )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(.secondarySystemBackground))
        .clipShape(.rect(cornerRadius: 18, style: .continuous))
    }

    private func metricsTag(_ metrics: GenerationMetrics) -> some View {
        HStack(spacing: 8) {
            HStack(spacing: 3) {
                Image(systemName: "gauge.with.dots.needle.67percent")
                Text("\(metrics.decodeTokensPerSecond, specifier: "%.1f") tok/s")
            }

            HStack(spacing: 3) {
                Image(systemName: "number")
                Text("\(metrics.totalTokens)")
            }

            HStack(spacing: 3) {
                Image(systemName: "timer")
                Text("\(metrics.totalDuration, specifier: "%.1f")s")
            }

            if metrics.acceptedSpeculativeTokens > 0 {
                let total = metrics.acceptedSpeculativeTokens + metrics.rejectedSpeculativeTokens
                let rate = Double(metrics.acceptedSpeculativeTokens) / Double(max(total, 1)) * 100
                HStack(spacing: 3) {
                    Image(systemName: "bolt.fill")
                    Text("\(rate, specifier: "%.0f")%")
                }
                .foregroundStyle(.blue)
            }
        }
        .font(.caption2.monospacedDigit())
        .foregroundStyle(.tertiary)
    }
}
