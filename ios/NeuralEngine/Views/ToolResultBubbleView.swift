import SwiftUI

struct ToolResultBubbleView: View {
    let message: Message

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.orange, .yellow],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 28, height: 28)

                Image(systemName: "wrench.and.screwdriver.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .padding(.top, 2)

            VStack(alignment: .leading, spacing: 6) {
                if message.toolResults.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Executing tools…")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(.tertiarySystemBackground))
                    .clipShape(.rect(cornerRadius: 14, style: .continuous))
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(message.toolResults.enumerated()), id: \.offset) { _, result in
                            toolResultCard(result)
                        }
                    }
                }
            }

            Spacer(minLength: 48)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }

    private func toolResultCard(_ result: ToolResult) -> some View {
        HStack(spacing: 10) {
            Image(systemName: result.displayIcon)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(result.success ? .orange : .red)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(formatToolName(result.toolName))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)

                Text(formatToolData(result.data))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }

            Spacer(minLength: 0)

            Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.caption)
                .foregroundStyle(result.success ? .green : .red)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.tertiarySystemBackground))
        .clipShape(.rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(result.success ? Color.orange.opacity(0.2) : Color.red.opacity(0.2), lineWidth: 1)
        )
    }

    private func formatToolName(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ").capitalized
    }

    private func formatToolData(_ data: String) -> String {
        guard let jsonData = data.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            return data
        }
        return obj.map { "\($0.key): \($0.value)" }.joined(separator: " · ")
    }
}
