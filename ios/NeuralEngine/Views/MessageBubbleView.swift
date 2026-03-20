import SwiftUI

struct MessageBubbleView: View {
    let message: Message
    var onReaction: ((MessageReaction) -> Void)?

    @State private var showReactions: Bool = false

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
                    HStack(spacing: 6) {
                        metricsTag(metrics)

                        if message.role == .assistant, let reaction = message.reaction {
                            reactionBadge(reaction)
                        }
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                if message.role == .assistant && message.metrics != nil && message.reaction == nil {
                    reactionBar
                        .transition(.opacity)
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
        VStack(alignment: .leading, spacing: 0) {
            let blocks = MarkdownBlockParser.parse(message.content)
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .text(let content):
                    Text(renderInlineMarkdown(content))
                        .font(.body)
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .textSelection(.enabled)

                case .heading(let level, let content):
                    headingView(level: level, content: content)
                        .padding(.top, level == 1 ? 8 : 4)
                        .padding(.bottom, 2)

                case .bulletList(let items):
                    listView(items: items, ordered: false)

                case .numberedList(let items):
                    listView(items: items, ordered: true)

                case .blockquote(let content):
                    blockquoteView(content)

                case .horizontalRule:
                    Divider()
                        .padding(.vertical, 6)

                case .codeBlock(let language, let code):
                    codeBlockView(language: language, code: code)

                case .link(let title, let url):
                    if URLSafetyChecker.isSafe(url) {
                        Link(destination: url) {
                            HStack(spacing: 6) {
                                Image(systemName: "link")
                                    .font(.caption)
                                Text(title.isEmpty ? url.host ?? url.absoluteString : title)
                                    .font(.subheadline)
                                    .lineLimit(1)
                            }
                            .foregroundStyle(message.role == .user ? .white : .blue)
                        }
                    } else {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.shield")
                                .font(.caption)
                            Text("Blocked: unsafe URL")
                                .font(.caption)
                        }
                        .foregroundStyle(.secondary)
                    }
                }
            }

            if message.isStreaming && !message.content.isEmpty {
                StreamingCursorView()
            }
        }
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

    private func headingView(level: Int, content: String) -> some View {
        let font: Font = switch level {
        case 1: .title3.bold()
        case 2: .headline
        case 3: .subheadline.bold()
        default: .subheadline.weight(.semibold)
        }
        return Text(renderInlineMarkdown(content))
            .font(font)
            .foregroundStyle(message.role == .user ? .white : .primary)
            .textSelection(.enabled)
    }

    private func listView(items: [String], ordered: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    if ordered {
                        Text("\(index + 1).")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(message.role == .user ? .white.opacity(0.7) : .secondary)
                            .frame(width: 22, alignment: .trailing)
                    } else {
                        Text("•")
                            .font(.callout.bold())
                            .foregroundStyle(message.role == .user ? .white.opacity(0.7) : .blue)
                            .frame(width: 16, alignment: .center)
                    }
                    Text(renderInlineMarkdown(item))
                        .font(.callout)
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .textSelection(.enabled)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func blockquoteView(_ content: String) -> some View {
        HStack(spacing: 0) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(message.role == .user ? Color.white.opacity(0.4) : Color.blue.opacity(0.5))
                .frame(width: 3)

            Text(renderInlineMarkdown(content))
                .font(.callout.italic())
                .foregroundStyle(message.role == .user ? .white.opacity(0.85) : .secondary)
                .textSelection(.enabled)
                .padding(.leading, 10)
        }
        .padding(.vertical, 4)
    }

    private var reactionBar: some View {
        HStack(spacing: 12) {
            ForEach(MessageReaction.allCases, id: \.self) { reaction in
                Button {
                    onReaction?(reaction)
                } label: {
                    Image(systemName: reaction.icon)
                        .font(.system(size: 12))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.top, 2)
    }

    private func reactionBadge(_ reaction: MessageReaction) -> some View {
        HStack(spacing: 3) {
            Image(systemName: reaction.icon)
            Text(reaction.label)
        }
        .font(.caption2)
        .foregroundStyle(reaction.color)
    }

    private var bubbleBackground: AnyShapeStyle {
        if message.role == .user {
            AnyShapeStyle(.blue.gradient)
        } else {
            AnyShapeStyle(Color(.secondarySystemBackground))
        }
    }

    private func renderInlineMarkdown(_ text: String) -> AttributedString {
        if let parsed = try? AttributedString(
            markdown: text,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            return parsed
        }
        return AttributedString(text)
    }

    private func codeBlockView(language: String, code: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if !language.isEmpty {
                HStack {
                    Text(language)
                        .font(.caption2.bold())
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        UIPasteboard.general.string = code
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color(.quaternarySystemFill))
            }

            Text(code)
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(message.role == .user ? .white.opacity(0.9) : .primary)
                .textSelection(.enabled)
                .padding(10)
        }
        .background(message.role == .user ? Color.white.opacity(0.1) : Color(.tertiarySystemBackground))
        .clipShape(.rect(cornerRadius: 10))
        .padding(.vertical, 4)
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

struct StreamingCursorView: View {
    @State private var visible: Bool = true

    var body: some View {
        Rectangle()
            .fill(Color.blue)
            .frame(width: 2, height: 16)
            .opacity(visible ? 1 : 0)
            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: visible)
            .onAppear { visible = false }
    }
}

enum MessageReaction: String, CaseIterable, Sendable {
    case good
    case bad
    case insightful

    var icon: String {
        switch self {
        case .good: return "hand.thumbsup.fill"
        case .bad: return "hand.thumbsdown.fill"
        case .insightful: return "lightbulb.fill"
        }
    }

    var label: String {
        switch self {
        case .good: return "Good"
        case .bad: return "Poor"
        case .insightful: return "Insightful"
        }
    }

    var color: Color {
        switch self {
        case .good: return .green
        case .bad: return .red
        case .insightful: return .orange
        }
    }
}

enum MarkdownBlock {
    case text(String)
    case heading(level: Int, content: String)
    case bulletList(items: [String])
    case numberedList(items: [String])
    case blockquote(String)
    case horizontalRule
    case codeBlock(language: String, code: String)
    case link(title: String, url: URL)
}

enum MarkdownBlockParser {
    static func parse(_ content: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var currentText = ""
        let lines = content.components(separatedBy: "\n")
        var i = 0
        var inCodeBlock = false
        var codeLanguage = ""
        var codeContent = ""

        while i < lines.count {
            let line = lines[i]

            if line.hasPrefix("```") && !inCodeBlock {
                flushText(&currentText, into: &blocks)
                inCodeBlock = true
                codeLanguage = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                codeContent = ""
                i += 1
                continue
            }

            if line.hasPrefix("```") && inCodeBlock {
                inCodeBlock = false
                blocks.append(.codeBlock(language: codeLanguage, code: codeContent.trimmingCharacters(in: .newlines)))
                i += 1
                continue
            }

            if inCodeBlock {
                if !codeContent.isEmpty { codeContent += "\n" }
                codeContent += line
                i += 1
                continue
            }

            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if let headingMatch = parseHeading(trimmed) {
                flushText(&currentText, into: &blocks)
                blocks.append(.heading(level: headingMatch.0, content: headingMatch.1))
                i += 1
                continue
            }

            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                flushText(&currentText, into: &blocks)
                blocks.append(.horizontalRule)
                i += 1
                continue
            }

            if trimmed.hasPrefix("> ") {
                flushText(&currentText, into: &blocks)
                var quoteLines: [String] = []
                while i < lines.count {
                    let ql = lines[i].trimmingCharacters(in: .whitespaces)
                    if ql.hasPrefix("> ") {
                        quoteLines.append(String(ql.dropFirst(2)))
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.blockquote(quoteLines.joined(separator: "\n")))
                continue
            }

            if isBulletItem(trimmed) {
                flushText(&currentText, into: &blocks)
                var items: [String] = []
                while i < lines.count {
                    let bl = lines[i].trimmingCharacters(in: .whitespaces)
                    if isBulletItem(bl) {
                        items.append(stripBullet(bl))
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.bulletList(items: items))
                continue
            }

            if isNumberedItem(trimmed) {
                flushText(&currentText, into: &blocks)
                var items: [String] = []
                while i < lines.count {
                    let nl = lines[i].trimmingCharacters(in: .whitespaces)
                    if isNumberedItem(nl) {
                        items.append(stripNumber(nl))
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.numberedList(items: items))
                continue
            }

            if !currentText.isEmpty { currentText += "\n" }
            currentText += line
            i += 1
        }

        if inCodeBlock && !codeContent.isEmpty {
            blocks.append(.codeBlock(language: codeLanguage, code: codeContent.trimmingCharacters(in: .newlines)))
        }

        flushText(&currentText, into: &blocks)

        return blocks.isEmpty ? [.text(content)] : blocks
    }

    private static func flushText(_ text: inout String, into blocks: inout [MarkdownBlock]) {
        guard !text.isEmpty else { return }
        let extracted = extractLinks(from: text)
        blocks.append(contentsOf: extracted)
        text = ""
    }

    private static func parseHeading(_ line: String) -> (Int, String)? {
        if line.hasPrefix("### ") { return (3, String(line.dropFirst(4))) }
        if line.hasPrefix("## ") { return (2, String(line.dropFirst(3))) }
        if line.hasPrefix("# ") { return (1, String(line.dropFirst(2))) }
        return nil
    }

    private static func isBulletItem(_ line: String) -> Bool {
        line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("+ ")
    }

    private static func stripBullet(_ line: String) -> String {
        String(line.dropFirst(2))
    }

    private static func isNumberedItem(_ line: String) -> Bool {
        guard let dotIdx = line.firstIndex(of: ".") else { return false }
        let prefix = line[line.startIndex..<dotIdx]
        guard !prefix.isEmpty, prefix.allSatisfy(\.isNumber) else { return false }
        let afterDot = line.index(after: dotIdx)
        return afterDot < line.endIndex && line[afterDot] == " "
    }

    private static func stripNumber(_ line: String) -> String {
        guard let dotIdx = line.firstIndex(of: ".") else { return line }
        let afterDot = line.index(after: dotIdx)
        guard afterDot < line.endIndex else { return line }
        return String(line[line.index(after: afterDot)...])
    }

    private static func extractLinks(from text: String) -> [MarkdownBlock] {
        let pattern = #"\[([^\]]*)\]\((https?://[^\)]+)\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return [.text(text)]
        }

        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))

        guard !matches.isEmpty else {
            return [.text(text)]
        }

        var blocks: [MarkdownBlock] = []
        var lastEnd = 0

        for match in matches {
            let fullRange = match.range
            if fullRange.location > lastEnd {
                let before = nsText.substring(with: NSRange(location: lastEnd, length: fullRange.location - lastEnd))
                if !before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    blocks.append(.text(before))
                }
            }

            let title = nsText.substring(with: match.range(at: 1))
            let urlString = nsText.substring(with: match.range(at: 2))
            if let url = URL(string: urlString) {
                blocks.append(.link(title: title, url: url))
            } else {
                blocks.append(.text("[\(title)](\(urlString))"))
            }

            lastEnd = fullRange.location + fullRange.length
        }

        if lastEnd < nsText.length {
            let remaining = nsText.substring(from: lastEnd)
            if !remaining.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                blocks.append(.text(remaining))
            }
        }

        return blocks
    }
}

enum URLSafetyChecker {
    private static let blockedSchemes: Set<String> = ["javascript", "data", "ftp", "file"]
    private static let blockedPatterns: [String] = [
        "bit.ly", "tinyurl.com", "goo.gl",
        "phishing", "malware", "exploit"
    ]

    static func isSafe(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else { return false }

        if blockedSchemes.contains(scheme) { return false }

        guard scheme == "https" || scheme == "http" else { return false }

        guard let host = url.host?.lowercased() else { return false }

        if host.isEmpty { return false }

        for pattern in blockedPatterns {
            if host.contains(pattern) { return false }
        }

        if host.filter({ $0 == "." }).count > 5 { return false }

        return true
    }
}
