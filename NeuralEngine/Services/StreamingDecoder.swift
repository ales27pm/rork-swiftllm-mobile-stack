import Foundation

final class StreamingDecoder: @unchecked Sendable {
    private var pendingTokens: [Int] = []
    private let lock = NSLock()

    func append(_ token: Int, tokenizer: TokenizerService) -> String? {
        lock.lock()
        defer { lock.unlock() }

        pendingTokens.append(token)

        let text = tokenizer.decode(pendingTokens)

        if text.contains(" ") || text.contains("\n") || text.hasSuffix(".") || text.hasSuffix(",") || text.hasSuffix("!") || text.hasSuffix("?") || text.hasSuffix(":") || text.hasSuffix(";") {
            pendingTokens.removeAll(keepingCapacity: true)
            return text
        }

        if pendingTokens.count > 6 {
            pendingTokens.removeAll(keepingCapacity: true)
            return text
        }

        return nil
    }

    func flush(tokenizer: TokenizerService) -> String {
        lock.lock()
        defer { lock.unlock() }

        let text = tokenizer.decode(pendingTokens)
        pendingTokens.removeAll(keepingCapacity: true)
        return text
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        pendingTokens.removeAll(keepingCapacity: true)
    }
}
