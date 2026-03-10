import Foundation
import Tokenizers

final class TokenizerService: @unchecked Sendable {
    private let lock = NSLock()
    private var tokenizer: Tokenizer?
    private var fallbackVocabulary: [Int: String] = [:]
    private var fallbackReverse: [String: Int] = [:]
    private var _vocabularySize: Int = 32000
    private var isRealTokenizer: Bool = false

    static let bosToken = 1
    static let eosToken = 2
    static let padToken = 0
    static let unknownToken = 3
    static let specialTokens: Set<Int> = [0, 1, 2, 3]

    init() {
        buildFallbackVocabulary()
    }

    func loadFromHub(repoID: String) async throws {
        let loaded = try await AutoTokenizer.from(pretrained: repoID)
        lock.lock()
        tokenizer = loaded
        isRealTokenizer = true
        lock.unlock()
    }

    func loadFromDirectory(_ url: URL) async throws {
        let loaded = try await AutoTokenizer.from(modelFolder: url)
        lock.lock()
        tokenizer = loaded
        isRealTokenizer = true
        lock.unlock()
    }

    func unloadTokenizer() {
        lock.lock()
        tokenizer = nil
        isRealTokenizer = false
        lock.unlock()
    }

    var hasRealTokenizer: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isRealTokenizer
    }

    func encode(_ text: String) -> [Int] {
        lock.lock()
        defer { lock.unlock() }

        if let tokenizer {
            let encoded = tokenizer(text)
            return encoded
        }

        return fallbackEncode(text)
    }

    func decode(_ tokenIDs: [Int]) -> String {
        lock.lock()
        defer { lock.unlock() }

        if let tokenizer {
            return tokenizer.decode(tokens: tokenIDs)
        }

        return fallbackDecode(tokenIDs)
    }

    func decodeIncremental(_ tokenID: Int) -> String? {
        lock.lock()
        defer { lock.unlock() }

        if let tokenizer {
            return tokenizer.decode(tokens: [tokenID])
        }

        guard !Self.specialTokens.contains(tokenID) else { return nil }
        guard let piece = fallbackVocabulary[tokenID] else { return nil }
        return piece.replacingOccurrences(of: "▁", with: " ")
    }

    var vocabularySize: Int {
        lock.lock()
        defer { lock.unlock() }
        if isRealTokenizer {
            return _vocabularySize
        }
        return fallbackVocabulary.count
    }

    func applyTemplate(messages: [[String: String]]) -> String? {
        lock.lock()
        defer { lock.unlock() }

        guard let tokenizer else { return nil }

        if let encoded = try? tokenizer.applyChatTemplate(messages: messages) {
            return tokenizer.decode(tokens: encoded)
        }

        return nil
    }

    private func fallbackEncode(_ text: String) -> [Int] {
        var tokens: [Int] = [Self.bosToken]
        let words = text.components(separatedBy: .whitespacesAndNewlines)
        for word in words where !word.isEmpty {
            let key = "▁" + word.lowercased()
            if let id = fallbackReverse[key] {
                tokens.append(id)
            } else {
                for char in word {
                    let charStr = String(char)
                    if let id = fallbackReverse[charStr] {
                        tokens.append(id)
                    } else {
                        tokens.append(Self.unknownToken)
                    }
                }
            }
        }
        return tokens
    }

    private func fallbackDecode(_ tokenIDs: [Int]) -> String {
        var result = ""
        for id in tokenIDs {
            guard !Self.specialTokens.contains(id) else { continue }
            if let piece = fallbackVocabulary[id] {
                result += piece
            }
        }
        return result.replacingOccurrences(of: "▁", with: " ")
    }

    private func buildFallbackVocabulary() {
        var id = 4
        let special = ["<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3]
        for (token, tokenID) in special {
            fallbackVocabulary[tokenID] = token
            fallbackReverse[token] = tokenID
        }

        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" {
            fallbackVocabulary[id] = String(c)
            fallbackReverse[String(c)] = id
            id += 1
        }

        let common = ["▁the", "▁a", "▁is", "▁and", "▁to", "▁of", "▁in", "▁for", "▁that", "▁it",
                       "▁this", "▁with", "▁on", "▁not", "▁are", "▁be", "▁was", "▁have", "▁has",
                       "▁model", "▁token", "▁cache", "▁inference", "▁device", "▁memory"]
        for word in common {
            fallbackVocabulary[id] = word
            fallbackReverse[word] = id
            id += 1
        }

        _vocabularySize = id
    }
}
