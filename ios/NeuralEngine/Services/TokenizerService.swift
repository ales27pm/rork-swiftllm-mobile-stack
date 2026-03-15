import Foundation
import Tokenizers

nonisolated struct TokenizerSchemaInfo: Sendable {
    let version: String?
    let modelType: String?
    let tokenizerClass: String?
    let isCompatible: Bool
    let diagnosticCode: TokenizerDiagnostic
}

nonisolated enum TokenizerDiagnostic: String, Sendable {
    case valid = "TOKENIZER_VALID"
    case missingConfig = "TOKENIZER_MISSING_CONFIG"
    case schemaMismatch = "TOKENIZER_SCHEMA_MISMATCH"
    case corruptedEncoding = "TOKENIZER_CORRUPTED_ENCODING"
    case unknownFormat = "TOKENIZER_UNKNOWN_FORMAT"
}

final class TokenizerService: @unchecked Sendable {
    private let lock = NSLock()
    private var tokenizer: Tokenizer?
    private var fallbackVocabulary: [Int: String] = [:]
    private var fallbackReverse: [String: Int] = [:]
    private var _vocabularySize: Int = 32000
    private var isRealTokenizer: Bool = false
    private var schemaInfo: TokenizerSchemaInfo?

    static let bosToken = 1
    static let eosToken = 2
    static let padToken = 0
    static let unknownToken = 3
    static let specialTokens: Set<Int> = [0, 1, 2, 3]

    static let supportedTokenizerClasses: Set<String> = [
        "PreTrainedTokenizerFast",
        "LlamaTokenizer",
        "LlamaTokenizerFast",
        "GPT2Tokenizer",
        "GPT2TokenizerFast",
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
        "GemmaTokenizer",
        "GemmaTokenizerFast"
    ]

    private var eosTokenIDs: Set<Int> = [2]

    var effectiveEOSTokens: Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        return eosTokenIDs
    }

    init() {
        buildFallbackVocabulary()
    }

    func loadFromHub(repoID: String) async throws {
        let loaded = try await AutoTokenizer.from(pretrained: repoID)
        lock.lock()
        tokenizer = loaded
        isRealTokenizer = true
        detectEOSTokens(loaded)
        lock.unlock()
    }

    func loadFromDirectory(_ url: URL) async throws {
        let loaded = try await AutoTokenizer.from(modelFolder: url)
        lock.lock()
        tokenizer = loaded
        isRealTokenizer = true
        detectEOSTokens(loaded)
        lock.unlock()
    }

    private func detectEOSTokens(_ tok: Tokenizer) {
        var eos: Set<Int> = []
        let candidates = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
        for candidate in candidates {
            let encoded = tok(candidate)
            if encoded.count == 1 {
                eos.insert(encoded[0])
            }
        }
        if let eosID = tok.eosTokenId {
            eos.insert(eosID)
        }
        if eos.isEmpty {
            eos.insert(Self.eosToken)
        }
        eosTokenIDs = eos
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

    var currentSchemaInfo: TokenizerSchemaInfo? {
        lock.lock()
        defer { lock.unlock() }
        return schemaInfo
    }

    func validateSchema(in directory: URL) -> TokenizerSchemaInfo {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")

        var resolvedConfigURL = configURL
        if !FileManager.default.fileExists(atPath: configURL.path) {
            if let nested = findFile(named: "tokenizer_config.json", in: directory) {
                resolvedConfigURL = nested
            } else {
                return TokenizerSchemaInfo(
                    version: nil,
                    modelType: nil,
                    tokenizerClass: nil,
                    isCompatible: false,
                    diagnosticCode: .missingConfig
                )
            }
        }

        guard let configData = try? Data(contentsOf: resolvedConfigURL),
              let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            return TokenizerSchemaInfo(
                version: nil,
                modelType: nil,
                tokenizerClass: nil,
                isCompatible: false,
                diagnosticCode: .corruptedEncoding
            )
        }

        let version = config["version"] as? String
            ?? config["tokenizer_version"] as? String
            ?? config["transformers_version"] as? String
        let modelType = config["model_type"] as? String
        let tokenizerClass = config["tokenizer_class"] as? String

        var resolvedTokenizerURL = tokenizerURL
        if !FileManager.default.fileExists(atPath: tokenizerURL.path) {
            if let nested = findFile(named: "tokenizer.json", in: directory) {
                resolvedTokenizerURL = nested
            }
        }

        if FileManager.default.fileExists(atPath: resolvedTokenizerURL.path) {
            guard let tokData = try? Data(contentsOf: resolvedTokenizerURL),
                  let tokJSON = try? JSONSerialization.jsonObject(with: tokData) as? [String: Any] else {
                return TokenizerSchemaInfo(
                    version: version,
                    modelType: modelType,
                    tokenizerClass: tokenizerClass,
                    isCompatible: false,
                    diagnosticCode: .corruptedEncoding
                )
            }

            let hasModel = tokJSON["model"] != nil
            let hasAddedTokens = tokJSON["added_tokens"] != nil
            if !hasModel && !hasAddedTokens {
                return TokenizerSchemaInfo(
                    version: version,
                    modelType: modelType,
                    tokenizerClass: tokenizerClass,
                    isCompatible: false,
                    diagnosticCode: .corruptedEncoding
                )
            }
        }

        if let tokClass = tokenizerClass, !Self.supportedTokenizerClasses.contains(tokClass) {
            return TokenizerSchemaInfo(
                version: version,
                modelType: modelType,
                tokenizerClass: tokenizerClass,
                isCompatible: false,
                diagnosticCode: .schemaMismatch
            )
        }

        let info = TokenizerSchemaInfo(
            version: version,
            modelType: modelType,
            tokenizerClass: tokenizerClass,
            isCompatible: true,
            diagnosticCode: .valid
        )

        lock.lock()
        schemaInfo = info
        lock.unlock()

        return info
    }

    private func findFile(named name: String, in directory: URL) -> URL? {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        while let url = enumerator.nextObject() as? URL {
            if url.lastPathComponent == name {
                return url
            }
        }
        return nil
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
