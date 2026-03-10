import Foundation

nonisolated struct PromptPrefixKey: Hashable, Sendable {
    let modelID: String
    let tokenizerID: String
    let prefixHash: String

    init(modelID: String, tokenizerID: String, prefix: [Int]) {
        self.modelID = modelID
        self.tokenizerID = tokenizerID
        var hasher = Hasher()
        for token in prefix {
            hasher.combine(token)
        }
        self.prefixHash = String(hasher.finalize())
    }
}
