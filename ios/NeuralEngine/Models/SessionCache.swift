import Foundation

struct SessionCache: Sendable {
    var currentToken: Int = 0
    var allTokens: [Int] = []
    var activeLength: Int = 0
    var targetPages: [KVPage] = []
    var draftPages: [KVPage] = []
    var prefixKey: PromptPrefixKey?
    var sequencePosition: Int = 0

    mutating func accept(tokens: [Int]) {
        allTokens.append(contentsOf: tokens)
        activeLength += tokens.count
        sequencePosition += tokens.count
        if let last = tokens.last {
            currentToken = last
        }
    }

    mutating func reset() {
        currentToken = 0
        allTokens.removeAll(keepingCapacity: true)
        activeLength = 0
        targetPages.removeAll(keepingCapacity: true)
        draftPages.removeAll(keepingCapacity: true)
        prefixKey = nil
        sequencePosition = 0
    }
}
