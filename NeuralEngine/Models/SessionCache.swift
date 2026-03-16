import Foundation

struct SessionCache: Sendable {
    var currentToken: Int = 0
    var allTokens: [Int] = []
    var activeLength: Int = 0
    var targetPages: [KVPage] = []
    var draftPages: [KVPage] = []
    var prefixKey: PromptPrefixKey?
    var sequencePosition: Int = 0
    var sequenceID: UUID?
    var snapshotID: UUID?
    var prefillComplete: Bool = false

    mutating func accept(tokens: [Int]) {
        allTokens.append(contentsOf: tokens)
        activeLength += tokens.count
        sequencePosition += tokens.count
        if let last = tokens.last {
            currentToken = last
        }
    }

    mutating func rollbackTokens(_ count: Int) {
        guard count > 0, count <= allTokens.count else { return }
        allTokens.removeLast(count)
        activeLength = max(0, activeLength - count)
        sequencePosition = max(0, sequencePosition - count)
        currentToken = allTokens.last ?? 0
    }

    mutating func applySlidingWindow(keepPrefix: Int, keepSuffix: Int) {
        let total = allTokens.count
        guard total > keepPrefix + keepSuffix else { return }

        let prefix = Array(allTokens.prefix(keepPrefix))
        let suffix = Array(allTokens.suffix(keepSuffix))
        allTokens = prefix + suffix
        activeLength = allTokens.count
    }

    mutating func reset() {
        currentToken = 0
        allTokens.removeAll(keepingCapacity: true)
        activeLength = 0
        targetPages.removeAll(keepingCapacity: true)
        draftPages.removeAll(keepingCapacity: true)
        prefixKey = nil
        sequencePosition = 0
        sequenceID = nil
        snapshotID = nil
        prefillComplete = false
    }

    var pageCount: Int { targetPages.count }

    var totalMemoryEstimateBytes: Int64 {
        Int64(targetPages.reduce(0) { $0 + $1.memoryEstimateBytes })
            + Int64(draftPages.reduce(0) { $0 + $1.memoryEstimateBytes })
    }

    var sharedPageCount: Int {
        targetPages.filter(\.isShared).count
    }

    var hasActiveSequence: Bool {
        sequenceID != nil
    }
}
