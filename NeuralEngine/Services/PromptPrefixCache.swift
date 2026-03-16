import Foundation

nonisolated struct CachedPrefix: Sendable {
    let key: PromptPrefixKey
    let tokenizedPrefix: [Int]
    let pageCount: Int
    let sequencePosition: Int
    let timestamp: Date
}

actor PromptPrefixCache {
    private var cache: [PromptPrefixKey: CachedPrefix] = [:]
    private let maxEntries: Int

    init(maxEntries: Int = 8) {
        self.maxEntries = maxEntries
    }

    func lookup(key: PromptPrefixKey) -> CachedPrefix? {
        cache[key]
    }

    func store(prefix: CachedPrefix) {
        if cache.count >= maxEntries {
            evictOldest()
        }
        cache[prefix.key] = prefix
    }

    func invalidate(key: PromptPrefixKey) {
        cache.removeValue(forKey: key)
    }

    func clear() {
        cache.removeAll()
    }

    var entryCount: Int { cache.count }

    private func evictOldest() {
        guard let oldest = cache.min(by: { $0.value.timestamp < $1.value.timestamp }) else { return }
        cache.removeValue(forKey: oldest.key)
    }
}
