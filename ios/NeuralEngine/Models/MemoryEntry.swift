import Foundation

nonisolated enum MemoryCategory: String, Codable, Sendable, CaseIterable {
    case preference
    case fact
    case context
    case instruction
    case emotion
    case skill
}

nonisolated enum MemorySource: String, Codable, Sendable {
    case conversation
    case manual
    case system
}

nonisolated struct MemoryEntry: Identifiable, Codable, Sendable {
    let id: String
    var content: String
    var keywords: [String]
    var category: MemoryCategory
    var timestamp: Double
    var importance: Int
    var source: MemorySource
    var accessCount: Int
    var lastAccessed: Double
    var relations: [String]
    var consolidated: Bool
    var decay: Double
    var activationLevel: Double
    var emotionalValence: Double

    init(
        id: String = Conversation.generateId(),
        content: String,
        keywords: [String] = [],
        category: MemoryCategory = .context,
        timestamp: Double = Date().timeIntervalSince1970 * 1000,
        importance: Int = 3,
        source: MemorySource = .conversation,
        accessCount: Int = 0,
        lastAccessed: Double = Date().timeIntervalSince1970 * 1000,
        relations: [String] = [],
        consolidated: Bool = false,
        decay: Double = 1.0,
        activationLevel: Double = 0,
        emotionalValence: Double = 0
    ) {
        self.id = id
        self.content = content
        self.keywords = keywords
        self.category = category
        self.timestamp = timestamp
        self.importance = importance
        self.source = source
        self.accessCount = accessCount
        self.lastAccessed = lastAccessed
        self.relations = relations
        self.consolidated = consolidated
        self.decay = decay
        self.activationLevel = activationLevel
        self.emotionalValence = emotionalValence
    }
}

nonisolated struct AssociativeLink: Codable, Sendable {
    let sourceId: String
    let targetId: String
    var strength: Double
    var type: LinkType
    var createdAt: Double
    var reinforcements: Int

    nonisolated enum LinkType: String, Codable, Sendable {
        case semantic
        case temporal
        case topical
    }
}

nonisolated struct RetrievalResult: Sendable {
    let memory: MemoryEntry
    let score: Double
    let matchType: MatchType

    nonisolated enum MatchType: String, Sendable {
        case semantic
        case keyword
        case vector
        case primed
        case associative
    }
}
