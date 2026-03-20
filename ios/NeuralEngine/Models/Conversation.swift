import Foundation

nonisolated struct Conversation: Identifiable, Codable, Sendable {
    let id: String
    var title: String
    var lastMessage: String
    var messageCount: Int
    var createdAt: Date
    var updatedAt: Date
    var modelId: String?

    init(
        id: String = Self.generateId(),
        title: String = "New Chat",
        lastMessage: String = "",
        messageCount: Int = 0,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        modelId: String? = nil
    ) {
        self.id = id
        self.title = title
        self.lastMessage = lastMessage
        self.messageCount = messageCount
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.modelId = modelId
    }

    static func generateId() -> String {
        let ts = String(Int(Date().timeIntervalSince1970), radix: 36)
        let rand = String(Int.random(in: 0..<1_000_000), radix: 36)
        return ts + rand
    }
}

nonisolated struct ConversationSearchResult: Identifiable, Sendable {
    let id: String = Conversation.generateId()
    let conversationId: String
    let conversationTitle: String
    let matchedContent: String
    let role: String
    let isFirstMatchInConversation: Bool
}
