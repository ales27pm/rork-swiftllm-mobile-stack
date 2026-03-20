import Foundation

@MainActor
@Observable
class ConversationService {
    private let database: DatabaseService
    private let keyValueStore: KeyValueStore

    var conversations: [Conversation] = []

    init(database: DatabaseService, keyValueStore: KeyValueStore) {
        self.database = database
        self.keyValueStore = keyValueStore
        createTables()
        loadConversations()
    }

    private func createTables() {
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                last_message TEXT NOT NULL DEFAULT '',
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                model_id TEXT
            );
        """)
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                model_id TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)
        _ = database.execute("CREATE INDEX IF NOT EXISTS idx_conv_msgs ON conversation_messages(conversation_id, timestamp);")
        _ = database.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts USING fts5(
                content, conversation_id UNINDEXED, role UNINDEXED,
                content_rowid='rowid'
            );
        """)
    }

    func loadConversations() {
        let rows = database.query("SELECT * FROM conversations ORDER BY updated_at DESC;")
        conversations = rows.compactMap { row in
            guard let id = row["id"] as? String,
                  let title = row["title"] as? String,
                  let createdAt = row["created_at"] as? Double,
                  let updatedAt = row["updated_at"] as? Double else { return nil }
            return Conversation(
                id: id,
                title: title,
                lastMessage: (row["last_message"] as? String) ?? "",
                messageCount: (row["message_count"] as? Int64).map { Int($0) } ?? 0,
                createdAt: Date(timeIntervalSince1970: createdAt),
                updatedAt: Date(timeIntervalSince1970: updatedAt),
                modelId: row["model_id"] as? String
            )
        }
    }

    func createConversation(modelId: String?) -> Conversation {
        let conv = Conversation(modelId: modelId)
        _ = database.execute(
            "INSERT INTO conversations (id, title, last_message, message_count, created_at, updated_at, model_id) VALUES (?, ?, ?, ?, ?, ?, ?);",
            params: [conv.id, conv.title, conv.lastMessage, conv.messageCount, conv.createdAt.timeIntervalSince1970, conv.updatedAt.timeIntervalSince1970, conv.modelId ?? ""]
        )
        conversations.insert(conv, at: 0)
        return conv
    }

    func updateConversation(_ conv: Conversation) {
        _ = database.execute(
            "UPDATE conversations SET title = ?, last_message = ?, message_count = ?, updated_at = ?, model_id = ? WHERE id = ?;",
            params: [conv.title, conv.lastMessage, conv.messageCount, conv.updatedAt.timeIntervalSince1970, conv.modelId ?? "", conv.id]
        )
        if let idx = conversations.firstIndex(where: { $0.id == conv.id }) {
            conversations[idx] = conv
            let updated = conversations.remove(at: idx)
            conversations.insert(updated, at: 0)
        }
    }

    func deleteConversation(_ id: String) {
        _ = database.execute("DELETE FROM conversation_messages WHERE conversation_id = ?;", params: [id])
        _ = database.execute("DELETE FROM conversations WHERE id = ?;", params: [id])
        conversations.removeAll { $0.id == id }
    }

    func clearAllConversations() {
        _ = database.execute("DELETE FROM conversation_messages;")
        _ = database.execute("DELETE FROM conversations;")
        conversations.removeAll()
    }

    func saveMessage(_ message: Message, conversationId: String) {
        _ = database.execute(
            "INSERT OR REPLACE INTO conversation_messages (id, conversation_id, role, content, timestamp, model_id) VALUES (?, ?, ?, ?, ?, ?);",
            params: [message.id.uuidString, conversationId, message.role.rawValue, message.content, message.timestamp.timeIntervalSince1970, ""]
        )
        _ = database.execute(
            "INSERT OR REPLACE INTO conversation_messages_fts (rowid, content, conversation_id, role) VALUES ((SELECT rowid FROM conversation_messages WHERE id = ?), ?, ?, ?);",
            params: [message.id.uuidString, message.content, conversationId, message.role.rawValue]
        )
    }

    func loadMessages(for conversationId: String) -> [Message] {
        let rows = database.query(
            "SELECT * FROM conversation_messages WHERE conversation_id = ? ORDER BY timestamp ASC;",
            params: [conversationId]
        )
        return rows.compactMap { row in
            guard let idStr = row["id"] as? String,
                  let id = UUID(uuidString: idStr),
                  let roleStr = row["role"] as? String,
                  let role = MessageRole(rawValue: roleStr),
                  let content = row["content"] as? String,
                  let timestamp = row["timestamp"] as? Double else { return nil }
            return Message(id: id, role: role, content: content, timestamp: Date(timeIntervalSince1970: timestamp))
        }
    }

    func searchMessages(query: String) -> [ConversationSearchResult] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let sanitized = query.replacingOccurrences(of: "\"", with: "")
        let ftsQuery = sanitized.split(separator: " ").map { "\($0)*" }.joined(separator: " ")
        let rows = database.query(
            "SELECT conversation_messages_fts.content, conversation_messages_fts.conversation_id, conversation_messages_fts.role FROM conversation_messages_fts WHERE conversation_messages_fts MATCH ? ORDER BY rank LIMIT 30;",
            params: [ftsQuery]
        )

        var results: [ConversationSearchResult] = []
        var seenConversations: Set<String> = []

        for row in rows {
            guard let content = row["content"] as? String,
                  let conversationId = row["conversation_id"] as? String,
                  let role = row["role"] as? String else { continue }

            let isNew = seenConversations.insert(conversationId).inserted
            let conversation = conversations.first { $0.id == conversationId }

            results.append(ConversationSearchResult(
                conversationId: conversationId,
                conversationTitle: conversation?.title ?? "Conversation",
                matchedContent: String(content.prefix(200)),
                role: role,
                isFirstMatchInConversation: isNew
            ))
        }

        return results
    }

    func generateTitle(from firstMessage: String) -> String {
        let trimmed = firstMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= 40 { return trimmed }
        let words = trimmed.prefix(60).split(separator: " ")
        if words.count > 1 {
            return words.dropLast().joined(separator: " ") + "…"
        }
        return String(trimmed.prefix(40)) + "…"
    }
}
