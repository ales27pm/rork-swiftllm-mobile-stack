import SwiftUI

@Observable
@MainActor
class HistoryViewModel {
    let conversationService: ConversationService

    var searchText: String = ""

    init(conversationService: ConversationService) {
        self.conversationService = conversationService
    }

    var filteredConversations: [Conversation] {
        if searchText.isEmpty {
            return conversationService.conversations
        }
        let query = searchText.lowercased()
        return conversationService.conversations.filter {
            $0.title.lowercased().contains(query) ||
            $0.lastMessage.lowercased().contains(query)
        }
    }

    func deleteConversation(_ id: String) {
        conversationService.deleteConversation(id)
    }

    func clearAll() {
        conversationService.clearAllConversations()
    }
}
