import SwiftUI

@Observable
@MainActor
class HistoryViewModel {
    let conversationService: ConversationService

    var searchText: String = ""
    var searchResults: [ConversationSearchResult] = []
    var isDeepSearching: Bool = false

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

    var hasDeepSearchResults: Bool {
        !searchText.isEmpty && !searchResults.isEmpty
    }

    func performDeepSearch() {
        guard !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            searchResults = []
            return
        }
        isDeepSearching = true
        searchResults = conversationService.searchMessages(query: searchText)
        isDeepSearching = false
    }

    func deleteConversation(_ id: String) {
        conversationService.deleteConversation(id)
    }

    func clearAll() {
        conversationService.clearAllConversations()
    }
}
