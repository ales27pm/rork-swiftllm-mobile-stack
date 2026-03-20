import SwiftUI

struct HistoryView: View {
    @Bindable var viewModel: HistoryViewModel
    let onSelectConversation: (String) -> Void
    @State private var showClearConfirmation: Bool = false

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.filteredConversations.isEmpty && !viewModel.hasDeepSearchResults {
                    emptyState
                } else {
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            if viewModel.hasDeepSearchResults {
                                deepSearchSection
                            }
                            conversationListContent
                        }
                    }
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.large)
            .searchable(text: $viewModel.searchText, prompt: "Search conversations")
            .onChange(of: viewModel.searchText) { _, newValue in
                if newValue.count >= 2 {
                    viewModel.performDeepSearch()
                } else {
                    viewModel.searchResults = []
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if !viewModel.conversationService.conversations.isEmpty {
                        Button("Clear All", role: .destructive) {
                            showClearConfirmation = true
                        }
                        .font(.subheadline)
                    }
                }
            }
            .alert("Clear All History", isPresented: $showClearConfirmation) {
                Button("Delete All", role: .destructive) {
                    withAnimation { viewModel.clearAll() }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will permanently delete all conversation history.")
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 48))
                .foregroundStyle(.quaternary)
            Text("No Conversations Yet")
                .font(.title3.bold())
            Text("Your chat history will appear here")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var deepSearchSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "text.magnifyingglass")
                    .font(.caption)
                    .foregroundStyle(.blue)
                Text("Message Matches")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(viewModel.searchResults.count) found")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)

            ForEach(viewModel.searchResults) { result in
                Button {
                    onSelectConversation(result.conversationId)
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: result.role == "user" ? "person.fill" : "sparkles")
                            .font(.caption)
                            .foregroundStyle(result.role == "user" ? .blue : .purple)
                            .frame(width: 20)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(result.conversationTitle)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                            Text(result.matchedContent)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }

                        Spacer(minLength: 0)

                        Image(systemName: "chevron.right")
                            .font(.caption2)
                            .foregroundStyle(.quaternary)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }
            }

            Divider()
                .padding(.horizontal, 16)
                .padding(.vertical, 4)
        }
    }

    private var conversationListContent: some View {
        LazyVStack(spacing: 0) {
            if !viewModel.filteredConversations.isEmpty {
                HStack {
                    Text("Conversations")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 4)
            }

            ForEach(viewModel.filteredConversations) { conv in
                Button {
                    onSelectConversation(conv.id)
                } label: {
                    conversationRow(conv)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 6)
                }
                .contextMenu {
                    Button("Delete", systemImage: "trash", role: .destructive) {
                        withAnimation {
                            viewModel.deleteConversation(conv.id)
                        }
                    }
                }
            }
        }
    }

    private func conversationRow(_ conv: Conversation) -> some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(LinearGradient(
                        colors: [.blue.opacity(0.2), .purple.opacity(0.2)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 40, height: 40)
                Image(systemName: "bubble.left.and.text.bubble.right.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(.blue)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(conv.title)
                    .font(.subheadline.bold())
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                if !conv.lastMessage.isEmpty {
                    Text(conv.lastMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                HStack(spacing: 8) {
                    Text(conv.updatedAt.formatted(.relative(presentation: .named)))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)

                    Text("\(conv.messageCount) messages")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.quaternary)
        }
        .padding(.vertical, 4)
    }
}
