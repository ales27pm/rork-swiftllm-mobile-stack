import SwiftUI

struct HistoryView: View {
    @Bindable var viewModel: HistoryViewModel
    let onSelectConversation: (String) -> Void
    @State private var showClearConfirmation: Bool = false

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.filteredConversations.isEmpty {
                    emptyState
                } else {
                    conversationList
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.large)
            .searchable(text: $viewModel.searchText, prompt: "Search conversations")
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

    private var conversationList: some View {
        List {
            ForEach(viewModel.filteredConversations) { conv in
                Button {
                    onSelectConversation(conv.id)
                } label: {
                    conversationRow(conv)
                }
                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                    Button(role: .destructive) {
                        withAnimation {
                            viewModel.deleteConversation(conv.id)
                        }
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
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
