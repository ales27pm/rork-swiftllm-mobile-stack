import SwiftUI

struct WebSearchViewWithInjection: View {
    @Bindable var chatViewModel: ChatViewModel
    let onDismiss: () -> Void

    @State private var searchService = WebSearchService()
    @State private var searchText: String = ""
    @State private var browserURL: URL?
    @State private var browserTitle: String = ""
    @State private var showBrowser: Bool = false
    @State private var injectedFeedback: Bool = false
    @FocusState private var isSearchFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if searchService.searchResults.isEmpty && !searchService.isSearching {
                    emptyState
                } else {
                    resultsList
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Browse")
            .navigationBarTitleDisplayMode(.large)
            .searchable(text: $searchText, placement: .navigationBarDrawer(displayMode: .always), prompt: "Search the web...")
            .onSubmit(of: .search) {
                performSearch()
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { onDismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if searchService.isSearching {
                        ProgressView()
                            .controlSize(.small)
                    } else if !searchService.searchResults.isEmpty {
                        Button {
                            injectResults()
                        } label: {
                            Label("Send to Chat", systemImage: "arrow.turn.down.left")
                        }
                    }
                }
            }
            .sheet(isPresented: $showBrowser) {
                if let url = browserURL {
                    InAppBrowserView(url: url, title: browserTitle)
                }
            }
            .overlay(alignment: .bottom) {
                if injectedFeedback {
                    Text("Results sent to chat context")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(.green.gradient)
                        .clipShape(Capsule())
                        .padding(.bottom, 16)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
    }

    private var emptyState: some View {
        ScrollView {
            VStack(spacing: 32) {
                Spacer(minLength: 40)

                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [.blue.opacity(0.12), .cyan.opacity(0.12)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 100, height: 100)

                    Image(systemName: "globe.americas.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.blue, .cyan],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .symbolEffect(.breathe, options: .repeating)
                }

                VStack(spacing: 8) {
                    Text("Web Search")
                        .font(.title2.bold())

                    Text("Search the web and inject results\ninto your chat context")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                VStack(spacing: 10) {
                    quickSearch("Latest AI news", icon: "sparkles")
                    quickSearch("Swift programming tips", icon: "swift")
                    quickSearch("Weather forecast", icon: "cloud.sun.fill")
                    quickSearch("Top tech headlines", icon: "newspaper.fill")
                }
                .padding(.horizontal, 20)

                Spacer(minLength: 40)
            }
            .padding(.horizontal, 12)
        }
    }

    private func quickSearch(_ text: String, icon: String) -> some View {
        Button {
            searchText = text
            performSearch()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundStyle(.blue)
                    .frame(width: 24)

                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.primary)

                Spacer()

                Image(systemName: "magnifyingglass")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 12))
        }
    }

    private var resultsList: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if !searchService.searchResults.isEmpty {
                    Button {
                        injectResults()
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "arrow.turn.down.left")
                                .font(.subheadline.weight(.semibold))
                            Text("Inject \(searchService.searchResults.prefix(5).count) results into chat")
                                .font(.subheadline.weight(.medium))
                        }
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(.blue.gradient)
                        .clipShape(.rect(cornerRadius: 12))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                }

                if searchService.isSearching {
                    VStack(spacing: 12) {
                        ForEach(0..<4, id: \.self) { _ in
                            searchPlaceholder
                        }
                    }
                    .padding(16)
                } else {
                    HStack {
                        Text("\(searchService.searchResults.count) results")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        Spacer()
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 4)

                    ForEach(searchService.searchResults) { result in
                        searchResultRow(result)
                    }
                }
            }
        }
    }

    private var searchPlaceholder: some View {
        VStack(alignment: .leading, spacing: 8) {
            RoundedRectangle(cornerRadius: 4)
                .fill(Color(.tertiarySystemFill))
                .frame(height: 14)
                .frame(maxWidth: 200)

            RoundedRectangle(cornerRadius: 4)
                .fill(Color(.quaternarySystemFill))
                .frame(height: 10)

            RoundedRectangle(cornerRadius: 4)
                .fill(Color(.quaternarySystemFill))
                .frame(height: 10)
                .frame(maxWidth: 260)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 12))
        .shimmering()
    }

    private func searchResultRow(_ result: WebSearchResult) -> some View {
        VStack(spacing: 0) {
            Button {
                openInBrowser(result)
            } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(domainFrom(result.url))
                        .font(.system(size: 11).weight(.medium))
                        .foregroundStyle(.blue.opacity(0.8))
                        .lineLimit(1)

                    Text(result.title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    if !result.snippet.isEmpty {
                        Text(result.snippet)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                            .multilineTextAlignment(.leading)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }

            Divider()
                .padding(.leading, 16)
        }
    }

    private func openInBrowser(_ result: WebSearchResult) {
        guard let url = URL(string: result.url) else { return }
        browserURL = url
        browserTitle = result.title
        showBrowser = true
    }

    private func performSearch() {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }
        Task {
            _ = await searchService.search(query: query)
        }
    }

    private func injectResults() {
        chatViewModel.injectWebSearchContext(searchService.searchResults)
        withAnimation(.spring(duration: 0.3)) {
            injectedFeedback = true
        }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            withAnimation { injectedFeedback = false }
            onDismiss()
        }
    }

    private func domainFrom(_ urlString: String) -> String {
        guard let url = URL(string: urlString) else { return urlString }
        return url.host?.replacingOccurrences(of: "www.", with: "") ?? urlString
    }
}
