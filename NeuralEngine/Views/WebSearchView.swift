import SwiftUI

struct WebSearchView: View {
    @State private var searchService = WebSearchService()
    @State private var searchText: String = ""
    @State private var browserURL: URL?
    @State private var browserTitle: String = ""
    @State private var showBrowser: Bool = false
    @State private var recentSearches: [String] = []
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
                ToolbarItem(placement: .topBarTrailing) {
                    if searchService.isSearching {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
            }
            .sheet(isPresented: $showBrowser) {
                if let url = browserURL {
                    InAppBrowserView(url: url, title: browserTitle)
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

                    Text("Search the web and browse pages\ndirectly within Nexus")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                if let error = searchService.lastError {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .background(Color.orange.opacity(0.08))
                    .clipShape(.rect(cornerRadius: 10))
                }

                VStack(spacing: 10) {
                    quickSearch("Latest AI news", icon: "sparkles")
                    quickSearch("Swift programming tips", icon: "swift")
                    quickSearch("Weather forecast", icon: "cloud.sun.fill")
                    quickSearch("Top tech headlines", icon: "newspaper.fill")
                }
                .padding(.horizontal, 20)

                if !recentSearches.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Recent")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)

                            Spacer()

                            Button("Clear") {
                                withAnimation { recentSearches.removeAll() }
                            }
                            .font(.caption)
                        }

                        ForEach(recentSearches, id: \.self) { query in
                            Button {
                                searchText = query
                                performSearch()
                            } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: "clock.arrow.circlepath")
                                        .font(.caption)
                                        .foregroundStyle(.tertiary)

                                    Text(query)
                                        .font(.subheadline)
                                        .foregroundStyle(.primary)

                                    Spacer()

                                    Image(systemName: "arrow.up.left")
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                }

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
                if let error = searchService.lastError {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity)
                    .background(Color.orange.opacity(0.08))
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
                    .padding(.top, 12)
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
            .contextMenu {
                Button("Open in Browser", systemImage: "safari") {
                    openInBrowser(result)
                }
                Button("Open in Safari", systemImage: "arrow.up.right.square") {
                    if let url = URL(string: result.url) {
                        UIApplication.shared.open(url)
                    }
                }
                Button("Copy Link", systemImage: "doc.on.doc") {
                    UIPasteboard.general.string = result.url
                }
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

        if !recentSearches.contains(query) {
            recentSearches.insert(query, at: 0)
            if recentSearches.count > 10 {
                recentSearches.removeLast()
            }
        }

        Task {
            _ = await searchService.search(query: query)
        }
    }

    private func domainFrom(_ urlString: String) -> String {
        guard let url = URL(string: urlString) else { return urlString }
        return url.host?.replacingOccurrences(of: "www.", with: "") ?? urlString
    }
}

struct ShimmeringModifier: ViewModifier {
    @State private var isAnimating: Bool = false

    func body(content: Content) -> some View {
        content
            .opacity(isAnimating ? 0.5 : 1.0)
            .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)
            .onAppear { isAnimating = true }
    }
}

extension View {
    func shimmering() -> some View {
        modifier(ShimmeringModifier())
    }
}
