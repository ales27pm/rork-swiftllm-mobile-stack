import Foundation

@MainActor
@Observable
class WebSearchService {
    var searchResults: [WebSearchResult] = []
    var isSearching: Bool = false
    var isFetching: Bool = false
    var fetchedContent: String = ""
    var lastError: String?

    func search(query: String) async -> [WebSearchResult] {
        isSearching = true
        lastError = nil
        defer { isSearching = false }

        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let urlString = "https://html.duckduckgo.com/html/?q=\(encoded)"

        guard let url = URL(string: urlString) else {
            lastError = "Invalid search query"
            return []
        }

        do {
            var request = URLRequest(url: url, timeoutInterval: 15)
            request.setValue("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1", forHTTPHeaderField: "User-Agent")

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                lastError = "Search returned an error"
                return []
            }

            guard let html = String(data: data, encoding: .utf8) else {
                lastError = "Could not decode search results"
                return []
            }

            let results = parseDuckDuckGoHTML(html)
            searchResults = results
            return results
        } catch {
            lastError = "Search failed: \(error.localizedDescription)"
            return []
        }
    }

    func fetchURL(_ urlString: String) async -> String {
        isFetching = true
        lastError = nil
        fetchedContent = ""
        defer { isFetching = false }

        guard let url = URL(string: urlString) else {
            lastError = "Invalid URL"
            return ""
        }

        do {
            var request = URLRequest(url: url, timeoutInterval: 20)
            request.setValue("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1", forHTTPHeaderField: "User-Agent")

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                lastError = "Failed to fetch page (HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0))"
                return ""
            }

            guard let html = String(data: data, encoding: .utf8) else {
                lastError = "Could not decode page content"
                return ""
            }

            let text = extractTextFromHTML(html)
            let truncated = String(text.prefix(4000))
            fetchedContent = truncated
            return truncated
        } catch {
            lastError = "Fetch failed: \(error.localizedDescription)"
            return ""
        }
    }

    private func parseDuckDuckGoHTML(_ html: String) -> [WebSearchResult] {
        var results: [WebSearchResult] = []

        let resultPattern = #"<a rel="nofollow" class="result__a" href="([^"]*)"[^>]*>(.*?)</a>"#
        let snippetPattern = #"<a class="result__snippet"[^>]*>(.*?)</a>"#

        guard let resultRegex = try? NSRegularExpression(pattern: resultPattern, options: [.dotMatchesLineSeparators]),
              let snippetRegex = try? NSRegularExpression(pattern: snippetPattern, options: [.dotMatchesLineSeparators]) else {
            return []
        }

        let resultMatches = resultRegex.matches(in: html, range: NSRange(html.startIndex..., in: html))
        let snippetMatches = snippetRegex.matches(in: html, range: NSRange(html.startIndex..., in: html))

        for (index, match) in resultMatches.prefix(10).enumerated() {
            guard let urlRange = Range(match.range(at: 1), in: html),
                  let titleRange = Range(match.range(at: 2), in: html) else { continue }

            var rawURL = String(html[urlRange])
            let rawTitle = String(html[titleRange])

            if rawURL.hasPrefix("//duckduckgo.com/l/?uddg="),
               let components = URLComponents(string: "https:" + rawURL),
               let uddg = components.queryItems?.first(where: { $0.name == "uddg" })?.value {
                rawURL = uddg
            }

            let title = stripHTMLTags(rawTitle)
            let snippet: String
            if index < snippetMatches.count,
               let snipRange = Range(snippetMatches[index].range(at: 1), in: html) {
                snippet = stripHTMLTags(String(html[snipRange]))
            } else {
                snippet = ""
            }

            guard !rawURL.isEmpty, !title.isEmpty else { continue }

            results.append(WebSearchResult(
                title: title,
                url: rawURL,
                snippet: snippet
            ))
        }

        return results
    }

    private func extractTextFromHTML(_ html: String) -> String {
        var text = html

        if let scriptRegex = try? NSRegularExpression(pattern: #"<script[^>]*>[\s\S]*?</script>"#, options: .caseInsensitive) {
            text = scriptRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
        }
        if let styleRegex = try? NSRegularExpression(pattern: #"<style[^>]*>[\s\S]*?</style>"#, options: .caseInsensitive) {
            text = styleRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
        }
        if let navRegex = try? NSRegularExpression(pattern: #"<(nav|header|footer)[^>]*>[\s\S]*?</\1>"#, options: .caseInsensitive) {
            text = navRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
        }

        text = stripHTMLTags(text)

        text = text.replacingOccurrences(of: "&amp;", with: "&")
        text = text.replacingOccurrences(of: "&lt;", with: "<")
        text = text.replacingOccurrences(of: "&gt;", with: ">")
        text = text.replacingOccurrences(of: "&quot;", with: "\"")
        text = text.replacingOccurrences(of: "&#39;", with: "'")
        text = text.replacingOccurrences(of: "&nbsp;", with: " ")

        if let whitespaceRegex = try? NSRegularExpression(pattern: #"\n{3,}"#) {
            text = whitespaceRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "\n\n")
        }
        if let spaceRegex = try? NSRegularExpression(pattern: #" {2,}"#) {
            text = spaceRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: " ")
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func stripHTMLTags(_ string: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: #"<[^>]+>"#) else { return string }
        return regex.stringByReplacingMatches(in: string, range: NSRange(string.startIndex..., in: string), withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

nonisolated struct WebSearchResult: Identifiable, Sendable, Hashable {
    let id = UUID()
    let title: String
    let url: String
    let snippet: String
}
