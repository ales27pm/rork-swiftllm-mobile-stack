import Foundation

@MainActor
@Observable
class MemoryService {
    private let database: DatabaseService
    var memories: [MemoryEntry] = []
    var associativeLinks: [AssociativeLink] = []

    init(database: DatabaseService) {
        self.database = database
        createTables()
        loadMemories()
        loadLinks()
    }

    private func createTables() {
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                category TEXT NOT NULL DEFAULT 'context',
                timestamp REAL NOT NULL,
                importance INTEGER NOT NULL DEFAULT 3,
                source TEXT NOT NULL DEFAULT 'conversation',
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL,
                relations TEXT NOT NULL DEFAULT '[]',
                consolidated INTEGER NOT NULL DEFAULT 0,
                decay REAL NOT NULL DEFAULT 1.0,
                activation_level REAL NOT NULL DEFAULT 0,
                emotional_valence REAL NOT NULL DEFAULT 0
            );
        """)
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS associative_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                strength REAL NOT NULL,
                type TEXT NOT NULL DEFAULT 'semantic',
                created_at REAL NOT NULL,
                reinforcements INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (source_id, target_id)
            );
        """)
    }

    func loadMemories() {
        let rows = database.query("SELECT * FROM memories ORDER BY last_accessed DESC;")
        memories = rows.compactMap { row -> MemoryEntry? in
            guard let id = row["id"] as? String,
                  let content = row["content"] as? String,
                  let timestamp = row["timestamp"] as? Double,
                  let lastAccessed = row["last_accessed"] as? Double else { return nil }
            let keywordsJson = (row["keywords"] as? String) ?? "[]"
            let keywords = (try? JSONDecoder().decode([String].self, from: Data(keywordsJson.utf8))) ?? []
            let relationsJson = (row["relations"] as? String) ?? "[]"
            let relations = (try? JSONDecoder().decode([String].self, from: Data(relationsJson.utf8))) ?? []
            return MemoryEntry(
                id: id,
                content: content,
                keywords: keywords,
                category: MemoryCategory(rawValue: (row["category"] as? String) ?? "context") ?? .context,
                timestamp: timestamp,
                importance: (row["importance"] as? Int64).map { Int($0) } ?? 3,
                source: MemorySource(rawValue: (row["source"] as? String) ?? "conversation") ?? .conversation,
                accessCount: (row["access_count"] as? Int64).map { Int($0) } ?? 0,
                lastAccessed: lastAccessed,
                relations: relations,
                consolidated: ((row["consolidated"] as? Int64) ?? 0) != 0,
                decay: (row["decay"] as? Double) ?? 1.0,
                activationLevel: (row["activation_level"] as? Double) ?? 0,
                emotionalValence: (row["emotional_valence"] as? Double) ?? 0
            )
        }
    }

    func loadLinks() {
        let rows = database.query("SELECT * FROM associative_links;")
        associativeLinks = rows.compactMap { row -> AssociativeLink? in
            guard let sourceId = row["source_id"] as? String,
                  let targetId = row["target_id"] as? String,
                  let strength = row["strength"] as? Double,
                  let createdAt = row["created_at"] as? Double else { return nil }
            return AssociativeLink(
                sourceId: sourceId,
                targetId: targetId,
                strength: strength,
                type: AssociativeLink.LinkType(rawValue: (row["type"] as? String) ?? "semantic") ?? .semantic,
                createdAt: createdAt,
                reinforcements: (row["reinforcements"] as? Int64).map { Int($0) } ?? 0
            )
        }
    }

    func addMemory(_ memory: MemoryEntry) {
        let keywordsJson = (try? String(data: JSONEncoder().encode(memory.keywords), encoding: .utf8)) ?? "[]"
        let relationsJson = (try? String(data: JSONEncoder().encode(memory.relations), encoding: .utf8)) ?? "[]"
        _ = database.execute(
            "INSERT OR REPLACE INTO memories (id, content, keywords, category, timestamp, importance, source, access_count, last_accessed, relations, consolidated, decay, activation_level, emotional_valence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            params: [memory.id, memory.content, keywordsJson, memory.category.rawValue, memory.timestamp, memory.importance, memory.source.rawValue, memory.accessCount, memory.lastAccessed, relationsJson, memory.consolidated ? 1 : 0, memory.decay, memory.activationLevel, memory.emotionalValence]
        )
        if let idx = memories.firstIndex(where: { $0.id == memory.id }) {
            memories[idx] = memory
        } else {
            memories.insert(memory, at: 0)
        }
        let newLinks = buildAssociativeLinks(newMemory: memory)
        for link in newLinks {
            saveLink(link)
        }
    }

    func deleteMemory(_ id: String) {
        _ = database.execute("DELETE FROM memories WHERE id = ?;", params: [id])
        _ = database.execute("DELETE FROM associative_links WHERE source_id = ? OR target_id = ?;", params: [id, id])
        memories.removeAll { $0.id == id }
        associativeLinks.removeAll { $0.sourceId == id || $0.targetId == id }
    }

    func clearAllMemories() {
        _ = database.execute("DELETE FROM memories;")
        _ = database.execute("DELETE FROM associative_links;")
        memories.removeAll()
        associativeLinks.removeAll()
    }

    func reinforceMemory(_ id: String) {
        guard let idx = memories.firstIndex(where: { $0.id == id }) else { return }
        memories[idx].accessCount += 1
        memories[idx].lastAccessed = Date().timeIntervalSince1970 * 1000
        memories[idx].decay = min(1.0, memories[idx].decay + 0.1)
        memories[idx].activationLevel = min(1.0, memories[idx].activationLevel + 0.2)
        addMemory(memories[idx])
    }

    func searchMemories(query: String, maxResults: Int = 8, minScore: Double = 0.05, categoryFilter: [MemoryCategory]? = nil) -> [RetrievalResult] {
        guard !memories.isEmpty else { return [] }

        let filtered = categoryFilter != nil
            ? memories.filter { categoryFilter!.contains($0.category) }
            : memories

        guard !filtered.isEmpty else { return [] }

        let idf = buildIDF(filtered)
        let queryVec = computeTFIDF(text: query, idf: idf)

        var scored: [RetrievalResult] = filtered.map { m in
            let docText = m.content + " " + m.keywords.joined(separator: " ") + " " + m.category.rawValue
            let docVec = computeTFIDF(text: docText, idf: idf)
            let tfidfScore = cosineSimilarity(a: queryVec, b: docVec)

            let queryTerms = tokenize(query)
            var keywordBonus: Double = 0
            for term in queryTerms {
                for kw in m.keywords {
                    if kw.lowercased().contains(term) || term.contains(kw.lowercased()) {
                        keywordBonus += 0.15
                    }
                }
            }
            keywordBonus = min(keywordBonus, 0.4)

            let decay = computeDecay(m)
            let recencyScore = decay * 0.15
            let importanceScore = Double(m.importance) / 5.0 * 0.2
            let activationBonus = m.activationLevel * 0.15
            let totalScore = tfidfScore + keywordBonus + recencyScore + importanceScore + activationBonus

            var matchType: RetrievalResult.MatchType = tfidfScore > keywordBonus ? .semantic : .keyword
            if activationBonus > tfidfScore && activationBonus > keywordBonus {
                matchType = .primed
            }

            return RetrievalResult(memory: m, score: totalScore, matchType: matchType)
        }

        scored.sort { $0.score > $1.score }

        var selected: [RetrievalResult] = []
        var seenCategories: Set<String> = []

        for result in scored {
            guard result.score >= minScore else { continue }
            guard selected.count < maxResults else { break }

            var adjustedScore = result.score
            if seenCategories.contains(result.memory.category.rawValue) {
                adjustedScore -= 0.1
            }
            if adjustedScore >= minScore {
                selected.append(RetrievalResult(memory: result.memory, score: adjustedScore, matchType: result.matchType))
                seenCategories.insert(result.memory.category.rawValue)
            }
        }

        return selected
    }

    func getAssociativeMemories(query: String, directResults: [RetrievalResult]) -> [RetrievalResult] {
        guard !directResults.isEmpty, !associativeLinks.isEmpty else { return [] }

        let seedMemories = directResults.prefix(3).map(\.memory)
        var activations: [String: (level: Double, depth: Int)] = [:]

        for mem in seedMemories {
            activations[mem.id] = (1.0, 0)
        }

        for d in 0..<2 {
            let currentLevel = activations.filter { $0.value.depth == d }
            for (nodeId, activation) in currentLevel {
                let outgoing = associativeLinks.filter { $0.sourceId == nodeId || $0.targetId == nodeId }
                for link in outgoing {
                    let neighborId = link.sourceId == nodeId ? link.targetId : link.sourceId
                    guard memories.contains(where: { $0.id == neighborId }) else { continue }
                    let propagated = activation.level * link.strength * 0.5
                    guard propagated >= 0.05 else { continue }
                    if let existing = activations[neighborId], existing.level >= propagated { continue }
                    activations[neighborId] = (propagated, d + 1)
                }
            }
        }

        let directIds = Set(directResults.map(\.memory.id))
        var results: [RetrievalResult] = []

        for (nodeId, activation) in activations where activation.depth > 0 {
            guard !directIds.contains(nodeId) else { continue }
            guard let memory = memories.first(where: { $0.id == nodeId }) else { continue }
            results.append(RetrievalResult(memory: memory, score: activation.level * 0.6, matchType: .associative))
        }

        results.sort { $0.score > $1.score }
        return Array(results.prefix(4))
    }

    func shouldExtractMemory(userText: String, assistantText: String) -> Bool {
        userText.trimmingCharacters(in: .whitespacesAndNewlines).count >= 20 &&
        assistantText.trimmingCharacters(in: .whitespacesAndNewlines).count >= 20
    }

    func extractAndStoreMemory(userText: String, assistantText: String) {
        guard shouldExtractMemory(userText: userText, assistantText: assistantText) else { return }

        let combined = userText + " " + assistantText
        let keywords = extractKeywords(from: combined)
        let category = classifyCategory(userText)

        let memory = MemoryEntry(
            content: String(userText.prefix(200)),
            keywords: keywords,
            category: category,
            importance: keywords.count >= 3 ? 4 : 3,
            source: .conversation
        )
        addMemory(memory)
    }

    func buildContextInjection(query: String) -> String {
        let directResults = searchMemories(query: query, maxResults: 5)
        let associativeResults = getAssociativeMemories(query: query, directResults: directResults)
        let allResults = directResults + associativeResults

        guard !allResults.isEmpty else { return "" }

        var parts: [String] = ["[Memory Context]"]
        for result in allResults.prefix(6) {
            let tag = result.matchType == .associative ? "related" : "recall"
            parts.append("- [\(tag)] \(result.memory.content.prefix(120))")

            reinforceMemory(result.memory.id)
        }
        return parts.joined(separator: "\n")
    }

    private func extractKeywords(from text: String) -> [String] {
        let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "don", "now", "and", "but", "or", "if", "while", "this", "that", "these", "those", "it", "its", "i", "me", "my", "you", "your", "he", "she", "we", "they", "them", "his", "her", "our", "their", "what", "which", "who", "whom"]

        let tokens = tokenize(text)
        let filtered = tokens.filter { !stopWords.contains($0) && $0.count > 2 }

        var freq: [String: Int] = [:]
        for token in filtered { freq[token, default: 0] += 1 }

        return freq.sorted { $0.value > $1.value }.prefix(5).map(\.key)
    }

    private func classifyCategory(_ text: String) -> MemoryCategory {
        let lower = text.lowercased()
        if lower.contains("i like") || lower.contains("i prefer") || lower.contains("i love") || lower.contains("i hate") || lower.contains("favorite") {
            return .preference
        }
        if lower.contains("remember") || lower.contains("always") || lower.contains("never") || lower.contains("make sure") {
            return .instruction
        }
        if lower.contains("feel") || lower.contains("happy") || lower.contains("sad") || lower.contains("angry") || lower.contains("worried") {
            return .emotion
        }
        if lower.contains("how to") || lower.contains("tutorial") || lower.contains("learn") || lower.contains("teach") {
            return .skill
        }
        return .context
    }

    private func saveLink(_ link: AssociativeLink) {
        _ = database.execute(
            "INSERT OR REPLACE INTO associative_links (source_id, target_id, strength, type, created_at, reinforcements) VALUES (?, ?, ?, ?, ?, ?);",
            params: [link.sourceId, link.targetId, link.strength, link.type.rawValue, link.createdAt, link.reinforcements]
        )
        if !associativeLinks.contains(where: { $0.sourceId == link.sourceId && $0.targetId == link.targetId }) {
            associativeLinks.append(link)
        }
    }

    private func buildAssociativeLinks(newMemory: MemoryEntry) -> [AssociativeLink] {
        var newLinks: [AssociativeLink] = []
        let newTokens = Set(tokenize(newMemory.content + " " + newMemory.keywords.joined(separator: " ")))

        for existing in memories {
            guard existing.id != newMemory.id else { continue }
            let existingTokens = Set(tokenize(existing.content + " " + existing.keywords.joined(separator: " ")))
            var overlap = 0
            for t in newTokens { if existingTokens.contains(t) { overlap += 1 } }
            let union = newTokens.count + existingTokens.count - overlap
            guard union > 0 else { continue }
            let jaccard = Double(overlap) / Double(union)
            guard jaccard >= 0.1 else { continue }

            let alreadyLinked = associativeLinks.contains {
                ($0.sourceId == newMemory.id && $0.targetId == existing.id) ||
                ($0.sourceId == existing.id && $0.targetId == newMemory.id)
            }
            guard !alreadyLinked else { continue }

            var linkType: AssociativeLink.LinkType = .semantic
            let timeDiff = abs(newMemory.timestamp - existing.timestamp)
            if timeDiff < 300_000 {
                linkType = .temporal
            } else if newMemory.category == existing.category {
                linkType = .topical
            }

            let keywordOverlap = newMemory.keywords.filter { k in
                existing.keywords.contains { $0.lowercased() == k.lowercased() }
            }.count
            let strength = min(1, jaccard + Double(keywordOverlap) * 0.15)

            guard strength > 0.15 else { continue }

            newLinks.append(AssociativeLink(
                sourceId: newMemory.id,
                targetId: existing.id,
                strength: strength,
                type: linkType,
                createdAt: Date().timeIntervalSince1970 * 1000,
                reinforcements: 0
            ))
        }

        newLinks.sort { $0.strength > $1.strength }
        return Array(newLinks.prefix(8))
    }

    private func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: " ", options: .regularExpression)
            .split(separator: " ")
            .map(String.init)
            .filter { $0.count > 2 }
    }

    private func buildIDF(_ entries: [MemoryEntry]) -> [String: Double] {
        let docCount = Double(max(entries.count, 1))
        var termDocFreq: [String: Int] = [:]

        for m in entries {
            let terms = Set(tokenize(m.content + " " + m.keywords.joined(separator: " ")))
            for t in terms { termDocFreq[t, default: 0] += 1 }
        }

        var idf: [String: Double] = [:]
        for (term, freq) in termDocFreq {
            idf[term] = log((docCount + 1) / (Double(freq) + 1)) + 1
        }
        return idf
    }

    private func computeTFIDF(text: String, idf: [String: Double]) -> [String: Double] {
        let tokens = tokenize(text)
        var tf: [String: Double] = [:]
        for t in tokens { tf[t, default: 0] += 1 }
        let maxTF = max(tf.values.max() ?? 1, 1)

        var tfidf: [String: Double] = [:]
        for (term, freq) in tf {
            let normalizedTF = 0.5 + (0.5 * freq) / maxTF
            tfidf[term] = normalizedTF * (idf[term] ?? 1)
        }
        return tfidf
    }

    private func cosineSimilarity(a: [String: Double], b: [String: Double]) -> Double {
        var dot: Double = 0
        var magA: Double = 0
        var magB: Double = 0

        for (term, val) in a {
            magA += val * val
            if let bVal = b[term] { dot += val * bVal }
        }
        for val in b.values { magB += val * val }

        let denom = sqrt(magA) * sqrt(magB)
        return denom == 0 ? 0 : dot / denom
    }

    private func computeDecay(_ memory: MemoryEntry) -> Double {
        let now = Date().timeIntervalSince1970 * 1000
        let hoursSinceAccess = (now - memory.lastAccessed) / (1000 * 60 * 60)
        let hoursSinceCreation = (now - memory.timestamp) / (1000 * 60 * 60)

        let accessBoost = min(Double(memory.accessCount) * 0.1, 0.5)
        let importanceBoost = Double(memory.importance) / 5.0 * 0.3

        let halfLife = 168 * (1 + accessBoost + importanceBoost)
        let decayFactor = pow(0.5, hoursSinceAccess / halfLife)
        let freshnessBonus = hoursSinceCreation < 24 ? 0.2 : 0

        return max(0.05, min(1.0, decayFactor + freshnessBonus))
    }
}
