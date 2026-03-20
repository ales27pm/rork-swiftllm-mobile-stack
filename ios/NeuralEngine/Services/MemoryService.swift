import Foundation
import NaturalLanguage

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

    func reinforceFromReaction(messageContent: String) {
        let results = searchMemories(query: messageContent, maxResults: 3)
        for result in results {
            reinforceMemory(result.memory.id)
        }
    }

    func reinforceMemory(_ id: String) {
        guard let idx = memories.firstIndex(where: { $0.id == id }) else { return }
        memories[idx].accessCount += 1
        memories[idx].lastAccessed = Date().timeIntervalSince1970 * 1000
        memories[idx].decay = min(1.0, memories[idx].decay + 0.1)
        memories[idx].activationLevel = min(1.0, memories[idx].activationLevel + 0.2)
        addMemory(memories[idx])
    }

    func searchMemories(query: String, maxResults: Int = 8, minScore: Double = 0.05, categoryFilter: [MemoryCategory]? = nil, languageHint: String? = nil) -> [RetrievalResult] {
        guard !memories.isEmpty else { return [] }

        let filtered = categoryFilter != nil
            ? memories.filter { categoryFilter!.contains($0.category) }
            : memories

        guard !filtered.isEmpty else { return [] }

        let resolvedLanguageHint = languageHint ?? NLTextProcessing.detectLanguage(for: query)?.rawValue
        let queryTerms = tokenize(query, languageHint: resolvedLanguageHint)
        let idf = buildIDF(filtered, languageHint: resolvedLanguageHint)
        let queryVec = computeTFIDF(tokens: queryTerms, idf: idf)

        var scored: [RetrievalResult] = filtered.map { memory in
            let docText = memory.content + " " + memory.keywords.joined(separator: " ") + " " + memory.category.rawValue
            let documentLanguageHint = resolvedLanguageHint ?? NLTextProcessing.detectLanguage(for: docText)?.rawValue
            let docTerms = tokenize(docText, languageHint: documentLanguageHint)
            let docVec = computeTFIDF(tokens: docTerms, idf: idf)
            let tfidfScore = cosineSimilarity(a: queryVec, b: docVec)
            let lexicalOverlap = weightedOverlap(queryTerms: queryTerms, documentTerms: docTerms)

            var keywordBonus: Double = 0
            for term in queryTerms {
                for keyword in memory.keywords {
                    let normalizedKeyword = NLTextProcessing.normalizeForMatching(keyword, languageHint: documentLanguageHint)
                    if normalizedKeyword.contains(term) || term.contains(normalizedKeyword) {
                        keywordBonus += 0.15
                    }
                }
            }
            keywordBonus = min(keywordBonus, 0.4)

            let rawEmbeddingScore = NLTextProcessing.embeddingSimilarity(
                query: query,
                document: memory.content,
                languageHint: resolvedLanguageHint ?? documentLanguageHint
            )
            let keywordEmbeddingScore = NLTextProcessing.embeddingSimilarity(
                query: query,
                document: memory.keywords.joined(separator: " "),
                languageHint: resolvedLanguageHint ?? documentLanguageHint
            )
            let embeddingScore = max(rawEmbeddingScore ?? 0, keywordEmbeddingScore ?? 0)

            let decay = computeDecay(memory)
            let recencyScore = decay * 0.15
            let importanceScore = Double(memory.importance) / 5.0 * 0.2
            let activationBonus = memory.activationLevel * 0.15
            let hybridLexicalScore = (tfidfScore * 0.5) + (lexicalOverlap * 0.25) + (keywordBonus * 0.25)
            let semanticScore = embeddingScore * 0.35
            let totalScore = hybridLexicalScore + semanticScore + recencyScore + importanceScore + activationBonus

            var matchType: RetrievalResult.MatchType = semanticScore > hybridLexicalScore ? .semantic : .keyword
            if activationBonus > hybridLexicalScore && activationBonus > semanticScore {
                matchType = .primed
            }

            return RetrievalResult(memory: memory, score: totalScore, matchType: matchType)
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

        for depth in 0..<2 {
            let currentLevel = activations.filter { $0.value.depth == depth }
            for (nodeId, activation) in currentLevel {
                let outgoing = associativeLinks.filter { $0.sourceId == nodeId || $0.targetId == nodeId }
                for link in outgoing {
                    let neighborId = link.sourceId == nodeId ? link.targetId : link.sourceId
                    guard memories.contains(where: { $0.id == neighborId }) else { continue }
                    let propagated = activation.level * link.strength * 0.5
                    guard propagated >= 0.05 else { continue }
                    if let existing = activations[neighborId], existing.level >= propagated { continue }
                    activations[neighborId] = (propagated, depth + 1)
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
        userText.trimmingCharacters(in: .whitespacesAndNewlines).count >= 8 &&
        assistantText.trimmingCharacters(in: .whitespacesAndNewlines).count >= 4
    }

    func extractAndStoreMemory(userText: String, assistantText: String) {
        guard shouldExtractMemory(userText: userText, assistantText: assistantText) else { return }

        let extractions = extractMemorableContent(userText: userText, assistantText: assistantText)

        if extractions.isEmpty {
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
            addMemoryWithDedup(memory)
        } else {
            for extraction in extractions {
                addMemoryWithDedup(extraction)
            }
        }

        consolidateMemories()
    }

    private static let nonNameWords: Set<String> = [
        "learning", "working", "trying", "going", "doing", "feeling", "thinking",
        "looking", "getting", "making", "coming", "running", "playing", "reading",
        "writing", "living", "studying", "hoping", "planning", "building", "coding",
        "developing", "designing", "testing", "debugging", "happy", "sad", "tired",
        "excited", "frustrated", "confused", "sorry", "sure", "fine", "good",
        "great", "ok", "okay", "here", "there", "just", "not", "also", "very",
        "really", "quite", "pretty", "currently", "still", "always", "never",
        "wondering", "curious", "interested", "based", "located",
        "so", "well", "and", "but", "the", "from", "that", "this",
        "feeling", "honestly", "actually"
    ]

    private func normalizeApostrophes(_ text: String) -> String {
        text.replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
    }

    private func extractMemorableContent(userText: String, assistantText: String) -> [MemoryEntry] {
        var entries: [MemoryEntry] = []
        let normalized = normalizeApostrophes(userText)
        let lower = NLTextProcessing.normalizeForMatching(normalized)
        let processed = NLTextProcessing.process(text: normalized)

        let namePatterns = [
            #"(?i)my name is ([\w\s]+)"#,
            #"(?i)(?:i'm|i am) ([A-Z][\w]+)\b"#,
            #"(?i)call me ([\w]+)"#,
            #"(?i)(?:people |they |everyone )?calls? me ([\w]+)"#,
        ]
        for pattern in namePatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: normalized, range: NSRange(normalized.startIndex..., in: normalized)),
               let range = Range(match.range(at: 1), in: normalized) {
                let name = String(normalized[range]).trimmingCharacters(in: .whitespaces)
                if name.count > 1 && name.count < 30 && !Self.nonNameWords.contains(name.lowercased()) {
                    entries.append(MemoryEntry(
                        content: "User's name is \(name)",
                        keywords: ["name", name.lowercased()] + processed.namedEntities.map { $0.lowercased() },
                        category: .fact,
                        importance: 5,
                        source: .conversation
                    ))
                    break
                }
            }
        }

        let preferencePatterns: [(String, String)] = [
            (#"(?i)i (?:really |absolutely |truly )?(?:like|love|enjoy|prefer|adore) (.+?)(?:[.,!]|$)"#, "preference"),
            (#"(?i)i(?:'m| am) (?:a )?(?:big |huge )?fan of (.+?)(?:[.,!]|$)"#, "preference"),
            (#"(?i)i (?:hate|dislike|can't stand|don't like|detest|loathe) (.+?)(?:[.,!]|$)"#, "dislike"),
            (#"(?i)my favou?rite(?:s| \w+)? (?:is|are) (.+?)(?:[.,!]|$)"#, "favorite"),
            (#"(?i)i favou?r (.+?) over (.+?)(?:[.,!]|$)"#, "preference"),
        ]
        for (pattern, kind) in preferencePatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: normalized, range: NSRange(normalized.startIndex..., in: normalized)),
               let range = Range(match.range(at: 1), in: normalized) {
                let subject = String(normalized[range]).trimmingCharacters(in: .whitespaces)
                if subject.count > 2 && subject.count < 100 {
                    entries.append(MemoryEntry(
                        content: "User \(kind): \(subject)",
                        keywords: extractKeywords(from: subject) + [kind],
                        category: .preference,
                        importance: 4,
                        source: .conversation
                    ))
                }
            }
        }

        let goalPatterns = [
            #"(?i)i (?:want to|need to|plan to|am going to|will|aim to|wish to|hope to|intend to) (.+?)(?:[.,!]|$)"#,
            #"(?i)my goal is (.+?)(?:[.,!]|$)"#,
            #"(?i)i(?:'m| am) (?:working on|trying to|learning|studying|practicing|training for) (.+?)(?:[.,!]|$)"#,
            #"(?i)i(?:'m| am) (?:hoping|planning|aiming) to (.+?)(?:[.,!]|$)"#,
        ]
        for pattern in goalPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: normalized, range: NSRange(normalized.startIndex..., in: normalized)),
               let range = Range(match.range(at: 1), in: normalized) {
                let goal = String(normalized[range]).trimmingCharacters(in: .whitespaces)
                if goal.count > 3 && goal.count < 150 {
                    entries.append(MemoryEntry(
                        content: "User goal: \(goal)",
                        keywords: extractKeywords(from: goal) + ["goal"],
                        category: .context,
                        importance: 4,
                        source: .conversation
                    ))
                }
            }
        }

        if lower.contains("remember") || lower.contains("always") || lower.contains("make sure") || lower.contains("never") || lower.contains("dont forget") || lower.contains("don't forget") {
            entries.append(MemoryEntry(
                content: String(normalized.prefix(200)),
                keywords: extractKeywords(from: normalized) + ["instruction"],
                category: .instruction,
                importance: 5,
                source: .conversation
            ))
        }

        let factPatterns = [
            #"(?i)i (?:work|live|study|am from|was born|grew up) (.+?)(?:[.,!]|$)"#,
            #"(?i)i have (?:a |an )(.+?)(?:[.,!]|$)"#,
            #"(?i)i(?:'m| am) from (.+?)(?:[.,!]|$)"#,
            #"(?i)i(?:'m| am) (?:a |an )(.+?)(?:[.,!]|$)"#,
            #"(?i)i(?:'m| am) based (?:in|out of) (.+?)(?:[.,!]|$)"#,
            #"(?i)i (?:was raised|grew up) (?:in|near) (.+?)(?:[.,!]|$)"#,
            #"(?i)i(?:'m| am) (?:currently |now )?(?:working|living|studying|based) (?:at|in|on|for) (.+?)(?:[.,!]|$)"#,
            #"(?i)i (?:own|drive|speak|play) (.+?)(?:[.,!]|$)"#,
        ]
        for pattern in factPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: normalized, range: NSRange(normalized.startIndex..., in: normalized)),
               let range = Range(match.range(at: 1), in: normalized) {
                let fact = String(normalized[range]).trimmingCharacters(in: .whitespaces)
                if fact.count > 2 && fact.count < 100 {
                    entries.append(MemoryEntry(
                        content: "User fact: \(fact)",
                        keywords: extractKeywords(from: fact) + ["personal"],
                        category: .fact,
                        importance: 4,
                        source: .conversation
                    ))
                }
            }
        }

        return entries
    }

    private func addMemoryWithDedup(_ memory: MemoryEntry) {
        var bestMatch: (index: Int, similarity: Double)?
        for (idx, existing) in memories.enumerated() {
            let similarity = contentSimilarity(memory.content, existing.content)
            let categorySame = memory.category == existing.category
            let threshold: Double = categorySame ? 0.65 : 0.85
            if similarity > threshold {
                if bestMatch == nil || similarity > bestMatch!.similarity {
                    bestMatch = (idx, similarity)
                }
            }
        }

        if let match = bestMatch {
            let existing = memories[match.index]
            if memory.importance >= existing.importance || memory.content.count > existing.content.count {
                var updated = existing
                updated.content = memory.content
                updated.keywords = Array(Set(existing.keywords + memory.keywords))
                updated.importance = max(existing.importance, memory.importance)
                updated.lastAccessed = Date().timeIntervalSince1970 * 1000
                addMemory(updated)
            } else {
                reinforceMemory(existing.id)
            }
            return
        }
        addMemory(memory)
    }

    private func contentSimilarity(_ a: String, _ b: String) -> Double {
        let tokensA = Set(tokenize(a))
        let tokensB = Set(tokenize(b))
        guard !tokensA.isEmpty || !tokensB.isEmpty else { return 0 }
        let intersection = tokensA.intersection(tokensB).count
        let union = tokensA.union(tokensB).count
        let jaccardScore = union > 0 ? Double(intersection) / Double(union) : 0

        let normA = a.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let normB = b.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if normA == normB { return 1.0 }
        if normA.contains(normB) || normB.contains(normA) {
            let ratio = Double(min(normA.count, normB.count)) / Double(max(normA.count, normB.count))
            return max(jaccardScore, ratio)
        }

        return jaccardScore
    }

    private func consolidateMemories() {
        let maxPerCategory = 15
        let categoryCounts = Dictionary(grouping: memories, by: \.category)

        for (_, entries) in categoryCounts where entries.count > maxPerCategory {
            let sorted = entries.sorted { a, b in
                let scoreA = Double(a.importance) * 0.4 + a.decay * 0.3 + Double(a.accessCount) * 0.1 + (a.activationLevel * 0.2)
                let scoreB = Double(b.importance) * 0.4 + b.decay * 0.3 + Double(b.accessCount) * 0.1 + (b.activationLevel * 0.2)
                return scoreA > scoreB
            }

            let toRemove = sorted.suffix(from: maxPerCategory)
            for entry in toRemove {
                deleteMemory(entry.id)
            }
        }
    }

    func buildContextInjection(query: String, languageHint: String? = nil) -> String {
        let directResults = searchMemories(query: query, maxResults: 5, languageHint: languageHint)
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
        var freq: [String: Int] = [:]
        for token in tokenize(text) { freq[token, default: 0] += 1 }
        return freq.sorted { lhs, rhs in
            lhs.value == rhs.value ? lhs.key < rhs.key : lhs.value > rhs.value
        }.prefix(5).map(\.key)
    }

    private func classifyCategory(_ text: String) -> MemoryCategory {
        let lower = NLTextProcessing.normalizeForMatching(text)
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
            for token in newTokens where existingTokens.contains(token) { overlap += 1 }
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

            let keywordOverlap = newMemory.keywords.filter { keyword in
                existing.keywords.contains { $0.caseInsensitiveCompare(keyword) == .orderedSame }
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

    private func tokenize(_ text: String, languageHint: String? = nil) -> [String] {
        NLTextProcessing.stemmedTerms(text, languageHint: languageHint, droppingStopWords: true)
    }

    private func buildIDF(_ entries: [MemoryEntry], languageHint: String? = nil) -> [String: Double] {
        let docCount = Double(max(entries.count, 1))
        var termDocFreq: [String: Int] = [:]

        for memory in entries {
            let terms = Set(tokenize(memory.content + " " + memory.keywords.joined(separator: " "), languageHint: languageHint))
            for term in terms { termDocFreq[term, default: 0] += 1 }
        }

        var idf: [String: Double] = [:]
        for (term, freq) in termDocFreq {
            idf[term] = log((docCount + 1) / (Double(freq) + 1)) + 1
        }
        return idf
    }

    private func computeTFIDF(tokens: [String], idf: [String: Double]) -> [String: Double] {
        var tf: [String: Double] = [:]
        for token in tokens { tf[token, default: 0] += 1 }
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

    private func weightedOverlap(queryTerms: [String], documentTerms: [String]) -> Double {
        guard !queryTerms.isEmpty, !documentTerms.isEmpty else { return 0 }
        let querySet = Set(queryTerms)
        let documentSet = Set(documentTerms)
        let overlap = querySet.intersection(documentSet).count
        let denominator = max(querySet.count, documentSet.count)
        return denominator > 0 ? Double(overlap) / Double(denominator) : 0
    }
}
