import Foundation
import Accelerate

@MainActor
class VectorStore {
    private let database: DatabaseService
    private let embedder: VectorEmbeddingService
    private var index: HNSWIndex
    private var idToVector: [String: [Float]] = [:]
    private var loaded = false
    private let exactSearchThreshold = 2_048
    private let approximateCandidateMultiplier = 8

    init(database: DatabaseService, embedder: VectorEmbeddingService = .shared) {
        self.database = database
        self.embedder = embedder
        self.index = HNSWIndex(dimensions: VectorEmbeddingService.dimensions)
        createTable()
    }

    private func createTable() {
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                source_text TEXT NOT NULL DEFAULT '',
                augmentation_text TEXT,
                provider TEXT NOT NULL DEFAULT 'natural_language',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        migrateTableIfNeeded()
    }

    private func migrateTableIfNeeded() {
        let rows = database.query("PRAGMA table_info(vector_embeddings);")
        let columns = Set(rows.compactMap { $0["name"] as? String })

        if !columns.contains("source_text") {
            _ = database.execute("ALTER TABLE vector_embeddings ADD COLUMN source_text TEXT NOT NULL DEFAULT '';")
        }
        if !columns.contains("augmentation_text") {
            _ = database.execute("ALTER TABLE vector_embeddings ADD COLUMN augmentation_text TEXT;")
        }
        if !columns.contains("provider") {
            _ = database.execute("ALTER TABLE vector_embeddings ADD COLUMN provider TEXT NOT NULL DEFAULT 'natural_language';")
        }
    }

    func loadIndex() {
        guard !loaded else { return }
        let rows = database.query("SELECT id, vector, dimensions FROM vector_embeddings;")
        for row in rows {
            guard let id = row["id"] as? String,
                  let blob = row["vector"] as? Data,
                  let dims = row["dimensions"] as? Int64 else { continue }
            let rawVector = blobToFloats(blob, count: Int(dims))
            guard let vector = sanitizedVector(rawVector) else { continue }
            idToVector[id] = vector
            index.insert(id: id, vector: vector)
        }
        loaded = true
    }

    func upsert(id: String, text: String, languageHint: String? = nil) -> Bool {
        upsert(
            id: id,
            sourceText: text,
            augmentationText: nil,
            provider: .naturalLanguage,
            languageHint: languageHint
        )
    }

    func upsert(
        id: String,
        sourceText: String,
        augmentationText: String?,
        provider: EmbeddingProvider,
        languageHint: String? = nil
    ) -> Bool {
        ensureLoaded()

        let compositeText = [sourceText, augmentationText]
            .compactMap { part -> String? in
                guard let part else { return nil }
                let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : trimmed
            }
            .joined(separator: " ")

        guard let embedded = embedder.embed(compositeText, languageHint: languageHint),
              let vector = sanitizedVector(embedded) else { return false }

        let blob = floatsToBlob(vector)
        let now = Date().timeIntervalSince1970 * 1000
        let success = database.execute(
            "INSERT OR REPLACE INTO vector_embeddings (id, vector, dimensions, source_text, augmentation_text, provider, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            params: [id, blob, VectorEmbeddingService.dimensions, sourceText, augmentationText ?? NSNull(), provider.rawValue, now, now]
        )
        guard success else { return false }

        if idToVector[id] != nil {
            index.remove(id: id)
        }
        idToVector[id] = vector
        index.insert(id: id, vector: vector)
        return true
    }

    func upsertWithVector(id: String, vector: [Float]) -> Bool {
        ensureLoaded()
        guard let normalizedVector = sanitizedVector(vector) else { return false }
        let blob = floatsToBlob(normalizedVector)
        let now = Date().timeIntervalSince1970 * 1000
        let success = database.execute(
            "INSERT OR REPLACE INTO vector_embeddings (id, vector, dimensions, source_text, augmentation_text, provider, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            params: [id, blob, VectorEmbeddingService.dimensions, "", NSNull(), EmbeddingProvider.externalVector.rawValue, now, now]
        )
        guard success else { return false }
        if idToVector[id] != nil {
            index.remove(id: id)
        }
        idToVector[id] = normalizedVector
        index.insert(id: id, vector: normalizedVector)
        return true
    }

    func delete(id: String) {
        ensureLoaded()
        _ = database.execute("DELETE FROM vector_embeddings WHERE id = ?;", params: [id])
        idToVector.removeValue(forKey: id)
        index.remove(id: id)
    }

    func search(query: String, maxResults: Int = 10, minScore: Float = 0.0, allowedIDs: Set<String>? = nil, languageHint: String? = nil) -> [VectorSearchResult] {
        ensureLoaded()
        guard let queryVec = embedder.embed(query, languageHint: languageHint) else { return [] }
        return searchByVector(queryVec, maxResults: maxResults, minScore: minScore, allowedIDs: allowedIDs)
    }

    func searchByVector(_ queryVector: [Float], maxResults: Int = 10, minScore: Float = 0.0, allowedIDs: Set<String>? = nil) -> [VectorSearchResult] {
        ensureLoaded()
        guard let normalizedQuery = sanitizedVector(queryVector), !idToVector.isEmpty else { return [] }

        if let allowedIDs {
            let filteredCandidateIDs = Array(allowedIDs.filter { idToVector[$0] != nil })
            guard !filteredCandidateIDs.isEmpty else { return [] }
            return exactSearch(queryVector: normalizedQuery, candidateIDs: filteredCandidateIDs, maxResults: maxResults, minScore: minScore)
        }

        if idToVector.count <= exactSearchThreshold {
            return exactSearch(queryVector: normalizedQuery, candidateIDs: Array(idToVector.keys), maxResults: maxResults, minScore: minScore)
        }

        return approximateSearchByVector(normalizedQuery, maxResults: maxResults, minScore: minScore, queryIsNormalized: true, allowedIDs: nil, fallbackToExact: true)
    }

    func approximateSearchByVector(_ queryVector: [Float], maxResults: Int = 10, minScore: Float = 0.0, allowedIDs: Set<String>? = nil) -> [VectorSearchResult] {
        ensureLoaded()
        guard !idToVector.isEmpty else { return [] }
        return approximateSearchByVector(queryVector, maxResults: maxResults, minScore: minScore, queryIsNormalized: false, allowedIDs: allowedIDs, fallbackToExact: false)
    }

    func getVector(for id: String) -> [Float]? {
        ensureLoaded()
        return idToVector[id]
    }

    func metadata(for id: String) -> StoredEmbeddingMetadata? {
        ensureLoaded()
        guard let row = database.query(
            "SELECT id, source_text, augmentation_text, provider, updated_at FROM vector_embeddings WHERE id = ? LIMIT 1;",
            params: [id]
        ).first,
        let rowID = row["id"] as? String,
        let sourceText = row["source_text"] as? String,
        let providerRaw = row["provider"] as? String,
        let provider = EmbeddingProvider(rawValue: providerRaw) else {
            return nil
        }

        let augmentationText: String?
        if let rawValue = row["augmentation_text"] as? String, !rawValue.isEmpty {
            augmentationText = rawValue
        } else {
            augmentationText = nil
        }

        let updatedAt = row["updated_at"] as? Double ?? 0
        return StoredEmbeddingMetadata(
            id: rowID,
            sourceText: sourceText,
            augmentationText: augmentationText,
            provider: provider,
            updatedAt: updatedAt
        )
    }

    func hasVector(for id: String) -> Bool {
        ensureLoaded()
        return idToVector[id] != nil
    }

    var count: Int {
        ensureLoaded()
        return idToVector.count
    }

    var indexStats: VectorIndexStats {
        ensureLoaded()
        return VectorIndexStats(
            totalVectors: idToVector.count,
            dimensions: VectorEmbeddingService.dimensions,
            indexNodes: index.nodeCount,
            indexLayers: index.layerCount,
            memoryBytes: idToVector.count * VectorEmbeddingService.dimensions * MemoryLayout<Float>.size
        )
    }

    func clearAll() {
        _ = database.execute("DELETE FROM vector_embeddings;")
        idToVector.removeAll()
        index = HNSWIndex(dimensions: VectorEmbeddingService.dimensions)
        loaded = true
    }

    private func ensureLoaded() {
        if !loaded { loadIndex() }
    }

    private func exactSearch(queryVector: [Float], candidateIDs: [String], maxResults: Int, minScore: Float) -> [VectorSearchResult] {
        rerank(candidateIDs: candidateIDs, queryVector: queryVector, maxResults: maxResults, minScore: minScore)
    }

    private func approximateSearchByVector(_ queryVector: [Float], maxResults: Int, minScore: Float, queryIsNormalized: Bool, allowedIDs: Set<String>?, fallbackToExact: Bool) -> [VectorSearchResult] {
        let normalizedQuery: [Float]
        if queryIsNormalized {
            normalizedQuery = queryVector
        } else {
            guard let sanitizedQuery = sanitizedVector(queryVector) else { return [] }
            normalizedQuery = sanitizedQuery
        }

        let candidateCount = min(idToVector.count, max(maxResults * approximateCandidateMultiplier, maxResults * 4, 32))
        let approximate = index.search(query: normalizedQuery, k: candidateCount)
        let approximateCandidateIDs: [String]
        if let allowedIDs {
            approximateCandidateIDs = approximate.map(\.id).filter { allowedIDs.contains($0) }
        } else {
            approximateCandidateIDs = approximate.map(\.id)
        }

        let reranked = rerank(candidateIDs: approximateCandidateIDs, queryVector: normalizedQuery, maxResults: maxResults, minScore: minScore)

        guard fallbackToExact else {
            return reranked
        }

        if reranked.count >= maxResults || approximate.count == idToVector.count {
            return reranked
        }

        let fallbackCandidateIDs: [String]
        if let allowedIDs {
            fallbackCandidateIDs = allowedIDs.filter { idToVector[$0] != nil }
        } else {
            fallbackCandidateIDs = Array(idToVector.keys)
        }
        return exactSearch(queryVector: normalizedQuery, candidateIDs: fallbackCandidateIDs, maxResults: maxResults, minScore: minScore)
    }

    private func rerank(candidateIDs: [String], queryVector: [Float], maxResults: Int, minScore: Float) -> [VectorSearchResult] {
        var results: [VectorSearchResult] = []
        var seen: Set<String> = []

        for id in candidateIDs {
            guard seen.insert(id).inserted,
                  let stored = idToVector[id] else { continue }
            let similarity = embedder.cosineSimilarity(queryVector, stored)
            guard similarity >= minScore else { continue }
            results.append(VectorSearchResult(id: id, score: similarity))
        }

        results.sort { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.id < rhs.id
            }
            return lhs.score > rhs.score
        }
        return Array(results.prefix(maxResults))
    }

    private func sanitizedVector(_ vector: [Float]) -> [Float]? {
        guard !vector.isEmpty else { return nil }

        let resized: [Float]
        if vector.count == VectorEmbeddingService.dimensions {
            resized = vector
        } else if vector.count > VectorEmbeddingService.dimensions {
            resized = Array(vector.prefix(VectorEmbeddingService.dimensions))
        } else {
            resized = vector + [Float](repeating: 0, count: VectorEmbeddingService.dimensions - vector.count)
        }

        guard resized.allSatisfy(\.isFinite) else { return nil }
        return normalize(resized)
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        var magnitudeSquared: Float = 0
        vDSP_svesq(vector, 1, &magnitudeSquared, vDSP_Length(vector.count))
        let magnitude = sqrtf(magnitudeSquared)
        guard magnitude > 0 else { return vector }

        var normalized = [Float](repeating: 0, count: vector.count)
        var divisor = magnitude
        vDSP_vsdiv(vector, 1, &divisor, &normalized, 1, vDSP_Length(vector.count))
        return normalized
    }

    private func floatsToBlob(_ floats: [Float]) -> Data {
        floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func blobToFloats(_ data: Data, count: Int) -> [Float] {
        guard data.count >= count * MemoryLayout<Float>.size else { return [] }
        return data.withUnsafeBytes { ptr in
            let bound = ptr.bindMemory(to: Float.self)
            return Array(bound.prefix(count))
        }
    }
}

nonisolated struct VectorSearchResult: Sendable {
    let id: String
    let score: Float
}

nonisolated struct VectorIndexStats: Sendable {
    let totalVectors: Int
    let dimensions: Int
    let indexNodes: Int
    let indexLayers: Int
    let memoryBytes: Int

    var memoryMB: Double { Double(memoryBytes) / (1024 * 1024) }
}

final class HNSWIndex: @unchecked Sendable {
    private struct Node {
        let id: String
        let vector: [Float]
        var neighbors: [[String]]
    }

    private let dimensions: Int
    private let maxConnections: Int
    private let efConstruction: Int
    private let maxLevel: Int
    private let levelMultiplier: Double

    private var nodes: [String: Node] = [:]
    private var entryPoint: String?
    private var topLevel: Int = 0

    init(dimensions: Int, maxConnections: Int = 16, efConstruction: Int = 100) {
        self.dimensions = dimensions
        self.maxConnections = maxConnections
        self.efConstruction = efConstruction
        self.maxLevel = 6
        self.levelMultiplier = 1.0 / log(Double(maxConnections))
    }

    var nodeCount: Int { nodes.count }
    var layerCount: Int { topLevel + 1 }

    func insert(id: String, vector: [Float]) {
        let level = randomLevel()
        var node = Node(id: id, vector: vector, neighbors: Array(repeating: [], count: level + 1))

        guard let ep = entryPoint, nodes[ep] != nil else {
            nodes[id] = node
            entryPoint = id
            topLevel = level
            return
        }

        var currentEP = ep
        for l in stride(from: topLevel, through: level + 1, by: -1) {
            let nearest = searchLayer(query: vector, entryId: currentEP, level: l, ef: 1)
            if let best = nearest.first { currentEP = best.id }
        }

        for l in stride(from: min(level, topLevel), through: 0, by: -1) {
            let candidates = searchLayer(query: vector, entryId: currentEP, level: l, ef: efConstruction)
            let neighbors = selectNeighbors(candidates: candidates, maxCount: maxConnections)

            node.neighbors[l] = neighbors.map(\.id)

            for neighbor in neighbors {
                guard var nNode = nodes[neighbor.id] else { continue }
                while nNode.neighbors.count <= l { nNode.neighbors.append([]) }
                nNode.neighbors[l].append(id)
                if nNode.neighbors[l].count > maxConnections * 2 {
                    let kept = pruneNeighbors(nodeId: neighbor.id, neighbors: nNode.neighbors[l], level: l)
                    nNode.neighbors[l] = kept
                }
                nodes[neighbor.id] = nNode
            }

            if let best = candidates.first { currentEP = best.id }
        }

        nodes[id] = node

        if level > topLevel {
            topLevel = level
            entryPoint = id
        }
    }

    func remove(id: String) {
        guard let node = nodes[id] else { return }
        for (level, neighborIds) in node.neighbors.enumerated() {
            for nId in neighborIds {
                guard var nNode = nodes[nId] else { continue }
                if level < nNode.neighbors.count {
                    nNode.neighbors[level].removeAll { $0 == id }
                    nodes[nId] = nNode
                }
            }
        }
        nodes.removeValue(forKey: id)
        if entryPoint == id {
            entryPoint = nodes.keys.first
            topLevel = nodes.values.map { $0.neighbors.count - 1 }.max() ?? 0
        }
    }

    func search(query: [Float], k: Int) -> [VectorSearchResult] {
        guard let ep = entryPoint else { return [] }

        var currentEP = ep
        for l in stride(from: topLevel, through: 1, by: -1) {
            let nearest = searchLayer(query: query, entryId: currentEP, level: l, ef: 1)
            if let best = nearest.first { currentEP = best.id }
        }

        let ef = max(k, efConstruction)
        let candidates = searchLayer(query: query, entryId: currentEP, level: 0, ef: ef)
        return Array(candidates.prefix(k))
    }

    private func searchLayer(query: [Float], entryId: String, level: Int, ef: Int) -> [VectorSearchResult] {
        guard let entryNode = nodes[entryId] else { return [] }
        let entryDist = distance(query, entryNode.vector)

        var visited: Set<String> = [entryId]
        var candidates: [(id: String, dist: Float)] = [(entryId, entryDist)]
        var results: [(id: String, dist: Float)] = [(entryId, entryDist)]

        while !candidates.isEmpty {
            candidates.sort { $0.dist < $1.dist }
            let current = candidates.removeFirst()

            let farthestResult = results.max(by: { $0.dist < $1.dist })?.dist ?? Float.greatestFiniteMagnitude
            if current.dist > farthestResult && results.count >= ef { break }

            guard let currentNode = nodes[current.id], level < currentNode.neighbors.count else { continue }

            for neighborId in currentNode.neighbors[level] {
                guard !visited.contains(neighborId) else { continue }
                visited.insert(neighborId)

                guard let neighborNode = nodes[neighborId] else { continue }
                let dist = distance(query, neighborNode.vector)

                let worstResult = results.max(by: { $0.dist < $1.dist })?.dist ?? Float.greatestFiniteMagnitude
                if results.count < ef || dist < worstResult {
                    candidates.append((neighborId, dist))
                    results.append((neighborId, dist))
                    if results.count > ef {
                        results.sort { $0.dist < $1.dist }
                        results.removeLast()
                    }
                }
            }
        }

        results.sort { $0.dist < $1.dist }
        return results.map { VectorSearchResult(id: $0.id, score: max(0, 1.0 - $0.dist)) }
    }

    private func selectNeighbors(candidates: [VectorSearchResult], maxCount: Int) -> [VectorSearchResult] {
        Array(candidates.sorted { $0.score > $1.score }.prefix(maxCount))
    }

    private func pruneNeighbors(nodeId: String, neighbors: [String], level: Int) -> [String] {
        guard let node = nodes[nodeId] else { return Array(neighbors.prefix(maxConnections)) }
        var scored: [(id: String, dist: Float)] = []
        for nId in neighbors {
            guard let nNode = nodes[nId] else { continue }
            scored.append((nId, distance(node.vector, nNode.vector)))
        }
        scored.sort { $0.dist < $1.dist }
        return scored.prefix(maxConnections).map(\.id)
    }

    private func distance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 2.0 }
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        return max(0, 1.0 - dot)
    }

    private func randomLevel() -> Int {
        var level = 0
        while Double.random(in: 0..<1) < exp(-Double(level + 1) * levelMultiplier) && level < maxLevel {
            level += 1
        }
        return level
    }
}
