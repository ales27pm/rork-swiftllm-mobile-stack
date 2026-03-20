import Foundation
import Accelerate

@MainActor
class VectorStore {
    private let database: DatabaseService
    private let embedder: VectorEmbeddingService
    private var index: HNSWIndex
    private var idToVector: [String: [Float]] = [:]
    private var loaded = false

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
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
    }

    func loadIndex() {
        guard !loaded else { return }
        let rows = database.query("SELECT id, vector, dimensions FROM vector_embeddings;")
        for row in rows {
            guard let id = row["id"] as? String,
                  let blob = row["vector"] as? Data,
                  let dims = row["dimensions"] as? Int64 else { continue }
            let vector = blobToFloats(blob, count: Int(dims))
            idToVector[id] = vector
            index.insert(id: id, vector: vector)
        }
        loaded = true
    }

    func upsert(id: String, text: String, languageHint: String? = nil) -> Bool {
        ensureLoaded()
        guard let vector = embedder.embed(text, languageHint: languageHint) else { return false }
        let blob = floatsToBlob(vector)
        let now = Date().timeIntervalSince1970 * 1000
        let success = database.execute(
            "INSERT OR REPLACE INTO vector_embeddings (id, vector, dimensions, created_at, updated_at) VALUES (?, ?, ?, ?, ?);",
            params: [id, blob, VectorEmbeddingService.dimensions, now, now]
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
        let blob = floatsToBlob(vector)
        let now = Date().timeIntervalSince1970 * 1000
        let success = database.execute(
            "INSERT OR REPLACE INTO vector_embeddings (id, vector, dimensions, created_at, updated_at) VALUES (?, ?, ?, ?, ?);",
            params: [id, blob, vector.count, now, now]
        )
        guard success else { return false }
        if idToVector[id] != nil {
            index.remove(id: id)
        }
        idToVector[id] = vector
        index.insert(id: id, vector: vector)
        return true
    }

    func delete(id: String) {
        ensureLoaded()
        _ = database.execute("DELETE FROM vector_embeddings WHERE id = ?;", params: [id])
        idToVector.removeValue(forKey: id)
        index.remove(id: id)
    }

    func search(query: String, maxResults: Int = 10, minScore: Float = 0.0, languageHint: String? = nil) -> [VectorSearchResult] {
        ensureLoaded()
        guard let queryVec = embedder.embed(query, languageHint: languageHint) else { return [] }
        return searchByVector(queryVec, maxResults: maxResults, minScore: minScore)
    }

    func searchByVector(_ queryVector: [Float], maxResults: Int = 10, minScore: Float = 0.0) -> [VectorSearchResult] {
        ensureLoaded()
        let candidates = index.search(query: queryVector, k: maxResults * 2)
        var results: [VectorSearchResult] = []
        for candidate in candidates {
            guard let stored = idToVector[candidate.id] else { continue }
            let similarity = embedder.cosineSimilarity(queryVector, stored)
            guard similarity >= minScore else { continue }
            results.append(VectorSearchResult(id: candidate.id, score: similarity))
        }
        results.sort { $0.score > $1.score }
        return Array(results.prefix(maxResults))
    }

    func getVector(for id: String) -> [Float]? {
        ensureLoaded()
        return idToVector[id]
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

        guard let ep = entryPoint, let epNode = nodes[ep] else {
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
