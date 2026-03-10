import Foundation

nonisolated struct KVPage: Identifiable, Sendable {
    let id: UUID
    let tokenStart: Int
    let tokenCount: Int
    let layerCount: Int
    let pageSize: Int
    var isActive: Bool

    init(id: UUID = UUID(), tokenStart: Int, tokenCount: Int, layerCount: Int, pageSize: Int = 128, isActive: Bool = true) {
        self.id = id
        self.tokenStart = tokenStart
        self.tokenCount = tokenCount
        self.layerCount = layerCount
        self.pageSize = pageSize
        self.isActive = isActive
    }

    var tokenEnd: Int { tokenStart + tokenCount }
    var memoryEstimateBytes: Int { layerCount * pageSize * 2 * MemoryLayout<Float>.size }
}
