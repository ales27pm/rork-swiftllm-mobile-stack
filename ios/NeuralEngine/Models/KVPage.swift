import Foundation

nonisolated struct KVPage: Identifiable, Sendable {
    let id: UUID
    let tokenStart: Int
    let tokenCount: Int
    let layerCount: Int
    let pageSize: Int
    var isActive: Bool
    var referenceCount: Int
    var isDirty: Bool
    var lastAccessDate: Date
    var creationDate: Date
    var sequenceID: UUID?

    init(
        id: UUID = UUID(),
        tokenStart: Int,
        tokenCount: Int,
        layerCount: Int,
        pageSize: Int = 128,
        isActive: Bool = true,
        sequenceID: UUID? = nil
    ) {
        self.id = id
        self.tokenStart = tokenStart
        self.tokenCount = tokenCount
        self.layerCount = layerCount
        self.pageSize = pageSize
        self.isActive = isActive
        self.referenceCount = 1
        self.isDirty = true
        self.lastAccessDate = Date()
        self.creationDate = Date()
        self.sequenceID = sequenceID
    }

    var tokenEnd: Int { tokenStart + tokenCount }

    var memoryEstimateBytes: Int {
        layerCount * pageSize * 2 * MemoryLayout<Float>.size
    }

    var ageSeconds: TimeInterval {
        Date().timeIntervalSince(creationDate)
    }

    var idleSeconds: TimeInterval {
        Date().timeIntervalSince(lastAccessDate)
    }

    var isShared: Bool { referenceCount > 1 }

    mutating func touch() {
        lastAccessDate = Date()
    }

    mutating func retain() {
        referenceCount += 1
    }

    mutating func release() -> Bool {
        referenceCount = max(0, referenceCount - 1)
        return referenceCount == 0
    }

    mutating func markClean() {
        isDirty = false
    }
}

nonisolated enum KVPageState: Sendable {
    case active
    case frozen
    case evictable
    case freed
}
