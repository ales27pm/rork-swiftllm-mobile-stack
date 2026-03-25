#if DEBUG
import Foundation

nonisolated struct LlamaSyntheticTestingConfiguration: Sendable {
    let plannedTokens: [Int]
    let eogTokens: Set<Int>
    let tokenPieces: [Int: String]
    let decodeDelaySeconds: TimeInterval
    let unloadDelaySeconds: TimeInterval
    let loadDelaySeconds: TimeInterval
    let vocabSize: Int

    init(
        plannedTokens: [Int],
        eogTokens: Set<Int> = [0],
        tokenPieces: [Int: String] = [:],
        decodeDelaySeconds: TimeInterval = 0,
        unloadDelaySeconds: TimeInterval = 0,
        loadDelaySeconds: TimeInterval = 0,
        vocabSize: Int = 32
    ) {
        self.plannedTokens = plannedTokens
        self.eogTokens = eogTokens
        self.tokenPieces = tokenPieces
        self.decodeDelaySeconds = decodeDelaySeconds
        self.unloadDelaySeconds = unloadDelaySeconds
        self.loadDelaySeconds = loadDelaySeconds
        self.vocabSize = vocabSize
    }
}

nonisolated private struct LlamaSyntheticSerializedState: Codable, Sendable {
    let generationCursor: Int
    let decodedTokens: [Int]
}

nonisolated final class LlamaSyntheticTestingState: @unchecked Sendable {
    let configuration: LlamaSyntheticTestingConfiguration

    private let lock = NSLock()
    private var generationCursor: Int = 0
    private var decodedTokens: [Int] = []
    private var activeDecodeCount: Int = 0
    private var decodeCallCountValue: Int = 0

    init(configuration: LlamaSyntheticTestingConfiguration) {
        self.configuration = configuration
    }

    var hasActiveDecode: Bool {
        lock.lock()
        defer { lock.unlock() }
        return activeDecodeCount > 0
    }

    var decodeCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return decodeCallCountValue
    }

    var generationCursorValue: Int {
        lock.lock()
        defer { lock.unlock() }
        return generationCursor
    }

    func beginDecode() {
        lock.lock()
        activeDecodeCount += 1
        decodeCallCountValue += 1
        lock.unlock()
    }

    func endDecode() {
        lock.lock()
        activeDecodeCount = max(activeDecodeCount - 1, 0)
        lock.unlock()
    }

    func recordDecodedToken(_ token: Int) {
        lock.lock()
        decodedTokens.append(token)
        if configuration.plannedTokens.indices.contains(generationCursor), token == configuration.plannedTokens[generationCursor] {
            generationCursor += 1
        }
        lock.unlock()
    }

    func resetForNewSequence() {
        lock.lock()
        generationCursor = 0
        decodedTokens.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    func serializedState() -> Data {
        lock.lock()
        let snapshot = LlamaSyntheticSerializedState(
            generationCursor: generationCursor,
            decodedTokens: decodedTokens
        )
        lock.unlock()
        return (try? JSONEncoder().encode(snapshot)) ?? Data()
    }

    func restore(from data: Data) -> Bool {
        guard let snapshot = try? JSONDecoder().decode(LlamaSyntheticSerializedState.self, from: data) else {
            return false
        }
        lock.lock()
        generationCursor = snapshot.generationCursor
        decodedTokens = snapshot.decodedTokens
        lock.unlock()
        return true
    }
}
#endif
