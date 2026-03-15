import Foundation

nonisolated struct PrefillPhaseResult: Sendable {
    let processedTokens: Int
    let durationSeconds: Double
    let pages: [KVPage]
}

nonisolated struct DecodePhaseResult: Sendable {
    let token: Int
    let logits: [Float]
    let durationSeconds: Double
}

nonisolated struct VerificationResult: Sendable {
    let acceptedPrefix: Int
    let rejectedAt: Int?
    let correctedToken: Int
}

protocol PrefillEngineProtocol: Sendable {
    func prefill(inputIDs: [Int], cache: SessionCache) async throws -> (PrefillPhaseResult, SessionCache)
}

protocol DecodeEngineProtocol: Sendable {
    func decodeNext(token: Int, cache: SessionCache) async throws -> (DecodePhaseResult, SessionCache)
}

protocol DraftEngineProtocol: Sendable {
    func propose(from token: Int, cache: SessionCache, count: Int) async throws -> ([Int], SessionCache)
}

protocol TargetVerifier: Sendable {
    func verify(proposed: [Int], cache: SessionCache) async throws -> (VerificationResult, SessionCache)
}
