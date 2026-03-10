import Foundation

nonisolated struct PrefillResult: Sendable {
    let processedTokens: Int
    let durationSeconds: Double
    let pages: [KVPage]
}

nonisolated struct DecodeResult: Sendable {
    let token: Int
    let logits: [Float]
    let durationSeconds: Double
}

nonisolated struct VerificationResult: Sendable {
    let acceptedPrefix: Int
    let rejectedAt: Int?
    let correctedToken: Int
}

protocol PrefillEngine: Sendable {
    func prefill(inputIDs: [Int], cache: SessionCache) async throws -> (PrefillResult, SessionCache)
}

protocol DecodeEngine: Sendable {
    func decodeNext(token: Int, cache: SessionCache) async throws -> (DecodeResult, SessionCache)
}

protocol DraftEngine: Sendable {
    func propose(from token: Int, cache: SessionCache, count: Int) async throws -> ([Int], SessionCache)
}

protocol TargetVerifier: Sendable {
    func verify(proposed: [Int], cache: SessionCache) async throws -> (VerificationResult, SessionCache)
}
