import Foundation

struct SamplingConfig: Sendable {
    var temperature: Float = 0.8
    var topK: Int = 40
    var topP: Float = 0.95
    var repetitionPenalty: Float = 1.1
    var maxTokens: Int = 2048
    var stopSequences: [String] = []
    var samplerSeed: UInt64? = nil
}
