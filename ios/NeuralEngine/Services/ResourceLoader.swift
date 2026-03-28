import Foundation

enum ResourceLoader {
    static func load<T: Decodable>(_ type: T.Type, from filename: String) -> T? {
        guard let url = Bundle.neuralEngineResources.url(forResource: filename, withExtension: "json") else { return nil }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }
}

nonisolated struct EmotionLexiconEntry: Codable, Sendable {
    let pattern: String
    let valence: Double
    let arousal: Double
    let label: String
}

nonisolated struct StylePatternEntry: Codable, Sendable {
    let pattern: String
    let style: String
}

nonisolated struct IntentPatternEntry: Codable, Sendable {
    let pattern: String
    let intent: String
    let weight: Double
}

nonisolated struct CuriosityPatternEntry: Codable, Sendable {
    let pattern: String
    let weight: Double
}
