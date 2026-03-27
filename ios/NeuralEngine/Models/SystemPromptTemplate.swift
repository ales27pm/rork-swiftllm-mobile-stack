import Foundation

nonisolated struct SystemPromptTemplate: Identifiable, Codable, Sendable, Equatable {
    let id: String
    let name: String
    let icon: String
    let description: String
    let prompt: String
    let accentColorName: String

    static let builtIn: [SystemPromptTemplate] = {
        ResourceLoader.load([SystemPromptTemplate].self, from: "system_prompts") ?? []
    }()
}
