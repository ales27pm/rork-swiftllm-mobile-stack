import Testing
@testable import NeuralEngine

struct SettingsViewTests {
    @Test func samplingDraftRoundTripsSamplingConfig() {
        let config = SamplingConfig(
            temperature: 1.25,
            topK: 72,
            topP: 0.88,
            repetitionPenalty: 1.35,
            maxTokens: 4096,
            stopSequences: ["END"],
            samplerSeed: 42
        )

        let draft = SamplingDraft(config: config)
        let rebuilt = draft.makeConfig()

        #expect(rebuilt.temperature == 1.25)
        #expect(rebuilt.topK == 72)
        #expect(rebuilt.topP == 0.88)
        #expect(rebuilt.repetitionPenalty == 1.35)
        #expect(rebuilt.maxTokens == 4096)
        #expect(rebuilt.stopSequences.isEmpty)
        #expect(rebuilt.samplerSeed == nil)
    }

    @Test func toolCategoriesExposeExpectedSections() {
        #expect(SettingsToolCategory.defaults.count == 8)
        #expect(SettingsToolCategory.defaults.map(\.title).contains("Location & Maps"))
        #expect(SettingsToolCategory.defaults.map(\.title).contains("Sharing & Messaging"))
    }
}
