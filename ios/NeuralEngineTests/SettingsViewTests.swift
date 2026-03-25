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

    @Test func samplingDraftResetRestoresDefaults() {
        var draft = SamplingDraft(
            temperature: 1.6,
            topK: 90,
            topP: 0.5,
            repetitionPenalty: 1.8,
            maxTokens: 1024
        )

        draft.resetToDefaults()

        #expect(draft == SamplingDraft.defaults)
        #expect(draft.makeConfig().temperature == SamplingConfig().temperature)
        #expect(draft.makeConfig().topK == SamplingConfig().topK)
        #expect(draft.makeConfig().topP == SamplingConfig().topP)
        #expect(draft.makeConfig().repetitionPenalty == SamplingConfig().repetitionPenalty)
        #expect(draft.makeConfig().maxTokens == SamplingConfig().maxTokens)
    }

    @Test func settingsDestinationsRemainHashable() {
        let destinations: Set<SettingsDestination> = [.personas, .customPrompt, .models, .weather, .diagnostics]

        #expect(destinations.count == 5)
        #expect(destinations.contains(.diagnostics))
    }
}
