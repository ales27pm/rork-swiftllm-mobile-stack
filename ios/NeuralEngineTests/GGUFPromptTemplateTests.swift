import Testing
@testable import NeuralEngine

struct GGUFPromptTemplateTests {
    @Test func buildGGUFPrompt_lfm25PrependsStartOfTextAndUsesChatMLBlocks() {
        let messages: [[String: String]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "Say hi."]
        ]

        let prompt = InferenceEngine.buildGGUFPrompt(messages: messages, style: .lfm25)

        #expect(prompt.hasPrefix("<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"))
        #expect(prompt.contains("<|im_start|>user\nSay hi.<|im_end|>\n"))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test func manifest_usesLFM25TemplateStyleForLFMArchitecture() {
        let manifest = ModelManifest(
            id: "lfm2.5-test",
            name: "LFM2.5",
            variant: "Q4",
            parameterCount: "1.2B",
            quantization: "Q4_K_M",
            sizeBytes: 1_000,
            contextLength: 32_768,
            architecture: .lfm2,
            repoID: "unsloth/LFM2.5-1.2B-Instruct-GGUF",
            tokenizerRepoID: "LiquidAI/LFM2.5-1.2B-Instruct",
            modelFilePattern: "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            isDraft: false,
            format: .gguf
        )

        #expect(manifest.ggufChatTemplateStyle == .lfm25)
    }
}
