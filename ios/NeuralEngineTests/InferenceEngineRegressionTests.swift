import Foundation
import Testing
@testable import NeuralEngine

@MainActor
struct InferenceEngineRegressionTests {
    @Test func generate_whenBusy_stillCompletes() async {
        let engine = InferenceEngine(metricsLogger: MetricsLogger(), thermalGovernor: ThermalGovernor())
        engine.isGenerating = true

        let completion = CompletionBox()
        engine.generate(
            messages: [["role": "user", "content": "Hello"]],
            systemPrompt: "",
            samplingConfig: SamplingConfig(),
            onToken: { _ in },
            onComplete: { metrics in
                Task {
                    await completion.store(metrics)
                }
            }
        )

        let completed = await waitUntil(timeout: .seconds(1)) {
            await completion.hasValue
        }
        #expect(completed)
        let metrics = await completion.value
        #expect(metrics?.fallbackMode == "engineBusy")
    }

    @Test func ggufCancellation_preservesPartialMetrics() async {
        let runner = LlamaModelRunner()
        runner.installSyntheticModelForTesting(
            configuration: LlamaSyntheticTestingConfiguration(
                plannedTokens: [4, 5, 6, 7, 0],
                eogTokens: [0],
                tokenPieces: [4: "a", 5: "b", 6: "c", 7: "d"],
                decodeDelaySeconds: 0.03,
                vocabSize: 16
            )
        )

        let engine = InferenceEngine(metricsLogger: MetricsLogger(), thermalGovernor: ThermalGovernor())
        engine.attachRunner(
            CoreMLModelRunner(),
            llamaRunner: runner,
            draftLlamaRunner: nil,
            tokenizer: TokenizerService(),
            format: .gguf
        )

        let completion = CompletionBox()
        engine.generate(
            messages: [["role": "user", "content": "Explain this quickly."]],
            systemPrompt: "",
            samplingConfig: SamplingConfig(
                temperature: 0.7,
                topK: 8,
                topP: 1.0,
                repetitionPenalty: 1.0,
                maxTokens: 16,
                stopSequences: [],
                samplerSeed: 42
            ),
            onToken: { _ in },
            onComplete: { metrics in
                Task {
                    await completion.store(metrics)
                }
            }
        )

        let enteredDecode = await waitUntil(timeout: .seconds(5)) {
            runner.isSyntheticDecodeActiveForTesting()
        }
        #expect(enteredDecode)
        guard enteredDecode else { return }

        try? await Task.sleep(for: .milliseconds(80))
        engine.cancel(reason: "testCancel")

        let completed = await waitUntil(timeout: .seconds(5)) {
            await completion.hasValue
        }
        #expect(completed)
        let metrics = await completion.value
        #expect(metrics?.fallbackMode == "generationCancelled")
        #expect((metrics?.totalTokens ?? 0) > 0)
        #expect((metrics?.timeToFirstToken ?? 0) >= 0)
    }
}

private actor CompletionBox {
    private var stored: GenerationMetrics?

    var hasValue: Bool { stored != nil }
    var value: GenerationMetrics? { stored }

    func store(_ metrics: GenerationMetrics) {
        stored = metrics
    }
}

private func waitUntil(
    timeout: Duration = .seconds(3),
    interval: Duration = .milliseconds(10),
    _ condition: @escaping @Sendable () async -> Bool
) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)

    while clock.now < deadline {
        if await condition() {
            return true
        }
        try? await Task.sleep(for: interval)
    }

    return await condition()
}
