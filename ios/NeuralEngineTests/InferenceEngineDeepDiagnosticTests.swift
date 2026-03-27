import Foundation
import Testing
@testable import NeuralEngine

@MainActor
struct InferenceEngineDeepDiagnosticTests {
    @Test func ggufGenerationProducesTokensAndResetsFlag() async {
        let runner = LlamaModelRunner()
        runner.installSyntheticModelForTesting(
            configuration: LlamaSyntheticTestingConfiguration(
                plannedTokens: [4, 5, 6, 0],
                eogTokens: [0],
                tokenPieces: [4: "A", 5: "B", 6: "C"],
                decodeDelaySeconds: 0.02,
                vocabSize: 16
            )
        )

        let engine = InferenceEngine(
            metricsLogger: MetricsLogger(),
            thermalGovernor: ThermalGovernor()
        )
        engine.attachRunner(
            CoreMLModelRunner(),
            llamaRunner: runner,
            draftLlamaRunner: nil,
            tokenizer: TokenizerService(),
            format: .gguf
        )

        var concatenated = ""
        var observedBusyDuringCompletion: Bool? = nil
        let completion = CompletionBox()

        engine.generate(
            messages: [["role": "user", "content": "Say ABC"]],
            systemPrompt: "",
            samplingConfig: SamplingConfig(
                temperature: 0.7,
                topK: 8,
                topP: 1.0,
                repetitionPenalty: 1.0,
                maxTokens: 8,
                stopSequences: [],
                samplerSeed: 42
            ),
            onToken: { token in
                concatenated += token
            },
            onComplete: { metrics in
                observedBusyDuringCompletion = engine.isGenerating
                Task {
                    await completion.store(metrics)
                }
            }
        )

        let finished = await waitUntil(timeout: .seconds(5)) {
            await completion.hasValue
        }
        #expect(finished)
        let metrics = await completion.value
        #expect(observedBusyDuringCompletion == false)
        #expect((metrics?.totalTokens ?? 0) > 0)
        #expect(concatenated.count > 0)
    }
}

private actor CompletionBox {
    private var stored: GenerationMetrics?
    var hasValue: Bool { stored != nil }
    var value: GenerationMetrics? { stored }
    func store(_ metrics: GenerationMetrics?) {
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
