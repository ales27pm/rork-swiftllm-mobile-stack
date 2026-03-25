import Foundation
import Testing
@testable import NeuralEngine

struct EmbeddingRunnerConcurrencyTests {
    @Test
    func unloadDuringActiveSyntheticEmbed_waitsAndBlocksNewWork() async {
        let runner = EmbeddingModelRunner()
        runner.installSyntheticModelForTesting(dimensions: 16, embedDelaySeconds: 0.25)

        let firstEmbedTask = Task.detached(priority: .userInitiated) {
            runner.embed("first embedding payload")
        }

        let enteredEmbed = await waitUntilForEmbeddingTest {
            runner.isSyntheticEmbedActiveForTesting()
        }
        #expect(enteredEmbed)
        guard enteredEmbed else { return }

        let startedAt = Date()
        let unloadTask = Task.detached(priority: .userInitiated) {
            runner.unload()
        }

        try? await Task.sleep(for: .milliseconds(40))
        let blockedEmbed = await Task.detached(priority: .userInitiated) {
            runner.embed("should be rejected while unload drains")
        }.value

        let firstResult = await firstEmbedTask.value
        _ = await unloadTask.value
        let elapsed = Date().timeIntervalSince(startedAt)

        #expect(firstResult != nil)
        #expect(blockedEmbed == nil)
        #expect(elapsed >= 0.15)
        #expect(!runner.isLoaded)
        #expect(runner.syntheticEmbedCallCountForTesting() == 1)
    }

    @Test
    func concurrentSyntheticEmbeds_serializeInsteadOfOverlapping() async {
        let runner = EmbeddingModelRunner()
        runner.installSyntheticModelForTesting(dimensions: 12, embedDelaySeconds: 0.18)

        let firstStarted = Date()
        let firstTask = Task.detached(priority: .userInitiated) {
            runner.embed("alpha")
        }

        let enteredEmbed = await waitUntilForEmbeddingTest {
            runner.isSyntheticEmbedActiveForTesting()
        }
        #expect(enteredEmbed)
        guard enteredEmbed else { return }

        let secondFinished = EmbeddingFlagBox()
        let secondTask = Task.detached(priority: .userInitiated) { () -> [Float]? in
            let result = runner.embed("beta")
            await secondFinished.setTrue()
            return result
        }

        try? await Task.sleep(for: .milliseconds(60))
        #expect(!(await secondFinished.value))

        let firstResult = await firstTask.value
        let firstElapsed = Date().timeIntervalSince(firstStarted)
        let secondResult = await secondTask.value

        #expect(firstResult != nil)
        #expect(secondResult != nil)
        #expect(firstElapsed >= 0.12)
        #expect(runner.syntheticEmbedCallCountForTesting() == 2)
        #expect(!runner.isSyntheticEmbedActiveForTesting())
    }
}

private actor EmbeddingFlagBox {
    private(set) var value: Bool = false

    func setTrue() {
        value = true
    }
}

private func waitUntilForEmbeddingTest(
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
