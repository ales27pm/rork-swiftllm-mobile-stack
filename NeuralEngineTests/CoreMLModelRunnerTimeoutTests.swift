import Foundation
import CoreML
import Testing
@testable import NeuralEngine

struct CoreMLModelRunnerTimeoutTests {
    @Test func loadModelWithTimeout_waitsForCancelledLoadAndPreventsLateReadyState() async throws {
        let fakeURL = FileManager.default.temporaryDirectory.appendingPathComponent("timeout-fixture.mlmodelc")
        let runner = CoreMLModelRunner { runner, url, computeUnits in
            try? await Task.sleep(for: .milliseconds(200))
            runner.completeSyntheticLoadForTesting(at: url, computeUnits: computeUnits)
        }

        do {
            try await runner.loadModelWithTimeout(at: fakeURL, computeUnits: .cpuOnly, timeoutSeconds: 0.05)
            Issue.record("Expected loadModelWithTimeout to throw a timeout error")
        } catch let error as CoreMLRunnerError {
            switch error {
            case .loadTimeout(let seconds):
                #expect(seconds == 0.05)
            default:
                Issue.record("Expected load timeout error, got \(error)")
            }
        }

        #expect(runner.currentState == .idle)
        #expect(!runner.isLoaded)

        try await Task.sleep(for: .milliseconds(300))

        #expect(runner.currentState == .idle)
        #expect(!runner.isLoaded)
    }
}
