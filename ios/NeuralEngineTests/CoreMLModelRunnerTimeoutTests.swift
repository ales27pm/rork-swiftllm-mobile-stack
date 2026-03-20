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

    @Test func restorePrefillState_succeedsForIdenticalPrefixAndContext() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("snapshot-success.mlmodelc")
        let runner = CoreMLModelRunner()
        let prefixTokens = [11, 22, 33]

        runner.completeSyntheticLoadForTesting(at: url, computeUnits: .cpuAndNeuralEngine)
        runner.configurePrefixSnapshotContext(modelID: "model-A", tokenizerID: "tokenizer-A")

        let exported = runner.installSyntheticPrefillStateForTesting(
            prefixTokens: prefixTokens,
            modelID: "model-A",
            tokenizerID: "tokenizer-A",
            computeUnits: .cpuAndNeuralEngine
        )

        guard case .available(let snapshot) = exported else {
            Issue.record("Expected synthetic snapshot export to succeed")
            return
        }

        #expect(runner.restorePrefillState(from: snapshot, expectedPrefixTokens: prefixTokens))
    }

    @Test func restorePrefillState_rejectsTokenMismatch() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("snapshot-token-mismatch.mlmodelc")
        let runner = CoreMLModelRunner()

        runner.completeSyntheticLoadForTesting(at: url, computeUnits: .cpuAndNeuralEngine)
        runner.configurePrefixSnapshotContext(modelID: "model-A", tokenizerID: "tokenizer-A")

        let exported = runner.installSyntheticPrefillStateForTesting(
            prefixTokens: [1, 2, 3],
            modelID: "model-A",
            tokenizerID: "tokenizer-A",
            computeUnits: .cpuAndNeuralEngine
        )

        guard case .available(let snapshot) = exported else {
            Issue.record("Expected synthetic snapshot export to succeed")
            return
        }

        #expect(!runner.restorePrefillState(from: snapshot, expectedPrefixTokens: [1, 2, 4]))
    }

    @Test func restorePrefillState_rejectsAfterReloadOrConfigMismatch() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("snapshot-reload-mismatch.mlmodelc")
        let runner = CoreMLModelRunner()
        let prefixTokens = [7, 8, 9]

        runner.completeSyntheticLoadForTesting(at: url, computeUnits: .cpuAndNeuralEngine)
        runner.configurePrefixSnapshotContext(modelID: "model-A", tokenizerID: "tokenizer-A")

        let exported = runner.installSyntheticPrefillStateForTesting(
            prefixTokens: prefixTokens,
            modelID: "model-A",
            tokenizerID: "tokenizer-A",
            computeUnits: .cpuAndNeuralEngine
        )

        guard case .available(let snapshot) = exported else {
            Issue.record("Expected synthetic snapshot export to succeed")
            return
        }

        runner.completeSyntheticLoadForTesting(at: url, computeUnits: .cpuOnly)
        runner.configurePrefixSnapshotContext(modelID: "model-B", tokenizerID: "tokenizer-B")

        #expect(!runner.restorePrefillState(from: snapshot, expectedPrefixTokens: prefixTokens))
    }

    @Test func restorePrefillState_unavailableSnapshotFallsBackToColdPrefill() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("snapshot-fallback.mlmodelc")
        let runner = CoreMLModelRunner()
        let prefixTokens = [42, 43]

        runner.completeSyntheticLoadForTesting(at: url, computeUnits: .cpuOnly)
        runner.configurePrefixSnapshotContext(modelID: "model-A", tokenizerID: "tokenizer-A")

        #expect(!runner.restorePrefillState(
            from: .unavailable(reason: "not cached"),
            expectedPrefixTokens: prefixTokens
        ))
    }
}
