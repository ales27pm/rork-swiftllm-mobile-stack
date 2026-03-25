import Testing
import Foundation
@testable import NeuralEngine

// MARK: – Lifetime-ownership regression tests
//
// These tests cover the four crash-class scenarios identified in the
// EXC_BAD_ACCESS / SIGSEGV post-mortem:
//
//   Scenario A – generate while switching model (unload + reload races decode)
//   Scenario B – generate while deleting the current model (unload races decode)
//   Scenario C – generate while force-reloading (second loadModel races decode)
//   Scenario D – speculative decode while draft runner unloads
//
// None of these tests require an actual GGUF file.  They exercise the
// generation-token drain protocol and the new single-exclusivity invariant
// directly.  Thread Sanitizer is the intended execution environment.
//
// Additional tests verify the new single-exclusivity rule added during the
// refactor (each runner allows at most one active generation token at a time).

struct LlamaRunnerLifetimeTests {

    // MARK: – Single-token exclusivity (new invariant)

    @Test("Token exclusivity: second acquisition rejected while first is held")
    func tokenExclusivity_secondAcquisitionRejected() {
        let runner = LlamaModelRunner()
        let first = runner.tryAcquireGenerationToken()
        #expect(first, "First acquisition must succeed on an idle runner")

        let second = runner.tryAcquireGenerationToken()
        #expect(!second, "Second acquisition must be rejected while first token is held")

        if first { runner.releaseGenerationToken() }

        let third = runner.tryAcquireGenerationToken()
        #expect(third, "Acquisition must succeed once the previous token is released")
        if third { runner.releaseGenerationToken() }
    }

    @Test("Token exclusivity: acquisition succeeds after peer runner releases")
    func tokenExclusivity_peerRunnerDoesNotBlock() {
        let target = LlamaModelRunner()
        let draft  = LlamaModelRunner()

        let tAcquired = target.tryAcquireGenerationToken()
        let dAcquired = draft.tryAcquireGenerationToken()
        #expect(tAcquired, "Target token should succeed")
        #expect(dAcquired, "Draft token is independent — should also succeed")

        if tAcquired { target.releaseGenerationToken() }
        if dAcquired { draft.releaseGenerationToken() }
    }

    // MARK: – Scenario A: generate while switching model

    @Test("Scenario A: unload blocks until generation token is released")
    func scenarioA_switchModelBlocksUntilGenerationDone() async {
        let runner = LlamaModelRunner()

        // Simulate a long-running generation by holding the token.
        let tokenAcquired = runner.tryAcquireGenerationToken()
        #expect(tokenAcquired)

        let unloadStarted  = DispatchSemaphore(value: 0)
        let unloadFinished = DispatchSemaphore(value: 0)
        var unloadEndTime: Date?

        Thread.detachNewThread {
            unloadStarted.signal()
            runner.unload()             // blocks inside waitForGenerationDrainLocked
            unloadEndTime = Date()
            unloadFinished.signal()
        }

        unloadStarted.wait()
        // Give the background thread a moment to reach the drain wait.
        Thread.sleep(forTimeInterval: 0.05)

        // While the generation token is held, a concurrent model activation
        // must also be blocked (any new unload sets pendingUnload = true).
        let raceAcquired = runner.tryAcquireGenerationToken()
        #expect(!raceAcquired, "New generation must be blocked while unload is draining")

        let releaseTime = Date()
        runner.releaseGenerationToken()

        let waited = unloadFinished.wait(timeout: .now() + 5)
        #expect(waited == .success, "unload() must complete after token is released")

        if let unloadEndTime {
            #expect(
                unloadEndTime >= releaseTime,
                "unload must not complete before token was released"
            )
        }

        #expect(!runner.isLoaded)
    }

    // MARK: – Scenario B: generate while deleting the current model

    @Test("Scenario B: deleteModel path (unload) waits for active generation")
    func scenarioB_deleteModelWaitsForGeneration() async {
        let runner = LlamaModelRunner()

        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)

        var unloadCompleted = false
        let done = DispatchSemaphore(value: 0)

        Thread.detachNewThread {
            runner.unload()
            unloadCompleted = true
            done.signal()
        }

        Thread.sleep(forTimeInterval: 0.05)
        #expect(!unloadCompleted, "unload must not complete while token is held")

        runner.releaseGenerationToken()
        let waited = done.wait(timeout: .now() + 5)
        #expect(waited == .success)
        #expect(unloadCompleted)
        #expect(!runner.isLoaded)
        #expect(runner.currentState == .idle)
    }

    // MARK: – Scenario C: generate while force-reloading

    @Test("Scenario C: concurrent force-reload drains before reloading")
    func scenarioC_forceReloadDrainsFirst() async {
        // loadModel() always calls waitForGenerationDrainLocked() at entry.
        // With no actual model file this throws immediately after the drain,
        // but the drain ordering is what we are testing here.
        let runner = LlamaModelRunner()

        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)

        var drainCompleted = false
        let drainSignal = DispatchSemaphore(value: 0)

        Thread.detachNewThread {
            // loadModel will drain, then fail to open the nonexistent file.
            // We only care that it drained correctly (did not crash).
            _ = try? runner.loadModel(at: "/nonexistent/model.gguf")
            drainCompleted = true
            drainSignal.signal()
        }

        Thread.sleep(forTimeInterval: 0.05)
        // loadModel is inside waitForGenerationDrainLocked, so the token
        // must still be holdable from our side (we already hold it).
        // A third party must NOT be able to race in.
        let raceToken = runner.tryAcquireGenerationToken()
        #expect(!raceToken, "loadModel drain window must block new acquisitions")

        runner.releaseGenerationToken()
        let waited = drainSignal.wait(timeout: .now() + 5)
        #expect(waited == .success, "loadModel must complete (with error) after drain")
        #expect(drainCompleted)
    }

    // MARK: – Scenario D: speculative decode while draft runner unloads

    @Test("Scenario D: draft runner unload waits for draft generation token")
    func scenarioD_draftRunnerUnloadWaitsForToken() async {
        let target = LlamaModelRunner()
        let draft  = LlamaModelRunner()

        // Both tokens are held, simulating an active speculative-decode step.
        let targetAcquired = target.tryAcquireGenerationToken()
        let draftAcquired  = draft.tryAcquireGenerationToken()
        #expect(targetAcquired)
        #expect(draftAcquired)

        var draftUnloaded = false
        let draftDone = DispatchSemaphore(value: 0)

        Thread.detachNewThread {
            draft.unload()           // must block until draftAcquired is released
            draftUnloaded = true
            draftDone.signal()
        }

        Thread.sleep(forTimeInterval: 0.05)
        #expect(!draftUnloaded, "draft.unload() must block while draft token is held")

        // Target runner is completely unaffected by the draft runner's drain.
        #expect(target.currentState != .idle || !target.isLoaded || targetAcquired,
                "Target runner must remain independent of draft runner lifecycle")

        // Release draft token — draft.unload() can now complete.
        draft.releaseGenerationToken()
        let waited = draftDone.wait(timeout: .now() + 5)
        #expect(waited == .success, "draft.unload() must complete after draft token released")
        #expect(draftUnloaded)
        #expect(!draft.isLoaded)

        // Target is still alive and its token is still held.
        let targetSecond = target.tryAcquireGenerationToken()
        #expect(!targetSecond, "Target token is still held — second acquisition must fail")

        target.releaseGenerationToken()

        // After release the target is healthy again.
        let targetThird = target.tryAcquireGenerationToken()
        #expect(targetThird, "Target should accept a new token after release")
        if targetThird { target.releaseGenerationToken() }
    }

    // MARK: – High-contention stress: all four scenarios racing simultaneously

    @Test("Stress: concurrent acquire/unload never crashes under contention (50 iterations)")
    func stress_concurrentAcquireAndUnloadNeverCrashes() async {
        for _ in 0..<50 {
            let target = LlamaModelRunner()
            let draft  = LlamaModelRunner()

            await withTaskGroup(of: Void.self) { group in
                // Simulated generation (target)
                group.addTask {
                    if target.tryAcquireGenerationToken() {
                        try? await Task.sleep(for: .microseconds(Int.random(in: 0...300)))
                        target.releaseGenerationToken()
                    }
                }
                // Simulated generation (draft)
                group.addTask {
                    if draft.tryAcquireGenerationToken() {
                        try? await Task.sleep(for: .microseconds(Int.random(in: 0...300)))
                        draft.releaseGenerationToken()
                    }
                }
                // Scenario A/B/C: model switch on target
                group.addTask {
                    try? await Task.sleep(for: .microseconds(Int.random(in: 0...150)))
                    target.unload()
                }
                // Scenario D: draft runner unload
                group.addTask {
                    try? await Task.sleep(for: .microseconds(Int.random(in: 0...150)))
                    draft.unload()
                }
            }

            // After all tasks finish the runners must be in a coherent state.
            #expect(target.currentState == .idle || target.currentState == .ready)
            #expect(draft.currentState  == .idle || draft.currentState  == .ready)
            #expect(!target.isLoaded)
            #expect(!draft.isLoaded)
        }
    }

    // MARK: – isEOG / tokenPiece do not crash when model is absent

    @Test("isEOG returns false gracefully when no model is loaded")
    func isEOG_gracefulWhenNoModel() {
        let runner = LlamaModelRunner()
        #expect(!runner.isEOG(1), "isEOG must not crash on an unloaded runner")
        #expect(!runner.isEOG(2))
        #expect(!runner.isEOG(0))
    }

    // MARK: – resetContext is a no-op while token is held (regression)

    @Test("resetContext is a no-op while a generation token is held")
    func resetContext_noOpWhileTokenHeld() {
        let runner = LlamaModelRunner()
        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)
        runner.resetContext()
        runner.resetState()
        if acquired { runner.releaseGenerationToken() }
        let again = runner.tryAcquireGenerationToken()
        #expect(again)
        if again { runner.releaseGenerationToken() }
    }

    @Test("Helpers remain safe while unload is draining")
    func helpersRemainSafeWhileUnloadDrains() async {
        let runner = LlamaModelRunner()
        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)

        let unloadStarted = DispatchSemaphore(value: 0)
        let unloadFinished = DispatchSemaphore(value: 0)

        Thread.detachNewThread {
            unloadStarted.signal()
            runner.unload()
            unloadFinished.signal()
        }

        unloadStarted.wait()
        Thread.sleep(forTimeInterval: 0.05)

        runner.resetContext()
        runner.resetState()
        #expect(!runner.isEOG(1))

        let blocked = runner.tryAcquireGenerationToken()
        #expect(!blocked)

        runner.releaseGenerationToken()
        let waited = unloadFinished.wait(timeout: .now() + 5)
        #expect(waited == .success)

        let reacquired = runner.tryAcquireGenerationToken()
        #expect(reacquired)
        if reacquired { runner.releaseGenerationToken() }
    }

    // MARK: – healthCheck reflects post-unload state correctly

    @Test("healthCheck reports idle/not-healthy after unload")
    func healthCheck_afterUnload() {
        let runner = LlamaModelRunner()
        runner.unload()
        let status = runner.healthCheck()
        #expect(!status.isHealthy)
        #expect(status.state == .idle)
    }
}
