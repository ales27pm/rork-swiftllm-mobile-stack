import Testing
import Foundation
@testable import NeuralEngine

// MARK: – Generation-token lifecycle (no model required)
//
// These tests exercise the drain / generation-token protocol that was added to
// fix the EXC_BAD_ACCESS / SIGSEGV use-after-free crash.  They do NOT require
// an actual GGUF file — the runner is used in a no-model state so all llama_*
// C calls are avoided.  The invariants under test are:
//
//   1. tryAcquireGenerationToken returns false when pendingUnload is set.
//   2. unload() sets pendingUnload = true before it returns so new generation
//      attempts are rejected immediately.
//   3. A token acquired before unload() keeps the runner alive until released.
//   4. Multiple unload() calls are race-free (idempotent on an idle runner).
//   5. resetContext() is a no-op while a generation token is held.
//   6. tryAcquireGenerationToken enforces exclusivity (one token at a time).
//   7. releaseGenerationToken signals blocked waiters promptly.
//   8. loadModel() drains before reloading (state machine).

struct LlamaRunnerConcurrencyTests {

    // MARK: – 1. tryAcquireGenerationToken / pendingUnload

    @Test
    func tokenAcquisition_succeedsOnIdleRunner() {
        let runner = LlamaModelRunner()
        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)
        if acquired { runner.releaseGenerationToken() }
    }

    @Test
    func tokenAcquisition_failsAfterUnloadStarted() {
        let runner = LlamaModelRunner()
        // unload() on an idle runner (no model) drains immediately and then
        // clears pendingUnload.  To observe the window where pendingUnload is
        // true we use a helper that sets the flag externally via the drain
        // protocol: acquire a token, start unload in background, verify that
        // a *second* token acquisition is rejected.
        let firstTokenAcquired = runner.tryAcquireGenerationToken()
        #expect(firstTokenAcquired)

        // unload() will block in background until we release the first token.
        let unloadDone = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            runner.unload()
            unloadDone.signal()
        }

        // Give the background thread time to reach waitForGenerationDrainLocked
        // and set pendingUnload = true.
        Thread.sleep(forTimeInterval: 0.05)

        // While unload is draining, a new acquisition must fail.
        let secondTokenAcquired = runner.tryAcquireGenerationToken()
        #expect(!secondTokenAcquired, "Generation token must be rejected while unload is draining")

        // Release first token — unload() can now complete.
        runner.releaseGenerationToken()
        let waited = unloadDone.wait(timeout: .now() + 3)
        #expect(waited == .success, "unload() should complete within 3 seconds after token released")
    }

    // MARK: – 2. Exclusivity: at most one generation at a time

    @Test
    func tokenAcquisition_secondAttemptFailsWhileFirstHeld() {
        let runner = LlamaModelRunner()

        let first = runner.tryAcquireGenerationToken()
        #expect(first)

        let second = runner.tryAcquireGenerationToken()
        #expect(!second, "Second acquisition must fail while first token is held")

        if first { runner.releaseGenerationToken() }
        // After release the slot is free again.
        let third = runner.tryAcquireGenerationToken()
        #expect(third, "Acquisition should succeed once the previous token is released")
        if third { runner.releaseGenerationToken() }
    }

    // MARK: – 3. releaseGenerationToken unblocks a waiting unload

    @Test
    func releaseGenerationToken_unblocksUnload() async {
        let runner = LlamaModelRunner()

        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)

        let unloadCompletedAt = ActorBox<Date?>(nil)
        let releaseAt = Date()

        // Start unload in background; it will block waiting for the token.
        let unloadTask = Task.detached {
            runner.unload()
            await unloadCompletedAt.set(Date())
        }

        // Let unload block for a short while.
        try? await Task.sleep(for: .milliseconds(60))
        runner.releaseGenerationToken()

        // Wait for unload to finish.
        _ = await unloadTask.value
        let completedAt = await unloadCompletedAt.value
        #expect(completedAt != nil)
        if let completedAt {
            let lag = completedAt.timeIntervalSince(releaseAt)
            #expect(lag < 2.0, "unload should complete within 2 s of token release, got \(lag)s")
        }
    }

    // MARK: – 4. Idempotent unload on idle runner

    @Test
    func unload_onIdleRunnerCompletesImmediately() async {
        let runner = LlamaModelRunner()
        let start = Date()
        runner.unload()   // no model, no tokens
        runner.unload()   // second call must also be safe
        let elapsed = Date().timeIntervalSince(start)
        #expect(elapsed < 0.5, "Double unload on idle runner should be fast, took \(elapsed)s")
    }

    // MARK: – 5. resetContext is a no-op while token is held

    @Test
    func resetContext_isNoOpWhenGenerationTokenIsHeld() {
        let runner = LlamaModelRunner()
        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)

        // resetContext() must not crash and must skip the llama_memory_clear
        // path when a generation is in flight.
        runner.resetContext()   // should be a no-op
        runner.resetState()     // ditto

        if acquired { runner.releaseGenerationToken() }
    }

    // MARK: – 6. State after unload

    @Test
    func unload_leavesRunnerInIdleNotLoadedState() {
        let runner = LlamaModelRunner()
        runner.unload()
        #expect(!runner.isLoaded)
        #expect(runner.currentState == .idle)
    }

    // MARK: – 7. Concurrent unload + acquire is race-free under contention

    @Test
    func concurrentAcquireAndUnload_neverCrashes() async {
        // Run many iterations racing token acquisition against unload to ensure
        // the NSCondition-based protocol is race-free under Thread Sanitizer.
        for _ in 0..<20 {
            let runner = LlamaModelRunner()

            await withTaskGroup(of: Void.self) { group in
                // Thread 1: tries to acquire a token
                group.addTask {
                    if runner.tryAcquireGenerationToken() {
                        try? await Task.sleep(for: .microseconds(Int.random(in: 0...200)))
                        runner.releaseGenerationToken()
                    }
                }
                // Thread 2: calls unload
                group.addTask {
                    try? await Task.sleep(for: .microseconds(Int.random(in: 0...200)))
                    runner.unload()
                }
            }
            // After both, runner must be in a coherent state.
            #expect(!runner.isLoaded || runner.currentState == .idle || runner.currentState == .ready)
        }
    }

    // MARK: – 8. Token acquisition succeeds after unload completes

    @Test
    func tokenAcquisition_succeedsAfterUnloadCompletes() {
        let runner = LlamaModelRunner()
        runner.unload()  // idle runner, instant drain
        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired, "Should be able to acquire token after a completed unload")
        if acquired { runner.releaseGenerationToken() }
    }

    // MARK: – 9. healthCheck reflects pendingUnload state correctly

    @Test
    func healthCheck_reportsNotLoadedAfterUnload() {
        let runner = LlamaModelRunner()
        runner.unload()
        let status = runner.healthCheck()
        #expect(!status.isHealthy)
        #expect(status.state == .idle)
    }

    // MARK: – 10. Draft-runner token is independent from target runner

    @Test
    func draftAndTargetTokensAreIndependent() {
        let target = LlamaModelRunner()
        let draft = LlamaModelRunner()

        let targetAcquired = target.tryAcquireGenerationToken()
        #expect(targetAcquired)

        // Draft runner is independent — its count is 0.
        let draftAcquired = draft.tryAcquireGenerationToken()
        #expect(draftAcquired, "Draft runner token should be independent of target runner token")

        if targetAcquired { target.releaseGenerationToken() }
        if draftAcquired { draft.releaseGenerationToken() }
    }

    // MARK: – 11. Drain completes within timeout even if token is held briefly

    @Test
    func drain_completesWithinTimeoutWithShortHold() async {
        let runner = LlamaModelRunner()
        let acquired = runner.tryAcquireGenerationToken()
        #expect(acquired)

        let start = Date()
        let unloadTask = Task.detached { runner.unload() }

        // Simulate a very short generation step (1 ms).
        try? await Task.sleep(for: .milliseconds(1))
        runner.releaseGenerationToken()

        _ = await unloadTask.value
        let elapsed = Date().timeIntervalSince(start)
        #expect(elapsed < 2.0, "Drain should complete quickly after token release, took \(elapsed)s")
    }
}

// MARK: – Helpers

private actor ActorBox<T> {
    private var _value: T
    var value: T { _value }
    init(_ value: T) { _value = value }
    func set(_ newValue: T) { _value = newValue }
}
