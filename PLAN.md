# Fix LLM diagnostic test failures caused by engine busy race condition

## Problem

All 7 LLM diagnostic test failures (Instruction Following, Factual Recall, Coherence, Emotional Tone, Multi-Turn, Temperature Sensitivity, Latency Profile) are caused by a single race condition: when a generation times out, the cleanup runs asynchronously but the next test starts immediately — finding the engine still "busy" and producing zero output.

## Root Cause

The `llmGenerate` helper calls `ie.cancel()` on timeout, which is **fire-and-forget**. The actual `isGenerating = false` happens in a detached task's `MainActor.run` block that hasn't executed yet when the next test begins.

## Fix

**1. Use `cancelAndDrain` instead of `cancel` in the timeout path**

- `cancelAndDrain` already exists and awaits the generation task + guarantees `isGenerating = false`
- Change the timeout handler in `llmGenerate` to call `await ie.cancelAndDrain()` instead of `ie.cancel()`

**2. Add an idle-wait guard before each `generate` call in `llmGenerate**`

- Before calling `ie.generate()`, briefly poll (up to ~500ms) for `!ie.isGenerating` to handle edge cases where the previous test's cleanup is still in-flight
- This makes the diagnostic suite robust against any residual async cleanup from prior tests

**3. Add a settling yield after the continuation resumes on cancellation**

- After `withCheckedContinuation` returns with `completed == false`, yield briefly to let any pending MainActor work drain before returning

## Expected Impact

- All 7 LLM diagnostic failures should become passes (assuming the model can generate coherent output)
- The 4 warnings (LLM Cognition Pipeline, Stop Sequence, Memory-Aware, Health Monitor) may also improve since they partially depend on generation working
- No changes to the inference engine itself — only the diagnostic test harness
- No impact on normal app usage (chat, voice, etc.)

## Files Changed

- `ios/NeuralEngine/Services/LLMDiagnosticTests.swift` — Update `llmGenerate` helper with drain-based cancellation and idle-wait guard

