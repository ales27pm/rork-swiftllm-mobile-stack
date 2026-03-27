# Lifecycle + Inference Signpost Logging Guide

This project emits `os_log` + signpost telemetry for app lifecycle transitions and inference requests.

## What is logged

- Scene lifecycle callbacks are logged as `SceneLifecycleCallback` intervals for:
  - `sceneWillResignActive`
  - `sceneDidEnterBackground`
  - `sceneWillEnterForeground`
  - `sceneDidBecomeActive`
- Inference requests are logged as `InferenceRequest` intervals with:
  - request start
  - first-token event (`InferenceFirstToken`)
  - completion/cancellation
- Structured fields include:
  - request/lifecycle event ID
  - queue label and thread identity
  - model parameters (`contextLength`, `batchSize`, `maxTokens`)
  - completion status and generated token count

## Watchdog-risk signal

Lifecycle callbacks log a warning when processing time exceeds **200ms**. This is emitted as a high-signal `os_log` error entry and is intended to surface potential watchdog-termination risk.

## Collecting logs

### Via Console.app (macOS)

1. Connect the iOS device.
2. Open **Console.app**.
3. Select the target device.
4. Filter by subsystem (bundle identifier) and category `InferenceEngine`.
5. Optionally filter for signpost names:
   - `SceneLifecycleCallback`
   - `InferenceRequest`
   - `InferenceFirstToken`

### Via Instruments

1. Open Instruments and choose **Points of Interest** (or **OS Signpost** template).
2. Attach to the app process.
3. Record activity while reproducing the issue.
4. Search for the request ID and correlate lifecycle spans with inference spans.

## Correlating with crash reports

1. Capture the approximate crash time from the crash report (`Incident Identifier` timestamp and termination reason).
2. Pull logs for ±2 minutes around that timestamp.
3. Identify matching lifecycle callback spans near the crash time.
4. Correlate with `InferenceRequest` spans using request IDs and status:
   - `completed`
   - `cancelled`
   - `failed`
5. If a lifecycle span exceeds 200ms and crash reason indicates watchdog/termination, treat it as a strong lead.

## Recommended triage checklist

- Confirm whether app was backgrounding/foregrounding at crash time.
- Confirm if an inference request was in-flight during lifecycle transition.
- Check queue/thread labels for unexpected execution context.
- Compare `contextLength`/`batchSize` against device constraints and thermal state.
