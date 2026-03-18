import Foundation
import CoreML

@Observable
@MainActor
class InferenceEngine {
    var isGenerating: Bool = false
    var currentText: String = ""
    var sessionCache = SessionCache()
    var lastHealthStatus: HealthStatus?

    private let kvCacheManager: KVCacheManager
    private let streamingDecoder = StreamingDecoder()
    private var speculationPolicy = SpeculationPolicy()
    private let metricsLogger: MetricsLogger
    private let thermalGovernor: ThermalGovernor

    private let prefillEngine = PrefillEngine()
    private let decodeEngine = DecodeEngine()
    private let draftEngine = DraftEngine()

    private var generationTask: Task<Void, Never>?
    private var healthMonitorTask: Task<Void, Never>?
    private var modelRunner: CoreMLModelRunner?
    private var llamaRunner: LlamaModelRunner?
    private var tokenizer: TokenizerService?
    private var activeFormat: ModelFormat = .coreML

    private let slidingWindowSize: Int = 1536
    private let evictionThreshold: Int = 1800
    private let zeroTokenProbeLatencyThresholdMS: Double = 350

    private var recoveryInProgress: Bool = false
    private var hasValidatedCurrentSession: Bool = false
    private var lastProbeLatencyMS: Double = 0
    private var lastRecoveryRetryCount: Int = 0
    private var activeFallbackMode: String = "none"
    private var onRecoveryNeeded: (() -> Void)?
    private var onRecoverableWarning: ((String) -> Void)?
    private var forceStopObserver: NSObjectProtocol?
    private var thermalEscalationObserver: NSObjectProtocol?
    private var memoryPressureObserver: NSObjectProtocol?

    init(metricsLogger: MetricsLogger, thermalGovernor: ThermalGovernor) {
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.kvCacheManager = KVCacheManager(
            pageSize: 128,
            layerCount: 32,
            memoryBudgetMB: 256,
            evictionPolicy: .lruWithPrefixProtection,
            slidingWindowSize: 1536,
            evictionThreshold: 1800
        )
        registerForceStopObserver()
        registerMemoryPressureObserver()
    }

    private func registerForceStopObserver() {
        forceStopObserver = NotificationCenter.default.addObserver(
            forName: .forceInferenceStop,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isGenerating else { return }
                self.cancel()
            }
        }

        thermalEscalationObserver = NotificationCenter.default.addObserver(
            forName: .thermalStateEscalated,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            let newState = notification.userInfo?["newState"] as? Int
            guard let self,
                  let newState,
                  newState >= ProcessInfo.ThermalState.serious.rawValue,
                  self.isGenerating else { return }

            self.metricsLogger.recordThermalState(self.thermalGovernor.thermalLevel)
        }
    }

    private func registerMemoryPressureObserver() {
        memoryPressureObserver = NotificationCenter.default.addObserver(
            forName: .memoryPressureEscalated,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let response = await self.kvCacheManager.handleMemoryPressure()
                self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                    code: .memoryPressure,
                    message: "KV cache pressure relief: freed \(response.pagesFreed) pages (\(response.bytesReclaimed / 1024)KB)",
                    severity: .warning,
                    metadata: [
                        "sequencesEvicted": "\(response.sequencesEvicted)",
                        "pagesFreed": "\(response.pagesFreed)"
                    ]
                ))
            }
        }
    }

    func attachRunner(_ runner: CoreMLModelRunner, llamaRunner: LlamaModelRunner, tokenizer: TokenizerService, format: ModelFormat) {
        self.modelRunner = runner
        self.llamaRunner = llamaRunner
        self.tokenizer = tokenizer
        self.activeFormat = format
        hasValidatedCurrentSession = false
        activeFallbackMode = "none"
        startHealthMonitor()
    }

    func attachDraftRunner(_ runner: CoreMLModelRunner) {
        draftEngine.attachRunner(runner)
    }

    func detachDraftRunner() {
        draftEngine.detachRunner()
    }

    var speculativeStats: (accepted: Int, rejected: Int, rate: Double, speedup: Double) {
        (speculationPolicy.totalAcceptedTokens,
         speculationPolicy.totalRejectedTokens,
         speculationPolicy.lifetimeAcceptanceRate,
         speculationPolicy.effectiveSpeedup)
    }

    func updateFormat(_ format: ModelFormat) {
        self.activeFormat = format
    }

    func setRecoveryHandler(_ handler: @escaping () -> Void) {
        self.onRecoveryNeeded = handler
    }

    func setRecoverableWarningHandler(_ handler: @escaping (String) -> Void) {
        self.onRecoverableWarning = handler
    }

    var hasModel: Bool {
        switch activeFormat {
        case .coreML: return modelRunner?.isLoaded ?? false
        case .gguf: return llamaRunner?.isLoaded ?? false
        }
    }

    var cacheStatistics: KVCacheStatistics {
        get async {
            await kvCacheManager.statistics()
        }
    }


    func prepareVoiceContext(messages: [[String: String]], systemPrompt: String) {
        guard !isGenerating else { return }
        guard let tokenizer else { return }

        let promptBody: String
        if let templated = tokenizer.applyTemplate(messages: messages) {
            promptBody = templated
        } else {
            promptBody = Self.buildChatMLPrompt(messages: messages)
        }

        _ = tokenizer.encode(systemPrompt)
        _ = tokenizer.encode(promptBody)
        currentText = ""
    }

    func generate(
        messages: [[String: String]],
        systemPrompt: String,
        samplingConfig: SamplingConfig,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (GenerationMetrics) -> Void
    ) {
        guard !isGenerating else { return }

        if activeFormat == .gguf {
            generateWithLlama(messages: messages, samplingConfig: samplingConfig, onToken: onToken, onComplete: onComplete)
            return
        }

        guard let runner = modelRunner, runner.isLoaded, let tokenizer else {
            onComplete(GenerationMetrics(
                timeToFirstToken: 0, prefillTokensPerSecond: 0,
                decodeTokensPerSecond: 0, totalTokens: 0,
                totalDuration: 0, acceptedSpeculativeTokens: 0,
                rejectedSpeculativeTokens: 0
            ))
            return
        }

        isGenerating = true
        currentText = ""
        streamingDecoder.reset()
        lastProbeLatencyMS = 0
        metricsLogger.beginGeneration()
        metricsLogger.currentMetrics.zeroTokenProbeLatencyMS = 0

        generationTask = Task { [weak self] in
            guard let self else { return }

            let shouldProbe = !self.hasValidatedCurrentSession || self.thermalGovernor.shouldRunZeroTokenProbe || runner.currentState == .evicted
            if shouldProbe {
                let probeResult = runner.runZeroTokenProbe()
                self.lastProbeLatencyMS = probeResult.latencyMS
                self.metricsLogger.currentMetrics.zeroTokenProbeLatencyMS = probeResult.latencyMS

                let exceededLatency = probeResult.latencyMS > self.zeroTokenProbeLatencyThresholdMS
                let evictedState = probeResult.state == .evicted

                if !probeResult.passed || exceededLatency || evictedState {
                    let forensicCode: ForensicDiagnosticCode = evictedState ? .coreMLEviction : .coreMLPlanFailure
                    let forensic = ForensicDiagnostic(
                        code: forensicCode,
                        domain: "CoreML",
                        phenomenon: exceededLatency ? "Zero-token probe exceeded latency threshold" : "Zero-token probe indicates model eviction",
                        recoveryAction: .reloadModel
                    )
                    let forensicEvent = ForensicValidator.bridgeToDiagnosticEvent(forensic)
                    self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                        code: forensicEvent.code,
                        message: forensicEvent.message,
                        severity: forensicEvent.severity,
                        metadata: forensicEvent.metadata.merging([
                            "probeLatencyMS": String(format: "%.1f", probeResult.latencyMS),
                            "latencyThresholdMS": String(format: "%.1f", self.zeroTokenProbeLatencyThresholdMS),
                            "probeState": probeResult.state.rawValue
                        ]) { _, new in new }
                    ))
                    self.thermalGovernor.recordDiagnosticCode(forensicEvent.code)

                    let recovered = await self.attemptInlineRecovery(runner: runner, policy: BackoffPolicy.exponential)
                    self.metricsLogger.recordRecovery(success: recovered, newComputeUnits: recovered ? self.computeUnitsLabel(runner) : self.metricsLogger.activeComputeLabel)
                    if !recovered {
                        self.isGenerating = false
                        self.notifyRecoverableWarning("Neural Engine recovery failed, including CPU fallback. Please reload the model.")
                        self.notifyRecoveryNeeded()
                        onComplete(self.failedMetrics(since: Date()))
                        return
                    }
                } else {
                    self.hasValidatedCurrentSession = true
                    self.lastRecoveryRetryCount = 0
                    self.activeFallbackMode = "none"
                    self.metricsLogger.currentMetrics.recoveryRetryCount = 0
                    self.metricsLogger.currentMetrics.fallbackMode = "none"
                    self.thermalGovernor.clearEvictionFlag()
                    self.thermalGovernor.resetRecoveryState()
                }
            }

            let mode = self.thermalGovernor.currentMode
            var thermalAdjustedConfig = samplingConfig
            let tempBoost = self.thermalGovernor.adaptiveTemperatureBoost
            if tempBoost > 0 {
                thermalAdjustedConfig.temperature = min(samplingConfig.temperature + tempBoost, 2.0)
                self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                    code: .inferenceThrottled,
                    message: "Adaptive temperature: \(String(format: "%.2f", samplingConfig.temperature)) → \(String(format: "%.2f", thermalAdjustedConfig.temperature))",
                    severity: .info,
                    metadata: ["boost": String(format: "%.2f", tempBoost)]
                ))
            }
            let sampler = Sampler(config: thermalAdjustedConfig)
            let maxTokens = min(samplingConfig.maxTokens, mode.maxContextLength)

            let fullPrompt: String
            if let templated = tokenizer.applyTemplate(messages: messages) {
                fullPrompt = templated
            } else {
                fullPrompt = Self.buildChatMLPrompt(messages: messages)
            }

            let systemTokens = tokenizer.encode(systemPrompt)
            let promptTokens = tokenizer.encode(fullPrompt)
            let modelIdentity = runner.currentPrefixSnapshotModelID ?? "active-model"
            let tokenizerIdentity = tokenizer.cacheIdentifier

            runner.configurePrefixSnapshotContext(modelID: modelIdentity, tokenizerID: tokenizerIdentity)

            let prefixKey = PromptPrefixKey(
                modelID: modelIdentity,
                tokenizerID: tokenizerIdentity,
                prefix: systemTokens
            )

            let sequenceID = await self.kvCacheManager.beginSequence(prefixHash: prefixKey.prefixHash)
            self.sessionCache.sequenceID = sequenceID

            let prefillStart = Date()
            var tokensToProcess: [Int] = promptTokens
            var prefixHit = false
            var restoredPrefixState = false

            if let cached = await self.kvCacheManager.lookupPrefix(key: prefixKey) {
                prefixHit = true
                restoredPrefixState = runner.restorePrefillState(from: cached.stateSnapshot, expectedPrefixTokens: cached.tokenizedPrefix)

                if restoredPrefixState {
                    let newTokens = Array(promptTokens.dropFirst(cached.tokenizedPrefix.count))
                    tokensToProcess = newTokens
                    self.sessionCache.sequencePosition = cached.sequencePosition
                    self.sessionCache.allTokens = cached.tokenizedPrefix
                    self.sessionCache.activeLength = cached.tokenizedPrefix.count
                    self.sessionCache.prefixKey = cached.key

                    let prefixPages = await self.kvCacheManager.allocatePages(
                        sequenceID: sequenceID,
                        tokens: cached.tokenizedPrefix,
                        startPosition: 0
                    )
                    self.sessionCache.targetPages.append(contentsOf: prefixPages)

                    self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                        code: .prefixCacheHit,
                        message: "Prefix cache hit: restored state for \(cached.tokenizedPrefix.count) tokens (\(prefixPages.count) pages)",
                        severity: .info,
                        metadata: ["prefixTokens": "\(cached.tokenizedPrefix.count)"]
                    ))
                } else {
                    runner.resetState()
                    self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                        code: .prefixCacheMiss,
                        message: "Prefix cache restore unavailable; prefilling full prompt",
                        severity: .info,
                        metadata: ["prefixTokens": "\(cached.tokenizedPrefix.count)"]
                    ))
                }
            } else {
                runner.resetState()
            }

            do {
                if !tokensToProcess.isEmpty {
                    let _ = try prefillEngine.prefill(tokens: tokensToProcess, runner: runner)
                }
            } catch let error as CoreMLRunnerError where error.isEviction {
                self.metricsLogger.recordModelEviction(reason: error.localizedDescription, computeUnits: self.metricsLogger.activeComputeLabel)
                let recovered = await self.attemptInlineRecovery(runner: runner, policy: BackoffPolicy.exponential)
                self.metricsLogger.recordRecovery(success: recovered, newComputeUnits: recovered ? self.computeUnitsLabel(runner) : self.metricsLogger.activeComputeLabel)
                if recovered {
                    runner.resetState()
                    do {
                        if !tokensToProcess.isEmpty {
                            let _ = try prefillEngine.prefill(tokens: tokensToProcess, runner: runner)
                        }
                    } catch {
                        self.isGenerating = false
                        self.notifyRecoveryNeeded()
                        await self.kvCacheManager.releaseSequence(sequenceID)
                        onComplete(self.failedMetrics(since: prefillStart))
                        return
                    }
                } else {
                    self.isGenerating = false
                    self.notifyRecoveryNeeded()
                    await self.kvCacheManager.releaseSequence(sequenceID)
                    onComplete(self.failedMetrics(since: prefillStart))
                    return
                }
            } catch {
                self.isGenerating = false
                await self.kvCacheManager.releaseSequence(sequenceID)
                onComplete(self.failedMetrics(since: prefillStart))
                return
            }

            let prefillPages = await self.kvCacheManager.allocatePages(
                sequenceID: sequenceID,
                tokens: tokensToProcess,
                startPosition: self.sessionCache.activeLength
            )
            self.sessionCache.targetPages.append(contentsOf: prefillPages)

            let prefillDuration = Date().timeIntervalSince(prefillStart)
            self.metricsLogger.recordPrefill(tokens: tokensToProcess.count, duration: prefillDuration)
            self.sessionCache.accept(tokens: tokensToProcess)
            self.sessionCache.prefillComplete = true

            if !prefixHit || !restoredPrefixState {
                let stateSnapshot: PrefixStateSnapshot
                switch runner.exportPrefillState(for: systemTokens) {
                case .available(let snapshot):
                    stateSnapshot = snapshot
                case .unavailable(let reason):
                    stateSnapshot = .unavailable(reason: reason)
                }

                let cached = CachedPrefix(
                    key: prefixKey,
                    tokenizedPrefix: systemTokens,
                    pageCount: (systemTokens.count + 127) / 128,
                    sequencePosition: systemTokens.count,
                    stateSnapshot: stateSnapshot,
                    timestamp: Date()
                )
                await self.kvCacheManager.storePrefix(cached)
            }

            self.metricsLogger.recordFirstToken()

            let eosTokens = tokenizer.effectiveEOSTokens
            var generatedCount = 0
            var specAccepted = 0
            var specRejected = 0
            var lastToken = self.sessionCache.currentToken
            var pageAccumulator = 0

            let useSpeculation = self.draftEngine.hasDraftModel && mode.speculativeEnabled

            while generatedCount < maxTokens {
                guard !Task.isCancelled else { break }

                if self.thermalGovernor.shouldSuspendInference {
                    break
                }

                let delay = self.thermalGovernor.tokenDelaySeconds
                if delay > 0 {
                    try? await Task.sleep(for: .seconds(delay))
                }

                if self.sessionCache.activeLength > self.evictionThreshold {
                    let result = await self.kvCacheManager.slidingWindowEvict(
                        sequenceID: sequenceID,
                        systemTokenCount: systemTokens.count,
                        currentLength: self.sessionCache.activeLength
                    )
                    if result.evictedTokens > 0 {
                        self.sessionCache.applySlidingWindow(
                            keepPrefix: systemTokens.count,
                            keepSuffix: self.slidingWindowSize
                        )
                        self.metricsLogger.recordContextEviction(evictedTokens: result.evictedTokens)
                        runner.resetState()
                        do {
                            let _ = try runner.predictLogits(inputIDs: self.sessionCache.allTokens)
                        } catch {
                            break
                        }
                    }
                }

                if useSpeculation && self.speculationPolicy.shouldUseSpeculation {
                    let specResult = await self.executeSpeculativeDecode(
                        lastToken: lastToken,
                        runner: runner,
                        sampler: sampler,
                        eosTokens: eosTokens,
                        tokenizer: tokenizer,
                        maxRemaining: maxTokens - generatedCount,
                        onToken: onToken
                    )

                    if let result = specResult {
                        generatedCount += result.tokensGenerated
                        specAccepted += result.accepted
                        specRejected += result.rejected
                        lastToken = result.lastToken
                        pageAccumulator += result.tokensGenerated

                        self.metricsLogger.recordSpeculative(accepted: result.accepted, rejected: result.rejected)

                        if result.hitEOS { break }
                        continue
                    }
                }

                do {
                    let logits = try runner.predictLogits(inputIDs: [lastToken])

                    let sampledToken = sampler.sample(
                        logits: logits,
                        recentTokens: Array(self.sessionCache.allTokens.suffix(64))
                    )

                    if eosTokens.contains(sampledToken) { break }

                    lastToken = sampledToken
                    self.sessionCache.accept(tokens: [sampledToken])
                    self.metricsLogger.recordToken()
                    generatedCount += 1
                    pageAccumulator += 1

                    if pageAccumulator >= 128 {
                        let newPages = await self.kvCacheManager.allocatePages(
                            sequenceID: sequenceID,
                            tokens: Array(self.sessionCache.allTokens.suffix(pageAccumulator)),
                            startPosition: self.sessionCache.activeLength - pageAccumulator
                        )
                        self.sessionCache.targetPages.append(contentsOf: newPages)
                        pageAccumulator = 0
                    }

                    if let text = self.streamingDecoder.append(sampledToken, tokenizer: tokenizer) {
                        self.currentText += text
                        onToken(text)
                    }

                    self.metricsLogger.recordContextLength(self.sessionCache.activeLength)
                    let kvPages = await self.kvCacheManager.activePageCount()
                    self.metricsLogger.recordKVPages(kvPages)
                    self.metricsLogger.recordThermalState(self.thermalGovernor.thermalLevel)
                    let estimatedMem = await self.kvCacheManager.estimatedMemoryBytes()
                    self.metricsLogger.recordMemory(estimatedMem)

                } catch let error as CoreMLRunnerError where error.isEviction {
                    self.metricsLogger.recordModelEviction(reason: error.localizedDescription, computeUnits: self.metricsLogger.activeComputeLabel)
                    let recovered = await self.attemptInlineRecovery(runner: runner, policy: BackoffPolicy.exponential)
                    self.metricsLogger.recordRecovery(success: recovered, newComputeUnits: recovered ? self.computeUnitsLabel(runner) : self.metricsLogger.activeComputeLabel)
                    if recovered {
                        runner.resetState()
                        do {
                            let _ = try runner.predictLogits(inputIDs: self.sessionCache.allTokens)
                            continue
                        } catch {
                            break
                        }
                    } else {
                        self.notifyRecoveryNeeded()
                        break
                    }
                } catch {
                    break
                }
            }

            if pageAccumulator > 0 {
                let finalPages = await self.kvCacheManager.allocatePages(
                    sequenceID: sequenceID,
                    tokens: Array(self.sessionCache.allTokens.suffix(pageAccumulator)),
                    startPosition: self.sessionCache.activeLength - pageAccumulator
                )
                self.sessionCache.targetPages.append(contentsOf: finalPages)
            }

            let remaining = self.streamingDecoder.flush(tokenizer: tokenizer)
            if !remaining.isEmpty {
                self.currentText += remaining
                onToken(remaining)
            }

            await self.kvCacheManager.touchSequence(sequenceID)
            self.metricsLogger.endGeneration()

            let metrics = GenerationMetrics(
                timeToFirstToken: self.metricsLogger.currentMetrics.timeToFirstTokenMS,
                prefillTokensPerSecond: self.metricsLogger.currentMetrics.prefillTokensPerSecond,
                decodeTokensPerSecond: self.metricsLogger.currentMetrics.decodeTokensPerSecond,
                totalTokens: generatedCount,
                totalDuration: Date().timeIntervalSince(prefillStart),
                acceptedSpeculativeTokens: specAccepted,
                rejectedSpeculativeTokens: specRejected,
                zeroTokenProbeLatencyMS: self.lastProbeLatencyMS,
                recoveryRetryCount: self.lastRecoveryRetryCount,
                fallbackMode: self.activeFallbackMode
            )

            onComplete(metrics)
            self.isGenerating = false
        }
    }

    private func attemptInlineRecovery(runner: CoreMLModelRunner, policy: BackoffPolicy) async -> Bool {
        guard !recoveryInProgress else { return false }
        recoveryInProgress = true
        defer { recoveryInProgress = false }

        for attempt in 0..<policy.maxRetries {
            let backoff = policy.delay(forAttempt: attempt)
            try? await Task.sleep(for: .seconds(backoff))
            thermalGovernor.markRecoveryStarted()

            do {
                try await runner.attemptRecovery()
                hasValidatedCurrentSession = true
                lastRecoveryRetryCount = attempt + 1
                activeFallbackMode = "none"
                metricsLogger.currentMetrics.recoveryRetryCount = lastRecoveryRetryCount
                metricsLogger.currentMetrics.fallbackMode = activeFallbackMode
                thermalGovernor.markRecoveryCompleted(success: true)
                thermalGovernor.clearEvictionFlag()
                thermalGovernor.resetRecoveryState()
                return true
            } catch {
                thermalGovernor.markRecoveryCompleted(success: false)
                continue
            }
        }

        do {
            try await runner.switchToCPUOnly()
            hasValidatedCurrentSession = true
            lastRecoveryRetryCount = policy.maxRetries
            activeFallbackMode = "cpuOnly"
            metricsLogger.currentMetrics.recoveryRetryCount = lastRecoveryRetryCount
            metricsLogger.currentMetrics.fallbackMode = activeFallbackMode
            metricsLogger.recordDiagnostic(DiagnosticEvent(
                code: .cpuFallbackTriggered,
                message: "Recovery exhausted. Using CPU-only fallback path.",
                severity: .warning,
                metadata: ["retryCount": "\(policy.maxRetries)", "fallbackMode": activeFallbackMode]
            ))
            notifyRecoverableWarning("Neural Engine retries exhausted. Running on CPU-only fallback; generation may be slower.")
            thermalGovernor.recordDiagnosticCode(.cpuFallbackTriggered)
            thermalGovernor.resetRecoveryState()
            return true
        } catch {
            lastRecoveryRetryCount = policy.maxRetries
            metricsLogger.currentMetrics.recoveryRetryCount = lastRecoveryRetryCount
            return false
        }
    }

    private func notifyRecoveryNeeded() {
        onRecoveryNeeded?()
    }

    private func notifyRecoverableWarning(_ message: String) {
        onRecoverableWarning?(message)
    }

    private func failedMetrics(since start: Date) -> GenerationMetrics {
        GenerationMetrics(
            timeToFirstToken: 0, prefillTokensPerSecond: 0,
            decodeTokensPerSecond: 0, totalTokens: 0,
            totalDuration: Date().timeIntervalSince(start),
            acceptedSpeculativeTokens: 0, rejectedSpeculativeTokens: 0,
            zeroTokenProbeLatencyMS: lastProbeLatencyMS,
            recoveryRetryCount: lastRecoveryRetryCount,
            fallbackMode: activeFallbackMode
        )
    }

    private func startHealthMonitor() {
        healthMonitorTask?.cancel()
        healthMonitorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(15))
                guard let self else { return }
                guard self.activeFormat == .coreML, let runner = self.modelRunner else { continue }

                let status = runner.healthCheck()
                self.lastHealthStatus = status

                if !status.isHealthy && status.state == .evicted && !self.isGenerating {
                    self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                        code: .healthCheckFailed,
                        message: "Health check: \(status.diagnosticSummary)",
                        severity: .warning
                    ))
                    let _ = await self.attemptInlineRecovery(runner: runner, policy: BackoffPolicy.exponential)
                }
            }
        }
    }

    private func generateWithLlama(
        messages: [[String: String]],
        samplingConfig: SamplingConfig,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (GenerationMetrics) -> Void
    ) {
        guard let llamaRunner, llamaRunner.isLoaded else {
            onComplete(GenerationMetrics(
                timeToFirstToken: 0, prefillTokensPerSecond: 0,
                decodeTokensPerSecond: 0, totalTokens: 0,
                totalDuration: 0, acceptedSpeculativeTokens: 0,
                rejectedSpeculativeTokens: 0
            ))
            return
        }

        isGenerating = true
        currentText = ""
        metricsLogger.beginGeneration()

        let prompt = Self.buildChatMLPrompt(messages: messages)
        let mode = thermalGovernor.currentMode
        let maxTokens = min(samplingConfig.maxTokens, mode.maxContextLength)

        if thermalGovernor.shouldSuspendInference {
            isGenerating = false
            onComplete(GenerationMetrics(
                timeToFirstToken: 0, prefillTokensPerSecond: 0,
                decodeTokensPerSecond: 0, totalTokens: 0,
                totalDuration: 0, acceptedSpeculativeTokens: 0,
                rejectedSpeculativeTokens: 0
            ))
            return
        }

        generationTask = Task.detached { [weak self] in
            guard let self else { return }

            do {
                let result = try llamaRunner.generate(
                    prompt: prompt,
                    maxTokens: maxTokens,
                    temperature: samplingConfig.temperature,
                    topK: Int32(samplingConfig.topK),
                    topP: samplingConfig.topP,
                    repetitionPenalty: samplingConfig.repetitionPenalty,
                    onToken: { token in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.currentText += token
                            self.metricsLogger.recordToken()
                            onToken(token)
                        }
                    },
                    shouldStop: {
                        Task.isCancelled
                    }
                )

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.metricsLogger.recordFirstToken()
                    self.metricsLogger.recordPrefill(tokens: result.promptTokenCount, duration: 0.001)
                    self.metricsLogger.endGeneration()

                    let metrics = GenerationMetrics(
                        timeToFirstToken: result.timeToFirstTokenMS,
                        prefillTokensPerSecond: result.prefillTokensPerSecond,
                        decodeTokensPerSecond: result.decodeTokensPerSecond,
                        totalTokens: result.generatedTokenCount,
                        totalDuration: result.totalDuration,
                        acceptedSpeculativeTokens: 0,
                        rejectedSpeculativeTokens: 0
                    )
                    onComplete(metrics)
                    self.isGenerating = false
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isGenerating = false
                    onComplete(GenerationMetrics(
                        timeToFirstToken: 0, prefillTokensPerSecond: 0,
                        decodeTokensPerSecond: 0, totalTokens: 0,
                        totalDuration: 0, acceptedSpeculativeTokens: 0,
                        rejectedSpeculativeTokens: 0
                    ))
                }
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        decodeEngine.stop()
        prefillEngine.cancel()
        isGenerating = false
    }

    func resetSession() {
        cancel()
        let oldSequenceID = sessionCache.sequenceID
        sessionCache.reset()
        streamingDecoder.reset()
        speculationPolicy.reset()
        currentText = ""
        modelRunner?.resetState()
        llamaRunner?.resetContext()
        draftEngine.resetDraftState()
        Task {
            if let seqID = oldSequenceID {
                await kvCacheManager.releaseSequence(seqID)
            }
            await kvCacheManager.reset()
        }
    }

    func stopHealthMonitor() {
        healthMonitorTask?.cancel()
        healthMonitorTask = nil
    }

    private func computeUnitsLabel(_ runner: CoreMLModelRunner) -> String {
        let health = runner.healthCheck()
        switch health.computeUnits {
        case .all: return "All"
        case .cpuAndNeuralEngine: return "CPU+ANE"
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU+GPU"
        @unknown default: return "Unknown"
        }
    }

    func removeObservers() {
        if let observer = forceStopObserver {
            NotificationCenter.default.removeObserver(observer)
            forceStopObserver = nil
        }
        if let observer = thermalEscalationObserver {
            NotificationCenter.default.removeObserver(observer)
            thermalEscalationObserver = nil
        }
        if let observer = memoryPressureObserver {
            NotificationCenter.default.removeObserver(observer)
            memoryPressureObserver = nil
        }
    }

    private struct SpeculativeResult {
        let tokensGenerated: Int
        let accepted: Int
        let rejected: Int
        let lastToken: Int
        let hitEOS: Bool
    }

    private func executeSpeculativeDecode(
        lastToken: Int,
        runner: CoreMLModelRunner,
        sampler: Sampler,
        eosTokens: Set<Int>,
        tokenizer: TokenizerService,
        maxRemaining: Int,
        onToken: @escaping (String) -> Void
    ) async -> SpeculativeResult? {
        let draftK = min(speculationPolicy.k, maxRemaining)
        guard draftK > 0 else { return nil }

        do {
            let draftSequence = try draftEngine.generateDraftBurst(
                seedToken: lastToken,
                count: draftK,
                sampler: sampler,
                recentTokens: Array(sessionCache.allTokens.suffix(64))
            )

            guard !draftSequence.tokens.isEmpty else { return nil }

            let verification = try decodeEngine.verifySpeculativeTokens(
                draftSequence: draftSequence,
                runner: runner,
                sampler: sampler,
                recentTokens: Array(sessionCache.allTokens.suffix(64))
            )

            speculationPolicy.recordVerification(
                draftCount: draftSequence.tokens.count,
                acceptedCount: verification.accepted.count,
                rejectedCount: verification.rejected.count,
                correctionCount: verification.correctionSampled ? 1 : 0,
                draftLatencyMS: draftSequence.draftLatencyMS,
                verifyLatencyMS: verification.verificationLatencyMS,
                committedCount: verification.committedTokens.count,
                mismatchIndex: verification.mismatchIndex
            )

            var hitEOS = false
            var currentLast = lastToken
            var committedTokens: [Int] = []

            for token in verification.accepted {
                if eosTokens.contains(token) {
                    hitEOS = true
                    break
                }
                committedTokens.append(token)
                currentLast = token
            }

            if !hitEOS, let correction = verification.correctionToken {
                if eosTokens.contains(correction) {
                    hitEOS = true
                } else {
                    committedTokens.append(correction)
                    currentLast = correction
                }
            }

            if !committedTokens.isEmpty, let sequenceID = sessionCache.sequenceID {
                let startPosition = sessionCache.activeLength
                sessionCache.accept(tokens: committedTokens)
                for _ in committedTokens {
                    metricsLogger.recordToken()
                }

                let newPages = await kvCacheManager.allocatePages(
                    sequenceID: sequenceID,
                    tokens: committedTokens,
                    startPosition: startPosition
                )
                sessionCache.targetPages.append(contentsOf: newPages)

                for token in committedTokens {
                    if let text = streamingDecoder.append(token, tokenizer: tokenizer) {
                        currentText += text
                        onToken(text)
                    }
                }

                metricsLogger.recordContextLength(sessionCache.activeLength)
                let kvPages = await kvCacheManager.activePageCount()
                metricsLogger.recordKVPages(kvPages)
                metricsLogger.recordThermalState(thermalGovernor.thermalLevel)
                let estimatedMem = await kvCacheManager.estimatedMemoryBytes()
                metricsLogger.recordMemory(estimatedMem)
            }

            return SpeculativeResult(
                tokensGenerated: committedTokens.count,
                accepted: verification.accepted.count,
                rejected: verification.rejected.count,
                lastToken: currentLast,
                hitEOS: hitEOS
            )
        } catch {
            return nil
        }
    }

    static func buildChatMLPrompt(messages: [[String: String]]) -> String {
        var result = ""
        for msg in messages {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            result += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        result += "<|im_start|>assistant\n"
        return result
    }
}

extension CoreMLRunnerError {
    var isEviction: Bool {
        if case .modelEvicted = self { return true }
        return false
    }
}
