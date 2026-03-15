import Foundation
import CoreML

@Observable
@MainActor
class InferenceEngine {
    var isGenerating: Bool = false
    var currentText: String = ""
    var sessionCache = SessionCache()
    var lastHealthStatus: HealthStatus?

    private let kvCache = KVCacheArena()
    private let prefixCache = PromptPrefixCache()
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
    private let maxRecoveryAttempts = 3

    private var recoveryInProgress: Bool = false
    private var onRecoveryNeeded: (() -> Void)?
    private var forceStopObserver: NSObjectProtocol?
    private var thermalEscalationObserver: NSObjectProtocol?

    init(metricsLogger: MetricsLogger, thermalGovernor: ThermalGovernor) {
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        registerForceStopObserver()
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
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let newState = notification.userInfo?["newState"] as? Int,
                   newState >= ProcessInfo.ThermalState.serious.rawValue,
                   self.isGenerating {
                    self.metricsLogger.recordThermalState(self.thermalGovernor.thermalLevel)
                }
            }
        }
    }

    func attachRunner(_ runner: CoreMLModelRunner, llamaRunner: LlamaModelRunner, tokenizer: TokenizerService, format: ModelFormat) {
        self.modelRunner = runner
        self.llamaRunner = llamaRunner
        self.tokenizer = tokenizer
        self.activeFormat = format
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

    var hasModel: Bool {
        switch activeFormat {
        case .coreML: return modelRunner?.isLoaded ?? false
        case .gguf: return llamaRunner?.isLoaded ?? false
        }
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
        metricsLogger.beginGeneration()

        generationTask = Task { [weak self] in
            guard let self else { return }

            if self.thermalGovernor.shouldRunZeroTokenProbe {
                let probeResult = runner.zeroTokenProbe()
                if !probeResult.passed {
                    self.metricsLogger.recordDiagnostic(DiagnosticEvent(
                        code: .modelEvicted,
                        message: "Zero-Token Probe failed (Ghost Model detected). Latency: \(String(format: "%.1f", probeResult.latencyMS))ms",
                        severity: .critical,
                        metadata: ["probeState": probeResult.state.rawValue]
                    ))
                    self.thermalGovernor.recordDiagnosticCode(.modelEvicted)

                    let recovered = await self.attemptInlineRecovery(runner: runner)
                    self.metricsLogger.recordRecovery(success: recovered, newComputeUnits: recovered ? self.computeUnitsLabel(runner) : self.metricsLogger.activeComputeLabel)
                    if !recovered {
                        self.isGenerating = false
                        self.notifyRecoveryNeeded()
                        onComplete(self.failedMetrics(since: Date()))
                        return
                    }
                    self.thermalGovernor.clearEvictionFlag()
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

            let prefixKey = PromptPrefixKey(
                modelID: "active-model",
                tokenizerID: "default",
                prefix: systemTokens
            )

            let prefillStart = Date()
            var tokensToProcess: [Int]
            var prefixHit = false

            if let cached = await self.prefixCache.lookup(key: prefixKey) {
                prefixHit = true
                let newTokens = Array(promptTokens.dropFirst(cached.tokenizedPrefix.count))
                tokensToProcess = newTokens
                self.sessionCache.sequencePosition = cached.sequencePosition
                self.sessionCache.allTokens = cached.tokenizedPrefix
                self.sessionCache.activeLength = cached.tokenizedPrefix.count

                for i in 0..<cached.pageCount {
                    let page = await self.kvCache.allocatePage(
                        tokenStart: i * 128,
                        tokenCount: min(128, cached.tokenizedPrefix.count - i * 128)
                    )
                    self.sessionCache.targetPages.append(page)
                }
            } else {
                tokensToProcess = promptTokens
                runner.resetState()
            }

            do {
                if !tokensToProcess.isEmpty {
                    let _ = try runner.predictLogits(inputIDs: tokensToProcess)
                }
            } catch let error as CoreMLRunnerError where error.isEviction {
                self.metricsLogger.recordModelEviction(reason: error.localizedDescription, computeUnits: self.metricsLogger.activeComputeLabel)
                let recovered = await self.attemptInlineRecovery(runner: runner)
                self.metricsLogger.recordRecovery(success: recovered, newComputeUnits: recovered ? self.computeUnitsLabel(runner) : self.metricsLogger.activeComputeLabel)
                if recovered {
                    runner.resetState()
                    do {
                        if !tokensToProcess.isEmpty {
                            let _ = try runner.predictLogits(inputIDs: tokensToProcess)
                        }
                    } catch {
                        self.isGenerating = false
                        self.notifyRecoveryNeeded()
                        onComplete(self.failedMetrics(since: prefillStart))
                        return
                    }
                } else {
                    self.isGenerating = false
                    self.notifyRecoveryNeeded()
                    onComplete(self.failedMetrics(since: prefillStart))
                    return
                }
            } catch {
                self.isGenerating = false
                onComplete(self.failedMetrics(since: prefillStart))
                return
            }

            let prefillPageCount = (tokensToProcess.count + 127) / 128
            for i in 0..<prefillPageCount {
                let pageTokenStart = self.sessionCache.activeLength + i * 128
                let pageTokenCount = min(128, tokensToProcess.count - i * 128)
                let page = await self.kvCache.allocatePage(
                    tokenStart: pageTokenStart,
                    tokenCount: pageTokenCount
                )
                self.sessionCache.targetPages.append(page)
            }

            let prefillDuration = Date().timeIntervalSince(prefillStart)
            self.metricsLogger.recordPrefill(tokens: tokensToProcess.count, duration: prefillDuration)
            self.sessionCache.accept(tokens: tokensToProcess)

            if !prefixHit {
                let cached = CachedPrefix(
                    key: prefixKey,
                    tokenizedPrefix: systemTokens,
                    pageCount: (systemTokens.count + 127) / 128,
                    sequencePosition: systemTokens.count,
                    timestamp: Date()
                )
                await self.prefixCache.store(prefix: cached)
            }

            self.metricsLogger.recordFirstToken()

            let eosTokens = tokenizer.effectiveEOSTokens
            var generatedCount = 0
            var specAccepted = 0
            var specRejected = 0
            var lastToken = self.sessionCache.currentToken

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
                    await self.evictSlidingWindow(systemTokenCount: systemTokens.count)
                    runner.resetState()
                    do {
                        let _ = try runner.predictLogits(inputIDs: self.sessionCache.allTokens)
                    } catch {
                        break
                    }
                }

                if useSpeculation && self.speculationPolicy.shouldUseSpeculation {
                    let specResult = self.executeSpeculativeDecode(
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

                    if generatedCount % 128 == 0 {
                        let page = await self.kvCache.allocatePage(
                            tokenStart: self.sessionCache.activeLength - 128,
                            tokenCount: 128
                        )
                        self.sessionCache.targetPages.append(page)
                    }

                    if let text = self.streamingDecoder.append(sampledToken, tokenizer: tokenizer) {
                        self.currentText += text
                        onToken(text)
                    }

                    self.metricsLogger.recordContextLength(self.sessionCache.activeLength)
                    let kvPages = await self.kvCache.activePageCount
                    self.metricsLogger.recordKVPages(kvPages)
                    self.metricsLogger.recordThermalState(self.thermalGovernor.thermalLevel)
                    let estimatedMem = await self.kvCache.estimatedMemoryBytes
                    self.metricsLogger.recordMemory(estimatedMem)

                } catch let error as CoreMLRunnerError where error.isEviction {
                    self.metricsLogger.recordModelEviction(reason: error.localizedDescription, computeUnits: self.metricsLogger.activeComputeLabel)
                    let recovered = await self.attemptInlineRecovery(runner: runner)
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

            let remaining = self.streamingDecoder.flush(tokenizer: tokenizer)
            if !remaining.isEmpty {
                self.currentText += remaining
                onToken(remaining)
            }

            self.metricsLogger.endGeneration()

            let metrics = GenerationMetrics(
                timeToFirstToken: self.metricsLogger.currentMetrics.timeToFirstTokenMS,
                prefillTokensPerSecond: self.metricsLogger.currentMetrics.prefillTokensPerSecond,
                decodeTokensPerSecond: self.metricsLogger.currentMetrics.decodeTokensPerSecond,
                totalTokens: generatedCount,
                totalDuration: Date().timeIntervalSince(prefillStart),
                acceptedSpeculativeTokens: specAccepted,
                rejectedSpeculativeTokens: specRejected
            )

            onComplete(metrics)
            self.isGenerating = false
        }
    }

    private func attemptInlineRecovery(runner: CoreMLModelRunner) async -> Bool {
        guard !recoveryInProgress else { return false }
        recoveryInProgress = true
        defer { recoveryInProgress = false }

        for attempt in 0..<maxRecoveryAttempts {
            let backoff = 0.5 * pow(2.0, Double(attempt))
            try? await Task.sleep(for: .seconds(backoff))

            do {
                try await runner.attemptRecovery()
                return true
            } catch {
                continue
            }
        }

        return false
    }

    private func notifyRecoveryNeeded() {
        onRecoveryNeeded?()
    }

    private func failedMetrics(since start: Date) -> GenerationMetrics {
        GenerationMetrics(
            timeToFirstToken: 0, prefillTokensPerSecond: 0,
            decodeTokensPerSecond: 0, totalTokens: 0,
            totalDuration: Date().timeIntervalSince(start),
            acceptedSpeculativeTokens: 0, rejectedSpeculativeTokens: 0
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
                    let _ = await self.attemptInlineRecovery(runner: runner)
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
        sessionCache.reset()
        streamingDecoder.reset()
        speculationPolicy.reset()
        currentText = ""
        modelRunner?.resetState()
        llamaRunner?.resetContext()
        draftEngine.resetDraftState()
        Task {
            await kvCache.reset()
            await prefixCache.clear()
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
    ) -> SpeculativeResult? {
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
                draftTokens: draftSequence.tokens,
                runner: runner,
                sampler: sampler,
                recentTokens: Array(sessionCache.allTokens.suffix(64))
            )

            speculationPolicy.recordVerification(
                draftCount: draftSequence.tokens.count,
                acceptedCount: verification.accepted.count,
                rejectedCount: verification.rejected.count,
                draftLatencyMS: draftSequence.draftLatencyMS,
                verifyLatencyMS: verification.verificationLatencyMS
            )

            var tokensGenerated = 0
            var hitEOS = false
            var currentLast = lastToken

            for token in verification.accepted {
                if eosTokens.contains(token) {
                    hitEOS = true
                    break
                }

                sessionCache.accept(tokens: [token])
                metricsLogger.recordToken()
                tokensGenerated += 1
                currentLast = token

                if let text = streamingDecoder.append(token, tokenizer: tokenizer) {
                    currentText += text
                    onToken(text)
                }
            }

            if !hitEOS, let correction = verification.correctionToken {
                if eosTokens.contains(correction) {
                    hitEOS = true
                } else {
                    sessionCache.accept(tokens: [correction])
                    metricsLogger.recordToken()
                    tokensGenerated += 1
                    currentLast = correction

                    if let text = streamingDecoder.append(correction, tokenizer: tokenizer) {
                        currentText += text
                        onToken(text)
                    }
                }
            }

            return SpeculativeResult(
                tokensGenerated: tokensGenerated,
                accepted: verification.accepted.count,
                rejected: verification.rejected.count,
                lastToken: currentLast,
                hitEOS: hitEOS
            )
        } catch {
            return nil
        }
    }

    private func evictSlidingWindow(systemTokenCount: Int) async {
        let keepCount = slidingWindowSize
        let totalTokens = sessionCache.allTokens.count

        guard totalTokens > keepCount + systemTokenCount else { return }

        let preservePrefix = systemTokenCount
        let evictCount = totalTokens - keepCount - preservePrefix

        guard evictCount > 0 else { return }

        let prefixTokens = Array(sessionCache.allTokens.prefix(preservePrefix))
        let recentTokens = Array(sessionCache.allTokens.suffix(keepCount))
        sessionCache.allTokens = prefixTokens + recentTokens
        sessionCache.activeLength = sessionCache.allTokens.count

        let pagesToFree = evictCount / 128
        for i in 0..<min(pagesToFree, sessionCache.targetPages.count) {
            let pageIndex = preservePrefix / 128 + i
            if pageIndex < sessionCache.targetPages.count {
                await kvCache.freePage(at: pageIndex)
            }
        }

        if pagesToFree > 0 {
            let removeStart = preservePrefix / 128
            let removeEnd = min(removeStart + pagesToFree, sessionCache.targetPages.count)
            if removeStart < removeEnd {
                sessionCache.targetPages.removeSubrange(removeStart..<removeEnd)
            }
        }

        metricsLogger.recordContextEviction(evictedTokens: evictCount)
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
