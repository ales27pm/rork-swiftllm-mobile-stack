import Foundation
import CoreML

@Observable
@MainActor
class InferenceEngine {
    var isGenerating: Bool = false
    var currentText: String = ""
    var sessionCache = SessionCache()

    private let kvCache = KVCacheArena()
    private let prefixCache = PromptPrefixCache()
    private let streamingDecoder = StreamingDecoder()
    private var speculationPolicy = SpeculationPolicy()
    private let metricsLogger: MetricsLogger
    private let thermalGovernor: ThermalGovernor

    private var generationTask: Task<Void, Never>?
    private var modelRunner: CoreMLModelRunner?
    private var tokenizer: TokenizerService?

    private let slidingWindowSize: Int = 1536
    private let evictionThreshold: Int = 1800

    init(metricsLogger: MetricsLogger, thermalGovernor: ThermalGovernor) {
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
    }

    func attachRunner(_ runner: CoreMLModelRunner, tokenizer: TokenizerService) {
        self.modelRunner = runner
        self.tokenizer = tokenizer
    }

    var hasModel: Bool {
        modelRunner?.isLoaded ?? false
    }

    func generate(
        prompt: String,
        systemPrompt: String,
        samplingConfig: SamplingConfig,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (GenerationMetrics) -> Void
    ) {
        guard !isGenerating else { return }
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

            let mode = self.thermalGovernor.currentMode
            let sampler = Sampler(config: samplingConfig)
            let maxTokens = min(samplingConfig.maxTokens, mode.maxContextLength)

            let fullPrompt = systemPrompt + "\n\nUser: " + prompt + "\n\nAssistant:"
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
            } catch {
                self.isGenerating = false
                onComplete(GenerationMetrics(
                    timeToFirstToken: 0, prefillTokensPerSecond: 0,
                    decodeTokensPerSecond: 0, totalTokens: 0,
                    totalDuration: Date().timeIntervalSince(prefillStart),
                    acceptedSpeculativeTokens: 0, rejectedSpeculativeTokens: 0
                ))
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

            let eosToken = TokenizerService.eosToken
            var generatedCount = 0
            var specAccepted = 0
            var specRejected = 0
            var lastToken = self.sessionCache.currentToken

            while generatedCount < maxTokens {
                guard !Task.isCancelled else { break }

                if self.sessionCache.activeLength > self.evictionThreshold {
                    await self.evictSlidingWindow(systemTokenCount: systemTokens.count)
                    runner.resetState()
                    do {
                        let _ = try runner.predictLogits(inputIDs: self.sessionCache.allTokens)
                    } catch {
                        break
                    }
                }

                do {
                    let logits = try runner.predictLogits(inputIDs: [lastToken])

                    let sampledToken = sampler.sample(
                        logits: logits,
                        recentTokens: Array(self.sessionCache.allTokens.suffix(64))
                    )

                    if sampledToken == eosToken { break }

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

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    func resetSession() {
        cancel()
        sessionCache.reset()
        streamingDecoder.reset()
        speculationPolicy.reset()
        currentText = ""
        modelRunner?.resetState()
        Task {
            await kvCache.reset()
            await prefixCache.clear()
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
}
