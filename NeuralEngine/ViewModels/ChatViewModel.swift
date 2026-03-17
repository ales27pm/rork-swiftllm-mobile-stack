import SwiftUI

@Observable
@MainActor
class ChatViewModel {
    var messages: [Message] = []
    var inputText: String = ""
    var isGenerating: Bool = false
    var isExecutingTools: Bool = false
    var samplingConfig = SamplingConfig()
    var systemPrompt: String = "You are Nexus, a helpful and intelligent AI assistant running locally on-device. You have access to memory from past conversations and use it to provide personalized, contextual responses."
    var toolsEnabled: Bool = true
    var isVoiceMode: Bool = false
    var lastCognitionFrame: CognitionFrame?
    var lastError: WrappedError?
    var loadingProgress: Double = 0.0
    var statusMessage: String = "Idle"
    var isModelLoading: Bool = false
    private let maxToolExecutionRounds: Int = 3

    var reasoningReplayLog: [ReasoningReplayEntry] = []
    var activeReasoningDepth: Int = 0
    var activeConvergence: Double = 0
    var expectedResponseLength: ResponseLength?
    var lastIntentClassification: IntentClassification?

    var currentConversationId: String?

    let inferenceEngine: InferenceEngine
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let keyValueStore: KeyValueStore
    let database: DatabaseService
    let conversationService: ConversationService
    let memoryService: MemoryService
    var toolExecutor: ToolExecutor

    init(
        inferenceEngine: InferenceEngine,
        metricsLogger: MetricsLogger,
        thermalGovernor: ThermalGovernor,
        modelLoader: ModelLoaderService,
        keyValueStore: KeyValueStore,
        database: DatabaseService,
        conversationService: ConversationService,
        memoryService: MemoryService,
        toolExecutor: ToolExecutor
    ) {
        self.inferenceEngine = inferenceEngine
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.modelLoader = modelLoader
        self.keyValueStore = keyValueStore
        self.database = database
        self.conversationService = conversationService
        self.memoryService = memoryService
        self.toolExecutor = toolExecutor
        inferenceEngine.attachRunner(modelLoader.modelRunner, llamaRunner: modelLoader.llamaRunner, tokenizer: modelLoader.tokenizer, format: modelLoader.activeFormat)
        inferenceEngine.setRecoverableWarningHandler { [weak self] warning in
            guard let self else { return }
            self.statusMessage = warning
            self.lastError = WrappedError(
                domain: .inference,
                severity: .warning,
                userMessage: warning,
                technicalDetail: "InferenceEngine entered recoverable CPU fallback mode",
                recoveryAction: .none
            )
        }
        restoreSettings()
    }

    func dismissError() {
        lastError = nil
    }

    func replayReasoningState() -> [ReasoningReplayEntry] {
        reasoningReplayLog
    }

    private func recordReasoningReplay(frame: CognitionFrame) {
        let entry = ReasoningReplayEntry(
            timestamp: Date(),
            status: statusMessage,
            convergence: frame.reasoningTrace.finalConvergence,
            activeBranches: frame.thoughtTree.branches.count - frame.thoughtTree.prunedBranches.count,
            prunedBranches: frame.thoughtTree.prunedBranches.count,
            strategy: frame.reasoningTrace.dominantStrategy.rawValue,
            treeDepth: frame.thoughtTree.maxDepthReached,
            complexityLevel: frame.metacognition.complexityLevel.rawValue,
            uncertaintyLevel: frame.metacognition.uncertaintyLevel,
            selfCorrectionCount: frame.metacognition.selfCorrectionFlags.count
        )
        reasoningReplayLog.append(entry)
        if reasoningReplayLog.count > 50 {
            reasoningReplayLog.removeFirst()
        }
        activeReasoningDepth = frame.thoughtTree.maxDepthReached
        activeConvergence = frame.reasoningTrace.finalConvergence
    }

    func safeModelLoad() async {
        guard !isModelLoading else { return }
        isModelLoading = true
        statusMessage = "Loading model..."
        loadingProgress = 0.1
        lastError = nil

        do {
            guard let modelID = modelLoader.activeModelID,
                  let manifest = modelLoader.availableModels.first(where: { $0.id == modelID }) else {
                isModelLoading = false
                statusMessage = "No model selected"
                loadingProgress = 0
                return
            }

            loadingProgress = 0.3
            statusMessage = "Initializing \(manifest.name)..."

            if manifest.format == .gguf {
                modelLoader.activateModel(modelID)
            } else {
                modelLoader.activateModel(modelID)
            }

            loadingProgress = 0.7
            statusMessage = "Attaching engine..."

            inferenceEngine.attachRunner(
                modelLoader.modelRunner,
                llamaRunner: modelLoader.llamaRunner,
                tokenizer: modelLoader.tokenizer,
                format: modelLoader.activeFormat
            )

            loadingProgress = 1.0
            statusMessage = "Ready"
        } catch {
            let wrapped = NativeErrorWrapper.synthesize(error)
            lastError = wrapped
            statusMessage = wrapped.userMessage
            loadingProgress = 0
        }

        isModelLoading = false
    }

    func sendIntent() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard !isGenerating else { return }
        lastError = nil

        let intent = IntentClassifier.classify(text: text, conversationHistory: messages)
        lastIntentClassification = intent
        expectedResponseLength = intent.expectedResponseLength

        if currentConversationId == nil {
            let conv = conversationService.createConversation(modelId: modelLoader.activeModelID)
            currentConversationId = conv.id
        }

        let userMessage = Message(role: .user, content: text)
        messages.append(userMessage)
        inputText = ""

        if let convId = currentConversationId {
            conversationService.saveMessage(userMessage, conversationId: convId)
        }

        statusMessage = statusForIntent(intent)
        generateResponse(userText: text, toolIteration: 0)
    }

    func sendMessage() {
        sendIntent()
    }

    private func statusForIntent(_ intent: IntentClassification) -> String {
        switch intent.expectedResponseLength {
        case .brief: return "Generating brief response..."
        case .moderate: return "Generating..."
        case .detailed: return "Generating detailed response..."
        case .comprehensive: return "Generating comprehensive response..."
        }
    }

    private func generateResponse(userText: String, toolContext: [[String: String]] = [], toolIteration: Int) {
        let assistantMessage = Message(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        isGenerating = true
        statusMessage = "Thinking..."

        let assistantIndex = messages.count - 1

        let frame = CognitionEngine.process(
            userText: userText,
            conversationHistory: messages,
            memoryService: memoryService
        )
        lastCognitionFrame = frame
        recordReasoningReplay(frame: frame)

        let memoryResults = memoryService.searchMemories(query: userText, maxResults: 8)
        let associativeResults = memoryService.getAssociativeMemories(query: userText, directResults: memoryResults)
        let allMemoryResults = memoryResults + associativeResults

        let enrichedSystemPrompt = ContextAssembler.assembleSystemPrompt(
            frame: frame,
            memoryResults: allMemoryResults,
            conversationHistory: messages,
            toolsEnabled: toolsEnabled,
            isVoiceMode: isVoiceMode
        )

        var chatMessages: [[String: String]] = [
            ["role": "system", "content": enrichedSystemPrompt]
        ]
        for msg in messages where msg.role != .system {
            if msg.role == .assistant && msg.content.isEmpty { continue }
            chatMessages.append(["role": msg.role.rawValue, "content": msg.content])
        }
        chatMessages.append(contentsOf: toolContext)

        inferenceEngine.generate(
            messages: chatMessages,
            systemPrompt: enrichedSystemPrompt,
            samplingConfig: samplingConfig,
            onToken: { [weak self] token in
                guard let self else { return }
                self.messages[assistantIndex].content += token
                self.statusMessage = "Streaming..."
            },
            onComplete: { [weak self] metrics in
                guard let self else { return }
                self.messages[assistantIndex].isStreaming = false
                self.messages[assistantIndex].metrics = metrics
                self.isGenerating = false
                self.statusMessage = "Ready"
                self.expectedResponseLength = nil

                let fullContent = self.messages[assistantIndex].content

                if fullContent.isEmpty && metrics.totalTokens == 0 {
                    let wrapped = WrappedError(
                        domain: .inference,
                        severity: .warning,
                        userMessage: "No response generated. The model may need to be reloaded.",
                        technicalDetail: "0 tokens generated, duration=\(metrics.totalDuration)s",
                        recoveryAction: .reloadModel
                    )
                    self.lastError = wrapped
                }

                if self.toolsEnabled && ToolCallParser.containsToolCall(fullContent) {
                    self.statusMessage = "Executing tools..."
                    self.handleToolCalls(assistantIndex: assistantIndex, userText: userText, toolIteration: toolIteration)
                } else {
                    self.finalizeResponse(assistantIndex: assistantIndex, userText: userText)
                }
            }
        )
    }

    private func handleToolCalls(assistantIndex: Int, userText: String, toolIteration: Int) {
        let fullContent = messages[assistantIndex].content
        let toolCalls = ToolCallParser.parse(from: fullContent)
        let cleanedContent = ToolCallParser.stripToolCalls(from: fullContent)
        messages[assistantIndex].content = cleanedContent

        guard !toolCalls.isEmpty else {
            finalizeResponse(assistantIndex: assistantIndex, userText: userText)
            return
        }

        guard toolIteration < maxToolExecutionRounds else {
            messages[assistantIndex].content += "\n\n(I stopped tool execution after \(maxToolExecutionRounds) rounds to avoid a loop.)"
            finalizeResponse(assistantIndex: assistantIndex, userText: userText)
            return
        }

        isExecutingTools = true

        let toolMessage = Message(role: .tool, content: "", isToolExecution: true)
        messages.append(toolMessage)
        let toolMsgIndex = messages.count - 1

        Task {
            var results: [ToolResult] = []
            for call in toolCalls {
                let result = await toolExecutor.execute(call)
                results.append(result)
            }

            messages[toolMsgIndex].toolResults = results
            messages[toolMsgIndex].content = Self.renderToolExecutionSummary(calls: toolCalls, results: results)
            isExecutingTools = false

            let toolContextMessages = results.map {
                ["role": "tool", "content": "Tool result for \($0.toolName): \($0.data)"]
            }

            generateResponse(userText: userText, toolContext: toolContextMessages, toolIteration: toolIteration + 1)
        }
    }

    private static func renderToolExecutionSummary(calls: [ToolCall], results: [ToolResult]) -> String {
        var lines: [String] = ["[Tool Execution Summary]"]

        var resultsByToolName: [String: [ToolResult]] = [:]
        for result in results {
            resultsByToolName[result.toolName, default: []].append(result)
        }

        var unmatchedResults: [ToolResult] = []

        for call in calls {
            var callResult: ToolResult?
            if var bucket = resultsByToolName[call.name], !bucket.isEmpty {
                callResult = bucket.removeFirst()
                resultsByToolName[call.name] = bucket
            }

            let status = (callResult?.success ?? false) ? "success" : "failure"
            lines.append("- \(call.name) (\(status))")
            if let callResult {
                lines.append("  \(callResult.data)")
            } else {
                lines.append("  No matching tool result returned.")
            }
        }

        for bucket in resultsByToolName.values where !bucket.isEmpty {
            unmatchedResults.append(contentsOf: bucket)
        }

        if !unmatchedResults.isEmpty {
            lines.append("[Unmatched Tool Results]")
            for result in unmatchedResults {
                lines.append("- \(result.toolName): \(result.data)")
            }
        }

        return lines.joined(separator: "\n")
    }

    private func finalizeResponse(assistantIndex: Int, userText: String) {
        let composedResponse = LocationResponseComposer.composeResponseIfNeeded(
            userText: userText,
            assistantDraft: messages[assistantIndex].content,
            messages: messages
        )
        messages[assistantIndex].content = composedResponse

        if let convId = currentConversationId {
            conversationService.saveMessage(messages[assistantIndex], conversationId: convId)

            let userMsgCount = messages.filter { $0.role == .user }.count
            let assistantContent = messages[assistantIndex].content
            var conv = conversationService.conversations.first { $0.id == convId } ?? Conversation(id: convId)
            conv.lastMessage = String(assistantContent.prefix(100))
            conv.messageCount = messages.filter { $0.role != .system }.count
            conv.updatedAt = Date()
            if userMsgCount == 1 {
                conv.title = conversationService.generateTitle(from: userText)
            }
            conversationService.updateConversation(conv)

            memoryService.extractAndStoreMemory(userText: userText, assistantText: assistantContent)
        }
    }

    func stopGeneration() {
        inferenceEngine.cancel()
        if let lastIndex = messages.indices.last, messages[lastIndex].role == .assistant {
            messages[lastIndex].isStreaming = false
        }
        isGenerating = false
        statusMessage = "Stopped"
    }

    func clearChat() {
        inferenceEngine.resetSession()
        messages.removeAll()
        isGenerating = false
        currentConversationId = nil
        lastError = nil
        lastCognitionFrame = nil
        CognitionEngine.resetSignature()
        statusMessage = "Idle"
    }

    func newConversation() {
        inferenceEngine.resetSession()
        messages.removeAll()
        isGenerating = false
        currentConversationId = nil
        lastError = nil
        lastCognitionFrame = nil
        CognitionEngine.resetSignature()
        statusMessage = "Ready"
    }

    func loadConversation(_ id: String) {
        inferenceEngine.resetSession()
        isGenerating = false
        currentConversationId = id
        messages = conversationService.loadMessages(for: id)
    }

    var activeModelName: String {
        modelLoader.activeModel?.name ?? "No Model"
    }

    var hasActiveModel: Bool {
        modelLoader.activeModelID != nil
    }

    func syncEngineFormat() {
        inferenceEngine.updateFormat(modelLoader.activeFormat)
    }

    func saveSettings() {
        keyValueStore.setDouble(Double(samplingConfig.temperature), forKey: "sampling_temperature")
        keyValueStore.setInt(samplingConfig.topK, forKey: "sampling_topK")
        keyValueStore.setDouble(Double(samplingConfig.topP), forKey: "sampling_topP")
        keyValueStore.setDouble(Double(samplingConfig.repetitionPenalty), forKey: "sampling_repPenalty")
        keyValueStore.setInt(samplingConfig.maxTokens, forKey: "sampling_maxTokens")
        keyValueStore.setString(systemPrompt, forKey: "system_prompt")
        keyValueStore.setInt(toolsEnabled ? 1 : 0, forKey: "tools_enabled")
    }

    func logGeneration(metrics: GenerationMetrics) {
        _ = database.execute(
            "INSERT INTO generation_logs (model_id, prompt_tokens, generated_tokens, prefill_tps, decode_tps, time_to_first_token, total_duration, thermal_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            params: [
                modelLoader.activeModelID ?? "",
                0,
                metrics.totalTokens,
                metrics.prefillTokensPerSecond,
                metrics.decodeTokensPerSecond,
                metrics.timeToFirstToken,
                metrics.totalDuration,
                thermalGovernor.thermalLevel.rawValue
            ]
        )
    }

    private func restoreSettings() {
        if let temp = keyValueStore.getDouble("sampling_temperature") {
            samplingConfig.temperature = Float(temp)
        }
        if let topK = keyValueStore.getInt("sampling_topK") {
            samplingConfig.topK = topK
        }
        if let topP = keyValueStore.getDouble("sampling_topP") {
            samplingConfig.topP = Float(topP)
        }
        if let rep = keyValueStore.getDouble("sampling_repPenalty") {
            samplingConfig.repetitionPenalty = Float(rep)
        }
        if let maxTok = keyValueStore.getInt("sampling_maxTokens") {
            samplingConfig.maxTokens = maxTok
        }
        if let prompt = keyValueStore.getString("system_prompt") {
            systemPrompt = prompt
        }
        if let toolsPref = keyValueStore.getInt("tools_enabled") {
            toolsEnabled = toolsPref == 1
        }
    }
}
