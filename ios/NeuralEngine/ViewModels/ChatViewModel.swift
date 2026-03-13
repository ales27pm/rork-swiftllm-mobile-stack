import SwiftUI

@Observable
@MainActor
class ChatViewModel {
    var messages: [Message] = []
    var inputText: String = ""
    var isGenerating: Bool = false
    var samplingConfig = SamplingConfig()
    var systemPrompt: String = "You are Nexus, a helpful and intelligent AI assistant running locally on-device. You have access to memory from past conversations and use it to provide personalized, contextual responses."

    var currentConversationId: String?

    let inferenceEngine: InferenceEngine
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let keyValueStore: KeyValueStore
    let database: DatabaseService
    let conversationService: ConversationService
    let memoryService: MemoryService

    init(
        inferenceEngine: InferenceEngine,
        metricsLogger: MetricsLogger,
        thermalGovernor: ThermalGovernor,
        modelLoader: ModelLoaderService,
        keyValueStore: KeyValueStore,
        database: DatabaseService,
        conversationService: ConversationService,
        memoryService: MemoryService
    ) {
        self.inferenceEngine = inferenceEngine
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.modelLoader = modelLoader
        self.keyValueStore = keyValueStore
        self.database = database
        self.conversationService = conversationService
        self.memoryService = memoryService
        inferenceEngine.attachRunner(modelLoader.modelRunner, llamaRunner: modelLoader.llamaRunner, tokenizer: modelLoader.tokenizer, format: modelLoader.activeFormat)
        restoreSettings()
    }

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard !isGenerating else { return }

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

        let assistantMessage = Message(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        isGenerating = true

        let assistantIndex = messages.count - 1

        let memoryContext = memoryService.buildContextInjection(query: text)

        var enrichedSystemPrompt = systemPrompt
        if !memoryContext.isEmpty {
            enrichedSystemPrompt += "\n\n" + memoryContext
        }

        var chatMessages: [[String: String]] = [
            ["role": "system", "content": enrichedSystemPrompt]
        ]
        for msg in messages where msg.role != .system {
            if msg.role == .assistant && msg.content.isEmpty { continue }
            chatMessages.append(["role": msg.role.rawValue, "content": msg.content])
        }

        inferenceEngine.generate(
            messages: chatMessages,
            systemPrompt: enrichedSystemPrompt,
            samplingConfig: samplingConfig,
            onToken: { [weak self] token in
                guard let self else { return }
                self.messages[assistantIndex].content += token
            },
            onComplete: { [weak self] metrics in
                guard let self else { return }
                self.messages[assistantIndex].isStreaming = false
                self.messages[assistantIndex].metrics = metrics
                self.isGenerating = false

                if let convId = self.currentConversationId {
                    self.conversationService.saveMessage(self.messages[assistantIndex], conversationId: convId)

                    let userMsgCount = self.messages.filter { $0.role == .user }.count
                    let assistantContent = self.messages[assistantIndex].content
                    var conv = self.conversationService.conversations.first { $0.id == convId } ?? Conversation(id: convId)
                    conv.lastMessage = String(assistantContent.prefix(100))
                    conv.messageCount = self.messages.filter { $0.role != .system }.count
                    conv.updatedAt = Date()
                    if userMsgCount == 1 {
                        conv.title = self.conversationService.generateTitle(from: text)
                    }
                    self.conversationService.updateConversation(conv)

                    self.memoryService.extractAndStoreMemory(userText: text, assistantText: assistantContent)
                }
            }
        )
    }

    func stopGeneration() {
        inferenceEngine.cancel()
        if let lastIndex = messages.indices.last, messages[lastIndex].role == .assistant {
            messages[lastIndex].isStreaming = false
        }
        isGenerating = false
    }

    func clearChat() {
        inferenceEngine.resetSession()
        messages.removeAll()
        isGenerating = false
        currentConversationId = nil
    }

    func newConversation() {
        inferenceEngine.resetSession()
        messages.removeAll()
        isGenerating = false
        currentConversationId = nil
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
    }
}
