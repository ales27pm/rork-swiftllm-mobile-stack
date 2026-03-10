import SwiftUI

@Observable
@MainActor
class ChatViewModel {
    var messages: [Message] = []
    var inputText: String = ""
    var isGenerating: Bool = false
    var samplingConfig = SamplingConfig()
    var systemPrompt: String = "You are a helpful, concise AI assistant running locally on-device."

    let inferenceEngine: InferenceEngine
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let keyValueStore: KeyValueStore
    let database: DatabaseService

    init(inferenceEngine: InferenceEngine, metricsLogger: MetricsLogger, thermalGovernor: ThermalGovernor, modelLoader: ModelLoaderService, keyValueStore: KeyValueStore, database: DatabaseService) {
        self.inferenceEngine = inferenceEngine
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.modelLoader = modelLoader
        self.keyValueStore = keyValueStore
        self.database = database
        inferenceEngine.attachRunner(modelLoader.modelRunner, tokenizer: modelLoader.tokenizer)
        restoreSettings()
    }

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard !isGenerating else { return }

        let userMessage = Message(role: .user, content: text)
        messages.append(userMessage)
        inputText = ""

        let assistantMessage = Message(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        isGenerating = true

        let assistantIndex = messages.count - 1

        var chatMessages: [[String: String]] = [
            ["role": "system", "content": systemPrompt]
        ]
        for msg in messages where msg.role != .system {
            if msg.role == .assistant && msg.content.isEmpty { continue }
            chatMessages.append(["role": msg.role.rawValue, "content": msg.content])
        }

        inferenceEngine.generate(
            messages: chatMessages,
            systemPrompt: systemPrompt,
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
        _ = database.execute("DELETE FROM chat_history;")
    }

    var activeModelName: String {
        modelLoader.activeModel?.name ?? "No Model"
    }

    var hasActiveModel: Bool {
        modelLoader.activeModelID != nil
    }

    func saveSettings() {
        keyValueStore.setDouble(Double(samplingConfig.temperature), forKey: "sampling_temperature")
        keyValueStore.setInt(samplingConfig.topK, forKey: "sampling_topK")
        keyValueStore.setDouble(Double(samplingConfig.topP), forKey: "sampling_topP")
        keyValueStore.setDouble(Double(samplingConfig.repetitionPenalty), forKey: "sampling_repPenalty")
        keyValueStore.setInt(samplingConfig.maxTokens, forKey: "sampling_maxTokens")
        keyValueStore.setString(systemPrompt, forKey: "system_prompt")
    }

    func persistMessage(_ message: Message) {
        _ = database.execute(
            "INSERT OR REPLACE INTO chat_history (id, role, content, timestamp, model_id) VALUES (?, ?, ?, ?, ?);",
            params: [message.id.uuidString, message.role.rawValue, message.content, message.timestamp.timeIntervalSince1970, modelLoader.activeModelID ?? ""]
        )
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
