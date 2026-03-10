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

    init(inferenceEngine: InferenceEngine, metricsLogger: MetricsLogger, thermalGovernor: ThermalGovernor, modelLoader: ModelLoaderService) {
        self.inferenceEngine = inferenceEngine
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.modelLoader = modelLoader
        inferenceEngine.attachRunner(modelLoader.modelRunner, tokenizer: modelLoader.tokenizer)
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
    }

    var activeModelName: String {
        modelLoader.activeModel?.name ?? "No Model"
    }

    var hasActiveModel: Bool {
        modelLoader.activeModelID != nil
    }
}
