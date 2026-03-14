import SwiftUI

nonisolated enum SpeechModeState: Sendable {
    case idle
    case listening
    case processing
    case speaking
}

@Observable
class SpeechViewModel {
    var state: SpeechModeState = .idle
    var displayText: String = ""
    var responseText: String = ""
    var audioLevel: Float = 0
    var isAuthorized: Bool = false
    var errorMessage: String?

    let recognitionService = SpeechRecognitionService()
    let synthesisService = SpeechSynthesisService()

    private var chatViewModel: ChatViewModel?

    func attach(to chatViewModel: ChatViewModel) {
        self.chatViewModel = chatViewModel
    }

    func requestPermissions() async {
        isAuthorized = await recognitionService.requestAuthorization()
        if !isAuthorized {
            errorMessage = recognitionService.error ?? "Permissions required for speech mode"
        }
    }

    func startConversation() {
        guard isAuthorized else { return }
        errorMessage = nil
        synthesisService.stop()
        startListening()
    }

    func stopConversation() {
        recognitionService.stopListening()
        synthesisService.stop()
        state = .idle
        displayText = ""
        responseText = ""
        audioLevel = 0
    }

    func startListening() {
        state = .listening
        displayText = ""
        responseText = ""

        do {
            try recognitionService.startListening { [weak self] in
                guard let self else { return }
                let transcript = self.recognitionService.transcript
                guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    self.state = .idle
                    return
                }
                self.displayText = transcript
                self.processTranscript(transcript)
            }
        } catch {
            errorMessage = error.localizedDescription
            state = .idle
        }
    }

    private func processTranscript(_ text: String) {
        state = .processing
        displayText = text

        guard let chatViewModel else {
            state = .idle
            return
        }

        chatViewModel.inputText = text
        chatViewModel.sendMessage()

        Task {
            await waitForResponse()
        }
    }

    private func waitForResponse() async {
        guard let chatViewModel else { return }

        while chatViewModel.isGenerating || chatViewModel.isExecutingTools {
            try? await Task.sleep(for: .milliseconds(100))
        }

        guard let lastAssistant = chatViewModel.messages.last(where: { $0.role == .assistant && !$0.isToolExecution }) else {
            state = .idle
            return
        }

        let response = lastAssistant.content
        responseText = response
        state = .speaking

        synthesisService.speak(response) { [weak self] in
            guard let self else { return }
            self.state = .idle
        }
    }

    func updateAudioLevel() {
        audioLevel = recognitionService.audioLevel
    }
}
