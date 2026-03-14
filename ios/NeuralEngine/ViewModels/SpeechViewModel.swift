import SwiftUI

nonisolated enum SpeechModeState: Sendable {
    case idle
    case listening
    case processing
    case speaking
}

nonisolated struct VoiceTranscriptEntry: Identifiable, Sendable {
    let id: UUID
    let role: MessageRole
    let text: String
    let timestamp: Date

    init(id: UUID = UUID(), role: MessageRole, text: String, timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.text = text
        self.timestamp = timestamp
    }
}

@Observable
class SpeechViewModel {
    var state: SpeechModeState = .idle
    var displayText: String = ""
    var responseText: String = ""
    var audioLevel: Float = 0
    var isAuthorized: Bool = false
    var errorMessage: String?
    var conversationTranscript: [VoiceTranscriptEntry] = []
    var isAutoListenEnabled: Bool = true
    var turnCount: Int = 0
    var sessionDuration: TimeInterval = 0

    let recognitionService = SpeechRecognitionService()
    let synthesisService = SpeechSynthesisService()

    private var chatViewModel: ChatViewModel?
    private var sessionStartTime: Date?
    private var sessionTimer: Timer?
    private var bargeInMonitor: Timer?

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
        chatViewModel?.isVoiceMode = true

        if sessionStartTime == nil {
            sessionStartTime = Date()
            startSessionTimer()
        }

        synthesisService.stop()
        startListening()
    }

    func stopConversation() {
        recognitionService.stopListening()
        synthesisService.stop()
        stopBargeInMonitor()
        sessionTimer?.invalidate()
        sessionTimer = nil
        chatViewModel?.isVoiceMode = false
        state = .idle
        displayText = ""
        responseText = ""
        audioLevel = 0
        sessionStartTime = nil
        sessionDuration = 0
    }

    func startListening() {
        state = .listening
        displayText = ""
        responseText = ""
        stopBargeInMonitor()

        do {
            try recognitionService.startListening { [weak self] in
                guard let self else { return }
                let transcript = self.recognitionService.transcript
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !transcript.isEmpty else {
                    if self.isAutoListenEnabled && self.turnCount > 0 {
                        self.startListening()
                    } else {
                        self.state = .idle
                    }
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

    func interruptAndListen() {
        synthesisService.stop()
        stopBargeInMonitor()
        startListening()
    }

    private func processTranscript(_ text: String) {
        state = .processing
        displayText = text
        turnCount += 1

        conversationTranscript.append(
            VoiceTranscriptEntry(role: .user, text: text)
        )

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
            try? await Task.sleep(for: .milliseconds(80))
        }

        guard let lastAssistant = chatViewModel.messages.last(where: {
            $0.role == .assistant && !$0.isToolExecution
        }) else {
            if isAutoListenEnabled {
                startListening()
            } else {
                state = .idle
            }
            return
        }

        let response = lastAssistant.content
        responseText = response
        state = .speaking

        conversationTranscript.append(
            VoiceTranscriptEntry(role: .assistant, text: response)
        )

        startBargeInMonitor()

        synthesisService.speak(
            response,
            bargeInCheck: { [weak self] in
                self?.recognitionService.detectBargeIn() ?? false
            },
            onComplete: { [weak self] in
                guard let self else { return }
                self.stopBargeInMonitor()

                if self.isAutoListenEnabled {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        if self.state == .speaking {
                            self.startListening()
                        }
                    }
                } else {
                    self.state = .idle
                }
            }
        )
    }

    private func startBargeInMonitor() {
        bargeInMonitor = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            guard let self else { return }
            guard self.state == .speaking else { return }

            if self.recognitionService.detectBargeIn() {
                self.interruptAndListen()
            }
        }
    }

    private func stopBargeInMonitor() {
        bargeInMonitor?.invalidate()
        bargeInMonitor = nil
    }

    private func startSessionTimer() {
        sessionTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self, let start = self.sessionStartTime else { return }
            self.sessionDuration = Date().timeIntervalSince(start)
        }
    }

    func updateAudioLevel() {
        audioLevel = recognitionService.audioLevel
    }

    func clearTranscript() {
        conversationTranscript.removeAll()
        turnCount = 0
    }

    var formattedDuration: String {
        let minutes = Int(sessionDuration) / 60
        let seconds = Int(sessionDuration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
