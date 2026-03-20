import SwiftUI
import AVFAudio

nonisolated enum SpeechModeState: Sendable {
    case idle
    case listening
    case processing
    case speaking
    case timedOut
    case error
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

nonisolated struct SpeechLanguageOption: Identifiable, Hashable, Sendable {
    let code: String
    let label: String

    var id: String { code }
}

@Observable
class SpeechViewModel {
    var state: SpeechModeState = .idle
    var displayText: String = ""
    var responseText: String = ""
    var audioLevel: Float = 0
    var isAuthorized: Bool = false
    var errorMessage: String?
    var lastError: WrappedError?
    var conversationTranscript: [VoiceTranscriptEntry] = []
    var isAutoListenEnabled: Bool = true
    var turnCount: Int = 0
    var sessionDuration: TimeInterval = 0
    var statusMessage: String = "Idle"
    var consecutiveTimeouts: Int = 0
    var lastTranscriptUpdate: TranscriptUpdate?
    var currentRecognitionLanguageCode: String?
    var pendingPreviewTranscript: String = ""

    let recognitionService = SpeechRecognitionService()
    let synthesisService = SpeechSynthesisService()
    var onSpeechSettingsChanged: ((String?, String?) -> Void)?

    private var chatViewModel: ChatViewModel?
    private var sessionStartTime: Date?
    private var sessionTimer: Timer?
    private var bargeInMonitor: Timer?
    private var responseWaitTask: Task<Void, Never>?

    private let inactivityTimeoutSeconds: TimeInterval = 15
    private let maxConsecutiveTimeouts = 3

    private let noiseWords: Set<String> = [
        "um", "uh", "ah", "eh", "hmm", "hm", "mm",
        "bye", "goodbye", "bye bye",
        "thank you", "thanks", "thank",
        "okay", "ok",
        "hello", "hi", "hey",
        "yeah", "yep", "nah", "nope",
        "oh", "ooh", "ugh",
    ]

    private let silenceThresholdDB: Float = -35
    private let previewPolicy = TranscriptStabilizationPolicy.default

    init() {
        recognitionService.onTranscriptUpdate = { [weak self] update in
            Task { @MainActor [weak self] in
                self?.handleTranscriptUpdate(update)
            }
        }
    }

    func attach(to chatViewModel: ChatViewModel) {
        self.chatViewModel = chatViewModel
    }

    func dismissError() {
        lastError = nil
        errorMessage = nil
        if state == .error || state == .timedOut {
            state = .idle
        }
    }

    func requestPermissions() async {
        isAuthorized = await recognitionService.requestAuthorization()
        if !isAuthorized {
            let msg = recognitionService.error ?? "Permissions required for speech mode"
            errorMessage = msg
            lastError = WrappedError(
                domain: .speech,
                severity: .warning,
                userMessage: msg,
                technicalDetail: "SFSpeechRecognizer or AVAudioSession authorization denied",
                recoveryAction: .none
            )
        }
    }

    func startConversation() {
        guard isAuthorized else { return }
        errorMessage = nil
        lastError = nil
        consecutiveTimeouts = 0
        chatViewModel?.isVoiceMode = true
        statusMessage = "Starting..."

        if sessionStartTime == nil {
            sessionStartTime = Date()
            startSessionTimer()
        }

        synthesisService.stop()
        let selectedLanguage = synthesisService.currentPreferredLanguageCode()
        currentRecognitionLanguageCode = recognitionService.setRecognitionLanguage(code: selectedLanguage)
        let synchronizedLanguage = ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: selectedLanguage, detectedRecognitionLanguageCode: currentRecognitionLanguageCode)
        synthesisService.syncDetectedLanguage(synchronizedLanguage)
        chatViewModel?.speechLanguageCode = synchronizedLanguage
        startListening()
    }

    func stopConversation() {
        responseWaitTask?.cancel()
        responseWaitTask = nil
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
        consecutiveTimeouts = 0
        statusMessage = "Idle"
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    func startListening() {
        state = .listening
        displayText = ""
        responseText = ""
        stopBargeInMonitor()
        statusMessage = "Listening..."
        consecutiveTimeouts = 0
        pendingPreviewTranscript = ""
        lastTranscriptUpdate = nil

        do {
            try recognitionService.startListening { [weak self] in
                guard let self else { return }
                let transcript = self.recognitionService.transcript
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                self.syncLanguageFromRecognitionUpdate(self.recognitionService.transcriptUpdate)
                guard !transcript.isEmpty else {
                    if self.isAutoListenEnabled && self.turnCount > 0 {
                        self.startListening()
                    } else {
                        self.state = .idle
                        self.statusMessage = "Ready"
                    }
                    return
                }
                self.displayText = transcript
                self.processTranscript(transcript)
            }
        } catch {
            let wrapped = wrappedSpeechError(error)
            lastError = wrapped
            errorMessage = wrapped.userMessage
            state = .error
            statusMessage = "Speech error"
        }
    }


    private func handleTranscriptUpdate(_ update: TranscriptUpdate) {
        lastTranscriptUpdate = update
        displayText = update.text
        syncLanguageFromRecognitionUpdate(update)

        if recognitionService.detectBargeIn() && state == .speaking {
            interruptAndListen()
            return
        }

        guard state == .listening else { return }

        if recognitionService.shouldEmitPreview(for: update, policy: previewPolicy) {
            let preview = update.stablePrefix.isEmpty ? update.text : update.stablePrefix
            guard !preview.isEmpty, preview != pendingPreviewTranscript else { return }
            pendingPreviewTranscript = preview
            statusMessage = "Preparing response..."
            prepareInferencePreview(for: preview, languageCode: update.languageCode)
        }
    }

    private func prepareInferencePreview(for text: String, languageCode: String?) {
        guard let chatViewModel else { return }
        let synchronizedLanguage = ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: chatViewModel.speechLanguageCode, detectedRecognitionLanguageCode: languageCode)
        let frame = CognitionEngine.process(userText: text, conversationHistory: chatViewModel.messages, memoryService: chatViewModel.memoryService)
        let memoryResults = chatViewModel.memoryService.searchMemories(query: text, maxResults: 4)
        let associativeResults = chatViewModel.memoryService.getAssociativeMemories(query: text, directResults: memoryResults)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: memoryResults + associativeResults, conversationHistory: chatViewModel.messages, toolsEnabled: chatViewModel.toolsEnabled, isVoiceMode: true, preferredResponseLanguageCode: synchronizedLanguage, detectedRecognitionLanguageCode: languageCode)
        var previewMessages = [["role": "system", "content": prompt]]
        for msg in chatViewModel.messages where msg.role != .system {
            previewMessages.append(["role": msg.role.rawValue, "content": msg.content])
        }
        previewMessages.append(["role": MessageRole.user.rawValue, "content": text])
        chatViewModel.inferenceEngine.prepareVoiceContext(messages: previewMessages, systemPrompt: prompt)
    }

    private func syncLanguageFromRecognitionUpdate(_ update: TranscriptUpdate?) {
        guard let update else { return }
        let synchronizedLanguage = ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: synthesisService.currentPreferredLanguageCode(), detectedRecognitionLanguageCode: update.languageCode)
        currentRecognitionLanguageCode = update.languageCode
        if let synchronizedLanguage {
            _ = synthesisService.syncDetectedLanguage(synchronizedLanguage)
            chatViewModel?.speechLanguageCode = synchronizedLanguage
        }
    }

    private func wrappedSpeechError(_ error: Error) -> WrappedError {
        if let reason = recognitionService.failureReason {
            switch reason {
            case .permissionDenied:
                return WrappedError(domain: .speech, severity: .warning, userMessage: "Speech recognition permission was denied.", technicalDetail: "SFSpeechRecognizer authorization denied", recoveryAction: .none, underlyingError: error)
            case .microphonePermissionDenied:
                return WrappedError(domain: .speech, severity: .warning, userMessage: "Microphone access is required for voice mode.", technicalDetail: "AVAudioSession record permission denied", recoveryAction: .none, underlyingError: error)
            case .offlineRecognizerUnavailable:
                return WrappedError(domain: .speech, severity: .warning, userMessage: "Offline speech recognition is unavailable for this language.", technicalDetail: "On-device recognizer missing for locale \(currentRecognitionLanguageCode ?? "unknown")", recoveryAction: .retry, underlyingError: error)
            case .recognizerBusy:
                return WrappedError(domain: .speech, severity: .warning, userMessage: "Speech service is busy. Please try again in a moment.", technicalDetail: "Recognizer contention / assistant service busy", recoveryAction: .retry, underlyingError: error)
            case .recognizerUnavailable:
                return WrappedError(domain: .speech, severity: .warning, userMessage: "Speech service is unavailable.", technicalDetail: "Speech recognizer is not available for the requested locale or current device state", recoveryAction: .retry, underlyingError: error)
            case .microphoneInputUnavailable:
                return WrappedError(domain: .speech, severity: .error, userMessage: "Microphone input is unavailable.", technicalDetail: "Audio engine input format was invalid or no microphone input route was available", recoveryAction: .none, underlyingError: error)
            case .transientServiceFailure, .noSpeechDetected, .unknown:
                break
            }
        }
        return NativeErrorWrapper.synthesize(error)
    }

    func interruptAndListen() {
        synthesisService.stop()
        stopBargeInMonitor()
        responseWaitTask?.cancel()
        responseWaitTask = nil
        startListening()
    }

    private func isNoiseTranscription(_ text: String) -> Bool {
        let cleaned = text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "[^a-z\\s]", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)

        if cleaned.isEmpty { return true }

        if noiseWords.contains(cleaned) { return true }

        let words = cleaned.split(separator: " ").map(String.init)
        if words.count <= 2 && words.allSatisfy({ noiseWords.contains($0) }) {
            return true
        }

        return false
    }

    private func processTranscript(_ text: String) {
        if isNoiseTranscription(text) {
            statusMessage = "Filtered noise"
            if isAutoListenEnabled {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
                    self?.startListening()
                }
            } else {
                state = .idle
                statusMessage = "Ready"
            }
            return
        }

        state = .processing
        displayText = text
        turnCount += 1
        statusMessage = "Processing..."

        conversationTranscript.append(
            VoiceTranscriptEntry(role: .user, text: text)
        )

        guard let chatViewModel else {
            state = .idle
            statusMessage = "No chat engine"
            return
        }

        chatViewModel.inputText = text
        chatViewModel.sendMessage()

        responseWaitTask?.cancel()
        responseWaitTask = Task {
            await waitForResponseWithTimeout()
        }
    }

    private func waitForResponseWithTimeout() async {
        guard let chatViewModel else { return }

        let deadline = Date().addingTimeInterval(inactivityTimeoutSeconds)

        while chatViewModel.isGenerating || chatViewModel.isExecutingTools {
            if Date() > deadline {
                handleResponseTimeout()
                return
            }
            if Task.isCancelled { return }
            try? await Task.sleep(for: .milliseconds(80))
        }

        if Task.isCancelled { return }

        if let chatError = chatViewModel.lastError, chatError.severity >= .error {
            lastError = WrappedError(
                domain: .inference,
                severity: .warning,
                userMessage: "Generation failed: \(chatError.userMessage)",
                technicalDetail: chatError.technicalDetail,
                recoveryAction: chatError.recoveryAction
            )
            statusMessage = "Generation error"
            resumeAfterError()
            return
        }

        guard let lastAssistant = chatViewModel.messages.last(where: {
            $0.role == .assistant && !$0.isToolExecution
        }) else {
            statusMessage = "No response"
            resumeAfterError()
            return
        }

        let response = lastAssistant.content
        guard !response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusMessage = "Empty response"
            resumeAfterError()
            return
        }

        consecutiveTimeouts = 0
        if let synchronizedLanguage = ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: chatViewModel.speechLanguageCode, detectedRecognitionLanguageCode: currentRecognitionLanguageCode) {
            _ = synthesisService.syncDetectedLanguage(synchronizedLanguage)
            chatViewModel.speechLanguageCode = synchronizedLanguage
        }
        responseText = response
        state = .speaking
        statusMessage = "Speaking..."

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
                self.statusMessage = "Ready"

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

    private func handleResponseTimeout() {
        consecutiveTimeouts += 1
        statusMessage = "Response timed out"

        if consecutiveTimeouts >= maxConsecutiveTimeouts {
            state = .timedOut
            lastError = WrappedError(
                domain: .inference,
                severity: .error,
                userMessage: "Response timed out repeatedly. The model may be stuck.",
                technicalDetail: "Inactivity timeout after \(Int(inactivityTimeoutSeconds))s, \(consecutiveTimeouts) consecutive timeouts",
                recoveryAction: .restartSession
            )
            chatViewModel?.stopGeneration()
            return
        }

        lastError = WrappedError(
            domain: .inference,
            severity: .warning,
            userMessage: "Response took too long. Retrying...",
            technicalDetail: "Inactivity timeout after \(Int(inactivityTimeoutSeconds))s",
            recoveryAction: .retry
        )
        chatViewModel?.stopGeneration()

        resumeAfterError()
    }

    private func resumeAfterError() {
        if isAutoListenEnabled && consecutiveTimeouts < maxConsecutiveTimeouts {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                guard let self else { return }
                if self.state == .processing || self.state == .error {
                    self.startListening()
                }
            }
        } else {
            state = .idle
        }
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

    private func normalizedSpeechLanguageCode(_ code: String?) -> String? {
        guard let trimmed = code?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    @discardableResult
    private func syncSpeechSettingsAndNotify() -> (voiceIdentifier: String?, languageCode: String?) {
        let resolvedLanguageCode = ContextAssembler.synchronizedResponseLanguage(preferredResponseLanguageCode: normalizedSpeechLanguageCode(synthesisService.effectiveSpeechLanguageCode()), detectedRecognitionLanguageCode: currentRecognitionLanguageCode)
        currentRecognitionLanguageCode = recognitionService.setRecognitionLanguage(code: resolvedLanguageCode)
        let resolvedSettings = (synthesisService.currentVoiceIdentifier(), resolvedLanguageCode)
        onSpeechSettingsChanged?(resolvedSettings.0, resolvedSettings.1)
        return resolvedSettings
    }

    @discardableResult
    func initializeFromPersistedSettings(voiceIdentifier: String?, languageCode: String?) -> (voiceIdentifier: String?, languageCode: String?) {
        let normalizedLanguageCode = normalizedSpeechLanguageCode(languageCode)
        _ = synthesisService.applyPersistedSettings(voiceIdentifier: voiceIdentifier, languageCode: normalizedLanguageCode)
        return syncSpeechSettingsAndNotify()
    }


    var selectedSpeechLanguageCode: String? {
        synthesisService.currentPreferredLanguageCode()
    }

    var selectedSpeechVoiceIdentifier: String? {
        synthesisService.currentVoiceIdentifier()
    }

    func speechLanguageOptions() -> [SpeechLanguageOption] {
        let languages = Set(synthesisService.availableVoices().map(\.language))
        return languages.sorted().map { code in
            let localeName = Locale.current.localizedString(forIdentifier: code) ?? code
            return SpeechLanguageOption(code: code, label: localeName)
        }
    }

    func speechVoices(for languageCode: String?) -> [SpeechSynthesisService.VoiceOption] {
        let voices = synthesisService.availableVoices()
        let filteredVoices: [SpeechSynthesisService.VoiceOption]

        if let languageCode {
            filteredVoices = voices.filter { $0.language == languageCode }
        } else {
            filteredVoices = voices
        }

        return filteredVoices.sorted { lhs, rhs in
            if lhs.quality == rhs.quality {
                return lhs.displayName < rhs.displayName
            }
            return lhs.quality.rawValue > rhs.quality.rawValue
        }
    }

    func updateSpeechLanguage(code: String?) {
        synthesisService.setLanguagePreferred(normalizedSpeechLanguageCode(code))
        _ = syncSpeechSettingsAndNotify()
    }

    func updateSpeechVoice(identifier: String?) {
        synthesisService.setVoice(identifier: identifier)
        _ = syncSpeechSettingsAndNotify()
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
