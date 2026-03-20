import Foundation
import Speech
import AVFoundation
import os.log

nonisolated enum SpeechRecognitionFailureReason: String, Sendable {
    case permissionDenied
    case microphonePermissionDenied
    case recognizerUnavailable
    case recognizerBusy
    case offlineRecognizerUnavailable
    case microphoneInputUnavailable
    case noSpeechDetected
    case transientServiceFailure
    case unknown
}

nonisolated enum SpeechRecognizerAvailabilityState: String, Sendable {
    case ready
    case permissionDenied
    case microphonePermissionDenied
    case unavailable
    case busy
    case offlineUnavailable
    case transientFailure
}

nonisolated struct TranscriptSegment: Identifiable, Equatable, Sendable {
    let id: UUID
    let substring: String
    let timestamp: TimeInterval
    let duration: TimeInterval
    let confidence: Float
    let alternativeSubstrings: [String]

    init(id: UUID = UUID(), substring: String, timestamp: TimeInterval, duration: TimeInterval, confidence: Float, alternativeSubstrings: [String] = []) {
        self.id = id
        self.substring = substring
        self.timestamp = timestamp
        self.duration = duration
        self.confidence = confidence
        self.alternativeSubstrings = alternativeSubstrings
    }
}

nonisolated struct TranscriptUpdate: Equatable, Sendable {
    let text: String
    let segments: [TranscriptSegment]
    let isFinal: Bool
    let languageCode: String?
    let averageConfidence: Float
    let stablePrefix: String
    let emittedAt: Date

    var trailingDuration: TimeInterval {
        guard let last = segments.last else { return 0 }
        return last.timestamp + last.duration
    }
}

nonisolated struct TranscriptStabilizationPolicy: Sendable {
    let minimumAverageConfidence: Float
    let silenceWindow: TimeInterval
    let repeatedStableMatches: Int

    static let `default` = TranscriptStabilizationPolicy(minimumAverageConfidence: 0.45, silenceWindow: 0.6, repeatedStableMatches: 2)
}

nonisolated struct TranscriptStabilizer: Sendable {
    private(set) var lastText: String = ""
    private(set) var stableCandidate: String = ""
    private(set) var repeatCount: Int = 0
    private(set) var lastUpdateAt: Date?

    mutating func register(text: String, at date: Date = Date()) -> String {
        if text == lastText {
            repeatCount += 1
            if stableCandidate.isEmpty {
                stableCandidate = text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        } else {
            stableCandidate = Self.sharedPrefix(lhs: stableCandidate.isEmpty ? lastText : stableCandidate, rhs: text)
            repeatCount = 1
            lastText = text
        }
        lastUpdateAt = date
        return stableCandidate
    }

    func shouldEmitPreview(update: TranscriptUpdate, now: Date = Date(), policy: TranscriptStabilizationPolicy = .default) -> Bool {
        if update.isFinal { return !update.text.isEmpty }
        guard !update.stablePrefix.isEmpty else { return false }
        guard update.averageConfidence >= policy.minimumAverageConfidence else { return false }
        guard repeatCount >= policy.repeatedStableMatches else { return false }
        guard let lastUpdateAt else { return false }
        return now.timeIntervalSince(lastUpdateAt) >= policy.silenceWindow
    }

    static func sharedPrefix(lhs: String, rhs: String) -> String {
        guard !lhs.isEmpty, !rhs.isEmpty else { return "" }
        let lhsChars = Array(lhs)
        let rhsChars = Array(rhs)
        let count = min(lhsChars.count, rhsChars.count)
        var index = 0
        while index < count, lhsChars[index] == rhsChars[index] {
            index += 1
        }
        return String(lhsChars.prefix(index)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

@Observable
class SpeechRecognitionService: NSObject {
    private enum SpeechRecognitionError: LocalizedError {
        case microphoneInputUnavailable
        case recognizerUnavailable(String)

        var errorDescription: String? {
            switch self {
            case .microphoneInputUnavailable:
                return "Microphone input unavailable"
            case .recognizerUnavailable(let message):
                return message
            }
        }
    }

    private static let supportedLocales: [(originalIdentifier: String, normalizedIdentifier: String)] =
        SFSpeechRecognizer.supportedLocales()
            .map { locale in
                let identifier = locale.identifier
                return (identifier, normalizedLocaleIdentifier(identifier))
            }
            .sorted { $0.originalIdentifier < $1.originalIdentifier }
    private static let supportedLocaleLookup: [String: String] = Dictionary(
        uniqueKeysWithValues: supportedLocales.map { ($0.normalizedIdentifier, $0.originalIdentifier) }
    )
    static let appContextPhrases: [String] = [
        "NEXUS", "Neural Engine", "Swift LLM", "Core ML", "GGUF", "tool call", "tool calls",
        "open maps", "get current time", "send SMS", "send email", "create calendar event",
        "speech mode", "voice mode", "Context Assembler", "Inference Engine", "KV cache"
    ]

    var transcript: String = ""
    var transcriptUpdate = TranscriptUpdate(text: "", segments: [], isFinal: false, languageCode: nil, averageConfidence: 0, stablePrefix: "", emittedAt: Date())
    var isListening: Bool = false
    var audioLevel: Float = 0
    var error: String?
    var isSpeechDetected: Bool = false
    var detectedLanguageCode: String?
    var availabilityState: SpeechRecognizerAvailabilityState = .ready
    var failureReason: SpeechRecognitionFailureReason?

    var onTranscriptUpdate: ((TranscriptUpdate) -> Void)?

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var isInputTapInstalled = false
    private var silenceTimer: Timer?
    private var lastSpeechTime: Date = Date()
    private var onSilenceDetected: (() -> Void)?
    private var speechStartTime: Date?
    private let logger: Logger
    private var recognitionLanguageCode: String?
    private var transcriptStabilizer = TranscriptStabilizer()

    private var recentLevels: [Float] = []
    private let levelHistorySize = 20
    private var adaptiveThreshold: Float = -40
    private var consecutiveSilenceFrames: Int = 0
    private var consecutiveSpeechFrames: Int = 0
    private let speechOnsetFrames = 3
    private let speechOffsetFrames = 4

    private var dynamicSilenceDuration: TimeInterval = 1.8
    private let minSilenceDuration: TimeInterval = 1.2
    private let maxSilenceDuration: TimeInterval = 3.0

    private static func normalizedLocaleIdentifier(_ identifier: String) -> String {
        identifier.replacingOccurrences(of: "_", with: "-").lowercased()
    }

    private static func primaryLanguageCode(for identifier: String) -> String {
        Locale(identifier: identifier).language.languageCode?.identifier.lowercased()
            ?? normalizedLocaleIdentifier(identifier).split(separator: "-").first.map { String($0) }
            ?? normalizedLocaleIdentifier(identifier)
    }

    override init() {
        let subsystem = Bundle.main.bundleIdentifier ?? "SpeechRecognitionService"
        logger = Logger(subsystem: subsystem, category: "SpeechRecognitionService")
        super.init()
        recognitionLanguageCode = Locale.current.identifier
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: recognitionLanguageCode ?? Locale.current.identifier))
        detectedLanguageCode = recognitionLanguageCode
    }

    @discardableResult
    func setRecognitionLanguage(code: String?) -> String? {
        let resolvedCode = resolveRecognitionLocaleCode(for: code)
        let wasListening = isListening
        let existingOnSilenceDetected = onSilenceDetected

        if wasListening { stopListening() }

        recognitionLanguageCode = resolvedCode
        detectedLanguageCode = resolvedCode
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: resolvedCode))

        if speechRecognizer == nil {
            logger.error("Failed to initialize recognizer for locale: \(resolvedCode, privacy: .public). Falling back to current locale.")
            let fallbackCode = Locale.current.identifier
            recognitionLanguageCode = fallbackCode
            detectedLanguageCode = fallbackCode
            speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: fallbackCode))
        }

        guard let recognizer = speechRecognizer else {
            updateAvailability(.unavailable, reason: .recognizerUnavailable, message: "Speech recognizer unavailable for selected language")
            return nil
        }

        if !recognizer.isAvailable {
            logger.warning("Speech recognizer currently unavailable for locale: \(recognizer.locale.identifier, privacy: .public)")
            updateAvailability(.unavailable, reason: .recognizerUnavailable, message: "Speech recognizer temporarily unavailable")
        } else {
            updateAvailability(.ready, reason: nil, message: nil)
        }

        if wasListening {
            do {
                try startListening(onSilence: existingOnSilenceDetected)
            } catch {
                logger.error("Failed to restart speech recognition after locale change: \(error.localizedDescription, privacy: .public)")
                self.error = error.localizedDescription
            }
        }

        return recognizer.locale.identifier
    }

    private func resolveRecognitionLocaleCode(for code: String?) -> String {
        guard let trimmedCode = code?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmedCode.isEmpty else {
            return Locale.current.identifier
        }

        let normalized = Self.normalizedLocaleIdentifier(trimmedCode)

        if let exactMatch = Self.supportedLocaleLookup[normalized] {
            return exactMatch
        }

        let targetLanguage = Self.primaryLanguageCode(for: normalized)

        if let match = Self.supportedLocales.first(where: {
            Self.primaryLanguageCode(for: $0.originalIdentifier) == targetLanguage
        }) {
            logger.warning("No exact speech recognition locale for \(trimmedCode, privacy: .public); using \(match.originalIdentifier, privacy: .public) instead.")
            return match.originalIdentifier
        }

        logger.warning("No supported speech recognition locale for \(trimmedCode, privacy: .public); using current locale.")
        return Locale.current.identifier
    }

    func requestAuthorization() async -> Bool {
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        guard speechStatus == .authorized else {
            updateAvailability(.permissionDenied, reason: .permissionDenied, message: "Speech recognition permission was denied")
            return false
        }

        let audioStatus = await AVAudioApplication.requestRecordPermission()
        guard audioStatus else {
            updateAvailability(.microphonePermissionDenied, reason: .microphonePermissionDenied, message: "Microphone access was denied")
            return false
        }

        updateAvailability(.ready, reason: nil, message: nil)
        return true
    }

    func startListening(onSilence: (() -> Void)? = nil) throws {
        guard let speechRecognizer else {
            updateAvailability(.unavailable, reason: .recognizerUnavailable, message: "Speech recognizer unavailable")
            throw SpeechRecognitionError.recognizerUnavailable("Speech recognizer unavailable")
        }
        guard speechRecognizer.isAvailable else {
            updateAvailability(.busy, reason: .recognizerBusy, message: "Speech service is busy. Please try again.")
            throw SpeechRecognitionError.recognizerUnavailable("Speech recognizer unavailable")
        }

        stopListening()
        onSilenceDetected = onSilence
        transcript = ""
        transcriptUpdate = TranscriptUpdate(text: "", segments: [], isFinal: false, languageCode: recognitionLanguageCode, averageConfidence: 0, stablePrefix: "", emittedAt: Date())
        error = nil
        isSpeechDetected = false
        consecutiveSilenceFrames = 0
        consecutiveSpeechFrames = 0
        recentLevels.removeAll()
        dynamicSilenceDuration = 1.8
        speechStartTime = nil
        transcriptStabilizer = TranscriptStabilizer()
        updateAvailability(.ready, reason: nil, message: nil)

        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .duckOthers, .allowBluetooth])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        resetAudioEngine()

        let inputNode = audioEngine.inputNode
        let hardwareFormat = inputNode.inputFormat(forBus: 0)
        guard hardwareFormat.channelCount > 0, hardwareFormat.sampleRate > 0 else {
            logger.error("Invalid hardware input format for speech recognition tap. Channels: \(hardwareFormat.channelCount), sample rate: \(hardwareFormat.sampleRate, privacy: .public)")
            stopListening()
            let microphoneError = SpeechRecognitionError.microphoneInputUnavailable
            updateAvailability(.unavailable, reason: .microphoneInputUnavailable, message: microphoneError.localizedDescription)
            throw microphoneError
        }
        let tapFormat = inputNode.outputFormat(forBus: 0)

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }

        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = speechRecognizer.supportsOnDeviceRecognition
        recognitionRequest.taskHint = .dictation
        recognitionRequest.contextualStrings = Self.appContextPhrases
        if #available(iOS 18, *) {
            recognitionRequest.addsPunctuation = true
        }

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, taskError in
            DispatchQueue.main.async {
                guard let self else { return }

                if let result {
                    self.handleRecognitionResult(result)
                }

                if let taskError {
                    self.handleRecognitionError(taskError)
                }
            }
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
            self?.processAudioLevel(buffer: buffer)
        }
        isInputTapInstalled = true

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
            startAdaptiveSilenceDetection()
        } catch {
            logger.error("Failed to start audio engine for speech recognition: \(error.localizedDescription, privacy: .public)")
            stopListening()
            updateAvailability(.transientFailure, reason: .transientServiceFailure, message: error.localizedDescription)
            throw error
        }
    }

    private func handleRecognitionResult(_ result: SFSpeechRecognitionResult) {
        transcript = result.bestTranscription.formattedString
        lastSpeechTime = Date()
        detectedLanguageCode = speechRecognizer?.locale.identifier ?? recognitionLanguageCode

        if !isSpeechDetected {
            isSpeechDetected = true
            speechStartTime = Date()
        }

        updateDynamicSilenceDuration()

        let segments = result.bestTranscription.segments.map {
            TranscriptSegment(
                substring: $0.substring,
                timestamp: $0.timestamp,
                duration: $0.duration,
                confidence: $0.confidence,
                alternativeSubstrings: $0.alternativeSubstrings
            )
        }
        let avgConfidence = segments.isEmpty ? 0 : segments.reduce(0) { $0 + $1.confidence } / Float(segments.count)
        let emittedAt = Date()
        let stablePrefix = transcriptStabilizer.register(text: transcript, at: emittedAt)
        let update = TranscriptUpdate(
            text: transcript,
            segments: segments,
            isFinal: result.isFinal,
            languageCode: detectedLanguageCode ?? recognitionLanguageCode,
            averageConfidence: avgConfidence,
            stablePrefix: stablePrefix,
            emittedAt: emittedAt
        )
        transcriptUpdate = update
        onTranscriptUpdate?(update)

        if result.isFinal {
            stopListening()
            onSilenceDetected?()
        }
    }

    private func handleRecognitionError(_ taskError: Error) {
        let nsError = taskError as NSError
        switch (nsError.domain, nsError.code) {
        case ("kAFAssistantErrorDomain", 1110):
            logger.info("Speech recognizer reported no speech detected (1110). Transcript length: \(self.transcript.count, privacy: .public)")
            updateAvailability(.ready, reason: .noSpeechDetected, message: transcript.isEmpty ? "No speech detected" : nil)
            if !transcript.isEmpty {
                onSilenceDetected?()
            } else {
                stopListening()
            }
        case ("kAFAssistantErrorDomain", 203):
            logger.warning("Speech recognizer transient failure / timeout (203): \(taskError.localizedDescription, privacy: .public)")
            stopListening()
            updateAvailability(.transientFailure, reason: .transientServiceFailure, message: taskError.localizedDescription)
        case ("kAFAssistantErrorDomain", 209):
            logger.warning("Speech recognizer operation failed / contention (209): \(taskError.localizedDescription, privacy: .public)")
            stopListening()
            updateAvailability(.transientFailure, reason: .transientServiceFailure, message: taskError.localizedDescription)
        case (_, 216):
            logger.warning("Speech recognizer reported cancellation/interruption (216): \(taskError.localizedDescription, privacy: .public)")
            stopListening()
            updateAvailability(.transientFailure, reason: .transientServiceFailure, message: taskError.localizedDescription)
        default:
            logger.error("Speech recognizer error \(nsError.domain, privacy: .public):\(nsError.code) - \(taskError.localizedDescription, privacy: .public)")
            stopListening()
            updateAvailability(.transientFailure, reason: .transientServiceFailure, message: taskError.localizedDescription)
        }
    }

    func stopListening() {
        silenceTimer?.invalidate()
        silenceTimer = nil

        resetAudioEngine()

        recognitionRequest?.endAudio()
        recognitionRequest = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        isListening = false
        audioLevel = 0
        isSpeechDetected = false
    }

    private func updateAvailability(_ state: SpeechRecognizerAvailabilityState, reason: SpeechRecognitionFailureReason?, message: String?) {
        availabilityState = state
        failureReason = reason
        error = message
    }

    private func resetAudioEngine() {
        if audioEngine.isRunning { audioEngine.stop() }
        if isInputTapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            isInputTapInstalled = false
        }
        audioEngine.reset()
    }

    private func processAudioLevel(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)
        var sumOfSquares: Float = 0
        for i in 0..<frameLength { sumOfSquares += channelData[i] * channelData[i] }
        let rms = sqrt(sumOfSquares / Float(frameLength))
        let db = 20 * log10(max(rms, 0.00001))

        recentLevels.append(db)
        if recentLevels.count > levelHistorySize { recentLevels.removeFirst() }
        updateAdaptiveThreshold()

        let isSpeechFrame = db > adaptiveThreshold
        if isSpeechFrame {
            consecutiveSpeechFrames += 1
            consecutiveSilenceFrames = 0
        } else {
            consecutiveSilenceFrames += 1
            if consecutiveSilenceFrames > speechOffsetFrames { consecutiveSpeechFrames = 0 }
        }

        let normalized = max(0, min(1, (db + 50) / 50))
        Task { @MainActor in self.audioLevel = normalized }
    }

    private func updateAdaptiveThreshold() {
        guard recentLevels.count >= 5 else { return }
        let sorted = recentLevels.sorted()
        let noiseFloor = sorted[sorted.count / 4]
        adaptiveThreshold = noiseFloor + 8
    }

    private func updateDynamicSilenceDuration() {
        let wordCount = transcript.split(separator: " ").count
        if wordCount <= 3 {
            dynamicSilenceDuration = minSilenceDuration
        } else if wordCount <= 10 {
            dynamicSilenceDuration = 1.5
        } else {
            dynamicSilenceDuration = 2.0
        }

        let lowered = transcript.lowercased()
        let endsWithConjunction = lowered.hasSuffix(" and") || lowered.hasSuffix(" but") || lowered.hasSuffix(" or") || lowered.hasSuffix(" because") || lowered.hasSuffix(" so") || lowered.hasSuffix(" then")
        if endsWithConjunction { dynamicSilenceDuration = maxSilenceDuration }
        if transcript.hasSuffix(",") { dynamicSilenceDuration = min(dynamicSilenceDuration + 0.5, maxSilenceDuration) }
    }

    private func startAdaptiveSilenceDetection() {
        lastSpeechTime = Date()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
            guard let self else { return }
            let elapsed = Date().timeIntervalSince(self.lastSpeechTime)

            guard self.isSpeechDetected else {
                if elapsed > 8.0 {
                    self.stopListening()
                    self.onSilenceDetected?()
                }
                return
            }

            let trimmed = self.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }

            if elapsed > self.dynamicSilenceDuration {
                self.stopListening()
                self.onSilenceDetected?()
            }
        }
    }


    func shouldEmitPreview(for update: TranscriptUpdate, policy: TranscriptStabilizationPolicy = .default, now: Date = Date()) -> Bool {
        transcriptStabilizer.shouldEmitPreview(update: update, now: now, policy: policy)
    }

    func detectBargeIn() -> Bool {
        consecutiveSpeechFrames >= speechOnsetFrames && audioLevel > 0.15
    }
}
