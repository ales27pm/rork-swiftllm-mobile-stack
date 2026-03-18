import Foundation
import Speech
import AVFoundation
import os.log

@Observable
class SpeechRecognitionService: NSObject {
    var transcript: String = ""
    var isListening: Bool = false
    var audioLevel: Float = 0
    var error: String?
    var isSpeechDetected: Bool = false

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var silenceTimer: Timer?
    private var lastSpeechTime: Date = Date()
    private var onSilenceDetected: (() -> Void)?
    private var speechStartTime: Date?
    private let logger: Logger
    private var recognitionLanguageCode: String?

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

    override init() {
        let subsystem = Bundle.main.bundleIdentifier ?? "SpeechRecognitionService"
        logger = Logger(subsystem: subsystem, category: "SpeechRecognitionService")
        super.init()
        recognitionLanguageCode = Locale.current.identifier
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: recognitionLanguageCode ?? Locale.current.identifier))
    }


    @discardableResult
    func setRecognitionLanguage(code: String?) -> String? {
        let resolvedCode = resolveRecognitionLocaleCode(for: code)
        recognitionLanguageCode = resolvedCode
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: resolvedCode))

        if speechRecognizer == nil {
            logger.error("Failed to initialize recognizer for locale: \(resolvedCode, privacy: .public). Falling back to current locale.")
            let fallbackCode = Locale.current.identifier
            recognitionLanguageCode = fallbackCode
            speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: fallbackCode))
        }

        guard let recognizer = speechRecognizer else {
            error = "Speech recognizer unavailable for selected language"
            return nil
        }

        if !recognizer.isAvailable {
            logger.warning("Speech recognizer currently unavailable for locale: \(recognizer.locale.identifier, privacy: .public)")
        }

        return recognizer.locale.identifier
    }

    private func resolveRecognitionLocaleCode(for code: String?) -> String {
        guard let code, !code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return Locale.current.identifier
        }

        let normalized = code.replacingOccurrences(of: "_", with: "-")
        let supportedLocales = Set(SFSpeechRecognizer.supportedLocales().map { $0.identifier })

        if supportedLocales.contains(normalized) {
            return normalized
        }

        if supportedLocales.contains(code) {
            return code
        }

        let targetLanguage = Locale(identifier: normalized).language.languageCode?.identifier.lowercased() ?? normalized.split(separator: "-").first.map { String($0).lowercased() }

        if let targetLanguage,
           let match = supportedLocales.sorted().first(where: {
               Locale(identifier: $0).language.languageCode?.identifier.lowercased() == targetLanguage
           }) {
            logger.warning("No exact speech recognition locale for \(normalized, privacy: .public); using \(match, privacy: .public) instead.")
            return match
        }

        logger.warning("No supported speech recognition locale for \(normalized, privacy: .public); using current locale.")
        return Locale.current.identifier
    }

    func requestAuthorization() async -> Bool {
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        guard speechStatus == .authorized else {
            error = "Speech recognition not authorized"
            return false
        }

        let audioStatus = await AVAudioApplication.requestRecordPermission()
        guard audioStatus else {
            error = "Microphone access not granted"
            return false
        }

        return true
    }

    func startListening(onSilence: (() -> Void)? = nil) throws {
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            error = "Speech recognizer unavailable"
            return
        }

        stopListening()
        onSilenceDetected = onSilence
        transcript = ""
        error = nil
        isSpeechDetected = false
        consecutiveSilenceFrames = 0
        consecutiveSpeechFrames = 0
        recentLevels.removeAll()
        dynamicSilenceDuration = 1.8
        speechStartTime = nil

        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }

        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = speechRecognizer.supportsOnDeviceRecognition
        if #available(iOS 18, *) {
            recognitionRequest.addsPunctuation = true
        }

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, taskError in
            guard let self else { return }

            if let result {
                self.transcript = result.bestTranscription.formattedString
                self.lastSpeechTime = Date()

                if !self.isSpeechDetected {
                    self.isSpeechDetected = true
                    self.speechStartTime = Date()
                }

                self.updateDynamicSilenceDuration()

                if result.isFinal {
                    self.stopListening()
                    self.onSilenceDetected?()
                }
            }

            if let taskError {
                let nsError = taskError as NSError
                if nsError.domain == "kAFAssistantErrorDomain" && nsError.code == 1110 {
                    if !self.transcript.isEmpty {
                        self.onSilenceDetected?()
                    } else {
                        self.stopListening()
                    }
                } else if nsError.code != 216 {
                    self.error = taskError.localizedDescription
                    self.stopListening()
                }
            }
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
            self?.processAudioLevel(buffer: buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
        isListening = true

        startAdaptiveSilenceDetection()
    }

    func stopListening() {
        silenceTimer?.invalidate()
        silenceTimer = nil

        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }

        recognitionRequest?.endAudio()
        recognitionRequest = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        isListening = false
        audioLevel = 0
        isSpeechDetected = false

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func processAudioLevel(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)

        var sumOfSquares: Float = 0
        for i in 0..<frameLength {
            sumOfSquares += channelData[i] * channelData[i]
        }
        let rms = sqrt(sumOfSquares / Float(frameLength))
        let db = 20 * log10(max(rms, 0.00001))

        recentLevels.append(db)
        if recentLevels.count > levelHistorySize {
            recentLevels.removeFirst()
        }
        updateAdaptiveThreshold()

        let isSpeechFrame = db > adaptiveThreshold
        if isSpeechFrame {
            consecutiveSpeechFrames += 1
            consecutiveSilenceFrames = 0
        } else {
            consecutiveSilenceFrames += 1
            if consecutiveSilenceFrames > speechOffsetFrames {
                consecutiveSpeechFrames = 0
            }
        }

        let normalized = max(0, min(1, (db + 50) / 50))

        Task { @MainActor in
            self.audioLevel = normalized
        }
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

        let endsWithConjunction = transcript.lowercased().hasSuffix(" and") ||
            transcript.lowercased().hasSuffix(" but") ||
            transcript.lowercased().hasSuffix(" or") ||
            transcript.lowercased().hasSuffix(" because") ||
            transcript.lowercased().hasSuffix(" so") ||
            transcript.lowercased().hasSuffix(" then")

        if endsWithConjunction {
            dynamicSilenceDuration = maxSilenceDuration
        }

        let endsWithComma = transcript.hasSuffix(",")
        if endsWithComma {
            dynamicSilenceDuration = min(dynamicSilenceDuration + 0.5, maxSilenceDuration)
        }
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

    func detectBargeIn() -> Bool {
        return consecutiveSpeechFrames >= speechOnsetFrames && audioLevel > 0.15
    }
}
