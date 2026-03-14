import Foundation
import Speech
import AVFoundation

@Observable
class SpeechRecognitionService: NSObject {
    var transcript: String = ""
    var isListening: Bool = false
    var audioLevel: Float = 0
    var error: String?

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var silenceTimer: Timer?
    private var lastSpeechTime: Date = Date()
    private var onSilenceDetected: (() -> Void)?

    override init() {
        super.init()
        speechRecognizer = SFSpeechRecognizer(locale: Locale.current)
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

        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }

        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = speechRecognizer.supportsOnDeviceRecognition

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, taskError in
            guard let self else { return }

            if let result {
                self.transcript = result.bestTranscription.formattedString
                self.lastSpeechTime = Date()

                if result.isFinal {
                    self.stopListening()
                    self.onSilenceDetected?()
                }
            }

            if let taskError {
                let nsError = taskError as NSError
                if nsError.domain == "kAFAssistantErrorDomain" && nsError.code == 1110 {
                    self.onSilenceDetected?()
                } else if nsError.code != 216 {
                    self.error = taskError.localizedDescription
                }
                self.stopListening()
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

        startSilenceDetection()
    }

    func stopListening() {
        silenceTimer?.invalidate()
        silenceTimer = nil

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)

        recognitionRequest?.endAudio()
        recognitionRequest = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        isListening = false
        audioLevel = 0

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func processAudioLevel(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)

        var sum: Float = 0
        for i in 0..<frameLength {
            sum += abs(channelData[i])
        }
        let average = sum / Float(frameLength)
        let db = 20 * log10(max(average, 0.0001))
        let normalized = max(0, min(1, (db + 50) / 50))

        Task { @MainActor in
            self.audioLevel = normalized
        }
    }

    private func startSilenceDetection() {
        lastSpeechTime = Date()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self else { return }
            let elapsed = Date().timeIntervalSince(self.lastSpeechTime)
            if elapsed > 2.0 && !self.transcript.isEmpty {
                self.stopListening()
                self.onSilenceDetected?()
            }
        }
    }
}
