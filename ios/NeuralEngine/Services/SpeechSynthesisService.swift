import Foundation
import AVFoundation

@Observable
class SpeechSynthesisService: NSObject {
    var isSpeaking: Bool = false
    var progress: Double = 0
    var currentWord: String = ""

    private let synthesizer = AVSpeechSynthesizer()
    private var totalLength: Int = 0
    private var completion: (() -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String, onComplete: (() -> Void)? = nil) {
        stop()
        completion = onComplete

        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            onComplete?()
            return
        }

        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .voicePrompt, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            onComplete?()
            return
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        utterance.pitchMultiplier = 1.0
        utterance.preUtteranceDelay = 0.1
        utterance.postUtteranceDelay = 0.1

        if let voice = AVSpeechSynthesisVoice(identifier: "com.apple.voice.premium.en-US.Ava") {
            utterance.voice = voice
        } else if let voice = AVSpeechSynthesisVoice(identifier: "com.apple.voice.enhanced.en-US.Ava") {
            utterance.voice = voice
        } else {
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        }

        totalLength = text.count
        progress = 0
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
        progress = 0
        currentWord = ""
        completion = nil
    }

    func pause() {
        synthesizer.pauseSpeaking(at: .word)
    }

    func resume() {
        synthesizer.continueSpeaking()
    }
}

extension SpeechSynthesisService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = true
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = false
            progress = 1.0
            currentWord = ""
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            completion?()
            completion = nil
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = false
            progress = 0
            currentWord = ""
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, willSpeakRangeOfSpeechString characterRange: NSRange, utterance: AVSpeechUtterance) {
        Task { @MainActor in
            let text = utterance.speechString
            if let range = Range(characterRange, in: text) {
                currentWord = String(text[range])
            }
            guard totalLength > 0 else { return }
            progress = Double(characterRange.location + characterRange.length) / Double(totalLength)
        }
    }
}
