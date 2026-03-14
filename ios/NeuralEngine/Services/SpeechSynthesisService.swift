import Foundation
import AVFoundation

@Observable
class SpeechSynthesisService: NSObject {
    var isSpeaking: Bool = false
    var progress: Double = 0
    var currentWord: String = ""
    var currentSentenceIndex: Int = 0
    var spokenText: String = ""

    private let synthesizer = AVSpeechSynthesizer()
    private var totalLength: Int = 0
    private var completion: (() -> Void)?
    private var sentences: [String] = []
    private var currentSentence: Int = 0
    private var accumulatedLength: Int = 0

    private var selectedVoice: AVSpeechSynthesisVoice?
    private var bargeInDetection: (() -> Bool)?

    override init() {
        super.init()
        synthesizer.delegate = self
        prepareVoice()
    }

    private func prepareVoice() {
        let preferredVoices = [
            "com.apple.voice.premium.en-US.Ava",
            "com.apple.voice.premium.en-US.Zoe",
            "com.apple.voice.premium.en-US.Tom",
            "com.apple.voice.enhanced.en-US.Ava",
            "com.apple.voice.enhanced.en-US.Samantha",
        ]

        for voiceId in preferredVoices {
            if let voice = AVSpeechSynthesisVoice(identifier: voiceId) {
                selectedVoice = voice
                return
            }
        }

        let availableVoices = AVSpeechSynthesisVoice.speechVoices()
        let premiumUS = availableVoices.first { $0.language == "en-US" && $0.quality == .premium }
        let enhancedUS = availableVoices.first { $0.language == "en-US" && $0.quality == .enhanced }
        let defaultUS = availableVoices.first { $0.language == "en-US" }

        selectedVoice = premiumUS ?? enhancedUS ?? defaultUS ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    func speak(_ text: String, bargeInCheck: (() -> Bool)? = nil, onComplete: (() -> Void)? = nil) {
        stop()
        completion = onComplete
        bargeInDetection = bargeInCheck

        let cleaned = cleanTextForSpeech(text)
        guard !cleaned.isEmpty else {
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

        sentences = splitIntoSentences(cleaned)
        totalLength = cleaned.count
        progress = 0
        currentSentence = 0
        accumulatedLength = 0
        spokenText = ""
        isSpeaking = true

        speakNextSentence()
    }

    private func speakNextSentence() {
        guard currentSentence < sentences.count else {
            finishSpeaking()
            return
        }

        let sentence = sentences[currentSentence]
        currentSentenceIndex = currentSentence

        let utterance = AVSpeechUtterance(string: sentence)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.02
        utterance.pitchMultiplier = 1.0
        utterance.preUtteranceDelay = currentSentence == 0 ? 0.08 : 0.02
        utterance.postUtteranceDelay = 0.15
        utterance.volume = 1.0
        utterance.voice = selectedVoice

        synthesizer.speak(utterance)
    }

    private func finishSpeaking() {
        isSpeaking = false
        progress = 1.0
        currentWord = ""
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        let cb = completion
        completion = nil
        cb?()
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
        progress = 0
        currentWord = ""
        spokenText = ""
        sentences = []
        currentSentence = 0
        accumulatedLength = 0
        completion = nil
        bargeInDetection = nil
    }

    func pause() {
        synthesizer.pauseSpeaking(at: .word)
    }

    func resume() {
        synthesizer.continueSpeaking()
    }

    private func cleanTextForSpeech(_ text: String) -> String {
        var cleaned = text

        let codeBlockPattern = #"```[\s\S]*?```"#
        if let regex = try? NSRegularExpression(pattern: codeBlockPattern) {
            cleaned = regex.stringByReplacingMatches(in: cleaned, range: NSRange(cleaned.startIndex..., in: cleaned), withTemplate: "")
        }

        let patterns: [(String, String)] = [
            (#"\*\*(.+?)\*\*"#, "$1"),
            (#"\*(.+?)\*"#, "$1"),
            (#"__(.+?)__"#, "$1"),
            (#"_(.+?)_"#, "$1"),
            (#"`(.+?)`"#, "$1"),
            (#"#{1,6}\s+"#, ""),
            (#"^\s*[-*+]\s+"#, ""),
            (#"^\s*\d+\.\s+"#, ""),
            (#"\[(.+?)\]\(.+?\)"#, "$1"),
            (#"!\[.*?\]\(.+?\)"#, ""),
        ]

        for (pattern, replacement) in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .anchorsMatchLines) {
                cleaned = regex.stringByReplacingMatches(in: cleaned, range: NSRange(cleaned.startIndex..., in: cleaned), withTemplate: replacement)
            }
        }

        cleaned = cleaned.replacingOccurrences(of: "---", with: "")
        cleaned = cleaned.replacingOccurrences(of: "***", with: "")

        let lines = cleaned.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        cleaned = lines.joined(separator: " ")

        while cleaned.contains("  ") {
            cleaned = cleaned.replacingOccurrences(of: "  ", with: " ")
        }

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func splitIntoSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }

        if sentences.isEmpty && !text.isEmpty {
            sentences = [text]
        }

        return sentences
    }
}

import NaturalLanguage

extension SpeechSynthesisService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in
            isSpeaking = true
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            let sentenceText = utterance.speechString
            accumulatedLength += sentenceText.count
            spokenText += (spokenText.isEmpty ? "" : " ") + sentenceText
            currentSentence += 1

            if currentSentence < sentences.count {
                speakNextSentence()
            } else {
                finishSpeaking()
            }
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
            let currentPos = accumulatedLength + characterRange.location + characterRange.length
            progress = Double(currentPos) / Double(totalLength)
        }
    }
}
