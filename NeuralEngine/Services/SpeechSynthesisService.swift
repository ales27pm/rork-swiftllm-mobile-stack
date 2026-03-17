import Foundation
import AVFoundation
import os.log

@Observable
class SpeechSynthesisService: NSObject {
    struct VoiceOption: Identifiable, Hashable {
        let identifier: String
        let language: String
        let quality: AVSpeechSynthesisVoiceQuality
        let displayName: String

        var id: String { identifier }
    }

    private enum StorageKey {
        static let selectedVoiceIdentifier = "SpeechSynthesisService.selectedVoiceIdentifier"
        static let preferredLanguageCode = "SpeechSynthesisService.preferredLanguageCode"
    }

    var isSpeaking: Bool = false
    var progress: Double = 0
    var currentWord: String = ""
    var currentSentenceIndex: Int = 0
    var spokenText: String = ""

    private let synthesizer = AVSpeechSynthesizer()
    private let userDefaults: UserDefaults
    private var totalLength: Int = 0
    private var completion: (() -> Void)?
    private var sentences: [String] = []
    private var currentSentence: Int = 0
    private var accumulatedLength: Int = 0
    private var cachedVoices: [AVSpeechSynthesisVoice] = []

    private var selectedVoice: AVSpeechSynthesisVoice?
    private var selectedVoiceIdentifier: String?
    private var preferredLanguageCode: String?
    private var autoSelectedVoiceIdentifier: String?
    private var bargeInDetection: (() -> Bool)?
    private let logger: Logger

    override init() {
        userDefaults = .standard
        let subsystem = Bundle.main.bundleIdentifier ?? "SpeechSynthesisService"
        logger = Logger(subsystem: subsystem, category: "SpeechSynthesisService")
        selectedVoiceIdentifier = userDefaults.string(forKey: StorageKey.selectedVoiceIdentifier)
        preferredLanguageCode = userDefaults.string(forKey: StorageKey.preferredLanguageCode)
        super.init()
        synthesizer.delegate = self
        prepareVoice(availableVoices: loadAvailableVoices())
    }

    init(userDefaults: UserDefaults) {
        self.userDefaults = userDefaults
        let subsystem = Bundle.main.bundleIdentifier ?? "SpeechSynthesisService"
        logger = Logger(subsystem: subsystem, category: "SpeechSynthesisService")
        selectedVoiceIdentifier = userDefaults.string(forKey: StorageKey.selectedVoiceIdentifier)
        preferredLanguageCode = userDefaults.string(forKey: StorageKey.preferredLanguageCode)
        super.init()
        synthesizer.delegate = self
        prepareVoice(availableVoices: loadAvailableVoices())
    }

    private func loadAvailableVoices(forceRefresh: Bool = false) -> [AVSpeechSynthesisVoice] {
        if forceRefresh || cachedVoices.isEmpty {
            cachedVoices = AVSpeechSynthesisVoice.speechVoices()
        }
        return cachedVoices
    }

    private func prepareVoice(
        availableVoices: [AVSpeechSynthesisVoice],
        preferredVoice: AVSpeechSynthesisVoice? = nil,
        skipPreferredLanguageLookup: Bool = false
    ) {
        let currentLocale = Locale.current
        let targetLanguageCode = preferredLanguageCode ?? currentLocale.language.languageCode?.identifier

        if let selectedVoiceIdentifier {
            if let selected = availableVoices.first(where: { $0.identifier == selectedVoiceIdentifier }) {
                selectedVoice = selected
                autoSelectedVoiceIdentifier = selected.identifier
                return
            }

            self.selectedVoiceIdentifier = nil
            userDefaults.removeObject(forKey: StorageKey.selectedVoiceIdentifier)
        }

        if let preferredVoice {
            selectedVoice = preferredVoice
            autoSelectedVoiceIdentifier = preferredVoice.identifier
            return
        }

        if
            !skipPreferredLanguageLookup,
            let targetLanguageCode,
            let localeMatched = bestVoice(in: availableVoices, forLanguageCode: targetLanguageCode)
        {
            selectedVoice = localeMatched
            autoSelectedVoiceIdentifier = localeMatched.identifier
            return
        }

        if let localeIdentifierVoice = AVSpeechSynthesisVoice(language: currentLocale.identifier) {
            selectedVoice = localeIdentifierVoice
            autoSelectedVoiceIdentifier = localeIdentifierVoice.identifier
            return
        }

        if let firstVoice = availableVoices.first {
            selectedVoice = firstVoice
            autoSelectedVoiceIdentifier = firstVoice.identifier
            return
        }

        let compatibilityVoice = AVSpeechSynthesisVoice(language: "en-US")
        selectedVoice = compatibilityVoice
        autoSelectedVoiceIdentifier = compatibilityVoice?.identifier
    }

    func availableVoices() -> [VoiceOption] {
        loadAvailableVoices(forceRefresh: true).map {
            VoiceOption(identifier: $0.identifier, language: $0.language, quality: $0.quality, displayName: $0.name)
        }
    }

    func currentVoiceIdentifier() -> String? {
        selectedVoiceIdentifier
    }

    func currentPreferredLanguageCode() -> String? {
        preferredLanguageCode
    }

    func setVoice(identifier: String?) {
        guard let identifier else {
            selectedVoiceIdentifier = nil
            userDefaults.removeObject(forKey: StorageKey.selectedVoiceIdentifier)
            prepareVoice(availableVoices: loadAvailableVoices(forceRefresh: true))
            return
        }

        var availableVoices = loadAvailableVoices()
        if !availableVoices.contains(where: { $0.identifier == identifier }) {
            availableVoices = loadAvailableVoices(forceRefresh: true)
        }

        guard let voice = availableVoices.first(where: { $0.identifier == identifier }) else {
            logger.error("Requested voice identifier not found: \(identifier, privacy: .public)")
            return
        }

        selectedVoice = voice
        selectedVoiceIdentifier = voice.identifier
        autoSelectedVoiceIdentifier = voice.identifier
        userDefaults.set(voice.identifier, forKey: StorageKey.selectedVoiceIdentifier)
    }

    func setLanguagePreferred(_ languageCode: String?) {
        guard let languageCode else {
            preferredLanguageCode = nil
            userDefaults.removeObject(forKey: StorageKey.preferredLanguageCode)
            prepareVoice(availableVoices: loadAvailableVoices(forceRefresh: true))
            return
        }

        selectedVoiceIdentifier = nil
        userDefaults.removeObject(forKey: StorageKey.selectedVoiceIdentifier)

        preferredLanguageCode = languageCode
        userDefaults.set(languageCode, forKey: StorageKey.preferredLanguageCode)

        let availableVoices = loadAvailableVoices(forceRefresh: true)
        let preferredVoice = bestVoice(in: availableVoices, forLanguageCode: languageCode)

        if let preferredVoice {
            selectedVoice = preferredVoice
            autoSelectedVoiceIdentifier = preferredVoice.identifier
            return
        }

        logger.error("No voices available for preferred language: \(languageCode, privacy: .public)")
        selectedVoice = nil
        autoSelectedVoiceIdentifier = nil
        prepareVoice(availableVoices: availableVoices, preferredVoice: nil, skipPreferredLanguageLookup: true)
    }

    private func bestVoice(in voices: [AVSpeechSynthesisVoice], forLanguageCode languageCode: String) -> AVSpeechSynthesisVoice? {
        let normalizedTargetLocale = normalizedLocaleIdentifier(languageCode)
        let normalizedTargetLanguage = normalizedLanguageCode(languageCode)

        let exactLocaleMatches = voices.filter {
            normalizedLocaleIdentifier($0.language) == normalizedTargetLocale
        }

        if let exactMatch = bestVoiceFromCandidates(exactLocaleMatches) {
            return exactMatch
        }

        let languageMatches = voices.filter {
            normalizedLanguageCode($0.language) == normalizedTargetLanguage
        }

        return bestVoiceFromCandidates(languageMatches)
    }

    private func bestVoiceFromCandidates(_ voices: [AVSpeechSynthesisVoice]) -> AVSpeechSynthesisVoice? {
        voices.sorted {
            if $0.quality == $1.quality {
                return $0.name < $1.name
            }
            return $0.quality.rawValue > $1.quality.rawValue
        }.first
    }

    private func normalizedLocaleIdentifier(_ identifier: String) -> String {
        identifier.replacingOccurrences(of: "_", with: "-").lowercased()
    }

    private func normalizedLanguageCode(_ identifier: String) -> String {
        let normalizedIdentifier = normalizedLocaleIdentifier(identifier)
        return normalizedIdentifier.split(separator: "-").first.map { String($0).lowercased() } ?? normalizedIdentifier.lowercased()
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
