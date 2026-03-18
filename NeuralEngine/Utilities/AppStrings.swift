import Foundation

enum AppStrings {
    static let tabNexus = localized("tab.nexus", fallback: "Nexus")
    static let tabModels = localized("tab.models", fallback: "Models")
    static let tabSettings = localized("tab.settings", fallback: "Settings")

    static let initializingNexus = localized("progress.initializingNexus", fallback: "Initializing Nexus...")

    static let settingsTitle = localized("settings.title", fallback: "Settings")
    static let samplingTitle = localized("settings.sampling.title", fallback: "Sampling")
    static let samplingFooter = localized("settings.sampling.footer", fallback: "Controls the randomness and diversity of generated text. Lower temperature = more focused, higher = more creative.")
    static let systemPromptTitle = localized("settings.systemPrompt.title", fallback: "System Prompt")
    static let systemPromptFooter = localized("settings.systemPrompt.footer", fallback: "This prompt is prepended to every conversation. Cached via prompt prefix cache for fast repeated inference.")
    static let runtimeTitle = localized("settings.runtime.title", fallback: "Runtime")
    static let runtimeFooter = localized("settings.runtime.footer", fallback: "Runtime mode adapts automatically based on device thermal state.")
    static let thermalState = localized("settings.runtime.thermalState", fallback: "Thermal State")
    static let runtimeMode = localized("settings.runtime.mode", fallback: "Runtime Mode")
    static let maxDraftTokens = localized("settings.runtime.maxDraftTokens", fallback: "Max Draft Tokens")
    static let speculativeDecoding = localized("settings.runtime.speculativeDecoding", fallback: "Speculative Decoding")
    static let enabled = localized("common.enabled", fallback: "Enabled")
    static let disabled = localized("common.disabled", fallback: "Disabled")

    static let temperature = localized("settings.sampling.temperature", fallback: "Temperature")
    static let topK = localized("settings.sampling.topK", fallback: "Top-K")
    static let topP = localized("settings.sampling.topP", fallback: "Top-P")
    static let repetitionPenalty = localized("settings.sampling.repetitionPenalty", fallback: "Repetition Penalty")
    static let maxTokens = localized("settings.sampling.maxTokens", fallback: "Max Tokens")

    static let recognizingText = localized("document.status.recognizingText", fallback: "Recognizing text...")
    static let noTextDetected = localized("document.status.noTextDetected", fallback: "No text detected")
    static let processingImage = localized("document.status.processingImage", fallback: "Processing image...")
    static let analysisComplete = localized("document.status.analysisComplete", fallback: "Analysis complete")
    static let analysisFailed = localized("document.status.analysisFailed", fallback: "Analysis failed")
    static let processingPDF = localized("document.status.processingPDF", fallback: "Processing PDF...")
    static let loadingImage = localized("document.status.loadingImage", fallback: "Loading image...")
    static let loadingTextFile = localized("document.status.loadingTextFile", fallback: "Loading text file...")
    static let textFileLoaded = localized("document.status.textFileLoaded", fallback: "Text file loaded")
    static let imageLoadFailed = localized("document.error.imageLoadFailed", fallback: "Could not load image from file")
    static let textFileReadFailed = localized("document.error.textFileReadFailed", fallback: "Could not read text file")
    static let screenCaptureFailed = localized("document.error.screenCaptureFailed", fallback: "Failed to capture screen")

    static func pdfAnalysisComplete(pageCount: Int) -> String {
        localizedFormat("document.status.pdfAnalysisComplete %@", fallback: "PDF analysis complete %@ pages", argument: String(pageCount))
    }

    static func pdfAnalysisFailed() -> String {
        localized("document.status.pdfAnalysisFailed", fallback: "PDF analysis failed")
    }

    static func unsupportedFormat(_ ext: String) -> String {
        localizedFormat("document.error.unsupportedFormat %@", fallback: "Unsupported file format: .%@", argument: ext)
    }

    private static func localized(_ key: String, fallback: String) -> String {
        Bundle.main.localizedString(forKey: key, value: fallback, table: nil)
    }

    private static func localizedFormat(_ key: String, fallback: String, argument: String) -> String {
        let format = localized(key, fallback: fallback)
        return String(format: format, locale: Locale.current, argument)
    }
}
