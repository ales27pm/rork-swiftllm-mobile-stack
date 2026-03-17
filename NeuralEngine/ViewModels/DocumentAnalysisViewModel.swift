import SwiftUI
import PDFKit

@Observable
@MainActor
class DocumentAnalysisViewModel {
    var selectedImage: UIImage?
    var analysisResult: DocumentAnalysisResult?
    var barcodeResults: [BarcodeResult] = []
    var isProcessing: Bool = false
    var progress: Double = 0.0
    var statusMessage: String = ""
    var errorMessage: String?
    var selectedTab: AnalysisTab = .text
    var showDocumentPicker: Bool = false
    var showImagePicker: Bool = false
    var showCamera: Bool = false
    var documentName: String = ""
    var showTextOverlay: Bool = false
    var copiedToClipboard: Bool = false

    private let service: DocumentAnalysisServicing
    private var analysisTask: Task<Void, Never>?

    init(service: DocumentAnalysisServicing = DocumentAnalysisService()) {
        self.service = service
    }

    func analyzeImage(_ image: UIImage) {
        cancelAnalysisTask()
        beginImageAnalysis(image)
    }

    private func beginImageAnalysis(_ image: UIImage) {
        selectedImage = image
        analysisResult = nil
        barcodeResults = []
        errorMessage = nil
        isProcessing = true
        statusMessage = "Recognizing text..."
        progress = 0.1

        analysisTask = Task {
            do {
                try Task.checkCancellation()
                progress = 0.3
                async let textResult = service.recognizeText(in: image)
                async let barcodes = service.detectBarcodes(in: image)

                progress = 0.6
                let text = try await textResult
                let codes = try await barcodes
                try Task.checkCancellation()

                progress = 0.9
                analysisResult = text
                barcodeResults = codes
                statusMessage = text.fullText.isEmpty ? "No text detected" : "Analysis complete"
                progress = 1.0

                try? await Task.sleep(for: .milliseconds(500))
                isProcessing = false
                analysisTask = nil
            } catch is CancellationError {
                if Task.isCancelled {
                    statusMessage = ""
                    isProcessing = false
                }
                analysisTask = nil
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = "Analysis failed"
                isProcessing = false
                analysisTask = nil
            }
        }
    }

    func analyzeDocument(at url: URL) {
        cancelAnalysisTask()
        analysisResult = nil
        barcodeResults = []
        errorMessage = nil
        documentName = url.lastPathComponent

        let ext = url.pathExtension.lowercased()

        if ext == "pdf" {
            isProcessing = true
            statusMessage = "Processing PDF..."
            progress = 0.1

            analysisTask = Task {
                do {
                    try Task.checkCancellation()
                    progress = 0.4
                    let result = try await service.recognizeTextFromPDF(at: url)
                    try Task.checkCancellation()
                    progress = 0.9
                    analysisResult = result
                    statusMessage = "PDF analysis complete (\(result.pageCount) pages)"
                    progress = 1.0

                    try? await Task.sleep(for: .milliseconds(500))
                    isProcessing = false
                    analysisTask = nil
                } catch is CancellationError {
                    if Task.isCancelled {
                        statusMessage = ""
                        isProcessing = false
                    }
                    analysisTask = nil
                } catch {
                    errorMessage = error.localizedDescription
                    statusMessage = "PDF analysis failed"
                    isProcessing = false
                    analysisTask = nil
                }
            }
        } else if ["png", "jpg", "jpeg", "heic", "tiff", "bmp", "gif", "webp"].contains(ext) {
            isProcessing = true
            statusMessage = "Loading image..."
            progress = 0.1
            analysisTask = Task {
                do {
                    let image = try await Self.loadImageFromDisk(at: url)
                    try Task.checkCancellation()
                    beginImageAnalysis(image)
                } catch is CancellationError {
                    self.isProcessing = false
                    self.statusMessage = ""
                    self.analysisTask = nil
                } catch {
                    self.errorMessage = "Could not load image from file"
                    self.isProcessing = false
                    self.analysisTask = nil
                }
            }
        } else if ["txt", "rtf", "md"].contains(ext) {
            isProcessing = true
            statusMessage = "Loading text file..."
            progress = 0.1
            analysisTask = Task {
                do {
                    let text = try await Self.loadTextFromDisk(at: url)
                    try Task.checkCancellation()

                    analysisResult = DocumentAnalysisResult(
                        fullText: text,
                        blocks: [TextBlock(text: text, confidence: 1.0, boundingBox: .zero, normalizedBox: .zero)],
                        pageCount: 1,
                        languageHint: nil
                    )
                    statusMessage = "Text file loaded"
                    progress = 1.0
                    isProcessing = false
                    analysisTask = nil
                } catch is CancellationError {
                    if Task.isCancelled {
                        statusMessage = ""
                        isProcessing = false
                    }
                    analysisTask = nil
                } catch {
                    errorMessage = "Could not read text file"
                    isProcessing = false
                    analysisTask = nil
                }
            }
        } else {
            errorMessage = "Unsupported file format: .\(ext)"
        }
    }

    func copyFullText() {
        guard let text = analysisResult?.fullText, !text.isEmpty else { return }
        UIPasteboard.general.string = text
        copiedToClipboard = true
        Task {
            try? await Task.sleep(for: .seconds(2))
            copiedToClipboard = false
        }
    }

    func reset() {
        cancelAnalysisTask()
        selectedImage = nil
        analysisResult = nil
        barcodeResults = []
        isProcessing = false
        progress = 0.0
        statusMessage = ""
        errorMessage = nil
        documentName = ""
        showTextOverlay = false
    }

    func captureViewShot() {
        guard let image = ViewShotService.captureScreen() else {
            errorMessage = "Failed to capture screen"
            return
        }
        analyzeImage(image)
    }

    private func cancelAnalysisTask() {
        analysisTask?.cancel()
        analysisTask = nil
    }

    nonisolated private static func loadTextFromDisk(at url: URL) async throws -> String {
        try await Task.detached(priority: .userInitiated) {
            try String(contentsOf: url, encoding: .utf8)
        }.value
    }

    nonisolated private static func loadImageFromDisk(at url: URL) async throws -> UIImage {
        try await Task.detached(priority: .userInitiated) {
            guard let image = UIImage(contentsOfFile: url.path) else {
                throw DocumentAnalysisError.invalidImage
            }
            return image
        }.value
    }
}

protocol DocumentAnalysisServicing: Sendable {
    func recognizeText(in image: UIImage) async throws -> DocumentAnalysisResult
    func recognizeTextFromPDF(at url: URL) async throws -> DocumentAnalysisResult
    func detectBarcodes(in image: UIImage) async throws -> [BarcodeResult]
}

nonisolated enum AnalysisTab: String, CaseIterable, Sendable {
    case text = "Text"
    case blocks = "Blocks"
    case barcodes = "Barcodes"
    case info = "Info"
}
