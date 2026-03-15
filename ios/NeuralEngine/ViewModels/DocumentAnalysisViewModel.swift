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

    private let service = DocumentAnalysisService()

    func analyzeImage(_ image: UIImage) {
        selectedImage = image
        analysisResult = nil
        barcodeResults = []
        errorMessage = nil
        isProcessing = true
        statusMessage = "Recognizing text..."
        progress = 0.1

        Task {
            do {
                progress = 0.3
                async let textResult = service.recognizeText(in: image)
                async let barcodes = service.detectBarcodes(in: image)

                progress = 0.6
                let text = try await textResult
                let codes = try await barcodes

                progress = 0.9
                analysisResult = text
                barcodeResults = codes
                statusMessage = text.fullText.isEmpty ? "No text detected" : "Analysis complete"
                progress = 1.0

                try? await Task.sleep(for: .milliseconds(500))
                isProcessing = false
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = "Analysis failed"
                isProcessing = false
            }
        }
    }

    func analyzeDocument(at url: URL) {
        analysisResult = nil
        barcodeResults = []
        errorMessage = nil
        documentName = url.lastPathComponent

        let ext = url.pathExtension.lowercased()

        if ext == "pdf" {
            isProcessing = true
            statusMessage = "Processing PDF..."
            progress = 0.1

            Task {
                do {
                    let data = try Data(contentsOf: url)
                    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(url.lastPathComponent)
                    try data.write(to: tempURL)

                    progress = 0.4
                    let result = try await service.recognizeTextFromPDF(at: tempURL)
                    progress = 0.9
                    analysisResult = result
                    statusMessage = "PDF analysis complete (\(result.pageCount) pages)"
                    progress = 1.0

                    try? await Task.sleep(for: .milliseconds(500))
                    isProcessing = false
                } catch {
                    errorMessage = error.localizedDescription
                    statusMessage = "PDF analysis failed"
                    isProcessing = false
                }
            }
        } else if ["png", "jpg", "jpeg", "heic", "tiff", "bmp", "gif", "webp"].contains(ext) {
            if let data = try? Data(contentsOf: url), let image = UIImage(data: data) {
                analyzeImage(image)
            } else {
                errorMessage = "Could not load image from file"
            }
        } else if ["txt", "rtf", "md"].contains(ext) {
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                analysisResult = DocumentAnalysisResult(
                    fullText: text,
                    blocks: [TextBlock(text: text, confidence: 1.0, boundingBox: .zero, normalizedBox: .zero)],
                    pageCount: 1,
                    languageHint: nil
                )
                statusMessage = "Text file loaded"
            } else {
                errorMessage = "Could not read text file"
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
}

nonisolated enum AnalysisTab: String, CaseIterable, Sendable {
    case text = "Text"
    case blocks = "Blocks"
    case barcodes = "Barcodes"
    case info = "Info"
}
