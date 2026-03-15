import Foundation
import Vision
import UIKit
import PDFKit
import NaturalLanguage

@Observable
@MainActor
class DocumentAnalysisService {
    var isProcessing: Bool = false
    var progress: Double = 0.0
    var statusMessage: String = ""

    nonisolated func recognizeText(in image: UIImage) async throws -> DocumentAnalysisResult {
        guard let cgImage = image.cgImage else {
            throw DocumentAnalysisError.invalidImage
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US", "fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR", "zh-Hans", "ja", "ko"]
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let observations = request.results else {
            return DocumentAnalysisResult(fullText: "", blocks: [], pageCount: 1, languageHint: nil)
        }

        var blocks: [TextBlock] = []
        var fullTextLines: [String] = []

        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let box = observation.boundingBox
            let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
            let rect = CGRect(
                x: box.origin.x * imageSize.width,
                y: (1 - box.origin.y - box.height) * imageSize.height,
                width: box.width * imageSize.width,
                height: box.height * imageSize.height
            )
            let block = TextBlock(
                text: candidate.string,
                confidence: candidate.confidence,
                boundingBox: rect,
                normalizedBox: box
            )
            blocks.append(block)
            fullTextLines.append(candidate.string)
        }

        let fullText = fullTextLines.joined(separator: "\n")
        return DocumentAnalysisResult(
            fullText: fullText,
            blocks: blocks,
            pageCount: 1,
            languageHint: detectLanguage(fullText)
        )
    }

    nonisolated func recognizeTextFromPDF(at url: URL) async throws -> DocumentAnalysisResult {
        guard let pdfDocument = PDFDocument(url: url) else {
            throw DocumentAnalysisError.unsupportedFormat
        }

        var allBlocks: [TextBlock] = []
        var allText: [String] = []
        let pageCount = pdfDocument.pageCount

        for i in 0..<pageCount {
            guard let page = pdfDocument.page(at: i) else { continue }

            if let pageText = page.string, !pageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                allText.append(pageText)
                let block = TextBlock(
                    text: pageText,
                    confidence: 1.0,
                    boundingBox: .zero,
                    normalizedBox: .zero
                )
                allBlocks.append(block)
            } else {
                let bounds = page.bounds(for: .mediaBox)
                let scale: CGFloat = 2.0
                let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
                let renderer = UIGraphicsImageRenderer(size: size)
                let image = renderer.image { ctx in
                    UIColor.white.setFill()
                    ctx.fill(CGRect(origin: .zero, size: size))
                    ctx.cgContext.translateBy(x: 0, y: size.height)
                    ctx.cgContext.scaleBy(x: scale, y: -scale)
                    page.draw(with: .mediaBox, to: ctx.cgContext)
                }

                if let result = try? await recognizeText(in: image) {
                    allBlocks.append(contentsOf: result.blocks)
                    allText.append(result.fullText)
                }
            }
        }

        let fullText = allText.joined(separator: "\n\n--- Page Break ---\n\n")
        return DocumentAnalysisResult(
            fullText: fullText,
            blocks: allBlocks,
            pageCount: pageCount,
            languageHint: detectLanguage(fullText)
        )
    }

    nonisolated func detectBarcodes(in image: UIImage) async throws -> [BarcodeResult] {
        guard let cgImage = image.cgImage else {
            throw DocumentAnalysisError.invalidImage
        }

        let request = VNDetectBarcodesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results else { return [] }

        return results.compactMap { observation in
            guard let payload = observation.payloadStringValue else { return nil }
            return BarcodeResult(
                payload: payload,
                symbology: observation.symbology.rawValue,
                confidence: observation.confidence
            )
        }
    }

    nonisolated func detectRectangles(in image: UIImage) async throws -> [CGRect] {
        guard let cgImage = image.cgImage else {
            throw DocumentAnalysisError.invalidImage
        }

        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.3
        request.maximumAspectRatio = 1.0
        request.minimumSize = 0.1
        request.maximumObservations = 10

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results else { return [] }

        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        return results.map { observation in
            let box = observation.boundingBox
            return CGRect(
                x: box.origin.x * imageSize.width,
                y: (1 - box.origin.y - box.height) * imageSize.height,
                width: box.width * imageSize.width,
                height: box.height * imageSize.height
            )
        }
    }

    nonisolated private func detectLanguage(_ text: String) -> String? {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage?.rawValue
    }
}

nonisolated struct DocumentAnalysisResult: Sendable {
    let fullText: String
    let blocks: [TextBlock]
    let pageCount: Int
    let languageHint: String?

    var averageConfidence: Float {
        guard !blocks.isEmpty else { return 0 }
        return blocks.reduce(0) { $0 + $1.confidence } / Float(blocks.count)
    }

    var wordCount: Int {
        fullText.split(separator: " ").count
    }

    var characterCount: Int {
        fullText.count
    }
}

nonisolated struct TextBlock: Identifiable, Sendable {
    let id = UUID()
    let text: String
    let confidence: Float
    let boundingBox: CGRect
    let normalizedBox: CGRect
}

nonisolated struct BarcodeResult: Identifiable, Sendable {
    let id = UUID()
    let payload: String
    let symbology: String
    let confidence: Float
}

nonisolated enum DocumentAnalysisError: Error, Sendable, LocalizedError {
    case invalidImage
    case unsupportedFormat
    case ocrFailed(String)
    case noTextFound

    var errorDescription: String? {
        switch self {
        case .invalidImage: return "Could not process the image"
        case .unsupportedFormat: return "Unsupported document format"
        case .ocrFailed(let detail): return "OCR failed: \(detail)"
        case .noTextFound: return "No text was detected in the document"
        }
    }
}
