import Foundation
import Testing
import UIKit
@testable import NeuralEngine

struct DocumentAnalysisViewModelTests {

    @Test @MainActor
    func analyzeImage_cancelsStaleTaskWhenNewAnalysisStarts() async throws {
        let service = MockDocumentAnalysisService(textDelayNanos: 300_000_000)
        let viewModel = DocumentAnalysisViewModel(service: service)
        let image = makeTestImage()

        viewModel.analyzeImage(image)
        try await Task.sleep(for: .milliseconds(40))
        viewModel.analyzeImage(image)

        try await Task.sleep(for: .milliseconds(700))

        #expect(viewModel.analysisResult?.fullText == "Result #2")
        #expect(viewModel.errorMessage == nil)
        #expect(!viewModel.isProcessing)
    }

    @Test @MainActor
    func analyzeDocument_largeTextFileStartsAsyncWithoutBlockingMainActor() async throws {
        let service = MockDocumentAnalysisService()
        let viewModel = DocumentAnalysisViewModel(service: service)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("large-file-\(UUID().uuidString).txt")
        let largePayload = String(repeating: "swiftllm mobile stack integration test line\n", count: 150_000)
        try largePayload.write(to: tempURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let start = ContinuousClock.now
        viewModel.analyzeDocument(at: tempURL)
        let elapsed = start.duration(to: ContinuousClock.now)

        #expect(elapsed < .milliseconds(120))
        #expect(viewModel.isProcessing)
        #expect(viewModel.statusMessage == "Loading text file...")

        for _ in 0..<100 {
            if !viewModel.isProcessing { break }
            try await Task.sleep(for: .milliseconds(20))
        }

        #expect(!viewModel.isProcessing)
        #expect(viewModel.analysisResult?.characterCount == largePayload.count)
        #expect(viewModel.statusMessage == "Text file loaded")
    }

    @MainActor
    private func makeTestImage() -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 8, height: 8))
        return renderer.image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        }
    }
}

private final class MockDocumentAnalysisService: @unchecked Sendable, DocumentAnalysisServicing {
    private var textCallCount: Int = 0
    private let textDelayNanos: UInt64
    private let lock = NSLock()

    init(textDelayNanos: UInt64 = 0) {
        self.textDelayNanos = textDelayNanos
    }

    func recognizeText(in image: UIImage) async throws -> DocumentAnalysisResult {
        let currentCall: Int
        lock.lock()
        defer { lock.unlock() }
        textCallCount += 1
        currentCall = textCallCount

        if textDelayNanos > 0 {
            try await Task.sleep(nanoseconds: textDelayNanos)
        }

        return DocumentAnalysisResult(
            fullText: "Result #\(currentCall)",
            blocks: [TextBlock(text: "Result #\(currentCall)", confidence: 1.0, boundingBox: .zero, normalizedBox: .zero)],
            pageCount: 1,
            languageHint: "en"
        )
    }

    func recognizeTextFromPDF(at url: URL) async throws -> DocumentAnalysisResult {
        DocumentAnalysisResult(fullText: "", blocks: [], pageCount: 1, languageHint: nil)
    }

    func detectBarcodes(in image: UIImage) async throws -> [BarcodeResult] {
        []
    }
}
