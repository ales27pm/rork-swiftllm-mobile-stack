import SwiftUI

struct DocumentAnalysisView: View {
    @State private var viewModel = DocumentAnalysisViewModel()

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.selectedImage != nil || viewModel.analysisResult != nil {
                    analysisResultView
                } else {
                    sourceSelectionView
                }
            }
            .navigationTitle("Document Analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if viewModel.selectedImage != nil || viewModel.analysisResult != nil {
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu {
                            Button("Copy All Text", systemImage: "doc.on.doc") {
                                viewModel.copyFullText()
                            }
                            .disabled(viewModel.analysisResult?.fullText.isEmpty ?? true)

                            Button("New Scan", systemImage: "arrow.counterclockwise") {
                                withAnimation { viewModel.reset() }
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .sheet(isPresented: $viewModel.showDocumentPicker) {
                DocumentPickerView { url in
                    viewModel.showDocumentPicker = false
                    viewModel.analyzeDocument(at: url)
                }
            }
            .sheet(isPresented: $viewModel.showImagePicker) {
                ImagePickerView { image in
                    viewModel.showImagePicker = false
                    viewModel.analyzeImage(image)
                }
            }
            .fullScreenCover(isPresented: $viewModel.showCamera) {
                CameraProxyView { image in
                    viewModel.showCamera = false
                    viewModel.analyzeImage(image)
                }
            }
            .overlay {
                if viewModel.copiedToClipboard {
                    copiedToast
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .animation(.spring(duration: 0.3), value: viewModel.copiedToClipboard)
        }
    }

    private var sourceSelectionView: some View {
        ScrollView {
            VStack(spacing: 32) {
                headerSection

                VStack(spacing: 12) {
                    sourceButton(
                        title: "Scan Document",
                        subtitle: "Use camera to capture text",
                        icon: "camera.viewfinder",
                        gradient: [.blue, .cyan]
                    ) {
                        viewModel.showCamera = true
                    }

                    sourceButton(
                        title: "Import Document",
                        subtitle: "PDF, images, or text files",
                        icon: "doc.badge.plus",
                        gradient: [.purple, .indigo]
                    ) {
                        viewModel.showDocumentPicker = true
                    }

                    sourceButton(
                        title: "Photo Library",
                        subtitle: "Select an image to analyze",
                        icon: "photo.on.rectangle.angled",
                        gradient: [.orange, .pink]
                    ) {
                        viewModel.showImagePicker = true
                    }

                    sourceButton(
                        title: "Screen Capture",
                        subtitle: "Analyze current screen content",
                        icon: "rectangle.dashed.badge.record",
                        gradient: [.green, .mint]
                    ) {
                        viewModel.captureViewShot()
                    }
                }
                .padding(.horizontal, 16)

                capabilitiesSection
            }
            .padding(.vertical, 24)
        }
        .background(Color(.systemGroupedBackground))
    }

    private var headerSection: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.blue.opacity(0.15), .purple.opacity(0.15)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 88, height: 88)

                Image(systemName: "doc.text.viewfinder")
                    .font(.system(size: 38))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .symbolEffect(.breathe, options: .repeating)
            }

            VStack(spacing: 6) {
                Text("Document Intelligence")
                    .font(.title3.bold())

                Text("On-device OCR, barcode detection,\nand text extraction powered by Vision")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
    }

    private func sourceButton(title: String, subtitle: String, icon: String, gradient: [Color], action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(LinearGradient(colors: gradient, startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 44, height: 44)

                    Image(systemName: icon)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.primary)

                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding(14)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 14))
        }
    }

    private var capabilitiesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Capabilities")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20)

            VStack(spacing: 1) {
                capabilityRow(icon: "text.viewfinder", title: "OCR Text Recognition", detail: "Accurate multi-language text extraction")
                capabilityRow(icon: "barcode.viewfinder", title: "Barcode & QR Detection", detail: "Reads QR, EAN, Code128, and more")
                capabilityRow(icon: "doc.richtext", title: "PDF Processing", detail: "Multi-page PDF text extraction")
                capabilityRow(icon: "globe", title: "Language Detection", detail: "Auto-detects document language")
                capabilityRow(icon: "lock.shield", title: "Private & On-Device", detail: "All processing happens locally")
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 12))
            .padding(.horizontal, 16)
        }
    }

    private func capabilityRow(icon: String, title: String, detail: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundStyle(.blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.subheadline.weight(.medium))

                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var analysisResultView: some View {
        VStack(spacing: 0) {
            if viewModel.isProcessing {
                processingOverlay
            }

            if let image = viewModel.selectedImage {
                imagePreviewHeader(image)
            } else if !viewModel.documentName.isEmpty {
                documentHeader
            }

            if let error = viewModel.errorMessage {
                errorBanner(error)
            }

            if let result = viewModel.analysisResult {
                resultTabs(result)
            }
        }
        .background(Color(.systemGroupedBackground))
    }

    private var processingOverlay: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {
                ProgressView()
                    .controlSize(.small)

                Text(viewModel.statusMessage)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)

                Spacer()

                Text("\(Int(viewModel.progress * 100))%")
                    .font(.caption.monospacedDigit().bold())
                    .foregroundStyle(.blue)
            }

            ProgressView(value: viewModel.progress)
                .tint(.blue)
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground))
    }

    private func imagePreviewHeader(_ image: UIImage) -> some View {
        ZStack(alignment: .bottomTrailing) {
            Color(.secondarySystemBackground)
                .frame(height: 200)
                .overlay {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .allowsHitTesting(false)
                }
                .clipShape(.rect(cornerRadius: 0))

            if viewModel.analysisResult != nil {
                Button {
                    withAnimation { viewModel.showTextOverlay.toggle() }
                } label: {
                    Label(
                        viewModel.showTextOverlay ? "Hide Overlay" : "Show Overlay",
                        systemImage: viewModel.showTextOverlay ? "eye.slash" : "eye"
                    )
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                }
                .padding(12)
            }
        }
    }

    private var documentHeader: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.red.opacity(0.12))
                    .frame(width: 40, height: 40)

                Image(systemName: "doc.fill")
                    .font(.system(size: 18))
                    .foregroundStyle(.red)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.documentName)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)

                if let result = viewModel.analysisResult {
                    Text("\(result.pageCount) page\(result.pageCount == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground))
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)

            Text(message)
                .font(.caption.weight(.medium))
                .foregroundStyle(.primary)

            Spacer()
        }
        .padding(12)
        .background(Color.orange.opacity(0.08))
    }

    private func resultTabs(_ result: DocumentAnalysisResult) -> some View {
        VStack(spacing: 0) {
            Picker("Tab", selection: $viewModel.selectedTab) {
                ForEach(AnalysisTab.allCases, id: \.self) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            TabView(selection: $viewModel.selectedTab) {
                fullTextTab(result)
                    .tag(AnalysisTab.text)

                blocksTab(result)
                    .tag(AnalysisTab.blocks)

                barcodesTab
                    .tag(AnalysisTab.barcodes)

                infoTab(result)
                    .tag(AnalysisTab.info)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
        }
    }

    private func fullTextTab(_ result: DocumentAnalysisResult) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if result.fullText.isEmpty {
                    emptyResultView(icon: "text.magnifyingglass", message: "No text was detected")
                } else {
                    HStack {
                        Text("Extracted Text")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)

                        Spacer()

                        Button {
                            viewModel.copyFullText()
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                                .font(.caption.weight(.medium))
                        }
                    }

                    Text(result.fullText)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(.rect(cornerRadius: 10))
                }
            }
            .padding(16)
        }
    }

    private func blocksTab(_ result: DocumentAnalysisResult) -> some View {
        ScrollView {
            if result.blocks.isEmpty {
                emptyResultView(icon: "rectangle.dashed", message: "No text blocks detected")
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(result.blocks) { block in
                        HStack(alignment: .top, spacing: 10) {
                            confidenceBadge(block.confidence)

                            Text(block.text)
                                .font(.subheadline)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding(12)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(.rect(cornerRadius: 10))
                    }
                }
                .padding(16)
            }
        }
    }

    private var barcodesTab: some View {
        ScrollView {
            if viewModel.barcodeResults.isEmpty {
                emptyResultView(icon: "barcode", message: "No barcodes detected")
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(viewModel.barcodeResults) { barcode in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: "barcode.viewfinder")
                                    .foregroundStyle(.blue)

                                Text(barcode.symbology)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)

                                Spacer()

                                confidenceBadge(barcode.confidence)
                            }

                            Text(barcode.payload)
                                .font(.system(.body, design: .monospaced))
                                .textSelection(.enabled)

                            Button {
                                UIPasteboard.general.string = barcode.payload
                            } label: {
                                Label("Copy", systemImage: "doc.on.doc")
                                    .font(.caption.weight(.medium))
                            }
                        }
                        .padding(12)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(.rect(cornerRadius: 10))
                    }
                }
                .padding(16)
            }
        }
    }

    private func infoTab(_ result: DocumentAnalysisResult) -> some View {
        ScrollView {
            VStack(spacing: 1) {
                infoRow(label: "Pages", value: "\(result.pageCount)")
                infoRow(label: "Words", value: "\(result.wordCount)")
                infoRow(label: "Characters", value: "\(result.characterCount)")
                infoRow(label: "Text Blocks", value: "\(result.blocks.count)")
                infoRow(label: "Avg Confidence", value: "\(Int(result.averageConfidence * 100))%")
                infoRow(label: "Barcodes Found", value: "\(viewModel.barcodeResults.count)")

                if let lang = result.languageHint {
                    infoRow(label: "Detected Language", value: Locale.current.localizedString(forLanguageCode: lang) ?? lang)
                }
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 12))
            .padding(16)
        }
    }

    private func infoRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()

            Text(value)
                .font(.subheadline.weight(.medium).monospacedDigit())
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func confidenceBadge(_ confidence: Float) -> some View {
        Text("\(Int(confidence * 100))%")
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundStyle(confidence > 0.9 ? .green : confidence > 0.7 ? .orange : .red)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                (confidence > 0.9 ? Color.green : confidence > 0.7 ? Color.orange : Color.red)
                    .opacity(0.1)
            )
            .clipShape(Capsule())
    }

    private func emptyResultView(icon: String, message: String) -> some View {
        VStack(spacing: 12) {
            Spacer()

            Image(systemName: icon)
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()
        }
        .frame(maxWidth: .infinity, minHeight: 200)
        .padding(16)
    }

    private var copiedToast: some View {
        VStack {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)

                Text("Copied to clipboard")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial)
            .clipShape(Capsule())
            .shadow(color: .black.opacity(0.1), radius: 8, y: 4)
            .padding(.top, 8)

            Spacer()
        }
    }
}
