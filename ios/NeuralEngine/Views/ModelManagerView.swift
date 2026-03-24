import SwiftUI

struct ModelManagerView: View {
    @Bindable var viewModel: ModelManagerViewModel
    @State private var modelToDelete: ModelManifest?
    @State private var showDeleteConfirmation: Bool = false
    @State private var sortOrder: ModelSortOrder = .name

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                filterBar
                sortBar
                modelList
            }
            .padding(.vertical, 12)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Models")
        .navigationBarTitleDisplayMode(.large)
        .searchable(text: $viewModel.searchText, prompt: "Search models")
        .alert("Delete Model", isPresented: $showDeleteConfirmation, presenting: modelToDelete) { model in
            Button("Delete", role: .destructive) {
                withAnimation(.snappy) {
                    viewModel.delete(model)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: { model in
            Text("Remove \(model.name) \(model.variant)? You can re-download it later.")
        }
    }

    private var filterBar: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(ModelFilter.allCases, id: \.self) { filter in
                    Button {
                        withAnimation(.snappy) {
                            viewModel.selectedFilter = filter
                        }
                    } label: {
                        Text(filter.rawValue)
                            .font(.subheadline.weight(.medium))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(
                                viewModel.selectedFilter == filter
                                    ? AnyShapeStyle(.blue)
                                    : AnyShapeStyle(Color(.tertiarySystemBackground))
                            )
                            .foregroundStyle(viewModel.selectedFilter == filter ? .white : .primary)
                            .clipShape(Capsule())
                    }
                }
            }
        }
        .contentMargins(.horizontal, 16)
        .scrollIndicators(.hidden)
    }

    private var sortBar: some View {
        HStack {
            Text("\(sortedModels.count) models")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            Menu {
                ForEach(ModelSortOrder.allCases, id: \.self) { order in
                    Button {
                        withAnimation(.snappy) { sortOrder = order }
                    } label: {
                        Label(order.rawValue, systemImage: sortOrder == order ? "checkmark" : "")
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.up.arrow.down")
                    Text(sortOrder.rawValue)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
    }

    private var sortedModels: [ModelManifest] {
        viewModel.filteredModels.sorted { lhs, rhs in
            let lhsRank = lhs.recommendation?.rank ?? Int.max
            let rhsRank = rhs.recommendation?.rank ?? Int.max
            if lhsRank != rhsRank {
                return lhsRank < rhsRank
            }

            switch sortOrder {
            case .name:
                return lhs.name < rhs.name
            case .size:
                return lhs.sizeBytes < rhs.sizeBytes
            case .context:
                return lhs.contextLength > rhs.contextLength
            case .parameters:
                return lhs.sizeBytes > rhs.sizeBytes
            }
        }
    }

    private var modelList: some View {
        LazyVStack(spacing: 10) {
            ForEach(sortedModels) { model in
                ModelCardView(
                    model: model,
                    status: viewModel.status(for: model),
                    isActive: viewModel.isActiveModel(model),
                    onDownload: { viewModel.download(model) },
                    onDelete: {
                        modelToDelete = model
                        showDeleteConfirmation = true
                    },
                    onActivate: {
                        withAnimation(.snappy) {
                            viewModel.activate(model)
                        }
                    }
                )
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .move(edge: .top)),
                    removal: .opacity
                ))
            }
        }
        .padding(.horizontal, 16)
    }
}

enum ModelSortOrder: String, CaseIterable {
    case name = "Name"
    case size = "Size"
    case context = "Context"
    case parameters = "Parameters"
}

struct ModelCardView: View {
    let model: ModelManifest
    let status: ModelStatus
    let isActive: Bool
    let onDownload: () -> Void
    let onDelete: () -> Void
    let onActivate: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(model.name)
                            .font(.headline)

                        if let recommendation = model.recommendation {
                            Text(recommendation.badge.uppercased())
                                .font(.caption2.bold())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.teal.opacity(0.18))
                                .foregroundStyle(.teal)
                                .clipShape(Capsule())
                        }

                        if model.isEmbedding {
                            Text("EMBEDDING")
                                .font(.caption2.bold())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.mint.opacity(0.2))
                                .foregroundStyle(.mint)
                                .clipShape(Capsule())
                        }

                        if model.isDraft {
                            Text("DRAFT")
                                .font(.caption2.bold())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.orange.opacity(0.2))
                                .foregroundStyle(.orange)
                                .clipShape(Capsule())
                        }

                        if isActive {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.subheadline)
                        }
                    }

                    HStack(spacing: 6) {
                        Text(model.variant)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Text(model.format == .gguf ? "GGUF" : "CoreML")
                            .font(.caption2.bold())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(model.format == .gguf ? Color.indigo.opacity(0.15) : Color.blue.opacity(0.15))
                            .foregroundStyle(model.format == .gguf ? .indigo : .blue)
                            .clipShape(Capsule())
                    }
                }

                Spacer()

                architectureTag
            }

            HStack(spacing: 16) {
                Label(model.sizeFormatted, systemImage: "internaldrive")
                if model.isEmbedding, let dims = model.embeddingDimensions {
                    Label("\(dims)d", systemImage: "arrow.triangle.branch")
                } else {
                    Label("\(model.contextLength) ctx", systemImage: "text.line.last.and.arrowtriangle.forward")
                }
                Label(model.quantization, systemImage: "cube")
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            if let recommendation = model.recommendation {
                Label(recommendation.reason, systemImage: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.teal)
            }

            statusBar
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 14))
        .overlay {
            if isActive {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(model.isEmbedding ? .mint.opacity(0.4) : .blue.opacity(0.4), lineWidth: 1.5)
            }
        }
        .sensoryFeedback(.selection, trigger: isActive)
    }

    private var architectureTag: some View {
        Text(model.architecture.rawValue.uppercased())
            .font(.caption2.bold().monospaced())
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(architectureColor.opacity(0.15))
            .foregroundStyle(architectureColor)
            .clipShape(.rect(cornerRadius: 6))
    }

    private var architectureColor: Color {
        switch model.architecture {
        case .llama: return .blue
        case .phi: return .purple
        case .gemma: return .green
        case .qwen: return .orange
        case .mistral: return .cyan
        case .smolLM: return .pink
        case .dolphin: return .teal
        case .bert: return .mint
        }
    }

    @ViewBuilder
    private var statusBar: some View {
        switch status {
        case .notDownloaded:
            Button {
                onDownload()
            } label: {
                Label("Download", systemImage: "arrow.down.circle")
                    .font(.subheadline.weight(.medium))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .tint(.blue)

        case .downloading(let progress):
            VStack(spacing: 6) {
                ProgressView(value: progress)
                    .tint(.blue)

                HStack {
                    Text("Downloading")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(progress * 100))%")
                        .font(.caption.monospacedDigit().bold())
                        .foregroundStyle(.blue)
                }
            }

        case .verifying:
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text("Verifying checksum...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        case .compiling:
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text("Compiling model...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        case .ready:
            HStack(spacing: 8) {
                if !isActive {
                    Button {
                        onActivate()
                    } label: {
                        Label("Activate", systemImage: "power")
                            .font(.subheadline.weight(.medium))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Label("Active", systemImage: "checkmark.circle.fill")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.green)
                        .frame(maxWidth: .infinity)
                }

                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Image(systemName: "trash")
                        .font(.subheadline)
                }
                .buttonStyle(.bordered)
            }

        case .checksumFailed(let error):
            VStack(spacing: 6) {
                Label(error, systemImage: "exclamationmark.shield.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)

                HStack(spacing: 8) {
                    Button(role: .destructive) {
                        onDelete()
                    } label: {
                        Label("Delete corrupted files", systemImage: "trash")
                            .font(.subheadline.weight(.medium))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        onDownload()
                    } label: {
                        Label("Re-download", systemImage: "arrow.clockwise")
                            .font(.subheadline.weight(.medium))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

        case .unsupported(let error):
            Label(error, systemImage: "nosign")
                .font(.caption)
                .foregroundStyle(.secondary)

        case .failed(let error):
            VStack(spacing: 6) {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)

                Button {
                    onDownload()
                } label: {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.subheadline.weight(.medium))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
    }
}
