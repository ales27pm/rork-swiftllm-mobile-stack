import SwiftUI

struct OnboardingView: View {
    let modelLoader: ModelLoaderService
    let onDismiss: () -> Void

    @State private var currentStep: Int = 0
    @State private var selectedModelID: String? = "dolphin3-3b-q4-gguf"
    @State private var hasStartedDownload: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            stepIndicator
                .padding(.top, 20)

            TabView(selection: $currentStep) {
                welcomeStep.tag(0)
                modelStep.tag(1)
                readyStep.tag(2)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .animation(.spring(duration: 0.4), value: currentStep)

            bottomBar
                .padding(.horizontal, 24)
                .padding(.bottom, 16)
        }
        .background(Color(.systemBackground))
        .interactiveDismissDisabled(currentStep == 1 && isDownloading)
    }

    private var stepIndicator: some View {
        HStack(spacing: 8) {
            ForEach(0..<3, id: \.self) { index in
                Capsule()
                    .fill(index <= currentStep ? Color.blue : Color(.tertiarySystemFill))
                    .frame(width: index == currentStep ? 24 : 8, height: 8)
                    .animation(.spring(duration: 0.3), value: currentStep)
            }
        }
    }

    private var welcomeStep: some View {
        VStack(spacing: 32) {
            Spacer()

            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [.blue.opacity(0.2), .purple.opacity(0.08), .clear],
                            center: .center,
                            startRadius: 20,
                            endRadius: 100
                        )
                    )
                    .frame(width: 200, height: 200)

                Image(systemName: "brain.filled.head.profile")
                    .font(.system(size: 72))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .symbolEffect(.breathe, options: .repeating)
            }

            VStack(spacing: 12) {
                Text("Welcome to Nexus")
                    .font(.largeTitle.bold())

                Text("Your private AI assistant that runs entirely on your device. No cloud. No data leaves your phone.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }

            VStack(spacing: 16) {
                featureRow(icon: "lock.shield.fill", color: .green, title: "100% Private", subtitle: "All processing stays on-device")
                featureRow(icon: "bolt.fill", color: .orange, title: "Fast Inference", subtitle: "Optimized for Apple Neural Engine")
                featureRow(icon: "brain", color: .purple, title: "Persistent Memory", subtitle: "Remembers context across conversations")
            }
            .padding(.horizontal, 32)

            Spacer()
        }
    }

    private func featureRow(icon: String, color: Color, title: String, subtitle: String) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundStyle(color)
                .frame(width: 36, height: 36)
                .background(color.opacity(0.12))
                .clipShape(.rect(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
    }

    private var modelStep: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 16)

            VStack(spacing: 8) {
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 40))
                    .foregroundStyle(.blue)

                Text("Download a Model")
                    .font(.title2.bold())

                Text("Pick a model to get started. You can download more later from the Models tab.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }

            ScrollView {
                VStack(spacing: 10) {
                    ForEach(onboardingModels) { model in
                        onboardingModelCard(model)
                    }
                }
                .padding(.horizontal, 24)
            }

            Spacer(minLength: 16)
        }
    }

    private var onboardingModels: [ModelManifest] {
        modelLoader.availableModels.filter { manifest in
            guard manifest.format == .gguf else { return false }
            guard !manifest.isDraft else { return false }
            guard !(modelLoader.modelStatuses[manifest.id]?.blocksDownload ?? false) else { return false }
            return true
        }.sorted { lhs, rhs in
            let lr = lhs.recommendation?.rank ?? Int.max
            let rr = rhs.recommendation?.rank ?? Int.max
            return lr < rr
        }.prefix(5).map { $0 }
    }

    private func onboardingModelCard(_ model: ModelManifest) -> some View {
        let isSelected = selectedModelID == model.id
        let status = modelLoader.modelStatuses[model.id] ?? .notDownloaded

        return Button {
            if case .downloading = status { return }
            selectedModelID = model.id
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 6) {
                            Text(model.name)
                                .font(.headline)
                            if let rec = model.recommendation {
                                Text(rec.badge.uppercased())
                                    .font(.caption2.bold())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(.teal.opacity(0.18))
                                    .foregroundStyle(.teal)
                                    .clipShape(Capsule())
                            }
                        }
                        Text("\(model.variant) · \(model.sizeFormatted)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if case .ready = status {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.title3)
                    } else {
                        Image(systemName: isSelected ? "circle.inset.filled" : "circle")
                            .foregroundStyle(isSelected ? .blue : Color(.tertiaryLabel))
                            .font(.title3)
                    }
                }

                if let rec = model.recommendation {
                    Text(rec.reason)
                        .font(.caption)
                        .foregroundStyle(.teal)
                }

                if case .downloading(let progress) = status {
                    ProgressView(value: progress)
                        .tint(.blue)
                    Text("\(Int(progress * 100))%")
                        .font(.caption2.monospacedDigit().bold())
                        .foregroundStyle(.blue)
                }

                if case .verifying = status {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.mini)
                        Text("Verifying...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(14)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 14))
            .overlay {
                if isSelected {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(.blue.opacity(0.5), lineWidth: 1.5)
                }
            }
        }
        .buttonStyle(.plain)
    }

    private var readyStep: some View {
        VStack(spacing: 32) {
            Spacer()

            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [.green.opacity(0.2), .clear],
                            center: .center,
                            startRadius: 20,
                            endRadius: 80
                        )
                    )
                    .frame(width: 160, height: 160)

                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 72))
                    .foregroundStyle(.green)
                    .symbolEffect(.bounce, value: currentStep == 2)
            }

            VStack(spacing: 12) {
                Text("You're All Set")
                    .font(.largeTitle.bold())

                Text("Nexus is ready to chat. Type a message or try voice mode to get started.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }

            VStack(spacing: 12) {
                tipRow(icon: "text.bubble", text: "Type or speak to chat with Nexus")
                tipRow(icon: "wrench.and.screwdriver", text: "Enable Device Tools for battery, location & more")
                tipRow(icon: "brain", text: "Nexus remembers details across conversations")
            }
            .padding(.horizontal, 32)

            Spacer()
        }
    }

    private func tipRow(icon: String, text: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundStyle(.blue)
                .frame(width: 28)

            Text(text)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()
        }
    }

    private var isDownloading: Bool {
        guard let id = selectedModelID,
              let status = modelLoader.modelStatuses[id] else { return false }
        switch status {
        case .downloading, .verifying, .compiling: return true
        default: return false
        }
    }

    private var isSelectedModelReady: Bool {
        guard let id = selectedModelID,
              let status = modelLoader.modelStatuses[id] else { return false }
        if case .ready = status { return true }
        return false
    }

    private var hasAnyReadyModel: Bool {
        modelLoader.availableModels.contains { model in
            if case .ready = modelLoader.modelStatuses[model.id] { return true }
            return false
        }
    }

    private var bottomBar: some View {
        VStack(spacing: 12) {
            switch currentStep {
            case 0:
                Button {
                    withAnimation { currentStep = 1 }
                } label: {
                    Text("Get Started")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)

            case 1:
                if isSelectedModelReady || hasAnyReadyModel {
                    Button {
                        if let id = selectedModelID, isSelectedModelReady {
                            modelLoader.activateModel(id)
                        }
                        withAnimation { currentStep = 2 }
                    } label: {
                        Text("Continue")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                } else if isDownloading {
                    Button {} label: {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small).tint(.white)
                            Text("Downloading...")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(true)
                } else {
                    Button {
                        guard let id = selectedModelID else { return }
                        hasStartedDownload = true
                        modelLoader.downloadModel(id)
                    } label: {
                        Text("Download \(selectedModelName)")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(selectedModelID == nil)
                }

                Button {
                    withAnimation { currentStep = 2 }
                } label: {
                    Text("Skip for Now")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

            case 2:
                Button {
                    onDismiss()
                } label: {
                    Text("Start Chatting")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)

            default:
                EmptyView()
            }
        }
    }

    private var selectedModelName: String {
        guard let id = selectedModelID,
              let model = modelLoader.availableModels.first(where: { $0.id == id }) else {
            return "Model"
        }
        return "\(model.name) (\(model.sizeFormatted))"
    }
}
