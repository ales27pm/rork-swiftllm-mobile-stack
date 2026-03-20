import SwiftUI
import AVFoundation

struct SettingsView: View {
    @Bindable var chatViewModel: ChatViewModel
    @Bindable var speechViewModel: SpeechViewModel
    let thermalGovernor: ThermalGovernor
    var metricsLogger: MetricsLogger?
    var modelLoader: ModelLoaderService?
    var inferenceEngine: InferenceEngine?
    var keyValueStore: KeyValueStore?
    var diagSecureStore: SecureStore?
    var fileSystem: FileSystemService?
    var database: DatabaseService?
    var memoryService: MemoryService?
    var conversationService: ConversationService?

    @State private var draftSampling = SamplingDraft()
    @State private var hfToken: String = ""
    @State private var isTokenVisible = false
    @State private var tokenSaveState: TokenSaveState = .idle
    @State private var selectedSpeechLanguageCode: String?
    @State private var selectedSpeechVoiceIdentifier: String?

    private let secureStore = SecureStore()
    private let hfTokenKey = "hf_api_token"

    var body: some View {
        Form {
            overviewSection
            modelBehaviorSection
            systemPromptSection
            toolsSection
            speechSection
            accessSection
            runtimeSection
            engineSection
            diagnosticsSection
        }
        .navigationTitle(AppStrings.settingsTitle)
        .navigationBarTitleDisplayMode(.large)
        .onAppear(perform: loadConfig)
        .onChange(of: chatViewModel.systemPrompt) { _, _ in
            chatViewModel.saveSettings()
        }
    }

    private func loadConfig() {
        draftSampling = SamplingDraft(config: chatViewModel.samplingConfig)
        hfToken = secureStore.getString(hfTokenKey) ?? ""
        selectedSpeechLanguageCode = speechViewModel.selectedSpeechLanguageCode
        selectedSpeechVoiceIdentifier = speechViewModel.selectedSpeechVoiceIdentifier
    }

    private func applySamplingChanges() {
        chatViewModel.samplingConfig = draftSampling.makeConfig()
        chatViewModel.saveSettings()
    }

    private func saveToken() {
        let trimmedToken = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        let didPersist: Bool

        if trimmedToken.isEmpty {
            secureStore.delete(hfTokenKey)
            hfToken = ""
            didPersist = true
        } else {
            didPersist = secureStore.setString(trimmedToken, forKey: hfTokenKey)
            if didPersist {
                hfToken = trimmedToken
            }
        }

        tokenSaveState = didPersist ? .saved : .failed

        guard didPersist else { return }

        Task {
            try? await Task.sleep(for: .seconds(2))
            if tokenSaveState == .saved {
                tokenSaveState = .idle
            }
        }
    }

    private var overviewSection: some View {
        Section {
            statusRow(
                title: "Runtime Mode",
                subtitle: runtimeSummaryText,
                icon: thermalGovernor.currentMode.icon,
                tint: runtimeAccentColor
            )

            statusRow(
                title: "Speech",
                subtitle: speechSummaryText,
                icon: "waveform",
                tint: .blue
            )

            statusRow(
                title: "Tools",
                subtitle: chatViewModel.toolsEnabled ? "Device capabilities are available during chat." : "Tool calling is disabled for generation.",
                icon: chatViewModel.toolsEnabled ? "hammer.fill" : "hammer",
                tint: chatViewModel.toolsEnabled ? .orange : .secondary
            )
        } header: {
            Label("Overview", systemImage: "slider.horizontal.below.rectangle")
        } footer: {
            Text("Review the current assistant configuration before changing generation, speech, or device access behavior.")
        }
    }

    private var modelBehaviorSection: some View {
        Section {
            samplingControl(
                title: AppStrings.temperature,
                description: "Balances determinism versus creativity.",
                value: $draftSampling.temperature,
                range: 0...2,
                step: 0.05,
                format: "%.2f"
            )
            samplingControl(
                title: AppStrings.topK,
                description: "Limits the number of tokens considered at each step.",
                value: $draftSampling.topK,
                range: 1...100,
                step: 1,
                format: "%.0f"
            )
            samplingControl(
                title: AppStrings.topP,
                description: "Narrows token selection using cumulative probability.",
                value: $draftSampling.topP,
                range: 0...1,
                step: 0.05,
                format: "%.2f"
            )
            samplingControl(
                title: AppStrings.repetitionPenalty,
                description: "Discourages repeated phrasing across long outputs.",
                value: $draftSampling.repetitionPenalty,
                range: 1...2,
                step: 0.05,
                format: "%.2f"
            )
            samplingControl(
                title: AppStrings.maxTokens,
                description: "Caps the maximum assistant response length.",
                value: $draftSampling.maxTokens,
                range: 128...8192,
                step: 128,
                format: "%.0f"
            )
        } header: {
            Label("Model Behavior", systemImage: "dial.high")
        } footer: {
            Text(AppStrings.samplingFooter)
        }
    }

    private func samplingControl(
        title: String,
        description: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        format: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline)
                    Text(description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Slider(value: value, in: range, step: step) { _ in
                applySamplingChanges()
            }
        }
        .padding(.vertical, 4)
    }

    private var systemPromptSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                Text("Assistant Instructions")
                    .font(.headline)
                Text("This prompt is prepended to every conversation and shapes the assistant’s voice, safety, and priorities.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $chatViewModel.systemPrompt)
                    .font(.subheadline)
                    .frame(minHeight: 140)
            }
        } header: {
            Label(AppStrings.systemPromptTitle, systemImage: "text.quote")
        } footer: {
            Text(AppStrings.systemPromptFooter)
        }
    }

    private var toolsSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { chatViewModel.toolsEnabled },
                set: {
                    chatViewModel.toolsEnabled = $0
                    chatViewModel.saveSettings()
                }
            )) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Device Tool Calling")
                    Text("Allow the assistant to use on-device capabilities when a conversation requires them.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if chatViewModel.toolsEnabled {
                ForEach(SettingsToolCategory.defaults) { category in
                    HStack(spacing: 12) {
                        Image(systemName: category.icon)
                            .foregroundStyle(.orange)
                            .frame(width: 22)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(category.title)
                            Text(category.description)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        } header: {
            Label("Tools", systemImage: "hammer.fill")
        } footer: {
            Text("Disable tool calling when you want pure text-only inference without using device integrations.")
        }
    }

    private var speechSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { speechViewModel.isAutoListenEnabled },
                set: { speechViewModel.isAutoListenEnabled = $0 }
            )) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Auto-Listen")
                    Text("Resume listening automatically after spoken responses complete.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Picker("Speech Language", selection: $selectedSpeechLanguageCode) {
                Text("System Default")
                    .tag(Optional<String>.none)

                ForEach(speechViewModel.speechLanguageOptions()) { option in
                    Text(option.label)
                        .tag(Optional(option.code))
                }
            }
            .onChange(of: selectedSpeechLanguageCode) { _, newValue in
                speechViewModel.updateSpeechLanguage(code: newValue)
                selectedSpeechVoiceIdentifier = nil
                speechViewModel.updateSpeechVoice(identifier: nil)
            }

            Picker("Speech Voice", selection: $selectedSpeechVoiceIdentifier) {
                Text("System Default")
                    .tag(Optional<String>.none)

                ForEach(speechViewModel.speechVoices(for: selectedSpeechLanguageCode), id: \.identifier) { voice in
                    Text(voiceRowLabel(for: voice))
                        .tag(Optional(voice.identifier))
                }
            }
            .onChange(of: selectedSpeechVoiceIdentifier) { _, newValue in
                speechViewModel.updateSpeechVoice(identifier: newValue)
            }
        } header: {
            Label("Speech", systemImage: "waveform")
        } footer: {
            Text("Language and voice changes apply to speech output, speech recognition input, and voice-mode replies.")
        }
    }

    private var accessSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    Image(systemName: "key.fill")
                        .foregroundStyle(.orange)
                        .frame(width: 24)

                    Group {
                        if isTokenVisible {
                            TextField("hf_...", text: $hfToken)
                        } else {
                            SecureField("hf_...", text: $hfToken)
                        }
                    }
                    .font(.subheadline.monospaced())
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()

                    Button {
                        isTokenVisible.toggle()
                    } label: {
                        Image(systemName: isTokenVisible ? "eye.slash" : "eye")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }

                Button(action: saveToken) {
                    HStack {
                        Spacer()
                        Label(tokenSaveState.buttonTitle, systemImage: tokenSaveState.buttonIcon)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(tokenSaveState.buttonColor)
                        Spacer()
                    }
                }
                .disabled(tokenSaveState == .saved)

                if tokenSaveState == .failed {
                    Text("The token could not be saved securely. Try again after checking device keychain availability.")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        } header: {
            Label("Access", systemImage: "server.rack")
        } footer: {
            Text("A Hugging Face token is only required for downloading gated models. Generate one at huggingface.co/settings/tokens.")
        }
    }

    private var runtimeSection: some View {
        Section {
            runtimeMetricRow(title: AppStrings.thermalState, value: thermalGovernor.thermalLevel.rawValue, icon: "thermometer.medium", tint: thermalAccentColor)
            runtimeMetricRow(title: AppStrings.runtimeMode, value: thermalGovernor.currentMode.rawValue, icon: thermalGovernor.currentMode.icon, tint: runtimeAccentColor)
            runtimeMetricRow(title: AppStrings.maxDraftTokens, value: "\(thermalGovernor.currentMode.maxDraftTokens)", icon: "number", tint: .blue)
            runtimeMetricRow(title: AppStrings.speculativeDecoding, value: thermalGovernor.currentMode.speculativeEnabled ? AppStrings.enabled : AppStrings.disabled, icon: "bolt.fill", tint: thermalGovernor.currentMode.speculativeEnabled ? .green : .secondary)
            runtimeMetricRow(title: "Memory Pressure", value: thermalGovernor.memoryPressureLevel.rawValue.capitalized, icon: "memorychip", tint: .purple)
            runtimeMetricRow(title: "Current Memory", value: String(format: "%.0f MB", thermalGovernor.currentMemoryUsageMB), icon: "internaldrive", tint: .teal)
        } header: {
            Label(AppStrings.runtimeTitle, systemImage: "cpu")
        } footer: {
            Text("Runtime mode adapts automatically from thermal and memory conditions, so these values are read-only diagnostics.")
        }
    }

    private func runtimeMetricRow(title: String, value: String, icon: String, tint: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(tint)
                .frame(width: 22)
            Text(title)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }

    private var diagnosticsSection: some View {
        Section {
            NavigationLink {
                DiagnosticsView(
                    database: database ?? DatabaseService(),
                    keyValueStore: keyValueStore ?? KeyValueStore(),
                    secureStore: diagSecureStore ?? SecureStore(),
                    fileSystem: fileSystem ?? FileSystemService(),
                    memoryService: memoryService,
                    conversationService: conversationService,
                    metricsLogger: metricsLogger ?? MetricsLogger(),
                    thermalGovernor: thermalGovernor,
                    modelLoader: modelLoader ?? ModelLoaderService(),
                    inferenceEngine: inferenceEngine
                )
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "stethoscope")
                        .foregroundStyle(.purple)
                        .frame(width: 24)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Run Diagnostics")
                            .font(.headline)
                        Text("Test all subsystems, generate a downloadable report.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            }
        } header: {
            Label("Diagnostics", systemImage: "waveform.path.ecg")
        } footer: {
            Text("Runs exhaustive real tests on every subsystem and produces a shareable log file.")
        }
    }

    private var engineSection: some View {
        Section {
            aboutRow(label: "Backends", value: "CoreML + llama.cpp (GGUF)")
            aboutRow(label: "Architecture", value: "Split Prefill / Decode + Paged KV")
            aboutRow(label: "KV Cache", value: "Paged Arena (128 tokens / page)")
            aboutRow(label: "Speculation", value: "Adaptive Draft Length")
            aboutRow(label: "Prefix Cache", value: "Hash-based (8 entries)")
        } header: {
            Label("Engine", systemImage: "gearshape.2")
        } footer: {
            Text("These values document the current local inference stack shipping with the app build.")
        }
    }

    private func aboutRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }

    private func statusRow(title: String, subtitle: String, icon: String, tint: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(tint)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private var runtimeSummaryText: String {
        "\(thermalGovernor.currentMode.rawValue) • \(thermalGovernor.thermalLevel.rawValue) thermal • \(thermalGovernor.memoryPressureLevel.rawValue.capitalized) memory"
    }

    private var speechSummaryText: String {
        let languageSummary = selectedSpeechLanguageCode ?? "System Default"
        let voiceSummary = selectedSpeechVoiceIdentifier == nil ? "Default Voice" : "Custom Voice"
        return "\(languageSummary) • \(voiceSummary)"
    }

    private var runtimeAccentColor: Color {
        switch thermalGovernor.currentMode {
        case .maxPerformance:
            return .green
        case .balanced:
            return .blue
        case .coolDown:
            return .orange
        case .emergency:
            return .red
        }
    }

    private var thermalAccentColor: Color {
        switch thermalGovernor.thermalLevel {
        case .nominal:
            return .green
        case .fair:
            return .yellow
        case .serious:
            return .orange
        case .critical:
            return .red
        }
    }

    private func voiceRowLabel(for voice: SpeechSynthesisService.VoiceOption) -> String {
        let quality = qualityLabel(for: voice.quality).map { " · \($0)" } ?? ""
        return "\(voice.displayName)\(quality)"
    }

    private func qualityLabel(for quality: AVSpeechSynthesisVoiceQuality) -> String? {
        switch quality {
        case .default:
            return "Default"
        case .enhanced:
            return "Enhanced"
        case .premium:
            return "Premium"
        @unknown default:
            return nil
        }
    }
}

struct SamplingDraft: Equatable {
    var temperature: Double
    var topK: Double
    var topP: Double
    var repetitionPenalty: Double
    var maxTokens: Double

    init(
        temperature: Double = 0.8,
        topK: Double = 40,
        topP: Double = 0.95,
        repetitionPenalty: Double = 1.1,
        maxTokens: Double = 2048
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
    }

    init(config: SamplingConfig) {
        self.temperature = Double(config.temperature)
        self.topK = Double(config.topK)
        self.topP = Double(config.topP)
        self.repetitionPenalty = Double(config.repetitionPenalty)
        self.maxTokens = Double(config.maxTokens)
    }

    func makeConfig() -> SamplingConfig {
        SamplingConfig(
            temperature: Float(temperature),
            topK: Int(topK.rounded()),
            topP: Float(topP),
            repetitionPenalty: Float(repetitionPenalty),
            maxTokens: Int(maxTokens.rounded()),
            stopSequences: [],
            samplerSeed: nil
        )
    }
}

struct SettingsToolCategory: Identifiable, Equatable {
    let id: String
    let title: String
    let description: String
    let icon: String

    static let defaults: [SettingsToolCategory] = [
        .init(id: "location", title: "Location & Maps", description: "Use map lookups, directions, and local context.", icon: "location.fill"),
        .init(id: "device", title: "Battery & Device", description: "Read battery, device, and environment signals.", icon: "battery.100percent"),
        .init(id: "calendar", title: "Calendar & Events", description: "Review and create event-related actions.", icon: "calendar"),
        .init(id: "contacts", title: "Contacts", description: "Reference people and communication targets.", icon: "person.2.fill"),
        .init(id: "notifications", title: "Notifications", description: "Schedule reminders and alerts when supported.", icon: "bell.fill"),
        .init(id: "screen", title: "Screen & Haptics", description: "Coordinate visual feedback and haptic responses.", icon: "sun.max.fill"),
        .init(id: "sharing", title: "Sharing & Messaging", description: "Send content into system share flows.", icon: "square.and.arrow.up"),
        .init(id: "time", title: "Date & Time", description: "Work with calendars, clocks, and scheduling context.", icon: "clock.fill")
    ]
}

enum TokenSaveState: Equatable {
    case idle
    case saved
    case failed

    var buttonTitle: String {
        switch self {
        case .idle:
            return "Save Token"
        case .saved:
            return "Saved"
        case .failed:
            return "Save Failed"
        }
    }

    var buttonIcon: String {
        switch self {
        case .idle:
            return "square.and.arrow.down"
        case .saved:
            return "checkmark.circle.fill"
        case .failed:
            return "exclamationmark.triangle.fill"
        }
    }

    var buttonColor: Color {
        switch self {
        case .idle:
            return .accentColor
        case .saved:
            return .green
        case .failed:
            return .red
        }
    }
}
