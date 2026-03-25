import SwiftUI
import AVFoundation
import UIKit

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

    @State private var samplingDraft: SamplingDraft = .defaults
    @State private var hfToken: String = ""
    @State private var isTokenVisible: Bool = false
    @State private var tokenSaveState: TokenSaveState = .idle
    @State private var selectedSpeechLanguageCode: String?
    @State private var selectedSpeechVoiceIdentifier: String?
    @State private var isAdvancedSamplingExpanded: Bool = false
    @State private var modelManagerViewModel: ModelManagerViewModel?
    @State private var didLoadSettings: Bool = false
    @State private var showsResetConfirmation: Bool = false

    private let secureStore = SecureStore()
    private let hfTokenKey = "hf_api_token"

    var body: some View {
        Form {
            assistantSection
            voiceSection
            generationSection

            if modelLoader != nil {
                modelsSection
            }

            privacySection
            supportSection
        }
        .navigationTitle(AppStrings.settingsTitle)
        .navigationBarTitleDisplayMode(.large)
        .navigationDestination(for: SettingsDestination.self) { destination in
            destinationView(for: destination)
        }
        .task {
            loadSettings()
        }
        .onChange(of: samplingDraft) { _, _ in
            guard didLoadSettings else { return }
            applySamplingChanges()
        }
        .onChange(of: selectedSpeechLanguageCode) { _, newValue in
            guard didLoadSettings else { return }
            speechViewModel.updateSpeechLanguage(code: newValue)
            selectedSpeechLanguageCode = speechViewModel.selectedSpeechLanguageCode
            selectedSpeechVoiceIdentifier = speechViewModel.selectedSpeechVoiceIdentifier
        }
        .onChange(of: selectedSpeechVoiceIdentifier) { oldValue, newValue in
            guard didLoadSettings, oldValue != newValue else { return }
            speechViewModel.updateSpeechVoice(identifier: newValue)
            selectedSpeechVoiceIdentifier = speechViewModel.selectedSpeechVoiceIdentifier
        }
        .onChange(of: chatViewModel.systemPrompt) { _, _ in
            guard didLoadSettings else { return }
            chatViewModel.saveSettings()
        }
        .onChange(of: hfToken) { oldValue, newValue in
            guard oldValue != newValue, tokenSaveState != .idle else { return }
            tokenSaveState = .idle
        }
        .alert("Reset Assistant Settings", isPresented: $showsResetConfirmation) {
            Button("Reset", role: .destructive) {
                resetAssistantSettings()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This resets assistant persona, custom instructions, tool calling, generation tuning, and voice preferences to their defaults.")
        }
    }

    private var assistantSection: some View {
        Section {
            NavigationLink(value: SettingsDestination.personas) {
                navigationRow(
                    title: "Assistant Persona",
                    subtitle: activePersonaDescription,
                    systemImage: activePersonaIcon,
                    tint: activePersonaTint,
                    value: activePersonaName
                )
            }

            NavigationLink(value: SettingsDestination.customPrompt) {
                navigationRow(
                    title: "Custom Instructions",
                    subtitle: customPromptSummary,
                    systemImage: "text.quote",
                    tint: .purple,
                    value: isUsingDefaultPrompt ? "Default" : "Customized"
                )
            }

            Toggle(isOn: Binding(
                get: { chatViewModel.toolsEnabled },
                set: {
                    chatViewModel.toolsEnabled = $0
                    chatViewModel.saveSettings()
                }
            )) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Device Tool Calling")
                    Text("Allow the assistant to use on-device capabilities like location, calendar, contacts, camera, notifications, and sharing when a request needs them.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Assistant")
        } footer: {
            Text("Keep frequently changed assistant behavior here. Use the iOS Settings app for permission grants.")
        }
    }

    private var voiceSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { speechViewModel.isAutoListenEnabled },
                set: { speechViewModel.isAutoListenEnabled = $0 }
            )) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Auto-Listen")
                    Text("Resume listening after spoken replies finish so voice mode can continue hands-free.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Picker("Response Language", selection: $selectedSpeechLanguageCode) {
                Text("System Default")
                    .tag(Optional<String>.none)

                ForEach(speechViewModel.speechLanguageOptions()) { option in
                    Text(option.label)
                        .tag(Optional(option.code))
                }
            }

            Picker("Voice", selection: $selectedSpeechVoiceIdentifier) {
                Text("System Default")
                    .tag(Optional<String>.none)

                ForEach(speechViewModel.speechVoices(for: selectedSpeechLanguageCode), id: \.identifier) { voice in
                    Text(voiceRowLabel(for: voice))
                        .tag(Optional(voice.identifier))
                }
            }
        } header: {
            Text("Voice")
        } footer: {
            Text("Voice selections apply to speech output and help align voice mode responses with your preferred language.")
        }
    }

    private var generationSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Temperature")
                            .font(.headline)
                        Text("Lower values stay focused. Higher values explore more varied phrasing.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(samplingDraft.temperature.formatted(.number.precision(.fractionLength(2))))
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                Slider(value: $samplingDraft.temperature, in: 0...2, step: 0.05)

                HStack {
                    Text("Focused")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("Creative")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)

            Stepper(value: $samplingDraft.maxTokens, in: 128...8192, step: 128) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Maximum Response Length")
                    Text("Cap long replies to control latency and memory use.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            LabeledContent("Current Limit") {
                Text("\(Int(samplingDraft.maxTokens)) tokens")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }

            DisclosureGroup("Advanced Sampling", isExpanded: $isAdvancedSamplingExpanded) {
                VStack(spacing: 18) {
                    advancedSlider(
                        title: AppStrings.topK,
                        description: "Limit how many next-token candidates are considered.",
                        value: $samplingDraft.topK,
                        range: 1...100,
                        step: 1,
                        fractionDigits: 0
                    )

                    advancedSlider(
                        title: AppStrings.topP,
                        description: "Keep only the most probable candidates until this cumulative threshold is reached.",
                        value: $samplingDraft.topP,
                        range: 0.2...1,
                        step: 0.05,
                        fractionDigits: 2
                    )

                    advancedSlider(
                        title: AppStrings.repetitionPenalty,
                        description: "Reduce repeated phrases across longer answers.",
                        value: $samplingDraft.repetitionPenalty,
                        range: 1...2,
                        step: 0.05,
                        fractionDigits: 2
                    )

                    Button("Reset Generation Controls") {
                        samplingDraft.resetToDefaults()
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.top, 12)
            }
        } header: {
            Text("Generation")
        } footer: {
            Text("Most people only need temperature and maximum response length. Open advanced sampling only when you are tuning model behavior intentionally.")
        }
    }

    private var modelsSection: some View {
        Section {
            NavigationLink(value: SettingsDestination.models) {
                navigationRow(
                    title: "Manage Models",
                    subtitle: "Switch chat, draft, and embedding models. Download or remove local models.",
                    systemImage: "square.stack.3d.up",
                    tint: .indigo,
                    value: chatModelSummary
                )
            }

            LabeledContent("Draft Model") {
                Text(draftModelSummary)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
            }

            LabeledContent("Embedding Model") {
                Text(embeddingModelSummary)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
            }

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
            Text("Models & Downloads")
        } footer: {
            Text("Use a Hugging Face token only for gated model downloads. It stays in the keychain, not in app defaults.")
        }
    }

    private var privacySection: some View {
        Section {
            NavigationLink(value: SettingsDestination.weather) {
                navigationRow(
                    title: "Location & Weather",
                    subtitle: "Check weather access and verify that location-based answers have the context they need.",
                    systemImage: "cloud.sun",
                    tint: .blue
                )
            }

            Button {
                openAppSettings()
            } label: {
                actionRow(
                    title: "Open App Settings",
                    subtitle: "Manage camera, microphone, speech, notifications, contacts, calendar, and location permissions in iOS Settings.",
                    systemImage: "gearshape",
                    tint: .secondary
                )
            }
            .buttonStyle(.plain)
        } header: {
            Text("Privacy & Permissions")
        }
    }

    private var supportSection: some View {
        Section {
            NavigationLink(value: SettingsDestination.diagnostics) {
                navigationRow(
                    title: "Run Diagnostics",
                    subtitle: "Test storage, model loading, memory, speech, and local services when something feels off.",
                    systemImage: "stethoscope",
                    tint: .green
                )
            }

            Button(role: .destructive) {
                showsResetConfirmation = true
            } label: {
                actionRow(
                    title: "Reset Assistant Settings",
                    subtitle: "Restore the app’s assistant behavior and voice preferences to their defaults without deleting models or conversations.",
                    systemImage: "arrow.counterclockwise",
                    tint: .red
                )
            }
        } header: {
            Text("Support")
        }
    }

    @ViewBuilder
    private func destinationView(for destination: SettingsDestination) -> some View {
        switch destination {
        case .personas:
            SystemPromptTemplatesView(chatViewModel: chatViewModel)
        case .customPrompt:
            CustomPromptEditor(chatViewModel: chatViewModel)
        case .models:
            if let modelManagerViewModel {
                ModelManagerView(viewModel: modelManagerViewModel)
            } else {
                ContentUnavailableView("Models Unavailable", systemImage: "square.stack.3d.up")
            }
        case .weather:
            WeatherView()
        case .diagnostics:
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
        }
    }

    private func navigationRow(
        title: String,
        subtitle: String,
        systemImage: String,
        tint: Color,
        value: String? = nil
    ) -> some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .foregroundStyle(tint)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .foregroundStyle(.primary)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            if let value {
                Text(value)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }

    private func actionRow(
        title: String,
        subtitle: String,
        systemImage: String,
        tint: Color
    ) -> some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .foregroundStyle(tint)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 4)
    }

    private func advancedSlider(
        title: String,
        description: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        fractionDigits: Int
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
                Text(value.wrappedValue.formatted(.number.precision(.fractionLength(fractionDigits))))
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Slider(value: value, in: range, step: step)
        }
    }

    private func loadSettings() {
        samplingDraft = SamplingDraft(config: chatViewModel.samplingConfig)
        hfToken = secureStore.getString(hfTokenKey) ?? ""
        selectedSpeechLanguageCode = speechViewModel.selectedSpeechLanguageCode
        selectedSpeechVoiceIdentifier = speechViewModel.selectedSpeechVoiceIdentifier

        if modelManagerViewModel == nil, let modelLoader {
            modelManagerViewModel = ModelManagerViewModel(modelLoader: modelLoader)
        }

        didLoadSettings = true
    }

    private func applySamplingChanges() {
        chatViewModel.samplingConfig = samplingDraft.makeConfig()
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

    private func resetAssistantSettings() {
        didLoadSettings = false
        samplingDraft.resetToDefaults()
        chatViewModel.samplingConfig = samplingDraft.makeConfig()
        chatViewModel.systemPrompt = SystemPromptTemplate.builtIn[0].prompt
        chatViewModel.toolsEnabled = true
        speechViewModel.isAutoListenEnabled = true
        speechViewModel.updateSpeechLanguage(code: nil)
        speechViewModel.updateSpeechVoice(identifier: nil)
        selectedSpeechLanguageCode = speechViewModel.selectedSpeechLanguageCode
        selectedSpeechVoiceIdentifier = speechViewModel.selectedSpeechVoiceIdentifier
        chatViewModel.saveSettings()
        didLoadSettings = true
    }

    private func openAppSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    private var isUsingDefaultPrompt: Bool {
        chatViewModel.systemPrompt == SystemPromptTemplate.builtIn[0].prompt
    }

    private var activePersona: SystemPromptTemplate? {
        SystemPromptTemplate.builtIn.first { $0.prompt == chatViewModel.systemPrompt }
    }

    private var activePersonaName: String {
        activePersona?.name ?? "Custom"
    }

    private var activePersonaDescription: String {
        activePersona?.description ?? "A custom prompt is shaping every response."
    }

    private var activePersonaIcon: String {
        activePersona?.icon ?? "slider.horizontal.below.rectangle"
    }

    private var activePersonaTint: Color {
        color(for: activePersona?.accentColorName)
    }

    private var customPromptSummary: String {
        let trimmedPrompt = chatViewModel.systemPrompt
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedPrompt.isEmpty else {
            return "No custom instructions yet."
        }

        return String(trimmedPrompt.prefix(90)) + (trimmedPrompt.count > 90 ? "…" : "")
    }

    private var chatModelSummary: String {
        guard let modelLoader else { return "None" }
        return modelSummary(modelLoader.activeModel) ?? "None"
    }

    private var draftModelSummary: String {
        guard let modelLoader else { return "Off" }
        return modelSummary(modelLoader.activeDraftModel) ?? "Off"
    }

    private var embeddingModelSummary: String {
        guard let modelLoader else { return "None" }
        return modelSummary(modelLoader.activeEmbeddingModel) ?? "None"
    }

    private func modelSummary(_ manifest: ModelManifest?) -> String? {
        guard let manifest else { return nil }
        return "\(manifest.name) \(manifest.variant)"
    }

    private func color(for name: String?) -> Color {
        switch name {
        case "green":
            return .green
        case "purple":
            return .purple
        case "orange":
            return .orange
        case "teal":
            return .teal
        case "red":
            return .red
        default:
            return .blue
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

nonisolated enum SettingsDestination: Hashable, Sendable {
    case personas
    case customPrompt
    case models
    case weather
    case diagnostics
}

nonisolated struct SamplingDraft: Equatable, Sendable {
    var temperature: Double
    var topK: Double
    var topP: Double
    var repetitionPenalty: Double
    var maxTokens: Double

    static let defaults: SamplingDraft = .init(config: SamplingConfig())

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

    mutating func resetToDefaults() {
        self = Self.defaults
    }
}

private enum TokenSaveState: Equatable {
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
            return .blue
        case .saved:
            return .green
        case .failed:
            return .red
        }
    }
}
