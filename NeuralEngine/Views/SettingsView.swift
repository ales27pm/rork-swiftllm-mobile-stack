import SwiftUI
import AVFoundation

struct SettingsView: View {
    @Bindable var chatViewModel: ChatViewModel
    @Bindable var speechViewModel: SpeechViewModel
    let thermalGovernor: ThermalGovernor

    @State private var temperature: Double = 0.8
    @State private var topK: Double = 40
    @State private var topP: Double = 0.95
    @State private var repetitionPenalty: Double = 1.1
    @State private var maxTokens: Double = 2048
    @State private var hfToken: String = ""
    @State private var isTokenVisible: Bool = false
    @State private var tokenSaved: Bool = false
    @State private var selectedSpeechLanguageCode: String?
    @State private var selectedSpeechVoiceIdentifier: String?

    private let secureStore = SecureStore()
    private let hfTokenKey = "hf_api_token"

    var body: some View {
        Form {
            hfTokenSection
            toolsSection
            samplingSection
            systemPromptSection
            runtimeSection
            speechSection
            aboutSection
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.large)
        .onAppear { loadConfig() }
    }

    private func loadConfig() {
        temperature = Double(chatViewModel.samplingConfig.temperature)
        topK = Double(chatViewModel.samplingConfig.topK)
        topP = Double(chatViewModel.samplingConfig.topP)
        repetitionPenalty = Double(chatViewModel.samplingConfig.repetitionPenalty)
        maxTokens = Double(chatViewModel.samplingConfig.maxTokens)
        hfToken = secureStore.getString(hfTokenKey) ?? ""
        selectedSpeechLanguageCode = speechViewModel.selectedSpeechLanguageCode
        selectedSpeechVoiceIdentifier = speechViewModel.selectedSpeechVoiceIdentifier
    }

    private func syncConfig() {
        chatViewModel.samplingConfig.temperature = Float(temperature)
        chatViewModel.samplingConfig.topK = Int(topK)
        chatViewModel.samplingConfig.topP = Float(topP)
        chatViewModel.samplingConfig.repetitionPenalty = Float(repetitionPenalty)
        chatViewModel.samplingConfig.maxTokens = Int(maxTokens)
        chatViewModel.saveSettings()
    }

    private var samplingSection: some View {
        Section {
            sliderRow(title: "Temperature", value: $temperature, range: 0...2, step: 0.05, format: "%.2f")
            sliderRow(title: "Top-K", value: $topK, range: 1...100, step: 1, format: "%.0f")
            sliderRow(title: "Top-P", value: $topP, range: 0...1, step: 0.05, format: "%.2f")
            sliderRow(title: "Repetition Penalty", value: $repetitionPenalty, range: 1...2, step: 0.05, format: "%.2f")
            sliderRow(title: "Max Tokens", value: $maxTokens, range: 128...8192, step: 128, format: "%.0f")
        } header: {
            Label("Sampling", systemImage: "slider.horizontal.3")
        } footer: {
            Text("Controls the randomness and diversity of generated text. Lower temperature = more focused, higher = more creative.")
        }
    }

    private func sliderRow(title: String, value: Binding<Double>, range: ClosedRange<Double>, step: Double, format: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step) { _ in
                syncConfig()
            }
        }
    }

    private var systemPromptSection: some View {
        Section {
            TextEditor(text: $chatViewModel.systemPrompt)
                .font(.subheadline)
                .frame(minHeight: 100)
        } header: {
            Label("System Prompt", systemImage: "text.quote")
        } footer: {
            Text("This prompt is prepended to every conversation. Cached via prompt prefix cache for fast repeated inference.")
        }
    }

    private var runtimeSection: some View {
        Section {
            runtimeRow(label: "Thermal State", icon: "thermometer.medium", value: thermalGovernor.thermalLevel.rawValue)
            runtimeRow(label: "Runtime Mode", icon: thermalGovernor.currentMode.icon, value: thermalGovernor.currentMode.rawValue)
            specRow
            runtimeRow(label: "Max Draft Tokens", icon: "number", value: "\(thermalGovernor.currentMode.maxDraftTokens)")
        } header: {
            Label("Runtime", systemImage: "cpu")
        } footer: {
            Text("Runtime mode adapts automatically based on device thermal state.")
        }
    }

    private func runtimeRow(label: String, icon: String, value: String) -> some View {
        HStack {
            Label(label, systemImage: icon)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
    }

    private var specRow: some View {
        HStack {
            Label("Speculative Decoding", systemImage: "bolt.fill")
            Spacer()
            Text(thermalGovernor.currentMode.speculativeEnabled ? "Enabled" : "Disabled")
                .foregroundStyle(thermalGovernor.currentMode.speculativeEnabled ? .green : .secondary)
        }
    }

    private var hfTokenSection: some View {
        Section {
            HStack(spacing: 12) {
                Image(systemName: "key.fill")
                    .foregroundStyle(.orange)
                    .frame(width: 24)
                if isTokenVisible {
                    TextField("hf_...", text: $hfToken)
                        .font(.subheadline.monospaced())
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } else {
                    SecureField("hf_...", text: $hfToken)
                        .font(.subheadline.monospaced())
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Button {
                    isTokenVisible.toggle()
                } label: {
                    Image(systemName: isTokenVisible ? "eye.slash" : "eye")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            Button {
                if hfToken.isEmpty {
                    secureStore.delete(hfTokenKey)
                } else {
                    _ = secureStore.setString(hfToken, forKey: hfTokenKey)
                }
                tokenSaved = true
                Task {
                    try? await Task.sleep(for: .seconds(2))
                    tokenSaved = false
                }
            } label: {
                HStack {
                    Spacer()
                    Label(tokenSaved ? "Saved" : "Save Token", systemImage: tokenSaved ? "checkmark.circle.fill" : "square.and.arrow.down")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(tokenSaved ? .green : .accentColor)
                    Spacer()
                }
            }
            .disabled(tokenSaved)
        } header: {
            Label("Hugging Face", systemImage: "server.rack")
        } footer: {
            Text("Required to download gated models. Get your token at huggingface.co/settings/tokens.")
        }
    }

    private var toolsSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { chatViewModel.toolsEnabled },
                set: { chatViewModel.toolsEnabled = $0; chatViewModel.saveSettings() }
            )) {
                Label("Device Tools", systemImage: "wrench.and.screwdriver.fill")
            }

            if chatViewModel.toolsEnabled {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(toolCategories.keys.sorted()), id: \.self) { category in
                        HStack(spacing: 6) {
                            Image(systemName: toolCategories[category] ?? "circle")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .frame(width: 20)
                            Text(category)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        } header: {
            Label("Tool Calling", systemImage: "hammer.fill")
        } footer: {
            Text("When enabled, the AI can access device capabilities like location, battery, calendar, contacts, and more through natural conversation.")
        }
    }

    private var toolCategories: [String: String] {
        [
            "Location & Maps": "location.fill",
            "Battery & Device": "battery.100percent",
            "Calendar & Events": "calendar",
            "Contacts": "person.2.fill",
            "Notifications": "bell.fill",
            "Screen & Haptics": "sun.max.fill",
            "Sharing & Messaging": "square.and.arrow.up",
            "Date & Time": "clock.fill"
        ]
    }


    private var speechSection: some View {
        Section {
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
                    HStack {
                        Text(voice.displayName)
                        Spacer()
                        if let badge = qualityLabel(for: voice.quality) {
                            Text(badge)
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                    }
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

    private var aboutSection: some View {
        Section {
            aboutRow(label: "Backends", value: "CoreML + llama.cpp (GGUF)")
            aboutRow(label: "Architecture", value: "Split Prefill/Decode + Paged KV")
            aboutRow(label: "KV Cache", value: "Paged Arena (128 tokens/page)")
            aboutRow(label: "Speculation", value: "Adaptive Draft Length")
            aboutRow(label: "Prefix Cache", value: "Hash-based (8 entries)")
        } header: {
            Label("Engine", systemImage: "gearshape.2")
        }
    }

    private func aboutRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
