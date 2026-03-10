import SwiftUI

struct SettingsView: View {
    @Bindable var chatViewModel: ChatViewModel
    let thermalGovernor: ThermalGovernor

    @State private var temperature: Double = 0.8
    @State private var topK: Double = 40
    @State private var topP: Double = 0.95
    @State private var repetitionPenalty: Double = 1.1
    @State private var maxTokens: Double = 2048

    var body: some View {
        Form {
            samplingSection
            systemPromptSection
            runtimeSection
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

    private var aboutSection: some View {
        Section {
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
