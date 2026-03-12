import SwiftUI

struct ContentView: View {
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let keyValueStore: KeyValueStore
    let secureStore: SecureStore
    let fileSystem: FileSystemService
    let database: DatabaseService

    @State private var selectedTab: AppTab = .chat
    @State private var inferenceEngine: InferenceEngine?
    @State private var chatViewModel: ChatViewModel?
    @State private var modelManagerViewModel: ModelManagerViewModel?

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Chat", systemImage: "bubble.left.and.text.bubble.right", value: .chat) {
                if let chatVM = chatViewModel {
                    ChatView(viewModel: chatVM)
                } else {
                    ProgressView()
                }
            }

            Tab("Models", systemImage: "square.stack.3d.up", value: .models) {
                NavigationStack {
                    if let modelVM = modelManagerViewModel {
                        ModelManagerView(viewModel: modelVM)
                    } else {
                        ProgressView()
                    }
                }
            }

            Tab("Metrics", systemImage: "chart.xyaxis.line", value: .metrics) {
                NavigationStack {
                    MetricsDashboardView(
                        metricsLogger: metricsLogger,
                        thermalGovernor: thermalGovernor,
                        inferenceEngine: inferenceEngine ?? InferenceEngine(metricsLogger: metricsLogger, thermalGovernor: thermalGovernor)
                    )
                }
            }

            Tab("Settings", systemImage: "gearshape", value: .settings) {
                NavigationStack {
                    if let chatVM = chatViewModel {
                        SettingsView(chatViewModel: chatVM, thermalGovernor: thermalGovernor)
                    } else {
                        ProgressView()
                    }
                }
            }
        }
        .onAppear {
            setupViewModels()
        }
        .onChange(of: modelLoader.activeModelID) { _, _ in
            chatViewModel?.syncEngineFormat()
        }
    }

    private func setupViewModels() {
        guard inferenceEngine == nil else { return }

        let engine = InferenceEngine(metricsLogger: metricsLogger, thermalGovernor: thermalGovernor)
        inferenceEngine = engine

        chatViewModel = ChatViewModel(
            inferenceEngine: engine,
            metricsLogger: metricsLogger,
            thermalGovernor: thermalGovernor,
            modelLoader: modelLoader,
            keyValueStore: keyValueStore,
            database: database
        )

        modelManagerViewModel = ModelManagerViewModel(modelLoader: modelLoader)
    }
}

enum AppTab: String {
    case chat
    case models
    case metrics
    case settings
}
