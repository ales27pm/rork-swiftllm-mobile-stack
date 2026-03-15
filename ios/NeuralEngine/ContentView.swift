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
    @State private var historyViewModel: HistoryViewModel?
    @State private var memoryViewModel: MemoryViewModel?
    @State private var conversationService: ConversationService?
    @State private var memoryService: MemoryService?
    @State private var toolExecutor = ToolExecutor()
    @State private var speechViewModel = SpeechViewModel()

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Chat", systemImage: "sparkles", value: .chat) {
                if let chatVM = chatViewModel {
                    ChatView(viewModel: chatVM, speechViewModel: speechViewModel)
                } else {
                    ProgressView()
                }
            }

            Tab("History", systemImage: "clock", value: .history) {
                if let historyVM = historyViewModel {
                    HistoryView(viewModel: historyVM) { conversationId in
                        chatViewModel?.loadConversation(conversationId)
                        selectedTab = .chat
                    }
                } else {
                    ProgressView()
                }
            }

            Tab("Memory", systemImage: "brain", value: .memory) {
                if let memoryVM = memoryViewModel {
                    MemoryView(viewModel: memoryVM)
                } else {
                    ProgressView()
                }
            }

            Tab("Browse", systemImage: "globe", value: .browse) {
                WebSearchView()
            }

            Tab("Map", systemImage: "map", value: .map) {
                MapView()
            }

            Tab("Scan", systemImage: "doc.text.viewfinder", value: .scan) {
                DocumentAnalysisView()
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

            Tab("Metrics", systemImage: "gauge.with.dots.needle.67percent", value: .metrics) {
                NavigationStack {
                    if let engine = inferenceEngine, let chatVM = chatViewModel {
                        MetricsDashboardView(
                            metricsLogger: metricsLogger,
                            thermalGovernor: thermalGovernor,
                            inferenceEngine: engine,
                            chatViewModel: chatVM
                        )
                    } else {
                        ProgressView()
                    }
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
        .sheet(isPresented: $toolExecutor.showInAppBrowser) {
            if let url = toolExecutor.browserURL {
                InAppBrowserView(url: url, title: toolExecutor.browserTitle)
            }
        }
        .onChange(of: modelLoader.activeModelID) { _, _ in
            chatViewModel?.syncEngineFormat()
        }
    }

    private func setupViewModels() {
        guard inferenceEngine == nil else { return }

        thermalGovernor.metricsLogger = metricsLogger

        let engine = InferenceEngine(metricsLogger: metricsLogger, thermalGovernor: thermalGovernor)
        inferenceEngine = engine

        let convService = ConversationService(database: database, keyValueStore: keyValueStore)
        conversationService = convService

        let memService = MemoryService(database: database)
        memoryService = memService

        chatViewModel = ChatViewModel(
            inferenceEngine: engine,
            metricsLogger: metricsLogger,
            thermalGovernor: thermalGovernor,
            modelLoader: modelLoader,
            keyValueStore: keyValueStore,
            database: database,
            conversationService: convService,
            memoryService: memService,
            toolExecutor: toolExecutor
        )

        modelManagerViewModel = ModelManagerViewModel(modelLoader: modelLoader)
        historyViewModel = HistoryViewModel(conversationService: convService)
        memoryViewModel = MemoryViewModel(memoryService: memService)

        if let chatVM = chatViewModel {
            speechViewModel.attach(to: chatVM)
        }
    }
}

enum AppTab: String {
    case chat
    case history
    case memory
    case browse
    case map
    case scan
    case models
    case metrics
    case settings
}
