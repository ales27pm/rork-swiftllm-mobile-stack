import SwiftUI

struct ContentView: View {
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let keyValueStore: KeyValueStore
    let secureStore: SecureStore
    let fileSystem: FileSystemService
    let database: DatabaseService

    @State private var selectedTab: AppTab = .agent
    @State private var inferenceEngine: InferenceEngine?
    @State private var chatViewModel: ChatViewModel?
    @State private var modelManagerViewModel: ModelManagerViewModel?
    @State private var historyViewModel: HistoryViewModel?
    @State private var memoryViewModel: MemoryViewModel?
    @State private var conversationService: ConversationService?
    @State private var memoryService: MemoryService?
    @State private var toolExecutor = ToolExecutor()
    @State private var speechViewModel = SpeechViewModel()
    @State private var assistantAgent: AssistantAgent?

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Nexus", systemImage: "sparkles", value: .agent) {
                if let chatVM = chatViewModel, let agent = assistantAgent {
                    AgentHubView(
                        chatViewModel: chatVM,
                        speechViewModel: speechViewModel,
                        agent: agent,
                        historyViewModel: historyViewModel,
                        memoryViewModel: memoryViewModel,
                        metricsLogger: metricsLogger,
                        thermalGovernor: thermalGovernor,
                        inferenceEngine: inferenceEngine,
                        onLoadConversation: { conversationId in
                            chatVM.loadConversation(conversationId)
                        }
                    )
                } else {
                    ProgressView("Initializing Nexus...")
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

            Tab("Settings", systemImage: "gearshape", value: .settings) {
                NavigationStack {
                    if let chatVM = chatViewModel {
                        SettingsView(chatViewModel: chatVM, speechViewModel: speechViewModel, thermalGovernor: thermalGovernor)
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

        let chatVM = ChatViewModel(
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
        chatViewModel = chatVM

        let agent = AssistantAgent(
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
        assistantAgent = agent

        modelManagerViewModel = ModelManagerViewModel(modelLoader: modelLoader)
        historyViewModel = HistoryViewModel(conversationService: convService)
        memoryViewModel = MemoryViewModel(memoryService: memService)

        speechViewModel.attach(to: chatVM)
    }
}

enum AppTab: String {
    case agent
    case models
    case settings
}
