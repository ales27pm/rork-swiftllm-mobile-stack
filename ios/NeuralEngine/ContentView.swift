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

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Chat", systemImage: "sparkles", value: .chat) {
                if let chatVM = chatViewModel {
                    ChatView(viewModel: chatVM)
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
            memoryService: memService
        )

        modelManagerViewModel = ModelManagerViewModel(modelLoader: modelLoader)
        historyViewModel = HistoryViewModel(conversationService: convService)
        memoryViewModel = MemoryViewModel(memoryService: memService)
    }
}

enum AppTab: String {
    case chat
    case history
    case memory
    case models
    case settings
}
