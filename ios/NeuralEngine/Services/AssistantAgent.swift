import Foundation

@Observable
@MainActor
class AssistantAgent {
    let inferenceEngine: InferenceEngine
    let metricsLogger: MetricsLogger
    let thermalGovernor: ThermalGovernor
    let modelLoader: ModelLoaderService
    let keyValueStore: KeyValueStore
    let database: DatabaseService
    let conversationService: ConversationService
    let memoryService: MemoryService
    let toolExecutor: ToolExecutor

    var activeCapability: AgentCapability?
    var capabilityResults: [CapabilityResult] = []

    init(
        inferenceEngine: InferenceEngine,
        metricsLogger: MetricsLogger,
        thermalGovernor: ThermalGovernor,
        modelLoader: ModelLoaderService,
        keyValueStore: KeyValueStore,
        database: DatabaseService,
        conversationService: ConversationService,
        memoryService: MemoryService,
        toolExecutor: ToolExecutor
    ) {
        self.inferenceEngine = inferenceEngine
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.modelLoader = modelLoader
        self.keyValueStore = keyValueStore
        self.database = database
        self.conversationService = conversationService
        self.memoryService = memoryService
        self.toolExecutor = toolExecutor
    }

    func activateCapability(_ capability: AgentCapability) {
        activeCapability = capability
    }

    func dismissCapability() {
        activeCapability = nil
    }

    func addCapabilityResult(_ result: CapabilityResult) {
        capabilityResults.append(result)
        if capabilityResults.count > 20 {
            capabilityResults.removeFirst()
        }
    }

    func clearResults() {
        capabilityResults.removeAll()
    }

    var memoryCount: Int {
        memoryService.memories.count
    }

    var conversationCount: Int {
        conversationService.conversations.count
    }

    var hasActiveModel: Bool {
        inferenceEngine.hasModel
    }

    var activeModelName: String {
        modelLoader.activeModel?.name ?? "No Model"
    }

    var systemHealthStatus: SystemHealthStatus {
        let latency = metricsLogger.currentMetrics.avgTokenLatencyMS
        let thermal = thermalGovernor.thermalLevel
        if thermal == .critical || latency > 100 {
            return .critical
        }
        if thermal == .serious || latency > 50 {
            return .degraded
        }
        return .optimal
    }
}

enum AgentCapability: String, CaseIterable, Identifiable {
    case history = "History"
    case memory = "Memory"
    case browse = "Browse"
    case map = "Map"
    case scan = "Scan"
    case models = "Models"
    case metrics = "Metrics"
    case reasoning = "Reasoning"
    case personas = "Personas"
    case context = "Context"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .history: return "clock.arrow.circlepath"
        case .memory: return "brain"
        case .browse: return "globe"
        case .map: return "map"
        case .scan: return "doc.text.viewfinder"
        case .models: return "square.stack.3d.up"
        case .metrics: return "gauge.with.dots.needle.67percent"
        case .reasoning: return "arrow.triangle.branch"
        case .personas: return "theatermasks.fill"
        case .context: return "circle.dashed"
        }
    }

    var tint: String {
        switch self {
        case .history: return "blue"
        case .memory: return "purple"
        case .browse: return "cyan"
        case .map: return "green"
        case .scan: return "orange"
        case .models: return "indigo"
        case .metrics: return "teal"
        case .reasoning: return "pink"
        case .personas: return "mint"
        case .context: return "yellow"
        }
    }
}

nonisolated struct CapabilityResult: Identifiable, Sendable {
    let id: UUID
    let capability: String
    let summary: String
    let detail: String
    let icon: String
    let timestamp: Date

    init(id: UUID = UUID(), capability: String, summary: String, detail: String = "", icon: String = "sparkles", timestamp: Date = Date()) {
        self.id = id
        self.capability = capability
        self.summary = summary
        self.detail = detail
        self.icon = icon
        self.timestamp = timestamp
    }
}

nonisolated enum SystemHealthStatus: String, Sendable {
    case optimal = "Optimal"
    case degraded = "Degraded"
    case critical = "Critical"
}
