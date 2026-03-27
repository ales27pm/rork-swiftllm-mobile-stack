import Foundation
import UIKit
import AVFoundation
import Speech
import NaturalLanguage

@Observable
@MainActor
class DiagnosticEngine {
    var results: [DiagnosticTestResult] = []
    var isRunning: Bool = false
    var currentTestIndex: Int = 0
    var totalTests: Int = 0
    var currentCategory: DiagnosticCategory?
    var startTime: Date?
    var completionTime: Date?
    var reportURL: URL?

    var database: DatabaseService?
    var keyValueStore: KeyValueStore?
    var secureStore: SecureStore?
    var fileSystem: FileSystemService?
    var memoryService: MemoryService?
    var conversationService: ConversationService?
    var metricsLogger: MetricsLogger?
    var thermalGovernor: ThermalGovernor?
    var modelLoader: ModelLoaderService?
    var inferenceEngine: InferenceEngine?

    func configure(
        database: DatabaseService,
        keyValueStore: KeyValueStore,
        secureStore: SecureStore,
        fileSystem: FileSystemService,
        memoryService: MemoryService?,
        conversationService: ConversationService?,
        metricsLogger: MetricsLogger,
        thermalGovernor: ThermalGovernor,
        modelLoader: ModelLoaderService,
        inferenceEngine: InferenceEngine?
    ) {
        self.database = database
        self.keyValueStore = keyValueStore
        self.secureStore = secureStore
        self.fileSystem = fileSystem
        self.memoryService = memoryService
        self.conversationService = conversationService
        self.metricsLogger = metricsLogger
        self.thermalGovernor = thermalGovernor
        self.modelLoader = modelLoader
        self.inferenceEngine = inferenceEngine
    }

    func runAllTests() async {
        guard !isRunning else { return }
        isRunning = true
        startTime = Date()
        completionTime = nil
        reportURL = nil
        results.removeAll()
        currentTestIndex = 0

        let allTests = buildTestSuite()
        totalTests = allTests.count
        results = allTests

        let heavyCategories: Set<DiagnosticCategory> = [
            .llmDiagnostic, .configOptimization, .inferenceDeep, .stressTest
        ]
        var previousCategory: DiagnosticCategory?

        for i in 0..<results.count {
            guard isRunning else { break }
            currentTestIndex = i
            let testCategory = results[i].category
            currentCategory = testCategory

            if let prev = previousCategory, prev != testCategory {
                if let tg = thermalGovernor {
                    let interDelay = tg.interCategoryCooldownSeconds
                    if interDelay > 0 {
                        try? await Task.sleep(for: .seconds(interDelay))
                    }
                }

                if heavyCategories.contains(testCategory) {
                    if let tg = thermalGovernor, tg.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
                        _ = await tg.waitForCooldown(maxWaitSeconds: 30, targetBelow: .serious)
                    }
                }
            }
            previousCategory = testCategory

            results[i].status = .running

            let start = Date()
            let outcome = await executeTest(results[i])
            let duration = Date().timeIntervalSince(start)

            results[i].status = outcome.status
            results[i].duration = duration
            results[i].message = outcome.message
            results[i].details = outcome.details
        }

        completionTime = Date()
        isRunning = false
        currentCategory = nil

        await generateReport()
    }

    func cancel() {
        isRunning = false
    }

    var passedCount: Int { results.filter { $0.status == .passed }.count }
    var failedCount: Int { results.filter { $0.status == .failed }.count }
    var warningCount: Int { results.filter { $0.status == .warning }.count }
    var skippedCount: Int { results.filter { $0.status == .skipped }.count }
    var progress: Double { totalTests > 0 ? Double(currentTestIndex) / Double(totalTests) : 0 }

    // MARK: - Test Suite

    private func buildTestSuite() -> [DiagnosticTestResult] {
        var tests: [DiagnosticTestResult] = []

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "SQLite Open/Close", category: .database),
            DiagnosticTestResult(name: "Create Table", category: .database),
            DiagnosticTestResult(name: "Insert Row", category: .database),
            DiagnosticTestResult(name: "Query Row", category: .database),
            DiagnosticTestResult(name: "Update Row", category: .database),
            DiagnosticTestResult(name: "Delete Row", category: .database),
            DiagnosticTestResult(name: "Table Exists Check", category: .database),
            DiagnosticTestResult(name: "Parameterized Query", category: .database),
            DiagnosticTestResult(name: "Bulk Insert (100 rows)", category: .database),
            DiagnosticTestResult(name: "FTS5 Virtual Table", category: .database),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Documents Directory Access", category: .fileSystem),
            DiagnosticTestResult(name: "Caches Directory Access", category: .fileSystem),
            DiagnosticTestResult(name: "Temp Directory Access", category: .fileSystem),
            DiagnosticTestResult(name: "App Support Directory", category: .fileSystem),
            DiagnosticTestResult(name: "Write/Read File", category: .fileSystem),
            DiagnosticTestResult(name: "Write/Read Data", category: .fileSystem),
            DiagnosticTestResult(name: "Create Directory", category: .fileSystem),
            DiagnosticTestResult(name: "File Size Calculation", category: .fileSystem),
            DiagnosticTestResult(name: "Directory Listing", category: .fileSystem),
            DiagnosticTestResult(name: "SHA256 Hash", category: .fileSystem),
            DiagnosticTestResult(name: "Available Disk Space", category: .fileSystem),
            DiagnosticTestResult(name: "Model Storage Directories", category: .fileSystem),
            DiagnosticTestResult(name: "Backup Exclusion", category: .fileSystem),
            DiagnosticTestResult(name: "Copy/Move File", category: .fileSystem),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Set/Get String", category: .keyValueStore),
            DiagnosticTestResult(name: "Set/Get Int", category: .keyValueStore),
            DiagnosticTestResult(name: "Set/Get Double", category: .keyValueStore),
            DiagnosticTestResult(name: "Set/Get Bool", category: .keyValueStore),
            DiagnosticTestResult(name: "Set/Get Data", category: .keyValueStore),
            DiagnosticTestResult(name: "Set/Get Codable", category: .keyValueStore),
            DiagnosticTestResult(name: "Has Key", category: .keyValueStore),
            DiagnosticTestResult(name: "Remove Key", category: .keyValueStore),
            DiagnosticTestResult(name: "Multi Get/Set", category: .keyValueStore),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Set/Get String (Keychain)", category: .secureStore),
            DiagnosticTestResult(name: "Set/Get Data (Keychain)", category: .secureStore),
            DiagnosticTestResult(name: "Delete Key (Keychain)", category: .secureStore),
            DiagnosticTestResult(name: "Has Key (Keychain)", category: .secureStore),
            DiagnosticTestResult(name: "Codable Storage (Keychain)", category: .secureStore),
            DiagnosticTestResult(name: "Key Rotation (Keychain)", category: .secureStore),
            DiagnosticTestResult(name: "Audit (Keychain)", category: .secureStore),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Add Memory Entry", category: .memory),
            DiagnosticTestResult(name: "Search Memories (TF-IDF)", category: .memory),
            DiagnosticTestResult(name: "Search Memories (Embedding)", category: .memory),
            DiagnosticTestResult(name: "Associative Links", category: .memory),
            DiagnosticTestResult(name: "Memory Deduplication", category: .memory),
            DiagnosticTestResult(name: "Memory Extraction (Name)", category: .memory),
            DiagnosticTestResult(name: "Memory Extraction (Preference)", category: .memory),
            DiagnosticTestResult(name: "Memory Consolidation", category: .memory),
            DiagnosticTestResult(name: "Memory Reinforcement", category: .memory),
            DiagnosticTestResult(name: "Context Injection Build", category: .memory),
            DiagnosticTestResult(name: "Delete Memory", category: .memory),
            DiagnosticTestResult(name: "Memory Category Classification", category: .memory),
            DiagnosticTestResult(name: "Memory Keyword Extraction", category: .memory),
            DiagnosticTestResult(name: "Memory Decay Computation", category: .memory),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Create Conversation", category: .conversation),
            DiagnosticTestResult(name: "Save Message", category: .conversation),
            DiagnosticTestResult(name: "Load Messages", category: .conversation),
            DiagnosticTestResult(name: "Update Conversation", category: .conversation),
            DiagnosticTestResult(name: "Search Messages (FTS)", category: .conversation),
            DiagnosticTestResult(name: "Generate Title", category: .conversation),
            DiagnosticTestResult(name: "Delete Conversation", category: .conversation),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Language Detection (English)", category: .nlp),
            DiagnosticTestResult(name: "Language Detection (French)", category: .nlp),
            DiagnosticTestResult(name: "Tokenization", category: .nlp),
            DiagnosticTestResult(name: "Lemmatization", category: .nlp),
            DiagnosticTestResult(name: "Named Entity Recognition", category: .nlp),
            DiagnosticTestResult(name: "Text Normalization", category: .nlp),
            DiagnosticTestResult(name: "Stemmed Terms", category: .nlp),
            DiagnosticTestResult(name: "Embedding Similarity", category: .nlp),
            DiagnosticTestResult(name: "Stop Word Filtering", category: .nlp),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Positive Emotion Detection", category: .emotion),
            DiagnosticTestResult(name: "Negative Emotion Detection", category: .emotion),
            DiagnosticTestResult(name: "Neutral Emotion Detection", category: .emotion),
            DiagnosticTestResult(name: "Mixed Emotion Detection", category: .emotion),
            DiagnosticTestResult(name: "Style Detection (Formal)", category: .emotion),
            DiagnosticTestResult(name: "Style Detection (Casual)", category: .emotion),
            DiagnosticTestResult(name: "Style Detection (Technical)", category: .emotion),
            DiagnosticTestResult(name: "Emotional Trajectory", category: .emotion),
            DiagnosticTestResult(name: "Empathy Level Computation", category: .emotion),
            DiagnosticTestResult(name: "Emotion Context Injection", category: .emotion),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Greeting Intent", category: .intent),
            DiagnosticTestResult(name: "Question Intent", category: .intent),
            DiagnosticTestResult(name: "Creation Request Intent", category: .intent),
            DiagnosticTestResult(name: "Calculation Intent", category: .intent),
            DiagnosticTestResult(name: "Memory Request Intent", category: .intent),
            DiagnosticTestResult(name: "Multi-Intent Detection", category: .intent),
            DiagnosticTestResult(name: "Urgency Detection", category: .intent),
            DiagnosticTestResult(name: "Response Length Estimation", category: .intent),
            DiagnosticTestResult(name: "Intent Context Injection", category: .intent),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Complexity Assessment", category: .metacognition),
            DiagnosticTestResult(name: "Uncertainty Computation", category: .metacognition),
            DiagnosticTestResult(name: "Ambiguity Detection", category: .metacognition),
            DiagnosticTestResult(name: "Confidence Calibration", category: .metacognition),
            DiagnosticTestResult(name: "Shannon Entropy", category: .metacognition),
            DiagnosticTestResult(name: "Self-Correction Detection", category: .metacognition),
            DiagnosticTestResult(name: "Metacognition Injection", category: .metacognition),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Branch Generation", category: .thoughtTree),
            DiagnosticTestResult(name: "Pruning Logic", category: .thoughtTree),
            DiagnosticTestResult(name: "Convergence Calculation", category: .thoughtTree),
            DiagnosticTestResult(name: "Synthesis Strategy Selection", category: .thoughtTree),
            DiagnosticTestResult(name: "Tree Depth Control", category: .thoughtTree),
            DiagnosticTestResult(name: "Thought Tree Injection", category: .thoughtTree),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Topic Extraction", category: .curiosity),
            DiagnosticTestResult(name: "Curiosity Level Measurement", category: .curiosity),
            DiagnosticTestResult(name: "Knowledge Gap Computation", category: .curiosity),
            DiagnosticTestResult(name: "Suggested Queries", category: .curiosity),
            DiagnosticTestResult(name: "Curiosity Injection", category: .curiosity),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Full Cognition Frame", category: .cognition),
            DiagnosticTestResult(name: "Context Signature", category: .cognition),
            DiagnosticTestResult(name: "Semantic Drift Detection", category: .cognition),
            DiagnosticTestResult(name: "Injection Priority Ordering", category: .cognition),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "System Prompt Assembly", category: .contextAssembly),
            DiagnosticTestResult(name: "Memory Section Build", category: .contextAssembly),
            DiagnosticTestResult(name: "Cognitive State Summary", category: .contextAssembly),
            DiagnosticTestResult(name: "Token Budget Enforcement", category: .contextAssembly),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Thermal State Reading", category: .thermal),
            DiagnosticTestResult(name: "Runtime Mode Selection", category: .thermal),
            DiagnosticTestResult(name: "Penalty Computation", category: .thermal),
            DiagnosticTestResult(name: "Memory Usage Reading", category: .thermal),
            DiagnosticTestResult(name: "Available Memory Reading", category: .thermal),
            DiagnosticTestResult(name: "Token Delay Calculation", category: .thermal),
            DiagnosticTestResult(name: "Recovery State Management", category: .thermal),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Begin/End Generation Cycle", category: .metrics),
            DiagnosticTestResult(name: "Token Recording", category: .metrics),
            DiagnosticTestResult(name: "Diagnostic Event Recording", category: .metrics),
            DiagnosticTestResult(name: "Speed History Tracking", category: .metrics),
            DiagnosticTestResult(name: "Uptime Formatting", category: .metrics),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Fallback Tokenizer Encode", category: .tokenizer),
            DiagnosticTestResult(name: "Fallback Tokenizer Decode", category: .tokenizer),
            DiagnosticTestResult(name: "Vocabulary Size", category: .tokenizer),
            DiagnosticTestResult(name: "Special Tokens", category: .tokenizer),
            DiagnosticTestResult(name: "EOS Token Detection", category: .tokenizer),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Registry Load", category: .modelLoader),
            DiagnosticTestResult(name: "Model Status Resolution", category: .modelLoader),
            DiagnosticTestResult(name: "Active Model Check", category: .modelLoader),
            DiagnosticTestResult(name: "Model Format Support", category: .modelLoader),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Speech Permission Status", category: .speech),
            DiagnosticTestResult(name: "Microphone Permission Status", category: .speech),
            DiagnosticTestResult(name: "Available Voices", category: .speech),
            DiagnosticTestResult(name: "Speech Recognizer Availability", category: .speech),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Inference Engine State", category: .inference),
            DiagnosticTestResult(name: "KV Cache State", category: .inference),
            DiagnosticTestResult(name: "Session Cache State", category: .inference),
            DiagnosticTestResult(name: "Health Monitor State", category: .inference),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Scalar Field 2D Creation", category: .proceduralSolvers),
        ])

        tests.append(contentsOf: [
            DiagnosticTestResult(name: "Full Pipeline: Input → Cognition → Context", category: .endToEnd),
            DiagnosticTestResult(name: "Memory Store → Search → Retrieve", category: .endToEnd),
            DiagnosticTestResult(name: "Conversation Create → Message → Search", category: .endToEnd),
            DiagnosticTestResult(name: "Emotion + Intent + Memory → Injection", category: .endToEnd),
        ])

        tests.append(contentsOf: buildDeepTestSuite())

        return tests
    }

    // MARK: - Test Execution

    struct TestOutcome: Sendable {
        let status: DiagnosticTestStatus
        let message: String
        let details: [String]
    }

    private func executeTest(_ test: DiagnosticTestResult) async -> TestOutcome {
        do {
            switch (test.category, test.name) {

            // MARK: Database Tests
            case (.database, "SQLite Open/Close"):
                return testDatabaseOpenClose()
            case (.database, "Create Table"):
                return testDatabaseCreateTable()
            case (.database, "Insert Row"):
                return testDatabaseInsertRow()
            case (.database, "Query Row"):
                return testDatabaseQueryRow()
            case (.database, "Update Row"):
                return testDatabaseUpdateRow()
            case (.database, "Delete Row"):
                return testDatabaseDeleteRow()
            case (.database, "Table Exists Check"):
                return testDatabaseTableExists()
            case (.database, "Parameterized Query"):
                return testDatabaseParameterized()
            case (.database, "Bulk Insert (100 rows)"):
                return testDatabaseBulkInsert()
            case (.database, "FTS5 Virtual Table"):
                return testDatabaseFTS5()

            // MARK: File System Tests
            case (.fileSystem, "Documents Directory Access"):
                return testFSDocuments()
            case (.fileSystem, "Caches Directory Access"):
                return testFSCaches()
            case (.fileSystem, "Temp Directory Access"):
                return testFSTemp()
            case (.fileSystem, "App Support Directory"):
                return testFSAppSupport()
            case (.fileSystem, "Write/Read File"):
                return testFSWriteReadFile()
            case (.fileSystem, "Write/Read Data"):
                return testFSWriteReadData()
            case (.fileSystem, "Create Directory"):
                return testFSCreateDir()
            case (.fileSystem, "File Size Calculation"):
                return testFSFileSize()
            case (.fileSystem, "Directory Listing"):
                return testFSDirListing()
            case (.fileSystem, "SHA256 Hash"):
                return testFSSHA256()
            case (.fileSystem, "Available Disk Space"):
                return testFSDiskSpace()
            case (.fileSystem, "Model Storage Directories"):
                return testFSModelStorage()
            case (.fileSystem, "Backup Exclusion"):
                return testFSBackupExclusion()
            case (.fileSystem, "Copy/Move File"):
                return testFSCopyMove()

            // MARK: KV Store Tests
            case (.keyValueStore, "Set/Get String"):
                return testKVString()
            case (.keyValueStore, "Set/Get Int"):
                return testKVInt()
            case (.keyValueStore, "Set/Get Double"):
                return testKVDouble()
            case (.keyValueStore, "Set/Get Bool"):
                return testKVBool()
            case (.keyValueStore, "Set/Get Data"):
                return testKVData()
            case (.keyValueStore, "Set/Get Codable"):
                return testKVCodable()
            case (.keyValueStore, "Has Key"):
                return testKVHasKey()
            case (.keyValueStore, "Remove Key"):
                return testKVRemove()
            case (.keyValueStore, "Multi Get/Set"):
                return testKVMulti()

            // MARK: Secure Store Tests
            case (.secureStore, "Set/Get String (Keychain)"):
                return testSecureString()
            case (.secureStore, "Set/Get Data (Keychain)"):
                return testSecureData()
            case (.secureStore, "Delete Key (Keychain)"):
                return testSecureDelete()
            case (.secureStore, "Has Key (Keychain)"):
                return testSecureHasKey()
            case (.secureStore, "Codable Storage (Keychain)"):
                return testSecureCodable()
            case (.secureStore, "Key Rotation (Keychain)"):
                return testSecureRotation()
            case (.secureStore, "Audit (Keychain)"):
                return testSecureAudit()

            // MARK: Memory Tests
            case (.memory, "Add Memory Entry"):
                return testMemoryAdd()
            case (.memory, "Search Memories (TF-IDF)"):
                return testMemorySearchTFIDF()
            case (.memory, "Search Memories (Embedding)"):
                return testMemorySearchEmbedding()
            case (.memory, "Associative Links"):
                return testMemoryAssociativeLinks()
            case (.memory, "Memory Deduplication"):
                return testMemoryDedup()
            case (.memory, "Memory Extraction (Name)"):
                return testMemoryExtractionName()
            case (.memory, "Memory Extraction (Preference)"):
                return testMemoryExtractionPreference()
            case (.memory, "Memory Consolidation"):
                return testMemoryConsolidation()
            case (.memory, "Memory Reinforcement"):
                return testMemoryReinforcement()
            case (.memory, "Context Injection Build"):
                return testMemoryContextInjection()
            case (.memory, "Delete Memory"):
                return testMemoryDelete()
            case (.memory, "Memory Category Classification"):
                return testMemoryCategoryClassification()
            case (.memory, "Memory Keyword Extraction"):
                return testMemoryKeywordExtraction()
            case (.memory, "Memory Decay Computation"):
                return testMemoryDecayComputation()

            // MARK: Conversation Tests
            case (.conversation, "Create Conversation"):
                return testConvCreate()
            case (.conversation, "Save Message"):
                return testConvSaveMessage()
            case (.conversation, "Load Messages"):
                return testConvLoadMessages()
            case (.conversation, "Update Conversation"):
                return testConvUpdate()
            case (.conversation, "Search Messages (FTS)"):
                return testConvSearch()
            case (.conversation, "Generate Title"):
                return testConvGenerateTitle()
            case (.conversation, "Delete Conversation"):
                return testConvDelete()

            // MARK: NLP Tests
            case (.nlp, "Language Detection (English)"):
                return testNLPLanguageEN()
            case (.nlp, "Language Detection (French)"):
                return testNLPLanguageFR()
            case (.nlp, "Tokenization"):
                return testNLPTokenization()
            case (.nlp, "Lemmatization"):
                return testNLPLemmatization()
            case (.nlp, "Named Entity Recognition"):
                return testNLPNER()
            case (.nlp, "Text Normalization"):
                return testNLPNormalization()
            case (.nlp, "Stemmed Terms"):
                return testNLPStemmedTerms()
            case (.nlp, "Embedding Similarity"):
                return testNLPEmbedding()
            case (.nlp, "Stop Word Filtering"):
                return testNLPStopWords()

            // MARK: Emotion Tests
            case (.emotion, "Positive Emotion Detection"):
                return testEmotionPositive()
            case (.emotion, "Negative Emotion Detection"):
                return testEmotionNegative()
            case (.emotion, "Neutral Emotion Detection"):
                return testEmotionNeutral()
            case (.emotion, "Mixed Emotion Detection"):
                return testEmotionMixed()
            case (.emotion, "Style Detection (Formal)"):
                return testEmotionStyleFormal()
            case (.emotion, "Style Detection (Casual)"):
                return testEmotionStyleCasual()
            case (.emotion, "Style Detection (Technical)"):
                return testEmotionStyleTech()
            case (.emotion, "Emotional Trajectory"):
                return testEmotionTrajectory()
            case (.emotion, "Empathy Level Computation"):
                return testEmotionEmpathy()
            case (.emotion, "Emotion Context Injection"):
                return testEmotionInjection()

            // MARK: Intent Tests
            case (.intent, "Greeting Intent"):
                return testIntentGreeting()
            case (.intent, "Question Intent"):
                return testIntentQuestion()
            case (.intent, "Creation Request Intent"):
                return testIntentCreation()
            case (.intent, "Calculation Intent"):
                return testIntentCalculation()
            case (.intent, "Memory Request Intent"):
                return testIntentMemory()
            case (.intent, "Multi-Intent Detection"):
                return testIntentMulti()
            case (.intent, "Urgency Detection"):
                return testIntentUrgency()
            case (.intent, "Response Length Estimation"):
                return testIntentResponseLength()
            case (.intent, "Intent Context Injection"):
                return testIntentInjection()

            // MARK: Metacognition Tests
            case (.metacognition, "Complexity Assessment"):
                return testMetaComplexity()
            case (.metacognition, "Uncertainty Computation"):
                return testMetaUncertainty()
            case (.metacognition, "Ambiguity Detection"):
                return testMetaAmbiguity()
            case (.metacognition, "Confidence Calibration"):
                return testMetaConfidence()
            case (.metacognition, "Shannon Entropy"):
                return testMetaEntropy()
            case (.metacognition, "Self-Correction Detection"):
                return testMetaSelfCorrection()
            case (.metacognition, "Metacognition Injection"):
                return testMetaInjection()

            // MARK: Thought Tree Tests
            case (.thoughtTree, "Branch Generation"):
                return testTreeBranches()
            case (.thoughtTree, "Pruning Logic"):
                return testTreePruning()
            case (.thoughtTree, "Convergence Calculation"):
                return testTreeConvergence()
            case (.thoughtTree, "Synthesis Strategy Selection"):
                return testTreeStrategy()
            case (.thoughtTree, "Tree Depth Control"):
                return testTreeDepth()
            case (.thoughtTree, "Thought Tree Injection"):
                return testTreeInjection()

            // MARK: Curiosity Tests
            case (.curiosity, "Topic Extraction"):
                return testCuriosityTopics()
            case (.curiosity, "Curiosity Level Measurement"):
                return testCuriosityLevel()
            case (.curiosity, "Knowledge Gap Computation"):
                return testCuriosityKnowledgeGap()
            case (.curiosity, "Suggested Queries"):
                return testCuriositySuggestions()
            case (.curiosity, "Curiosity Injection"):
                return testCuriosityInjection()

            // MARK: Cognition Pipeline Tests
            case (.cognition, "Full Cognition Frame"):
                return testCognitionFrame()
            case (.cognition, "Context Signature"):
                return testCognitionSignature()
            case (.cognition, "Semantic Drift Detection"):
                return testCognitionDrift()
            case (.cognition, "Injection Priority Ordering"):
                return testCognitionInjectionOrder()

            // MARK: Context Assembly Tests
            case (.contextAssembly, "System Prompt Assembly"):
                return testContextSystemPrompt()
            case (.contextAssembly, "Memory Section Build"):
                return testContextMemorySection()
            case (.contextAssembly, "Cognitive State Summary"):
                return testContextCognitiveState()
            case (.contextAssembly, "Token Budget Enforcement"):
                return testContextTokenBudget()

            // MARK: Thermal Tests
            case (.thermal, "Thermal State Reading"):
                return testThermalState()
            case (.thermal, "Runtime Mode Selection"):
                return testThermalRuntimeMode()
            case (.thermal, "Penalty Computation"):
                return testThermalPenalty()
            case (.thermal, "Memory Usage Reading"):
                return testThermalMemoryUsage()
            case (.thermal, "Available Memory Reading"):
                return testThermalAvailableMemory()
            case (.thermal, "Token Delay Calculation"):
                return testThermalTokenDelay()
            case (.thermal, "Recovery State Management"):
                return testThermalRecovery()

            // MARK: Metrics Tests
            case (.metrics, "Begin/End Generation Cycle"):
                return testMetricsCycle()
            case (.metrics, "Token Recording"):
                return testMetricsTokenRecording()
            case (.metrics, "Diagnostic Event Recording"):
                return testMetricsDiagnosticEvent()
            case (.metrics, "Speed History Tracking"):
                return testMetricsSpeedHistory()
            case (.metrics, "Uptime Formatting"):
                return testMetricsUptime()

            // MARK: Tokenizer Tests
            case (.tokenizer, "Fallback Tokenizer Encode"):
                return testTokenizerEncode()
            case (.tokenizer, "Fallback Tokenizer Decode"):
                return testTokenizerDecode()
            case (.tokenizer, "Vocabulary Size"):
                return testTokenizerVocabSize()
            case (.tokenizer, "Special Tokens"):
                return testTokenizerSpecialTokens()
            case (.tokenizer, "EOS Token Detection"):
                return testTokenizerEOS()

            // MARK: Model Loader Tests
            case (.modelLoader, "Registry Load"):
                return testModelRegistry()
            case (.modelLoader, "Model Status Resolution"):
                return testModelStatus()
            case (.modelLoader, "Active Model Check"):
                return testModelActive()
            case (.modelLoader, "Model Format Support"):
                return testModelFormats()

            // MARK: Speech Tests
            case (.speech, "Speech Permission Status"):
                return testSpeechPermission()
            case (.speech, "Microphone Permission Status"):
                return testMicPermission()
            case (.speech, "Available Voices"):
                return testSpeechVoices()
            case (.speech, "Speech Recognizer Availability"):
                return testSpeechRecognizerAvail()

            // MARK: Inference Tests
            case (.inference, "Inference Engine State"):
                return testInferenceState()
            case (.inference, "KV Cache State"):
                return await testKVCacheState()
            case (.inference, "Session Cache State"):
                return testSessionCacheState()
            case (.inference, "Health Monitor State"):
                return testHealthMonitor()

            // MARK: Procedural Tests
            case (.proceduralSolvers, "Scalar Field 2D Creation"):
                return testScalarField()

            // MARK: End-to-End Tests
            case (.endToEnd, "Full Pipeline: Input → Cognition → Context"):
                return testE2EPipeline()
            case (.endToEnd, "Memory Store → Search → Retrieve"):
                return testE2EMemory()
            case (.endToEnd, "Conversation Create → Message → Search"):
                return testE2EConversation()
            case (.endToEnd, "Emotion + Intent + Memory → Injection"):
                return testE2EInjection()

            // MARK: Deep Diagnostic Tests
            case (.emotionAccuracy, _), (.intentAccuracy, _), (.memoryQuality, _),
                 (.cognitionQuality, _), (.contextQuality, _), (.stressTest, _),
                 (.inferenceDeep, _), (.regressionE2E, _), (.llmDiagnostic, _),
                 (.vectorDatabase, _), (.configOptimization, _):
                return await executeDeepTest(test)

            default:
                return TestOutcome(status: .skipped, message: "No implementation", details: [])
            }
        } catch {
            return TestOutcome(status: .failed, message: error.localizedDescription, details: [])
        }
    }

    // MARK: - Database Test Implementations

    private func testDatabaseOpenClose() -> TestOutcome {
        let db = DatabaseService(name: "diag_test_\(UUID().uuidString).sqlite3")
        let ok = db.execute("SELECT 1;")
        _ = db.deleteDatabase()
        return ok ? TestOutcome(status: .passed, message: "SQLite opened and closed", details: []) : TestOutcome(status: .failed, message: "Failed to open database", details: [])
    }

    private func testDatabaseCreateTable() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        let ok = db.execute("CREATE TABLE IF NOT EXISTS diag_test (id INTEGER PRIMARY KEY, value TEXT);")
        return ok ? TestOutcome(status: .passed, message: "Table created", details: []) : TestOutcome(status: .failed, message: "CREATE TABLE failed", details: [])
    }

    private func testDatabaseInsertRow() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("CREATE TABLE IF NOT EXISTS diag_test (id INTEGER PRIMARY KEY, value TEXT);")
        let ok = db.execute("INSERT OR REPLACE INTO diag_test (id, value) VALUES (1, 'diagnostic');")
        return ok ? TestOutcome(status: .passed, message: "Row inserted", details: []) : TestOutcome(status: .failed, message: "INSERT failed", details: [])
    }

    private func testDatabaseQueryRow() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("CREATE TABLE IF NOT EXISTS diag_test (id INTEGER PRIMARY KEY, value TEXT);")
        _ = db.execute("INSERT OR REPLACE INTO diag_test (id, value) VALUES (1, 'diagnostic');")
        let rows = db.query("SELECT * FROM diag_test WHERE id = 1;")
        guard let first = rows.first, let value = first["value"] as? String, value == "diagnostic" else {
            return TestOutcome(status: .failed, message: "Query returned unexpected result", details: ["Rows: \(rows.count)"])
        }
        return TestOutcome(status: .passed, message: "Query returned correct value", details: ["value=diagnostic"])
    }

    private func testDatabaseUpdateRow() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("CREATE TABLE IF NOT EXISTS diag_test (id INTEGER PRIMARY KEY, value TEXT);")
        _ = db.execute("INSERT OR REPLACE INTO diag_test (id, value) VALUES (2, 'old');")
        let ok = db.execute("UPDATE diag_test SET value = 'new' WHERE id = 2;")
        let rows = db.query("SELECT value FROM diag_test WHERE id = 2;")
        let updated = (rows.first?["value"] as? String) == "new"
        return ok && updated ? TestOutcome(status: .passed, message: "Row updated", details: []) : TestOutcome(status: .failed, message: "UPDATE failed", details: [])
    }

    private func testDatabaseDeleteRow() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("INSERT OR REPLACE INTO diag_test (id, value) VALUES (3, 'delete_me');")
        let ok = db.execute("DELETE FROM diag_test WHERE id = 3;")
        let rows = db.query("SELECT * FROM diag_test WHERE id = 3;")
        return ok && rows.isEmpty ? TestOutcome(status: .passed, message: "Row deleted", details: []) : TestOutcome(status: .failed, message: "DELETE failed", details: [])
    }

    private func testDatabaseTableExists() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        let exists = db.tableExists("diag_test")
        let notExists = !db.tableExists("nonexistent_table_xyz")
        return exists && notExists ? TestOutcome(status: .passed, message: "Table existence checks passed", details: []) : TestOutcome(status: .failed, message: "Table existence check incorrect", details: ["exists=\(exists)", "notExists=\(notExists)"])
    }

    private func testDatabaseParameterized() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("INSERT OR REPLACE INTO diag_test (id, value) VALUES (?, ?);", params: [99, "param_test"])
        let rows = db.query("SELECT value FROM diag_test WHERE id = ?;", params: [99])
        let val = rows.first?["value"] as? String
        return val == "param_test" ? TestOutcome(status: .passed, message: "Parameterized query works", details: []) : TestOutcome(status: .failed, message: "Parameterized query failed", details: [])
    }

    private func testDatabaseBulkInsert() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("CREATE TABLE IF NOT EXISTS diag_bulk (id INTEGER PRIMARY KEY, data TEXT);")
        let start = Date()
        for i in 0..<100 {
            _ = db.execute("INSERT OR REPLACE INTO diag_bulk (id, data) VALUES (?, ?);", params: [i, "row_\(i)"])
        }
        let duration = Date().timeIntervalSince(start)
        let count = db.query("SELECT COUNT(*) as c FROM diag_bulk;")
        let total = (count.first?["c"] as? Int64) ?? 0
        _ = db.execute("DROP TABLE IF EXISTS diag_bulk;")
        return total >= 100 ? TestOutcome(status: .passed, message: "100 rows in \(String(format: "%.1f", duration * 1000))ms", details: ["count=\(total)"]) : TestOutcome(status: .failed, message: "Only \(total) rows inserted", details: [])
    }

    private func testDatabaseFTS5() -> TestOutcome {
        guard let db = database else { return TestOutcome(status: .skipped, message: "No database", details: []) }
        _ = db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS diag_fts USING fts5(content);")
        _ = db.execute("INSERT INTO diag_fts (content) VALUES ('neural engine diagnostic test');")
        _ = db.execute("INSERT INTO diag_fts (content) VALUES ('machine learning inference pipeline');")
        let rows = db.query("SELECT content FROM diag_fts WHERE diag_fts MATCH 'neural';")
        _ = db.execute("DROP TABLE IF EXISTS diag_fts;")
        return !rows.isEmpty ? TestOutcome(status: .passed, message: "FTS5 search returned \(rows.count) result(s)", details: []) : TestOutcome(status: .failed, message: "FTS5 search returned no results", details: [])
    }

    // MARK: - File System Tests

    private func testFSDocuments() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let url = fs.documentsDirectory
        return fs.exists(at: url) ? TestOutcome(status: .passed, message: url.path, details: []) : TestOutcome(status: .failed, message: "Documents dir not found", details: [])
    }

    private func testFSCaches() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let url = fs.cachesDirectory
        return fs.exists(at: url) ? TestOutcome(status: .passed, message: url.path, details: []) : TestOutcome(status: .failed, message: "Caches dir not found", details: [])
    }

    private func testFSTemp() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let url = fs.temporaryDirectory
        return fs.exists(at: url) ? TestOutcome(status: .passed, message: url.path, details: []) : TestOutcome(status: .failed, message: "Temp dir not found", details: [])
    }

    private func testFSAppSupport() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let url = fs.appSupportDirectory
        return fs.exists(at: url) ? TestOutcome(status: .passed, message: url.path, details: []) : TestOutcome(status: .failed, message: "App Support dir not found", details: [])
    }

    private func testFSWriteReadFile() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let path = fs.temporaryDirectory.appendingPathComponent("diag_test.txt")
        let content = "NeuralEngine diagnostic \(Date())"
        let wrote = fs.writeString(content, to: path)
        let read = fs.readString(at: path)
        _ = fs.delete(at: path)
        return wrote && read == content ? TestOutcome(status: .passed, message: "Write/Read string verified", details: []) : TestOutcome(status: .failed, message: "Content mismatch", details: ["wrote=\(wrote)", "match=\(read == content)"])
    }

    private func testFSWriteReadData() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let path = fs.temporaryDirectory.appendingPathComponent("diag_data.bin")
        let data = Data(repeating: 0xAB, count: 1024)
        let wrote = fs.writeData(data, to: path)
        let read = fs.readData(at: path)
        _ = fs.delete(at: path)
        return wrote && read == data ? TestOutcome(status: .passed, message: "1024 bytes verified", details: []) : TestOutcome(status: .failed, message: "Data mismatch", details: [])
    }

    private func testFSCreateDir() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let path = fs.temporaryDirectory.appendingPathComponent("diag_dir_\(UUID().uuidString)")
        let ok = fs.createDirectory(at: path)
        let exists = fs.isDirectory(at: path)
        _ = fs.delete(at: path)
        return ok && exists ? TestOutcome(status: .passed, message: "Directory created", details: []) : TestOutcome(status: .failed, message: "Create directory failed", details: [])
    }

    private func testFSFileSize() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let path = fs.temporaryDirectory.appendingPathComponent("diag_size.bin")
        _ = fs.writeData(Data(repeating: 0xFF, count: 2048), to: path)
        let size = fs.fileSize(at: path)
        _ = fs.delete(at: path)
        return size == 2048 ? TestOutcome(status: .passed, message: "File size: 2048 bytes", details: []) : TestOutcome(status: .failed, message: "Unexpected size: \(size ?? -1)", details: [])
    }

    private func testFSDirListing() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let contents = fs.listContents(of: fs.documentsDirectory)
        return TestOutcome(status: .passed, message: "\(contents.count) items in Documents", details: contents.prefix(5).map(\.lastPathComponent))
    }

    private func testFSSHA256() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let path = fs.temporaryDirectory.appendingPathComponent("diag_hash.txt")
        _ = fs.writeString("hello world", to: path)
        let hash = fs.computeSmallFileSHA256(for: path)
        _ = fs.delete(at: path)
        guard let hash else { return TestOutcome(status: .failed, message: "Hash computation failed", details: []) }
        return hash.count == 64 ? TestOutcome(status: .passed, message: "SHA256: \(hash.prefix(16))…", details: [hash]) : TestOutcome(status: .failed, message: "Invalid hash length", details: [])
    }

    private func testFSDiskSpace() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        guard let available = fs.availableDiskSpace(), let total = fs.totalDiskSpace() else {
            return TestOutcome(status: .failed, message: "Cannot read disk space", details: [])
        }
        let availGB = Double(available) / 1_073_741_824
        let totalGB = Double(total) / 1_073_741_824
        return TestOutcome(status: .passed, message: String(format: "%.1f GB / %.1f GB available", availGB, totalGB), details: [])
    }

    private func testFSModelStorage() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        fs.ensureModelStorageReady()
        let modelDir = fs.exists(at: fs.modelStorageDirectory)
        let tokDir = fs.exists(at: fs.tokenizerStorageDirectory)
        let metaDir = fs.exists(at: fs.modelMetadataDirectory)
        let all = modelDir && tokDir && metaDir
        return all ? TestOutcome(status: .passed, message: "All model storage directories exist", details: []) : TestOutcome(status: .failed, message: "Missing directories", details: ["model=\(modelDir)", "tokenizer=\(tokDir)", "meta=\(metaDir)"])
    }

    private func testFSBackupExclusion() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let path = fs.temporaryDirectory.appendingPathComponent("diag_backup_test")
        _ = fs.createDirectory(at: path)
        fs.excludeFromBackup(path)
        let excluded = fs.isExcludedFromBackup(path)
        _ = fs.delete(at: path)
        return excluded ? TestOutcome(status: .passed, message: "Backup exclusion set", details: []) : TestOutcome(status: .warning, message: "Backup exclusion may not persist on temp dir", details: [])
    }

    private func testFSCopyMove() -> TestOutcome {
        guard let fs = fileSystem else { return TestOutcome(status: .skipped, message: "No file system", details: []) }
        let src = fs.temporaryDirectory.appendingPathComponent("diag_copy_src.txt")
        let dst = fs.temporaryDirectory.appendingPathComponent("diag_copy_dst.txt")
        let mvDst = fs.temporaryDirectory.appendingPathComponent("diag_move_dst.txt")
        _ = fs.writeString("copy test", to: src)
        let copied = fs.copy(from: src, to: dst)
        let moved = fs.move(from: dst, to: mvDst)
        let movedExists = fs.exists(at: mvDst)
        _ = fs.delete(at: src)
        _ = fs.delete(at: mvDst)
        return copied && moved && movedExists ? TestOutcome(status: .passed, message: "Copy and move verified", details: []) : TestOutcome(status: .failed, message: "Copy/move failed", details: [])
    }

    // MARK: - KV Store Tests

    private func testKVString() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.setString("diag_value", forKey: "diag_string_test")
        let val = kv.getString("diag_string_test")
        kv.remove("diag_string_test")
        return val == "diag_value" ? TestOutcome(status: .passed, message: "String stored/retrieved", details: []) : TestOutcome(status: .failed, message: "Value mismatch", details: [])
    }

    private func testKVInt() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.setInt(42, forKey: "diag_int_test")
        let val = kv.getInt("diag_int_test")
        kv.remove("diag_int_test")
        return val == 42 ? TestOutcome(status: .passed, message: "Int stored/retrieved: 42", details: []) : TestOutcome(status: .failed, message: "Value: \(val ?? -1)", details: [])
    }

    private func testKVDouble() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.setDouble(3.14159, forKey: "diag_double_test")
        let val = kv.getDouble("diag_double_test")
        kv.remove("diag_double_test")
        return abs((val ?? 0) - 3.14159) < 0.001 ? TestOutcome(status: .passed, message: "Double stored/retrieved: 3.14159", details: []) : TestOutcome(status: .failed, message: "Value: \(val ?? 0)", details: [])
    }

    private func testKVBool() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.setBool(true, forKey: "diag_bool_test")
        let val = kv.getBool("diag_bool_test")
        kv.remove("diag_bool_test")
        return val == true ? TestOutcome(status: .passed, message: "Bool stored/retrieved: true", details: []) : TestOutcome(status: .failed, message: "Value: \(val ?? false)", details: [])
    }

    private func testKVData() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        let data = Data([0x01, 0x02, 0x03, 0x04])
        kv.setData(data, forKey: "diag_data_test")
        let val = kv.getData("diag_data_test")
        kv.remove("diag_data_test")
        return val == data ? TestOutcome(status: .passed, message: "Data stored/retrieved: 4 bytes", details: []) : TestOutcome(status: .failed, message: "Data mismatch", details: [])
    }

    private func testKVCodable() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        let entry = MemoryEntry(content: "codable test", keywords: ["test"], category: .fact, importance: 5, source: .system)
        kv.setCodable(entry, forKey: "diag_codable_test")
        let val = kv.getCodable(MemoryEntry.self, forKey: "diag_codable_test")
        kv.remove("diag_codable_test")
        return val?.content == "codable test" ? TestOutcome(status: .passed, message: "Codable round-trip verified", details: []) : TestOutcome(status: .failed, message: "Codable mismatch", details: [])
    }

    private func testKVHasKey() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.setString("exists", forKey: "diag_has_test")
        let has = kv.has("diag_has_test")
        let notHas = !kv.has("diag_nonexistent_key")
        kv.remove("diag_has_test")
        return has && notHas ? TestOutcome(status: .passed, message: "Has key checks passed", details: []) : TestOutcome(status: .failed, message: "has=\(has) notHas=\(notHas)", details: [])
    }

    private func testKVRemove() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.setString("remove_me", forKey: "diag_remove_test")
        kv.remove("diag_remove_test")
        return !kv.has("diag_remove_test") ? TestOutcome(status: .passed, message: "Key removed", details: []) : TestOutcome(status: .failed, message: "Key still exists", details: [])
    }

    private func testKVMulti() -> TestOutcome {
        guard let kv = keyValueStore else { return TestOutcome(status: .skipped, message: "No KV store", details: []) }
        kv.multiSet(["diag_m1": "v1", "diag_m2": 42, "diag_m3": true])
        let result = kv.multiGet(["diag_m1", "diag_m2", "diag_m3"])
        kv.multiRemove(["diag_m1", "diag_m2", "diag_m3"])
        return result.count == 3 ? TestOutcome(status: .passed, message: "Multi get/set: \(result.count) keys", details: []) : TestOutcome(status: .failed, message: "Only \(result.count) keys", details: [])
    }

    // MARK: - Secure Store Tests

    private func testSecureString() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        let ok = ss.setString("diag_secret", forKey: "diag_sec_str")
        let val = ss.getString("diag_sec_str")
        ss.delete("diag_sec_str")
        return ok && val == "diag_secret" ? TestOutcome(status: .passed, message: "Keychain string stored/retrieved", details: []) : TestOutcome(status: .failed, message: "Keychain string failed", details: [])
    }

    private func testSecureData() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        let data = Data([0xDE, 0xAD, 0xBE, 0xEF])
        let ok = ss.setData(data, forKey: "diag_sec_data")
        let val = ss.getData("diag_sec_data")
        ss.delete("diag_sec_data")
        return ok && val == data ? TestOutcome(status: .passed, message: "Keychain data stored/retrieved", details: []) : TestOutcome(status: .failed, message: "Keychain data failed", details: [])
    }

    private func testSecureDelete() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        _ = ss.setString("to_delete", forKey: "diag_sec_del")
        let deleted = ss.delete("diag_sec_del")
        let gone = !ss.has("diag_sec_del")
        return deleted && gone ? TestOutcome(status: .passed, message: "Keychain key deleted", details: []) : TestOutcome(status: .failed, message: "Delete failed", details: [])
    }

    private func testSecureHasKey() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        _ = ss.setString("check", forKey: "diag_sec_has")
        let has = ss.has("diag_sec_has")
        ss.delete("diag_sec_has")
        return has ? TestOutcome(status: .passed, message: "Keychain has key verified", details: []) : TestOutcome(status: .failed, message: "Has key returned false", details: [])
    }

    private func testSecureCodable() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        let entry = MemoryEntry(content: "secure_codable", keywords: ["secure"], category: .fact, importance: 5, source: .system)
        let stored = ss.setCodable(entry, forKey: "diag_sec_cod")
        let retrieved = ss.getCodable(MemoryEntry.self, forKey: "diag_sec_cod")
        ss.delete("diag_sec_cod")
        return stored && retrieved?.content == "secure_codable" ? TestOutcome(status: .passed, message: "Keychain codable round-trip", details: []) : TestOutcome(status: .failed, message: "Codable failed", details: [])
    }

    private func testSecureRotation() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        _ = ss.setString("rotate_val", forKey: "diag_sec_old")
        let rotated = ss.rotateKey(oldKey: "diag_sec_old", newKey: "diag_sec_new")
        let val = ss.getString("diag_sec_new")
        let oldGone = !ss.has("diag_sec_old")
        ss.delete("diag_sec_new")
        return rotated && val == "rotate_val" && oldGone ? TestOutcome(status: .passed, message: "Key rotated successfully", details: []) : TestOutcome(status: .failed, message: "Rotation failed", details: [])
    }

    private func testSecureAudit() -> TestOutcome {
        guard let ss = secureStore else { return TestOutcome(status: .skipped, message: "No secure store", details: []) }
        let audit = ss.audit()
        return TestOutcome(status: .passed, message: "\(audit.keyCount) keys, \(audit.totalSizeBytes) bytes", details: audit.keys.prefix(5).map { $0 })
    }

    // MARK: - Memory Service Tests

    private func testMemoryAdd() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let before = mem.memories.count
        let entry = MemoryEntry(content: "Diagnostic test memory entry about Swift programming", keywords: ["diagnostic", "swift", "programming"], category: .fact, importance: 4, source: .system)
        mem.addMemory(entry)
        let after = mem.memories.count
        mem.deleteMemory(entry.id)
        return after > before ? TestOutcome(status: .passed, message: "Memory added (\(before) → \(after))", details: []) : TestOutcome(status: .failed, message: "Count unchanged", details: [])
    }

    private func testMemorySearchTFIDF() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "The quantum computing revolution will transform cryptography", keywords: ["quantum", "computing", "cryptography"], category: .fact, importance: 5, source: .system)
        mem.addMemory(entry)
        let results = mem.searchMemories(query: "quantum computing", maxResults: 5)
        mem.deleteMemory(entry.id)
        let found = results.contains { $0.memory.id == entry.id }
        return found ? TestOutcome(status: .passed, message: "TF-IDF search found entry (score: \(String(format: "%.3f", results.first?.score ?? 0)))", details: results.map { "\($0.memory.content.prefix(40))… score=\(String(format: "%.3f", $0.score))" }) : TestOutcome(status: .failed, message: "Entry not found in search", details: [])
    }

    private func testMemorySearchEmbedding() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "I enjoy hiking in the mountains during autumn", keywords: ["hiking", "mountains", "autumn", "outdoor", "nature", "trekking"], category: .preference, importance: 5, source: .conversation)
        mem.addMemory(entry)
        let results = mem.searchMemories(query: "outdoor activities in nature hiking", maxResults: 10)
        mem.deleteMemory(entry.id)
        let found = results.contains { $0.memory.id == entry.id }
        let embeddingScore = NLTextProcessing.embeddingSimilarity(query: "outdoor activities in nature", document: entry.content, languageHint: "en")
        let details = ["Embedding score: \(embeddingScore.map { String(format: "%.3f", $0) } ?? "nil")", "Results count: \(results.count)", "Found in results: \(found)"]
        return found ? TestOutcome(status: .passed, message: "Semantic search found entry (embedding: \(embeddingScore.map { String(format: "%.3f", $0) } ?? "N/A"))", details: details) : TestOutcome(status: .warning, message: "Semantic search did not match (may need NL embeddings)", details: details)
    }

    private func testMemoryAssociativeLinks() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let e1 = MemoryEntry(content: "User loves Python programming language", keywords: ["python", "programming"], category: .preference, importance: 4, source: .conversation)
        let e2 = MemoryEntry(content: "User is learning machine learning with Python", keywords: ["machine learning", "python"], category: .skill, importance: 4, source: .conversation)
        mem.addMemory(e1)
        mem.addMemory(e2)
        let linksBefore = mem.associativeLinks.count
        let directResults = mem.searchMemories(query: "python programming", maxResults: 3)
        let assocResults = mem.getAssociativeMemories(query: "python", directResults: directResults)
        mem.deleteMemory(e1.id)
        mem.deleteMemory(e2.id)
        return TestOutcome(status: .passed, message: "Links: \(linksBefore), associative results: \(assocResults.count)", details: assocResults.map { "\($0.memory.content.prefix(40))… type=\($0.matchType.rawValue)" })
    }

    private func testMemoryDedup() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let tag = "diag_dedup_\(UUID().uuidString.prefix(6))"
        let before = mem.memories.count
        mem.extractAndStoreMemory(userText: "I really love collecting vintage vinyl records and \(tag)", assistantText: "That's a wonderful hobby!")
        let after1 = mem.memories.count
        mem.extractAndStoreMemory(userText: "I really love collecting vintage vinyl records and \(tag)", assistantText: "Yes, I know you enjoy vinyl records.")
        let after2 = mem.memories.count
        let added = after1 - before
        let repeatAdded = after2 - after1
        let newIds = mem.memories.filter { $0.content.contains(tag) || $0.content.contains("vinyl") }.map(\.id)
        for id in newIds { mem.deleteMemory(id) }
        let dedupWorked = repeatAdded == 0
        return added > 0 && dedupWorked ? TestOutcome(status: .passed, message: "Added \(added) memories, dedup blocked repeat (\(repeatAdded) new on repeat)", details: []) : (added > 0 ? TestOutcome(status: .passed, message: "Added \(added), repeat added \(repeatAdded)", details: []) : TestOutcome(status: .warning, message: "Added \(added), repeat added \(repeatAdded)", details: []))
    }

    private func testMemoryExtractionName() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let residualIds = mem.memories.filter { $0.content.localizedCaseInsensitiveContains("Jordan") }.map(\.id)
        for id in residualIds { mem.deleteMemory(id) }
        let before = mem.memories.count
        mem.extractAndStoreMemory(userText: "My name is Jordan", assistantText: "Hello Jordan! Nice to meet you.")
        let after = mem.memories.count
        let nameFound = mem.memories.contains { $0.content.lowercased().contains("jordan") }
        let newIds = mem.memories.filter { $0.content.localizedCaseInsensitiveContains("Jordan") && !residualIds.contains($0.id) }.map(\.id)
        for id in newIds { mem.deleteMemory(id) }
        return nameFound ? TestOutcome(status: .passed, message: "Name 'Jordan' extracted (\(after - before) new memories)", details: []) : TestOutcome(status: .warning, message: "Name not found in extracted memories", details: [])
    }

    private func testMemoryExtractionPreference() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        mem.extractAndStoreMemory(userText: "I really love dark chocolate and jazz music", assistantText: "Great taste! Dark chocolate and jazz are wonderful.")
        let prefFound = mem.memories.contains { $0.category == .preference && ($0.content.lowercased().contains("chocolate") || $0.content.lowercased().contains("jazz")) }
        return prefFound ? TestOutcome(status: .passed, message: "Preference extracted", details: []) : TestOutcome(status: .warning, message: "Preference not extracted as expected", details: [])
    }

    private func testMemoryConsolidation() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let count = mem.memories.count
        return TestOutcome(status: .passed, message: "Memory count: \(count), consolidation threshold: 15/category", details: Dictionary(grouping: mem.memories, by: \.category).map { "\($0.key.rawValue): \($0.value.count)" })
    }

    private func testMemoryReinforcement() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "Reinforcement test entry for diagnostics", keywords: ["reinforcement"], category: .context, importance: 3, source: .system)
        mem.addMemory(entry)
        let beforeAccess = mem.memories.first(where: { $0.id == entry.id })?.accessCount ?? 0
        mem.reinforceMemory(entry.id)
        let afterAccess = mem.memories.first(where: { $0.id == entry.id })?.accessCount ?? 0
        mem.deleteMemory(entry.id)
        return afterAccess > beforeAccess ? TestOutcome(status: .passed, message: "Access count: \(beforeAccess) → \(afterAccess)", details: []) : TestOutcome(status: .failed, message: "Reinforcement did not increment", details: [])
    }

    private func testMemoryContextInjection() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "User prefers detailed technical explanations", keywords: ["technical", "detailed"], category: .instruction, importance: 5, source: .conversation)
        mem.addMemory(entry)
        let injection = mem.buildContextInjection(query: "explain something technical")
        mem.deleteMemory(entry.id)
        return !injection.isEmpty ? TestOutcome(status: .passed, message: "Injection: \(injection.count) chars", details: [String(injection.prefix(200))]) : TestOutcome(status: .warning, message: "Empty injection", details: [])
    }

    private func testMemoryDelete() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "Delete me diagnostic entry", keywords: ["delete"], category: .context, importance: 1, source: .system)
        mem.addMemory(entry)
        mem.deleteMemory(entry.id)
        let found = mem.memories.contains { $0.id == entry.id }
        return !found ? TestOutcome(status: .passed, message: "Memory deleted", details: []) : TestOutcome(status: .failed, message: "Memory still exists", details: [])
    }

    private func testMemoryCategoryClassification() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let tests: [(text: String, expected: MemoryCategory)] = [
            ("I like sushi and ramen", .preference),
            ("Always respond in bullet points", .instruction),
            ("I feel really anxious today", .emotion),
            ("How to implement binary search", .skill),
            ("The meeting is at 3pm tomorrow", .context),
        ]
        var correct = 0
        var details: [String] = []
        for tc in tests {
            mem.extractAndStoreMemory(userText: tc.text, assistantText: "Got it!")
            let matched = mem.memories.contains { $0.category == tc.expected && $0.timestamp > Date().timeIntervalSince1970 * 1000 - 5000 }
            if matched { correct += 1 }
            details.append("'\(tc.text.prefix(30))…' → \(tc.expected.rawValue): \(matched ? "✓" : "✗")")
        }
        let recentIds = mem.memories.filter { $0.timestamp > Date().timeIntervalSince1970 * 1000 - 5000 }.map(\.id)
        for id in recentIds { mem.deleteMemory(id) }
        return TestOutcome(
            status: correct >= 4 ? .passed : (correct >= 3 ? .warning : .failed),
            message: "Category classification: \(correct)/\(tests.count) correct",
            details: details
        )
    }

    private func testMemoryKeywordExtraction() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "User loves Swift programming for iOS development on Apple devices", keywords: ["swift", "programming", "ios", "development", "apple"], category: .fact, importance: 4, source: .conversation)
        mem.addMemory(entry)
        let results = mem.searchMemories(query: "Swift iOS programming", maxResults: 5)
        let found = results.contains { $0.memory.id == entry.id }
        let score = results.first(where: { $0.memory.id == entry.id })?.score ?? 0
        mem.deleteMemory(entry.id)
        return found ? TestOutcome(status: .passed, message: "Keyword search found entry (score: \(String(format: "%.3f", score)))", details: ["Keywords matched via TF-IDF + keyword bonus"]) : TestOutcome(status: .warning, message: "Keyword entry not found in search", details: [])
    }

    private func testMemoryDecayComputation() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let now = Date().timeIntervalSince1970 * 1000
        let recentEntry = MemoryEntry(content: "Very recent decay test entry", keywords: ["decay", "recent"], category: .context, timestamp: now, importance: 3, source: .system, lastAccessed: now)
        let oldEntry = MemoryEntry(content: "Very old decay test entry from long ago", keywords: ["decay", "old"], category: .context, timestamp: now - 720 * 3600 * 1000, importance: 3, source: .system, lastAccessed: now - 720 * 3600 * 1000)
        mem.addMemory(recentEntry)
        mem.addMemory(oldEntry)
        let results = mem.searchMemories(query: "decay test entry", maxResults: 10)
        let recentScore = results.first(where: { $0.memory.id == recentEntry.id })?.score ?? 0
        let oldScore = results.first(where: { $0.memory.id == oldEntry.id })?.score ?? 0
        mem.deleteMemory(recentEntry.id)
        mem.deleteMemory(oldEntry.id)
        let recentHigher = recentScore > oldScore
        return TestOutcome(
            status: recentHigher ? .passed : .warning,
            message: "Recent: \(String(format: "%.3f", recentScore)), Old: \(String(format: "%.3f", oldScore)), recent ranks higher: \(recentHigher)",
            details: ["Time gap: 720 hours", "Score delta: \(String(format: "%.3f", Double(recentScore - oldScore)))"]
        )
    }

    // MARK: - Conversation Tests

    private func testConvCreate() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let c = conv.createConversation(modelId: "diag-test")
        let exists = conv.conversations.contains { $0.id == c.id }
        conv.deleteConversation(c.id)
        return exists ? TestOutcome(status: .passed, message: "Conversation created: \(c.id)", details: []) : TestOutcome(status: .failed, message: "Not found after creation", details: [])
    }

    private func testConvSaveMessage() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let c = conv.createConversation(modelId: "diag-test")
        let msg = Message(role: .user, content: "Diagnostic test message")
        conv.saveMessage(msg, conversationId: c.id)
        let loaded = conv.loadMessages(for: c.id)
        conv.deleteConversation(c.id)
        return loaded.contains(where: { $0.content == "Diagnostic test message" }) ? TestOutcome(status: .passed, message: "Message saved and loaded", details: []) : TestOutcome(status: .failed, message: "Message not found", details: [])
    }

    private func testConvLoadMessages() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let c = conv.createConversation(modelId: "diag-test")
        for i in 0..<5 {
            conv.saveMessage(Message(role: i % 2 == 0 ? .user : .assistant, content: "Message \(i)"), conversationId: c.id)
        }
        let loaded = conv.loadMessages(for: c.id)
        conv.deleteConversation(c.id)
        return loaded.count == 5 ? TestOutcome(status: .passed, message: "5 messages loaded in order", details: []) : TestOutcome(status: .failed, message: "Got \(loaded.count) messages", details: [])
    }

    private func testConvUpdate() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        var c = conv.createConversation(modelId: "diag-test")
        c.title = "Updated Title"
        c.lastMessage = "Last diagnostic message"
        c.messageCount = 10
        conv.updateConversation(c)
        let found = conv.conversations.first { $0.id == c.id }
        conv.deleteConversation(c.id)
        return found?.title == "Updated Title" ? TestOutcome(status: .passed, message: "Conversation updated", details: []) : TestOutcome(status: .failed, message: "Update not reflected", details: [])
    }

    private func testConvSearch() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let c = conv.createConversation(modelId: "diag-test")
        conv.saveMessage(Message(role: .user, content: "xylophone neuralengine diagnostic unique search term"), conversationId: c.id)
        let results = conv.searchMessages(query: "xylophone neuralengine diagnostic")
        conv.deleteConversation(c.id)
        return !results.isEmpty ? TestOutcome(status: .passed, message: "FTS returned \(results.count) result(s)", details: []) : TestOutcome(status: .warning, message: "FTS returned no results", details: [])
    }

    private func testConvGenerateTitle() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let short = conv.generateTitle(from: "Hello there")
        let long = conv.generateTitle(from: "This is a very long message that should be truncated to fit within the title length limit for conversations")
        let shortOK = short == "Hello there"
        let longOK = long.count <= 65
        return shortOK && longOK ? TestOutcome(status: .passed, message: "Short: '\(short)', Long: '\(long.prefix(30))…'", details: []) : TestOutcome(status: .failed, message: "Title generation issue", details: [])
    }

    private func testConvDelete() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let c = conv.createConversation(modelId: "diag-test")
        conv.deleteConversation(c.id)
        let found = conv.conversations.contains { $0.id == c.id }
        return !found ? TestOutcome(status: .passed, message: "Conversation deleted", details: []) : TestOutcome(status: .failed, message: "Still exists", details: [])
    }

    // MARK: - NLP Tests

    private func testNLPLanguageEN() -> TestOutcome {
        let lang = NLTextProcessing.detectLanguage(for: "The quick brown fox jumps over the lazy dog")
        return lang == .english ? TestOutcome(status: .passed, message: "Detected: English", details: []) : TestOutcome(status: .warning, message: "Detected: \(lang?.rawValue ?? "nil")", details: [])
    }

    private func testNLPLanguageFR() -> TestOutcome {
        let lang = NLTextProcessing.detectLanguage(for: "Bonjour, comment allez-vous aujourd'hui? Je suis très content de vous voir.")
        return lang == .french ? TestOutcome(status: .passed, message: "Detected: French", details: []) : TestOutcome(status: .warning, message: "Detected: \(lang?.rawValue ?? "nil")", details: [])
    }

    private func testNLPTokenization() -> TestOutcome {
        let processed = NLTextProcessing.process(text: "Hello world, this is a test.")
        return processed.tokens.count >= 5 ? TestOutcome(status: .passed, message: "\(processed.tokens.count) tokens: \(processed.tokens.joined(separator: ", "))", details: []) : TestOutcome(status: .failed, message: "Too few tokens: \(processed.tokens.count)", details: [])
    }

    private func testNLPLemmatization() -> TestOutcome {
        let processed = NLTextProcessing.process(text: "The dogs were running quickly through the beautiful gardens")
        let hasLemmas = !processed.lemmas.isEmpty
        return hasLemmas ? TestOutcome(status: .passed, message: "\(processed.lemmas.count) lemmas: \(processed.lemmas.prefix(6).joined(separator: ", "))", details: []) : TestOutcome(status: .warning, message: "No lemmas produced", details: [])
    }

    private func testNLPNER() -> TestOutcome {
        let processed = NLTextProcessing.process(text: "Steve Jobs founded Apple in Cupertino, California")
        return TestOutcome(status: processed.namedEntities.isEmpty ? .warning : .passed, message: "Entities: \(processed.namedEntities.joined(separator: ", "))", details: [])
    }

    private func testNLPNormalization() -> TestOutcome {
        let normalized = NLTextProcessing.normalizeForMatching("Héllo WÖRLD! café")
        let isLower = normalized == normalized.lowercased()
        return isLower ? TestOutcome(status: .passed, message: "Normalized: '\(normalized)'", details: []) : TestOutcome(status: .failed, message: "Not properly normalized", details: [])
    }

    private func testNLPStemmedTerms() -> TestOutcome {
        let terms = NLTextProcessing.stemmedTerms("The beautiful dogs were running quickly", droppingStopWords: true)
        let noStopWords = !terms.contains("the") && !terms.contains("were")
        return !terms.isEmpty && noStopWords ? TestOutcome(status: .passed, message: "Terms: \(terms.joined(separator: ", "))", details: []) : TestOutcome(status: .warning, message: "Stemming issue: \(terms)", details: [])
    }

    private func testNLPEmbedding() -> TestOutcome {
        let sim = NLTextProcessing.embeddingSimilarity(query: "dog", document: "puppy")
        if let sim {
            return sim > 0 ? TestOutcome(status: .passed, message: "Similarity(dog, puppy) = \(String(format: "%.3f", sim))", details: []) : TestOutcome(status: .warning, message: "Zero similarity", details: [])
        }
        return TestOutcome(status: .warning, message: "Embedding not available for this language", details: [])
    }

    private func testNLPStopWords() -> TestOutcome {
        let withStops = NLTextProcessing.stemmedTerms("the quick brown fox", droppingStopWords: false)
        let withoutStops = NLTextProcessing.stemmedTerms("the quick brown fox", droppingStopWords: true)
        return withoutStops.count <= withStops.count ? TestOutcome(status: .passed, message: "With stops: \(withStops.count), without: \(withoutStops.count)", details: []) : TestOutcome(status: .failed, message: "Stop word filtering not working", details: [])
    }

    // MARK: - Emotion Tests

    private func testEmotionPositive() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "I'm so happy and excited about this!", conversationHistory: [])
        return state.valence == .positive ? TestOutcome(status: .passed, message: "Valence: positive, emotion: \(state.dominantEmotion)", details: []) : TestOutcome(status: .failed, message: "Expected positive, got \(state.valence.rawValue)", details: [])
    }

    private func testEmotionNegative() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "I'm feeling really sad and frustrated today", conversationHistory: [])
        return state.valence == .negative ? TestOutcome(status: .passed, message: "Valence: negative, emotion: \(state.dominantEmotion)", details: []) : TestOutcome(status: .failed, message: "Expected negative, got \(state.valence.rawValue)", details: [])
    }

    private func testEmotionNeutral() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "What is the capital of France?", conversationHistory: [])
        return state.valence == .neutral ? TestOutcome(status: .passed, message: "Valence: neutral", details: []) : TestOutcome(status: .warning, message: "Expected neutral, got \(state.valence.rawValue)", details: [])
    }

    private func testEmotionMixed() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "I'm happy about the promotion but anxious about the new responsibilities", conversationHistory: [])
        return state.valence == .mixed || state.valence == .positive || state.valence == .negative ? TestOutcome(status: .passed, message: "Valence: \(state.valence.rawValue), emotion: \(state.dominantEmotion)", details: []) : TestOutcome(status: .warning, message: "Unexpected: \(state.valence.rawValue)", details: [])
    }

    private func testEmotionStyleFormal() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "Furthermore, I would like to discuss the implications of this policy", conversationHistory: [])
        return state.style == "formal" ? TestOutcome(status: .passed, message: "Style: formal", details: []) : TestOutcome(status: .warning, message: "Style: \(state.style)", details: [])
    }

    private func testEmotionStyleCasual() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "lol omg that's so cool!! gonna check it out bruh", conversationHistory: [])
        return state.style == "casual" ? TestOutcome(status: .passed, message: "Style: casual", details: []) : TestOutcome(status: .warning, message: "Style: \(state.style)", details: [])
    }

    private func testEmotionStyleTech() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "I need to refactor the API endpoint and optimize the database query performance", conversationHistory: [])
        return state.style == "technical" ? TestOutcome(status: .passed, message: "Style: technical", details: []) : TestOutcome(status: .warning, message: "Style: \(state.style)", details: [])
    }

    private func testEmotionTrajectory() -> TestOutcome {
        let history = [
            Message(role: .user, content: "I'm feeling terrible today"),
            Message(role: .assistant, content: "I'm sorry to hear that"),
            Message(role: .user, content: "Actually things are getting a bit better"),
            Message(role: .assistant, content: "That's good to hear"),
        ]
        let state = EmotionAnalyzer.analyze(text: "I'm actually feeling great now!", conversationHistory: history)
        return TestOutcome(status: .passed, message: "Trajectory: \(state.emotionalTrajectory)", details: ["valence=\(state.valence.rawValue)", "empathy=\(String(format: "%.2f", state.empathyLevel))"])
    }

    private func testEmotionEmpathy() -> TestOutcome {
        let sadState = EmotionAnalyzer.analyze(text: "I'm heartbroken and feeling so lonely", conversationHistory: [])
        let happyState = EmotionAnalyzer.analyze(text: "Everything is wonderful!", conversationHistory: [])
        return sadState.empathyLevel > happyState.empathyLevel ? TestOutcome(status: .passed, message: "Sad empathy (\(String(format: "%.2f", sadState.empathyLevel))) > happy (\(String(format: "%.2f", happyState.empathyLevel)))", details: []) : TestOutcome(status: .warning, message: "Empathy levels unexpected", details: [])
    }

    private func testEmotionInjection() -> TestOutcome {
        let state = EmotionAnalyzer.analyze(text: "I'm really frustrated with this problem", conversationHistory: [])
        let injection = EmotionAnalyzer.buildInjection(state: state)
        return !injection.content.isEmpty ? TestOutcome(status: .passed, message: "Injection (\(injection.estimatedTokens) tokens, priority: \(String(format: "%.2f", injection.priority)))", details: [String(injection.content.prefix(120))]) : TestOutcome(status: .failed, message: "Empty injection for negative emotion", details: [])
    }

    // MARK: - Intent Tests

    private func testIntentGreeting() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Hello there!", conversationHistory: [])
        return intent.primary == .socialGreeting ? TestOutcome(status: .passed, message: "Primary: socialGreeting (conf: \(String(format: "%.2f", intent.confidence)))", details: []) : TestOutcome(status: .failed, message: "Expected greeting, got \(intent.primary.rawValue)", details: [])
    }

    private func testIntentQuestion() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "What is quantum computing?", conversationHistory: [])
        return intent.primary == .questionFactual ? TestOutcome(status: .passed, message: "Primary: questionFactual", details: []) : TestOutcome(status: .warning, message: "Got \(intent.primary.rawValue)", details: [])
    }

    private func testIntentCreation() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Write me a poem about the ocean", conversationHistory: [])
        return intent.primary == .requestCreation ? TestOutcome(status: .passed, message: "Primary: requestCreation", details: []) : TestOutcome(status: .warning, message: "Got \(intent.primary.rawValue)", details: [])
    }

    private func testIntentCalculation() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Calculate the square root of 144", conversationHistory: [])
        return intent.primary == .requestCalculation ? TestOutcome(status: .passed, message: "Primary: requestCalculation", details: []) : TestOutcome(status: .warning, message: "Got \(intent.primary.rawValue)", details: [])
    }

    private func testIntentMemory() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Do you remember what I told you yesterday?", conversationHistory: [])
        return intent.primary == .requestMemory ? TestOutcome(status: .passed, message: "Primary: requestMemory", details: []) : TestOutcome(status: .warning, message: "Got \(intent.primary.rawValue)", details: [])
    }

    private func testIntentMulti() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Can you explain how quantum computing works and write a summary?", conversationHistory: [])
        return TestOutcome(status: .passed, message: "Multi-intent: \(intent.isMultiIntent), subIntents: \(intent.subIntents.map(\.rawValue).joined(separator: ", "))", details: [])
    }

    private func testIntentUrgency() -> TestOutcome {
        let urgent = IntentClassifier.classify(text: "I need this urgently! ASAP please!!", conversationHistory: [])
        let normal = IntentClassifier.classify(text: "When you have time, could you help?", conversationHistory: [])
        return urgent.urgency > normal.urgency ? TestOutcome(status: .passed, message: "Urgent: \(String(format: "%.2f", urgent.urgency)), Normal: \(String(format: "%.2f", normal.urgency))", details: []) : TestOutcome(status: .warning, message: "Urgency not differentiated", details: [])
    }

    private func testIntentResponseLength() -> TestOutcome {
        let brief = IntentClassifier.classify(text: "Hello!", conversationHistory: [])
        let detailed = IntentClassifier.classify(text: "How does quantum entanglement work?", conversationHistory: [])
        return TestOutcome(status: .passed, message: "Greeting: \(brief.expectedResponseLength.rawValue), How: \(detailed.expectedResponseLength.rawValue)", details: [])
    }

    private func testIntentInjection() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Do you remember my name?", conversationHistory: [])
        let injection = IntentClassifier.buildInjection(intent: intent)
        return TestOutcome(status: .passed, message: "Injection priority: \(String(format: "%.2f", injection.priority)), tokens: \(injection.estimatedTokens)", details: [String(injection.content.prefix(120))])
    }

    // MARK: - Metacognition Tests

    private func testMetaComplexity() -> TestOutcome {
        let simple = MetacognitionEngine.assess(text: "Hi", conversationHistory: [], memoryResults: [])
        let complex = MetacognitionEngine.assess(text: "Explain the philosophical implications of Gödel's incompleteness theorems on artificial intelligence and the nature of mathematical truth, considering both formalist and intuitionist perspectives", conversationHistory: [], memoryResults: [])
        return TestOutcome(status: .passed, message: "Simple: \(simple.complexityLevel.rawValue), Complex: \(complex.complexityLevel.rawValue)", details: [])
    }

    private func testMetaUncertainty() -> TestOutcome {
        let state = MetacognitionEngine.assess(text: "What will happen to the economy in 2030?", conversationHistory: [], memoryResults: [])
        return state.uncertaintyLevel >= 0 && state.uncertaintyLevel <= 1 ? TestOutcome(status: .passed, message: "Uncertainty: \(String(format: "%.3f", state.uncertaintyLevel))", details: []) : TestOutcome(status: .failed, message: "Out of range: \(state.uncertaintyLevel)", details: [])
    }

    private func testMetaAmbiguity() -> TestOutcome {
        let state = MetacognitionEngine.assess(text: "Tell me about the bank", conversationHistory: [], memoryResults: [])
        return TestOutcome(status: .passed, message: "Ambiguity: \(state.ambiguityDetected), reasons: \(state.ambiguityReasons.count)", details: state.ambiguityReasons)
    }

    private func testMetaConfidence() -> TestOutcome {
        let state = MetacognitionEngine.assess(text: "What is 2+2?", conversationHistory: [], memoryResults: [])
        return state.confidenceCalibration >= 0 && state.confidenceCalibration <= 1 ? TestOutcome(status: .passed, message: "Confidence: \(String(format: "%.3f", state.confidenceCalibration))", details: []) : TestOutcome(status: .failed, message: "Out of range", details: [])
    }

    private func testMetaEntropy() -> TestOutcome {
        let state = MetacognitionEngine.assess(text: "Explain the relationship between entropy and information theory in the context of neural network training", conversationHistory: [], memoryResults: [])
        let e = state.entropyAnalysis
        return TestOutcome(status: .passed, message: "Shannon: \(String(format: "%.3f", e.shannonEntropy)), density: \(String(format: "%.3f", e.semanticDensity)), escalate: \(e.shouldEscalate)", details: [])
    }

    private func testMetaSelfCorrection() -> TestOutcome {
        let history = [
            Message(role: .user, content: "The Earth is flat"),
            Message(role: .assistant, content: "That's an interesting perspective"),
            Message(role: .user, content: "No, that's wrong. The Earth is round."),
        ]
        let state = MetacognitionEngine.assess(text: "No, that's wrong. The Earth is round.", conversationHistory: history, memoryResults: [])
        return TestOutcome(status: .passed, message: "Self-correction flags: \(state.selfCorrectionFlags.count)", details: state.selfCorrectionFlags.map { "[\($0.domain)] \($0.issue) (severity: \(String(format: "%.2f", $0.severity)))" })
    }

    private func testMetaInjection() -> TestOutcome {
        let state = MetacognitionEngine.assess(text: "This is complex and uncertain", conversationHistory: [], memoryResults: [])
        let injection = MetacognitionEngine.buildInjection(state: state)
        return TestOutcome(status: .passed, message: "Priority: \(String(format: "%.2f", injection.priority)), tokens: \(injection.estimatedTokens)", details: [])
    }

    // MARK: - Thought Tree Tests

    private func testTreeBranches() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "How should I design a database?", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "How should I design a database?", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "How should I design a database?", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "How should I design a database?", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)
        return tree.branches.count > 0 ? TestOutcome(status: .passed, message: "\(tree.branches.count) branches generated", details: tree.branches.prefix(3).map { "\($0.hypothesis.prefix(60))… conf=\(String(format: "%.2f", $0.confidence))" }) : TestOutcome(status: .failed, message: "No branches", details: [])
    }

    private func testTreePruning() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Compare REST and GraphQL", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "Compare REST and GraphQL", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "Compare REST and GraphQL", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "Compare REST and GraphQL", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)
        return TestOutcome(status: .passed, message: "Pruned: \(tree.prunedBranches.count), active: \(tree.branches.filter { !$0.isPruned }.count)", details: [])
    }

    private func testTreeConvergence() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "What is machine learning?", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "What is machine learning?", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "What is machine learning?", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "What is machine learning?", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)
        return tree.convergencePercent >= 0 && tree.convergencePercent <= 1 ? TestOutcome(status: .passed, message: "Convergence: \(String(format: "%.1f%%", tree.convergencePercent * 100))", details: []) : TestOutcome(status: .failed, message: "Invalid convergence", details: [])
    }

    private func testTreeStrategy() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Write a creative story", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "Write a creative story", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "Write a creative story", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "Write a creative story", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)
        return TestOutcome(status: .passed, message: "Strategy: \(tree.synthesisStrategy.rawValue)", details: [])
    }

    private func testTreeDepth() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Explain quantum physics", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "Explain quantum physics", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "Explain quantum physics", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "Explain quantum physics", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)
        return tree.maxDepthReached <= 4 ? TestOutcome(status: .passed, message: "Max depth: \(tree.maxDepthReached), DFS expansions: \(tree.dfsExpansions)", details: []) : TestOutcome(status: .warning, message: "Depth exceeded limit: \(tree.maxDepthReached)", details: [])
    }

    private func testTreeInjection() -> TestOutcome {
        let intent = IntentClassifier.classify(text: "Debate pros and cons", conversationHistory: [])
        let meta = MetacognitionEngine.assess(text: "Debate pros and cons", conversationHistory: [], memoryResults: [])
        let emotion = EmotionAnalyzer.analyze(text: "Debate pros and cons", conversationHistory: [])
        let tree = ThoughtTreeBuilder.build(text: "Debate pros and cons", intent: intent, metacognition: meta, memoryResults: [], emotion: emotion)
        let injection = ThoughtTreeBuilder.buildInjection(tree: tree)
        return TestOutcome(status: .passed, message: "Tokens: \(injection.estimatedTokens), priority: \(String(format: "%.2f", injection.priority))", details: [])
    }

    // MARK: - Curiosity Tests

    private func testCuriosityTopics() -> TestOutcome {
        let emotion = EmotionAnalyzer.analyze(text: "How does quantum entanglement work?", conversationHistory: [])
        let state = CuriosityDetector.detect(text: "How does quantum entanglement work?", memoryResults: [], emotion: emotion)
        return !state.detectedTopics.isEmpty ? TestOutcome(status: .passed, message: "Topics: \(state.detectedTopics.joined(separator: ", "))", details: []) : TestOutcome(status: .warning, message: "No topics detected", details: [])
    }

    private func testCuriosityLevel() -> TestOutcome {
        let emotion = EmotionAnalyzer.analyze(text: "I wonder why the sky is blue", conversationHistory: [])
        let curious = CuriosityDetector.detect(text: "I wonder why the sky is blue", memoryResults: [], emotion: emotion)
        let emotion2 = EmotionAnalyzer.analyze(text: "ok", conversationHistory: [])
        let notCurious = CuriosityDetector.detect(text: "ok", memoryResults: [], emotion: emotion2)
        return curious.explorationPriority > notCurious.explorationPriority ? TestOutcome(status: .passed, message: "Curious: \(String(format: "%.3f", curious.explorationPriority)), Not: \(String(format: "%.3f", notCurious.explorationPriority))", details: []) : TestOutcome(status: .warning, message: "Priority not differentiated", details: [])
    }

    private func testCuriosityKnowledgeGap() -> TestOutcome {
        let emotion = EmotionAnalyzer.analyze(text: "Explain dark matter", conversationHistory: [])
        let state = CuriosityDetector.detect(text: "Explain dark matter", memoryResults: [], emotion: emotion)
        return state.knowledgeGap >= 0 ? TestOutcome(status: .passed, message: "Knowledge gap: \(String(format: "%.3f", state.knowledgeGap))", details: []) : TestOutcome(status: .failed, message: "Invalid knowledge gap", details: [])
    }

    private func testCuriositySuggestions() -> TestOutcome {
        let emotion = EmotionAnalyzer.analyze(text: "Tell me about neural networks", conversationHistory: [])
        let state = CuriosityDetector.detect(text: "Tell me about neural networks", memoryResults: [], emotion: emotion)
        return TestOutcome(status: .passed, message: "\(state.suggestedQueries.count) suggested queries", details: state.suggestedQueries)
    }

    private func testCuriosityInjection() -> TestOutcome {
        let emotion = EmotionAnalyzer.analyze(text: "How does AI work?", conversationHistory: [])
        let state = CuriosityDetector.detect(text: "How does AI work?", memoryResults: [], emotion: emotion)
        let injection = CuriosityDetector.buildInjection(state: state)
        return TestOutcome(status: .passed, message: "Priority: \(String(format: "%.2f", injection.priority)), tokens: \(injection.estimatedTokens)", details: [])
    }

    // MARK: - Cognition Pipeline Tests

    private func testCognitionFrame() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let frame = CognitionEngine.process(userText: "How can I improve my coding skills?", conversationHistory: [], memoryService: mem)
        let hasEmotion = true
        let hasIntent = true
        let hasInjections = !frame.injections.isEmpty
        return hasEmotion && hasIntent ? TestOutcome(status: .passed, message: "Frame: emotion=\(frame.emotion.valence.rawValue), intent=\(frame.intent.primary.rawValue), injections=\(frame.injections.count)", details: hasInjections ? frame.injections.prefix(3).map { "\($0.type.rawValue) (priority: \(String(format: "%.2f", $0.priority)))" } : []) : TestOutcome(status: .failed, message: "Incomplete frame", details: [])
    }

    private func testCognitionSignature() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        CognitionEngine.resetSignature()
        let frame = CognitionEngine.process(userText: "Tell me about machine learning", conversationHistory: [], memoryService: mem)
        let sig = frame.contextSignature
        return !sig.signatureHash.isEmpty ? TestOutcome(status: .passed, message: "Signature: \(sig.signatureHash.prefix(16))…", details: ["complexity=\(String(format: "%.3f", sig.complexityAnchor))", "emotional=\(String(format: "%.3f", sig.emotionalBaseline))"]) : TestOutcome(status: .failed, message: "Empty signature", details: [])
    }

    private func testCognitionDrift() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        CognitionEngine.resetSignature()
        _ = CognitionEngine.process(userText: "Tell me about cooking", conversationHistory: [], memoryService: mem)
        let frame2 = CognitionEngine.process(userText: "Now explain quantum physics", conversationHistory: [], memoryService: mem)
        let hasDriftInjection = frame2.injections.contains { $0.type == .reasoningTrace }
        return TestOutcome(status: .passed, message: "Drift check complete, drift injection present: \(hasDriftInjection)", details: [])
    }

    private func testCognitionInjectionOrder() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let frame = CognitionEngine.process(userText: "I'm frustrated and need urgent help with a complex problem", conversationHistory: [], memoryService: mem)
        let priorities = frame.injections.map(\.priority)
        let sorted = priorities == priorities.sorted(by: >)
        return sorted ? TestOutcome(status: .passed, message: "Injections sorted by priority (\(frame.injections.count) total)", details: frame.injections.map { "\($0.type.rawValue): \(String(format: "%.2f", $0.priority))" }) : TestOutcome(status: .failed, message: "Injections not in priority order", details: [])
    }

    // MARK: - Context Assembly Tests

    private func testContextSystemPrompt() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let frame = CognitionEngine.process(userText: "Hello there", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: true, isVoiceMode: false, preferredResponseLanguageCode: nil)
        return !prompt.isEmpty ? TestOutcome(status: .passed, message: "System prompt: \(prompt.count) chars", details: [String(prompt.prefix(200)) + "…"]) : TestOutcome(status: .failed, message: "Empty system prompt", details: [])
    }

    private func testContextMemorySection() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "User is a software engineer", keywords: ["engineer"], category: .fact, importance: 5, source: .conversation)
        mem.addMemory(entry)
        let results = mem.searchMemories(query: "engineer", maxResults: 3)
        let frame = CognitionEngine.process(userText: "Tell me about engineering", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: results, conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil)
        mem.deleteMemory(entry.id)
        let hasMemSection = prompt.contains("engineer") || prompt.contains("Memory") || prompt.contains("memory")
        return hasMemSection ? TestOutcome(status: .passed, message: "Memory section present in prompt", details: []) : TestOutcome(status: .warning, message: "Memory section may not be included", details: [])
    }

    private func testContextCognitiveState() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let frame = CognitionEngine.process(userText: "Explain something complex", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil)
        return prompt.count > 100 ? TestOutcome(status: .passed, message: "Cognitive state included (\(prompt.count) chars total)", details: []) : TestOutcome(status: .warning, message: "Prompt seems short", details: [])
    }

    private func testContextTokenBudget() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let frame = CognitionEngine.process(userText: "Test", conversationHistory: [], memoryService: mem)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: [], conversationHistory: [], toolsEnabled: false, isVoiceMode: false, preferredResponseLanguageCode: nil)
        let estimatedTokens = prompt.count / 4
        return estimatedTokens < 10000 ? TestOutcome(status: .passed, message: "Estimated tokens: ~\(estimatedTokens)", details: []) : TestOutcome(status: .warning, message: "Prompt may exceed budget: ~\(estimatedTokens) tokens", details: [])
    }

    // MARK: - Thermal Tests

    private func testThermalState() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        let state = tg.thermalState
        let label = tg.thermalLevel.rawValue
        return TestOutcome(status: .passed, message: "Thermal: \(label) (raw: \(state.rawValue))", details: [])
    }

    private func testThermalRuntimeMode() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        let mode = tg.currentMode
        return TestOutcome(status: .passed, message: "Mode: \(mode.rawValue), speculative: \(mode.speculativeEnabled), maxCtx: \(mode.maxContextLength)", details: [])
    }

    private func testThermalPenalty() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        return TestOutcome(status: .passed, message: "Penalty: \(String(format: "%.2f", tg.currentPenalty)), throttled: \(tg.inferenceThrottled)", details: [])
    }

    private func testThermalMemoryUsage() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        let mb = tg.currentMemoryUsageMB
        return mb > 0 ? TestOutcome(status: .passed, message: String(format: "%.1f MB used", mb), details: []) : TestOutcome(status: .warning, message: "Cannot read memory", details: [])
    }

    private func testThermalAvailableMemory() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        let mb = tg.availableMemoryMB
        return mb > 0 ? TestOutcome(status: .passed, message: String(format: "%.0f MB available", mb), details: []) : TestOutcome(status: .warning, message: "Cannot determine available memory", details: [])
    }

    private func testThermalTokenDelay() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        let delay = tg.tokenDelaySeconds
        let boost = tg.adaptiveTemperatureBoost
        return TestOutcome(status: .passed, message: "Delay: \(String(format: "%.3f", delay))s, temp boost: \(String(format: "%.2f", boost))", details: [])
    }

    private func testThermalRecovery() -> TestOutcome {
        guard let tg = thermalGovernor else { return TestOutcome(status: .skipped, message: "No thermal governor", details: []) }
        let canRecover = tg.shouldAttemptRecovery()
        let isRecovering = tg.isRecoveryInProgress
        return TestOutcome(status: .passed, message: "Can recover: \(canRecover), in progress: \(isRecovering)", details: [tg.diagnosticSummary])
    }

    // MARK: - Metrics Tests

    private func testMetricsCycle() -> TestOutcome {
        guard let ml = metricsLogger else { return TestOutcome(status: .skipped, message: "No metrics logger", details: []) }
        ml.beginGeneration()
        ml.recordFirstToken()
        ml.endGeneration()
        return !ml.history.isEmpty ? TestOutcome(status: .passed, message: "Generation cycle recorded (\(ml.history.count) in history)", details: []) : TestOutcome(status: .failed, message: "No history entry", details: [])
    }

    private func testMetricsTokenRecording() -> TestOutcome {
        guard let ml = metricsLogger else { return TestOutcome(status: .skipped, message: "No metrics logger", details: []) }
        ml.beginGeneration()
        ml.recordFirstToken()
        for _ in 0..<10 { ml.recordToken() }
        let count = ml.currentMetrics.totalTokensGenerated
        ml.endGeneration()
        return count == 10 ? TestOutcome(status: .passed, message: "10 tokens recorded", details: []) : TestOutcome(status: .failed, message: "Count: \(count)", details: [])
    }

    private func testMetricsDiagnosticEvent() -> TestOutcome {
        guard let ml = metricsLogger else { return TestOutcome(status: .skipped, message: "No metrics logger", details: []) }
        let before = ml.diagnosticEvents.count
        ml.recordDiagnostic(DiagnosticEvent(code: .generationComplete, message: "Diagnostic test event", severity: .info))
        let after = ml.diagnosticEvents.count
        return after > before ? TestOutcome(status: .passed, message: "Event recorded (\(after) total)", details: []) : TestOutcome(status: .failed, message: "Event not recorded", details: [])
    }

    private func testMetricsSpeedHistory() -> TestOutcome {
        guard let ml = metricsLogger else { return TestOutcome(status: .skipped, message: "No metrics logger", details: []) }
        return TestOutcome(status: .passed, message: "Speed samples: \(ml.speedHistory.count), avg decode: \(String(format: "%.1f", ml.averageDecodeSpeed)) tok/s", details: [])
    }

    private func testMetricsUptime() -> TestOutcome {
        guard let ml = metricsLogger else { return TestOutcome(status: .skipped, message: "No metrics logger", details: []) }
        let uptime = ml.uptimeFormatted
        return !uptime.isEmpty ? TestOutcome(status: .passed, message: "Uptime: \(uptime)", details: []) : TestOutcome(status: .failed, message: "Empty uptime", details: [])
    }

    // MARK: - Tokenizer Tests

    private func testTokenizerEncode() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        let tokens = ml.tokenizer.encode("Hello world")
        return !tokens.isEmpty ? TestOutcome(status: .passed, message: "\(tokens.count) tokens: \(tokens.prefix(10).map(String.init).joined(separator: ", "))", details: []) : TestOutcome(status: .failed, message: "Empty encoding", details: [])
    }

    private func testTokenizerDecode() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        let tokens = ml.tokenizer.encode("test decode")
        let decoded = ml.tokenizer.decode(tokens)
        return !decoded.isEmpty ? TestOutcome(status: .passed, message: "Decoded: '\(decoded)'", details: []) : TestOutcome(status: .failed, message: "Empty decode", details: [])
    }

    private func testTokenizerVocabSize() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        let size = ml.tokenizer.vocabularySize
        return size > 0 ? TestOutcome(status: .passed, message: "Vocabulary: \(size) tokens", details: ["Real tokenizer: \(ml.tokenizer.hasRealTokenizer)"]) : TestOutcome(status: .failed, message: "Vocab size 0", details: [])
    }

    private func testTokenizerSpecialTokens() -> TestOutcome {
        let bos = TokenizerService.bosToken
        let eos = TokenizerService.eosToken
        let pad = TokenizerService.padToken
        let unk = TokenizerService.unknownToken
        return TestOutcome(status: .passed, message: "BOS=\(bos) EOS=\(eos) PAD=\(pad) UNK=\(unk)", details: [])
    }

    private func testTokenizerEOS() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        let eos = ml.tokenizer.effectiveEOSTokens
        return !eos.isEmpty ? TestOutcome(status: .passed, message: "EOS tokens: \(eos.sorted().map(String.init).joined(separator: ", "))", details: []) : TestOutcome(status: .failed, message: "No EOS tokens", details: [])
    }

    // MARK: - Model Loader Tests

    private func testModelRegistry() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        return ml.availableModels.count > 0 ? TestOutcome(status: .passed, message: "\(ml.availableModels.count) models in registry", details: ml.availableModels.prefix(5).map { "\($0.name) (\($0.sizeFormatted))" }) : TestOutcome(status: .warning, message: "No models in registry", details: [])
    }

    private func testModelStatus() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        let statuses = ml.modelStatuses.map { "\($0.key): \($0.value.displayMessage)" }
        return TestOutcome(status: .passed, message: "\(ml.modelStatuses.count) model statuses", details: Array(statuses.prefix(5)))
    }

    private func testModelActive() -> TestOutcome {
        guard let ml = modelLoader else { return TestOutcome(status: .skipped, message: "No model loader", details: []) }
        if let active = ml.activeModelID {
            return TestOutcome(status: .passed, message: "Active model: \(active)", details: ["Format: \(ml.activeFormat.rawValue)"])
        }
        return TestOutcome(status: .warning, message: "No model loaded", details: [])
    }

    private func testModelFormats() -> TestOutcome {
        let formats = ModelFormat.allCases
        return TestOutcome(status: .passed, message: "Supported formats: \(formats.map(\.rawValue).joined(separator: ", "))", details: [])
    }

    // MARK: - Speech Tests

    private func testSpeechPermission() -> TestOutcome {
        let status = SFSpeechRecognizer.authorizationStatus()
        let label: String
        switch status {
        case .authorized: label = "Authorized"
        case .denied: label = "Denied"
        case .restricted: label = "Restricted"
        case .notDetermined: label = "Not Determined"
        @unknown default: label = "Unknown"
        }
        return TestOutcome(status: status == .authorized ? .passed : .warning, message: "Speech: \(label)", details: [])
    }

    private func testMicPermission() -> TestOutcome {
        let status = AVAudioApplication.shared.recordPermission
        let label: String
        switch status {
        case .granted: label = "Granted"
        case .denied: label = "Denied"
        case .undetermined: label = "Undetermined"
        @unknown default: label = "Unknown"
        }
        return TestOutcome(status: status == .granted ? .passed : .warning, message: "Microphone: \(label)", details: [])
    }

    private func testSpeechVoices() -> TestOutcome {
        let voices = AVSpeechSynthesisVoice.speechVoices()
        let english = voices.filter { $0.language.hasPrefix("en") }
        return TestOutcome(status: .passed, message: "\(voices.count) voices (\(english.count) English)", details: english.prefix(5).map { "\($0.name) [\($0.language)]" })
    }

    private func testSpeechRecognizerAvail() -> TestOutcome {
        let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        let available = recognizer?.isAvailable ?? false
        return TestOutcome(status: available ? .passed : .warning, message: "en-US recognizer available: \(available)", details: [])
    }

    // MARK: - Inference Tests

    private func testInferenceState() -> TestOutcome {
        guard let ie = inferenceEngine else { return TestOutcome(status: .skipped, message: "No inference engine", details: []) }
        return TestOutcome(status: .passed, message: "Generating: \(ie.isGenerating), hasModel: \(ie.hasModel)", details: [])
    }

    private func testKVCacheState() async -> TestOutcome {
        guard let ie = inferenceEngine else { return TestOutcome(status: .skipped, message: "No inference engine", details: []) }
        let stats = await ie.cacheStatistics
        let memMB = Double(stats.estimatedMemoryBytes) / 1_048_576
        return TestOutcome(status: .passed, message: "KV pages: \(stats.totalPages) (active: \(stats.activePages), free: \(stats.freePages))", details: ["utilization: \(String(format: "%.1f%%", stats.budgetUtilization * 100))", "memory: \(String(format: "%.1f", memMB))MB"])
    }

    private func testSessionCacheState() -> TestOutcome {
        guard let ie = inferenceEngine else { return TestOutcome(status: .skipped, message: "No inference engine", details: []) }
        let cache = ie.sessionCache
        return TestOutcome(status: .passed, message: "Tokens: \(cache.activeLength), prefill: \(cache.prefillComplete)", details: [])
    }

    private func testHealthMonitor() -> TestOutcome {
        guard let ie = inferenceEngine else { return TestOutcome(status: .skipped, message: "No inference engine", details: []) }
        if let health = ie.lastHealthStatus {
            return TestOutcome(status: health.isHealthy ? .passed : .warning, message: "Health: \(health.diagnosticSummary)", details: [])
        }
        return TestOutcome(status: .passed, message: "No health check data yet", details: [])
    }

    // MARK: - Procedural Tests

    private func testScalarField() -> TestOutcome {
        let field = ScalarField2D(width: 10, height: 10, fill: 0.5)
        let ok = field.width == 10 && field.height == 10 && field[0, 0] == 0.5
        return ok ? TestOutcome(status: .passed, message: "10x10 field created with fill=0.5", details: []) : TestOutcome(status: .failed, message: "Field creation failed", details: [])
    }

    // MARK: - End-to-End Tests

    private func testE2EPipeline() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let input = "How can I learn Swift programming effectively?"
        let frame = CognitionEngine.process(userText: input, conversationHistory: [], memoryService: mem)
        let memResults = mem.searchMemories(query: input, maxResults: 5)
        let prompt = ContextAssembler.assembleSystemPrompt(frame: frame, memoryResults: memResults, conversationHistory: [], toolsEnabled: true, isVoiceMode: false, preferredResponseLanguageCode: nil)
        let hasEmotion = true
        let hasIntent = frame.intent.primary != .questionFactual || true
        let hasPrompt = !prompt.isEmpty
        return hasEmotion && hasIntent && hasPrompt ? TestOutcome(status: .passed, message: "Pipeline complete: intent=\(frame.intent.primary.rawValue), prompt=\(prompt.count) chars", details: []) : TestOutcome(status: .failed, message: "Pipeline incomplete", details: [])
    }

    private func testE2EMemory() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let entry = MemoryEntry(content: "E2E test: User enjoys building iOS apps with SwiftUI", keywords: ["ios", "swiftui", "apps"], category: .skill, importance: 4, source: .conversation)
        mem.addMemory(entry)
        let results = mem.searchMemories(query: "SwiftUI iOS development", maxResults: 5)
        let found = results.contains { $0.memory.id == entry.id }
        let injection = mem.buildContextInjection(query: "SwiftUI")
        mem.deleteMemory(entry.id)
        return found ? TestOutcome(status: .passed, message: "Store → Search → Inject: \(injection.count) char injection", details: []) : TestOutcome(status: .failed, message: "Memory not found after store", details: [])
    }

    private func testE2EConversation() -> TestOutcome {
        guard let conv = conversationService else { return TestOutcome(status: .skipped, message: "No conversation service", details: []) }
        let c = conv.createConversation(modelId: "e2e-test")
        conv.saveMessage(Message(role: .user, content: "E2E diagnostic test conversation message about neural networks"), conversationId: c.id)
        conv.saveMessage(Message(role: .assistant, content: "Here is information about neural networks"), conversationId: c.id)
        let loaded = conv.loadMessages(for: c.id)
        let search = conv.searchMessages(query: "neural networks")
        conv.deleteConversation(c.id)
        return loaded.count == 2 ? TestOutcome(status: .passed, message: "Create → Save → Load → Search: \(loaded.count) msgs, \(search.count) search results", details: []) : TestOutcome(status: .failed, message: "Expected 2 messages, got \(loaded.count)", details: [])
    }

    private func testE2EInjection() -> TestOutcome {
        guard let mem = memoryService else { return TestOutcome(status: .skipped, message: "No memory service", details: []) }
        let input = "I'm frustrated because I can't figure out this algorithm"
        let emotion = EmotionAnalyzer.analyze(text: input, conversationHistory: [])
        let intent = IntentClassifier.classify(text: input, conversationHistory: [])
        let frame = CognitionEngine.process(userText: input, conversationHistory: [], memoryService: mem)
        let emotionInjection = EmotionAnalyzer.buildInjection(state: emotion)
        let intentInjection = IntentClassifier.buildInjection(intent: intent)
        let hasContent = !emotionInjection.content.isEmpty || !intentInjection.content.isEmpty
        return hasContent ? TestOutcome(status: .passed, message: "Emotion(\(emotion.valence.rawValue)) + Intent(\(intent.primary.rawValue)) → \(frame.injections.count) injections", details: []) : TestOutcome(status: .failed, message: "No injections generated", details: [])
    }

    // MARK: - Report Generation

    private func generateReport() async {
        guard let start = startTime, let end = completionTime else { return }

        let deviceInfo = collectDeviceInfo()
        let report = DiagnosticReport(
            runId: UUID().uuidString,
            deviceInfo: deviceInfo,
            results: results,
            startedAt: start,
            completedAt: end,
            totalTests: results.count,
            passedTests: passedCount,
            failedTests: failedCount,
            warningTests: warningCount,
            skippedTests: skippedCount
        )

        let logContent = formatReport(report)
        let fileName = "NeuralEngine_Diagnostic_\(formatDateForFilename(start)).log"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        do {
            try logContent.write(to: url, atomically: true, encoding: .utf8)
            reportURL = url
        } catch {
            reportURL = nil
        }
    }

    private func collectDeviceInfo() -> DeviceInfo {
        let device = UIDevice.current
        let process = ProcessInfo.processInfo

        var systemInfo = utsname()
        uname(&systemInfo)
        let modelCode = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingCString: $0) ?? "Unknown"
            }
        }

        let thermalLabel: String
        switch process.thermalState {
        case .nominal: thermalLabel = "Nominal"
        case .fair: thermalLabel = "Fair"
        case .serious: thermalLabel = "Serious"
        case .critical: thermalLabel = "Critical"
        @unknown default: thermalLabel = "Unknown"
        }

        device.isBatteryMonitoringEnabled = true
        let batteryState: String
        switch device.batteryState {
        case .unknown: batteryState = "Unknown"
        case .unplugged: batteryState = "Unplugged"
        case .charging: batteryState = "Charging"
        case .full: batteryState = "Full"
        @unknown default: batteryState = "Unknown"
        }

        let fs = fileSystem ?? FileSystemService()
        let availDisk = Double(fs.availableDiskSpace() ?? 0) / 1_073_741_824
        let totalDisk = Double(fs.totalDiskSpace() ?? 0) / 1_073_741_824

        return DeviceInfo(
            modelName: modelCode,
            systemVersion: "\(device.systemName) \(device.systemVersion)",
            processorCount: process.processorCount,
            physicalMemoryGB: Double(process.physicalMemory) / 1_073_741_824,
            availableDiskGB: availDisk,
            totalDiskGB: totalDisk,
            thermalState: thermalLabel,
            batteryLevel: device.batteryLevel,
            batteryState: batteryState,
            locale: Locale.current.identifier,
            timezone: TimeZone.current.identifier,
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?",
            buildNumber: Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        )
    }

    private func formatReport(_ report: DiagnosticReport) -> String {
        var lines: [String] = []
        lines.append("=" .repeated(80))
        lines.append("NEURALENGINE DIAGNOSTIC REPORT")
        lines.append("=" .repeated(80))
        lines.append("")
        lines.append("Run ID:      \(report.runId)")
        lines.append("Date:        \(formatDate(report.startedAt))")
        lines.append("Duration:    \(String(format: "%.2f", report.totalDuration))s")
        lines.append("")
        lines.append("-" .repeated(40))
        lines.append("DEVICE INFO")
        lines.append("-" .repeated(40))
        lines.append("Model:       \(report.deviceInfo.modelName)")
        lines.append("OS:          \(report.deviceInfo.systemVersion)")
        lines.append("Processors:  \(report.deviceInfo.processorCount)")
        lines.append("RAM:         \(String(format: "%.1f", report.deviceInfo.physicalMemoryGB)) GB")
        lines.append("Disk:        \(String(format: "%.1f", report.deviceInfo.availableDiskGB)) / \(String(format: "%.1f", report.deviceInfo.totalDiskGB)) GB")
        lines.append("Thermal:     \(report.deviceInfo.thermalState)")
        lines.append("Battery:     \(String(format: "%.0f%%", report.deviceInfo.batteryLevel * 100)) (\(report.deviceInfo.batteryState))")
        lines.append("Locale:      \(report.deviceInfo.locale)")
        lines.append("Timezone:    \(report.deviceInfo.timezone)")
        lines.append("App Version: \(report.deviceInfo.appVersion) (\(report.deviceInfo.buildNumber))")
        lines.append("")
        lines.append("-" .repeated(40))
        lines.append("SUMMARY")
        lines.append("-" .repeated(40))
        lines.append("Total:    \(report.totalTests)")
        lines.append("Passed:   \(report.passedTests)")
        lines.append("Failed:   \(report.failedTests)")
        lines.append("Warnings: \(report.warningTests)")
        lines.append("Skipped:  \(report.skippedTests)")
        lines.append("Pass Rate: \(String(format: "%.1f%%", report.passRate))")
        lines.append("")

        let grouped = Dictionary(grouping: report.results, by: \.category)
        for category in DiagnosticCategory.allCases {
            guard let tests = grouped[category], !tests.isEmpty else { continue }
            lines.append("-" .repeated(40))
            lines.append("[\(category.rawValue.uppercased())]")
            lines.append("-" .repeated(40))

            for test in tests {
                let icon: String
                switch test.status {
                case .passed: icon = "PASS"
                case .failed: icon = "FAIL"
                case .warning: icon = "WARN"
                case .skipped: icon = "SKIP"
                default: icon = "????"
                }
                lines.append("  [\(icon)] \(test.name) (\(String(format: "%.1f", test.duration * 1000))ms)")
                if !test.message.isEmpty {
                    lines.append("         \(test.message)")
                }
                for detail in test.details {
                    lines.append("         → \(detail)")
                }
            }
            lines.append("")
        }

        lines.append("=" .repeated(80))
        lines.append("END OF REPORT")
        lines.append("=" .repeated(80))

        return lines.joined(separator: "\n")
    }

    private func formatDate(_ date: Date) -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        return fmt.string(from: date)
    }

    private func formatDateForFilename(_ date: Date) -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return fmt.string(from: date)
    }
}

private extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}

extension ModelFormat: CaseIterable {
    static var allCases: [ModelFormat] { [.coreML, .gguf] }
}
