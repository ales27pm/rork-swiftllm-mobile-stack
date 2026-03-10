import SwiftUI

@main
struct NeuralEngineApp: App {
    @State private var metricsLogger = MetricsLogger()
    @State private var thermalGovernor = ThermalGovernor()
    @State private var modelLoader = ModelLoaderService()
    @State private var keyValueStore = KeyValueStore(suiteName: "com.neuralengine.storage")

    private let secureStore = SecureStore()
    private let fileSystem = FileSystemService()
    private let database = DatabaseService()

    var body: some Scene {
        WindowGroup {
            ContentView(
                metricsLogger: metricsLogger,
                thermalGovernor: thermalGovernor,
                modelLoader: modelLoader,
                keyValueStore: keyValueStore,
                secureStore: secureStore,
                fileSystem: fileSystem,
                database: database
            )
            .onAppear {
                thermalGovernor.startMonitoring()
                initializeDatabase()
            }
        }
    }

    private func initializeDatabase() {
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                model_id TEXT
            );
        """)
        _ = database.execute("""
            CREATE TABLE IF NOT EXISTS generation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                prompt_tokens INTEGER,
                generated_tokens INTEGER,
                prefill_tps REAL,
                decode_tps REAL,
                time_to_first_token REAL,
                total_duration REAL,
                thermal_state TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            );
        """)
    }
}
