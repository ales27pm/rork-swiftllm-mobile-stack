import SwiftUI

@main
struct NeuralEngineApp: App {
    @State private var metricsLogger = MetricsLogger()
    @State private var thermalGovernor = ThermalGovernor()
    @State private var modelLoader = ModelLoaderService()

    var body: some Scene {
        WindowGroup {
            ContentView(
                metricsLogger: metricsLogger,
                thermalGovernor: thermalGovernor,
                modelLoader: modelLoader
            )
            .onAppear {
                thermalGovernor.startMonitoring()
            }
        }
    }
}
