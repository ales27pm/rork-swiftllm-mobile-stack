import Foundation

extension Foundation.Bundle {
    static let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("NeuralEngine_NeuralEngine.bundle").path
        let buildPath = "/home/ales27pm/rork-swiftllm-mobile-stack/ios/NeuralEngine/NeuralEngine/.verify-full-app/arm64-apple-ios/debug/NeuralEngine_NeuralEngine.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}