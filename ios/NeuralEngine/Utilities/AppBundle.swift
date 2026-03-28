import Foundation

extension Bundle {
    static var neuralEngineResources: Bundle {
#if SWIFT_PACKAGE
        .module
#else
        .main
#endif
    }
}
