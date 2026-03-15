import Foundation

nonisolated enum ErrorDomain: String, Sendable {
    case coreML = "CoreML"
    case inference = "Inference"
    case speech = "Speech"
    case thermal = "Thermal"
    case model = "Model"
    case fileSystem = "FileSystem"
    case unknown = "Unknown"
}

nonisolated enum ErrorSeverity: Int, Sendable, Comparable {
    case info = 0
    case warning = 1
    case error = 2
    case critical = 3

    static func < (lhs: ErrorSeverity, rhs: ErrorSeverity) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

nonisolated struct WrappedError: Sendable, Identifiable {
    let id: UUID
    let domain: ErrorDomain
    let severity: ErrorSeverity
    let userMessage: String
    let technicalDetail: String
    let recoveryAction: RecoveryAction
    let timestamp: Date
    let underlyingError: String?

    init(
        domain: ErrorDomain,
        severity: ErrorSeverity,
        userMessage: String,
        technicalDetail: String,
        recoveryAction: RecoveryAction = .none,
        underlyingError: Error? = nil
    ) {
        self.id = UUID()
        self.domain = domain
        self.severity = severity
        self.userMessage = userMessage
        self.technicalDetail = technicalDetail
        self.recoveryAction = recoveryAction
        self.timestamp = Date()
        self.underlyingError = underlyingError?.localizedDescription
    }
}

nonisolated enum RecoveryAction: String, Sendable {
    case none
    case retry
    case reloadModel
    case reduceContext
    case switchToCPU
    case restartSession
    case clearCache
}

enum NativeErrorWrapper {
    static func synthesize(_ error: Error) -> WrappedError {
        if let coreMLError = error as? CoreMLRunnerError {
            return wrapCoreMLError(coreMLError)
        }

        if let loaderError = error as? ModelLoaderError {
            return wrapLoaderError(loaderError)
        }

        let nsError = error as NSError

        if nsError.domain == "com.apple.CoreML" {
            return wrapRawCoreMLError(nsError)
        }

        if nsError.domain == "kAFAssistantErrorDomain" {
            return wrapSpeechError(nsError)
        }

        return WrappedError(
            domain: .unknown,
            severity: .error,
            userMessage: "An unexpected error occurred.",
            technicalDetail: "\(nsError.domain):\(nsError.code) \(nsError.localizedDescription)",
            recoveryAction: .retry,
            underlyingError: error
        )
    }

    private static func wrapCoreMLError(_ error: CoreMLRunnerError) -> WrappedError {
        switch error {
        case .modelNotLoaded:
            return WrappedError(
                domain: .coreML,
                severity: .error,
                userMessage: "No model is loaded. Please select a model first.",
                technicalDetail: "CoreMLModelRunner state: no model reference",
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .modelEvicted:
            return WrappedError(
                domain: .coreML,
                severity: .critical,
                userMessage: "The model was reclaimed by the system. Recovering...",
                technicalDetail: "ANE/GPU eviction detected — Ghost Model state",
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .invalidState(let state):
            return WrappedError(
                domain: .coreML,
                severity: .warning,
                userMessage: "Model is busy (\(state.rawValue)). Please wait.",
                technicalDetail: "Runner state: \(state.rawValue)",
                recoveryAction: .retry,
                underlyingError: error
            )
        case .emptyInput:
            return WrappedError(
                domain: .inference,
                severity: .warning,
                userMessage: "Empty input provided.",
                technicalDetail: "predictLogits called with 0 tokens",
                recoveryAction: .none,
                underlyingError: error
            )
        case .invalidOutput(let keys):
            return WrappedError(
                domain: .coreML,
                severity: .error,
                userMessage: "Model output format mismatch.",
                technicalDetail: "Available output keys: \(keys)",
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .compilationFailed(let reason):
            return WrappedError(
                domain: .coreML,
                severity: .critical,
                userMessage: "Model compilation failed. Try re-downloading.",
                technicalDetail: reason,
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .recoveryFailed(let reason):
            return WrappedError(
                domain: .coreML,
                severity: .critical,
                userMessage: "Could not recover the model. Try restarting the app.",
                technicalDetail: reason,
                recoveryAction: .restartSession,
                underlyingError: error
            )
        }
    }

    private static func wrapLoaderError(_ error: ModelLoaderError) -> WrappedError {
        switch error {
        case .invalidPackage(let msg):
            return WrappedError(
                domain: .model,
                severity: .error,
                userMessage: "Model package is corrupted. Delete and re-download.",
                technicalDetail: msg,
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .noModelFound(let msg):
            return WrappedError(
                domain: .model,
                severity: .error,
                userMessage: "No compatible model found in the download.",
                technicalDetail: msg,
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .compilationFailed(let msg):
            return WrappedError(
                domain: .model,
                severity: .critical,
                userMessage: "Failed to compile the model for this device.",
                technicalDetail: msg,
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .integrityCheckFailed(let msg):
            return WrappedError(
                domain: .fileSystem,
                severity: .critical,
                userMessage: "Model integrity check failed. Delete and re-download.",
                technicalDetail: msg,
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .checksumMismatch(let expected, let actual):
            return WrappedError(
                domain: .fileSystem,
                severity: .critical,
                userMessage: "Model file checksum mismatch. The download may be corrupted.",
                technicalDetail: "Expected \(expected.prefix(16))… got \(actual.prefix(16))…",
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        case .assetRepairFailed(let msg):
            return WrappedError(
                domain: .fileSystem,
                severity: .critical,
                userMessage: "Automatic repair failed. Please delete and re-download the model.",
                technicalDetail: msg,
                recoveryAction: .clearCache,
                underlyingError: error
            )
        case .partialDownload(let msg):
            return WrappedError(
                domain: .fileSystem,
                severity: .error,
                userMessage: "Download appears incomplete. Please retry.",
                technicalDetail: msg,
                recoveryAction: .reloadModel,
                underlyingError: error
            )
        }
    }

    private static func wrapRawCoreMLError(_ nsError: NSError) -> WrappedError {
        let evictionCodes: Set<Int> = [-1, -2, -4, -6, 11]

        if evictionCodes.contains(nsError.code) {
            return WrappedError(
                domain: .coreML,
                severity: .critical,
                userMessage: "Hardware resources reclaimed. Recovering...",
                technicalDetail: "CoreML error \(nsError.code): probable ANE eviction",
                recoveryAction: .switchToCPU,
                underlyingError: nsError
            )
        }

        return WrappedError(
            domain: .coreML,
            severity: .error,
            userMessage: "A CoreML error occurred.",
            technicalDetail: "Code \(nsError.code): \(nsError.localizedDescription)",
            recoveryAction: .retry,
            underlyingError: nsError
        )
    }

    private static func wrapSpeechError(_ nsError: NSError) -> WrappedError {
        if nsError.code == 1110 {
            return WrappedError(
                domain: .speech,
                severity: .info,
                userMessage: "No speech detected.",
                technicalDetail: "kAFAssistantErrorDomain code 1110: no speech input",
                recoveryAction: .retry,
                underlyingError: nsError
            )
        }

        return WrappedError(
            domain: .speech,
            severity: .warning,
            userMessage: "Speech recognition encountered an issue.",
            technicalDetail: "Code \(nsError.code): \(nsError.localizedDescription)",
            recoveryAction: .retry,
            underlyingError: nsError
        )
    }
}
