import Foundation
import os

struct LocationProvenance: Equatable, Sendable {
    let source: String
    let timestamp: Date
    let confidence: Double
}

struct VerifiedLocationSignal: Equatable, Sendable {
    let latitude: Double
    let longitude: Double
    let address: String?
    let provenance: LocationProvenance
}

struct LocationResponseComposer {
    private static let logger = Logger(subsystem: "NeuralEngine", category: "LocationResponseComposer")
    private static let staleThresholdSeconds: TimeInterval = 120
    private static let directLocationQueryTokens = ["where am i right now", "where am i", "current location", "my location", "locate me"]

    static func composeResponseIfNeeded(userText: String, assistantDraft: String, messages: [Message], now: Date = Date()) -> String {
        guard isLocationQuery(userText) else { return assistantDraft }

        switch resolveSignal(from: messages, now: now) {
        case let .success(signal):
            return renderVerifiedLocationResponse(signal: signal)
        case let .failure(reason):
            logger.warning("Location fallback used. reason=\(reason.rawValue, privacy: .public)")
            return renderFallback(reason: reason)
        }
    }

    private static func isLocationQuery(_ text: String) -> Bool {
        let lowered = text.lowercased()
        return directLocationQueryTokens.contains(where: lowered.contains)
    }

    private static func resolveSignal(from messages: [Message], now: Date) -> Result<VerifiedLocationSignal, FallbackReason> {
        let locationResults = messages
            .reversed()
            .filter { $0.role == .tool }
            .flatMap(\.toolResults)
            .filter { $0.toolName == DeviceToolName.getLocation.rawValue }

        guard let latest = locationResults.first else { return .failure(.noSignal) }

        if !latest.success {
            if latest.data.lowercased().contains("permission") {
                return .failure(.missingPermission)
            }
            return .failure(.noSignal)
        }

        guard let payload = parsePayload(from: latest.data) else { return .failure(.noSignal) }
        guard payload.permissionGranted else { return .failure(.missingPermission) }
        guard let latitude = payload.latitude, let longitude = payload.longitude else { return .failure(.missingCoordinates) }
        guard let timestamp = payload.timestamp else { return .failure(.noSignal) }
        guard now.timeIntervalSince(timestamp) <= staleThresholdSeconds else { return .failure(.staleData) }
        guard let source = payload.source, let confidence = payload.confidence else { return .failure(.noSignal) }

        return .success(
            VerifiedLocationSignal(
                latitude: latitude,
                longitude: longitude,
                address: payload.address,
                provenance: LocationProvenance(source: source, timestamp: timestamp, confidence: confidence)
            )
        )
    }

    private static func renderVerifiedLocationResponse(signal: VerifiedLocationSignal) -> String {
        let ts = ISO8601DateFormatter().string(from: signal.provenance.timestamp)
        var response = "You are currently located at latitude \(signal.latitude), longitude \(signal.longitude)."
        if let address = signal.address, !address.isEmpty {
            response += " Nearby address: \(address)."
        }
        response += "\n\nProvenance: source=\(signal.provenance.source), timestamp=\(ts), confidence=\(String(format: "%.2f", signal.provenance.confidence))."
        return response
    }

    private static func renderFallback(reason: FallbackReason) -> String {
        let reasonText: String
        switch reason {
        case .missingPermission: reasonText = "location permission is not granted"
        case .staleData: reasonText = "the last location signal is stale"
        case .missingCoordinates: reasonText = "location coordinates are missing"
        case .noSignal: reasonText = "no verified live location signal is available"
        }

        return """
        Live location is currently unavailable because \(reasonText).
        To get your current location, please:
        1. Enable location permission for this app in Settings.
        2. Refresh location by asking me to check device GPS again.
        """
    }

    private static func parsePayload(from json: String) -> LocationPayload? {
        guard let data = json.data(using: .utf8) else {
            return nil
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(LocationPayload.self, from: data)
    }
}

private extension LocationResponseComposer {
  enum FallbackReason: String, Error {
        case missingPermission
        case staleData
        case missingCoordinates
        case noSignal
}

    struct LocationPayload: Codable {
        let latitude: Double?
        let longitude: Double?
        let address: String?
        let timestamp: Date?
        let permissionGranted: Bool
        let source: String?
        let confidence: Double?
    }
        }

}
