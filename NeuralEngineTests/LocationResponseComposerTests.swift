import Foundation
import Testing
@testable import NeuralEngine

struct LocationResponseComposerTests {
    @Test func locationQuery_withoutPermission_returnsSafeFallback() {
        let messages = [
            Message(role: .tool, content: "", toolResults: [
                ToolResult(
                    toolName: DeviceToolName.getLocation.rawValue,
                    success: false,
                    data: #"{"permissionGranted":false,"error":"location_permission_not_granted"}"#
                )
            ], isToolExecution: true)
        ]

        let result = LocationResponseComposer.composeResponseIfNeeded(
            userText: "Where am I right now?",
            assistantDraft: "Draft",
            messages: messages,
            now: Date()
        )

        #expect(result.contains("Live location is currently unavailable"))
        #expect(result.contains("permission"))
        #expect(result.contains("Enable location permission"))
        #expect(result.contains("Refresh location"))
    }

    @Test func locationQuery_withStaleLocation_returnsSafeFallback() {
        let staleTimestamp = ISO8601DateFormatter().string(from: Date().addingTimeInterval(-600))
        let messages = [
            Message(role: .tool, content: "", toolResults: [
                ToolResult(
                    toolName: DeviceToolName.getLocation.rawValue,
                    success: true,
                    data: """
                    {"latitude": 40.0, "longitude": -70.0, "address": "", "timestamp": "\(staleTimestamp)", "permissionGranted": true, "source": "core_location", "confidence": 0.91}
                    """
                )
            ], isToolExecution: true)
        ]

        let result = LocationResponseComposer.composeResponseIfNeeded(
            userText: "Where am I right now?",
            assistantDraft: "Draft",
            messages: messages,
            now: Date()
        )

        #expect(result.contains("Live location is currently unavailable"))
        #expect(result.contains("stale"))
    }

    @Test func locationQuery_withMissingCoordinates_returnsSafeFallback() {
        let freshTimestamp = ISO8601DateFormatter().string(from: Date())
        let messages = [
            Message(role: .tool, content: "", toolResults: [
                ToolResult(
                    toolName: DeviceToolName.getLocation.rawValue,
                    success: true,
                    data: """
                    {"timestamp": "\(freshTimestamp)", "permissionGranted": true, "source": "core_location", "confidence": 0.88}
                    """
                )
            ], isToolExecution: true)
        ]

        let result = LocationResponseComposer.composeResponseIfNeeded(
            userText: "Where am I right now?",
            assistantDraft: "Draft",
            messages: messages,
            now: Date()
        )

        #expect(result.contains("Live location is currently unavailable"))
        #expect(result.contains("coordinates are missing"))
    }

    @Test func locationQuery_withValidFreshLocation_returnsVerifiedClaimWithProvenance() {
        let freshTimestamp = ISO8601DateFormatter().string(from: Date())
        let messages = [
            Message(role: .tool, content: "", toolResults: [
                ToolResult(
                    toolName: DeviceToolName.getLocation.rawValue,
                    success: true,
                    data: """
                    {"latitude": 46.006164, "longitude": -73.1645294, "address": "Montreal", "timestamp": "\(freshTimestamp)", "permissionGranted": true, "source": "core_location", "confidence": 0.93}
                    """
                )
            ], isToolExecution: true)
        ]

        let result = LocationResponseComposer.composeResponseIfNeeded(
            userText: "Where am I right now?",
            assistantDraft: "Draft",
            messages: messages,
            now: Date()
        )

        #expect(result.contains("You are currently located"))
        #expect(result.contains("46.006164"))
        #expect(result.contains("Provenance:"))
        #expect(result.contains("source=core_location"))
        #expect(result.contains("confidence=0.93"))
    }
}
