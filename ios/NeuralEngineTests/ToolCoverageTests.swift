import Foundation
import Testing
@testable import NeuralEngine

struct ToolCoverageTests {
    @Test(arguments: DeviceToolName.allCases)
    func toolMetadata_isPopulatedAndRegisteredInPrompt(_ tool: DeviceToolName) {
        #expect(!tool.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!tool.parametersSchema.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(ToolExecutor.buildToolsPrompt().contains("- \(tool.rawValue): \(tool.description) Parameters: \(tool.parametersSchema)"))
    }

    @Test(arguments: DeviceToolName.allCases)
    func parser_acceptsSingleXMLCallForEveryRegisteredTool(_ tool: DeviceToolName) {
        let payload = #"<tool_call>{\"name\":\""# + tool.rawValue + #"\",\"parameters\":{}}</tool_call>"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.count == 1)
        #expect(calls[0].name == tool.rawValue)
        #expect(calls[0].parameters.isEmpty)
        #expect(ToolCallParser.containsToolCall(payload))
    }

    @Test(arguments: DeviceToolName.allCases)
    func parser_acceptsLegacyBracketedCallForEveryRegisteredTool(_ tool: DeviceToolName) {
        let payload = #"[TOOL_CALL]{\"name\":\""# + tool.rawValue + #"\",\"parameters\":{}}</tool_call>"#
        let normalizedPayload = payload.replacingOccurrences(of: "</tool_call>", with: "[/TOOL_CALL]")
        let calls = ToolCallParser.parse(from: normalizedPayload)

        #expect(calls.count == 1)
        #expect(calls[0].name == tool.rawValue)
        #expect(calls[0].parameters.isEmpty)
        #expect(ToolCallParser.containsToolCall(normalizedPayload))
    }

    @Test func parser_detectsRawJSONCallWithoutEnvelope() {
        let payload = #"prefix {"name":"web_search","parameters":{"query":"swift testing"}} suffix"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.count == 1)
        #expect(calls[0].name == DeviceToolName.webSearch.rawValue)
        #expect((calls[0].parameters["query"] as? String) == "swift testing")
        #expect(ToolCallParser.containsToolCall(payload))
    }

    @Test func parser_deduplicatesEquivalentCallsWhenParameterOrderDiffers() {
        let payload = #"<tool_calls>[{"name":"open_maps","parameters":{"query":"coffee","latitude":46.0}},{"name":"open_maps","parameters":{"latitude":46.0,"query":"coffee"}}]</tool_calls>"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.count == 1)
        #expect(calls[0].name == DeviceToolName.openMaps.rawValue)
        #expect((calls[0].parameters["query"] as? String) == "coffee")
        #expect((calls[0].parameters["latitude"] as? Double) == 46.0)
    }

    @Test func stripToolCalls_removesLegacyBracketedPayload() {
        let text = #"Before [TOOL_CALL]{"name":"get_current_time","parameters":{}}[/TOOL_CALL] After"#
        let stripped = ToolCallParser.stripToolCalls(from: text)

        #expect(stripped == "Before  After")
    }

    @Test func buildToolsPrompt_includesCallingContractRules() {
        let prompt = ToolExecutor.buildToolsPrompt()

        #expect(prompt.contains("[Tool Calling Contract]"))
        #expect(prompt.contains("<tool_call>{\"name\":\"tool_name\",\"parameters\":{}}</tool_call>"))
        #expect(prompt.contains("<tool_calls>[{\"name\":\"tool_a\",\"parameters\":{}},{\"name\":\"tool_b\",\"parameters\":{}}]</tool_calls>"))
        #expect(prompt.contains("Use ONLY tool names from the list below."))
        #expect(prompt.contains("Do not include markdown code fences around tool calls."))
        #expect(prompt.contains("After receiving tool results, synthesize a concise user-facing answer."))
    }

    @Test func toolCall_defaultsToEmptyParameters() {
        let call = ToolCall(name: DeviceToolName.getCurrentTime.rawValue)

        #expect(call.name == DeviceToolName.getCurrentTime.rawValue)
        #expect(call.parameters.isEmpty)
    }

    @Test func toolResult_defaultsToWrenchFillIcon() {
        let result = ToolResult(toolName: DeviceToolName.getCurrentTime.rawValue, success: true, data: "{}")

        #expect(result.displayIcon == "wrench.fill")
    }

    @MainActor
    @Test(arguments: [
        DeviceToolName.setScreenBrightness,
        .sendSMS,
        .sendEmail,
        .shareContent,
        .createCalendarEvent,
        .scheduleNotification,
        .openMaps,
        .webSearch,
        .fetchURL,
        .openURL
    ])
    func execute_parameterValidatedToolsRejectMissingOrInvalidPayload(_ tool: DeviceToolName) async {
        let executor = ToolExecutor()
        let call: ToolCall
        let expectedFragment: String

        switch tool {
        case .setScreenBrightness:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'level'"
        case .sendSMS:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'to' or 'body'"
        case .sendEmail:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'to'"
        case .shareContent:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'text'"
        case .createCalendarEvent:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing required parameters"
        case .scheduleNotification:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'title' or 'body'"
        case .openMaps:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'query' or 'latitude'/'longitude'"
        case .webSearch:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'query'"
        case .fetchURL:
            call = ToolCall(name: tool.rawValue)
            expectedFragment = "Missing 'url'"
        case .openURL:
            call = ToolCall(name: tool.rawValue, parameters: ["url": "not a valid url"])
            expectedFragment = "Missing or invalid 'url'"
        default:
            #expect(Bool(false))
            return
        }

        let result = await executor.execute(call)

        #expect(!result.success)
        #expect(result.toolName == tool.rawValue)
        #expect(result.data.contains(expectedFragment))
    }

    @MainActor
    @Test func execute_unknownTool_returnsFailure() async {
        let executor = ToolExecutor()
        let result = await executor.execute(ToolCall(name: "not_a_real_tool"))

        #expect(!result.success)
        #expect(result.toolName == "not_a_real_tool")
        #expect(result.data.contains("Unknown tool"))
        #expect(result.displayIcon == "exclamationmark.triangle.fill")
    }

    @MainActor
    @Test func execute_openURL_setsBrowserStateAndUsesExplicitTitle() async {
        let executor = ToolExecutor()
        let result = await executor.execute(ToolCall(name: DeviceToolName.openURL.rawValue, parameters: [
            "url": "https://example.com/docs",
            "title": "Docs"
        ]))
        let payload = Self.parseJSONObject(result.data)

        #expect(result.success)
        #expect(executor.showInAppBrowser)
        #expect(executor.browserURL?.absoluteString == "https://example.com/docs")
        #expect(executor.browserTitle == "Docs")
        #expect((payload["status"] as? String) == "browser_opened")
        #expect((payload["url"] as? String) == "https://example.com/docs")
    }

    @MainActor
    @Test func execute_openURL_usesHostFallbackTitleWhenTitleIsMissing() async {
        let executor = ToolExecutor()
        let result = await executor.execute(ToolCall(name: DeviceToolName.openURL.rawValue, parameters: [
            "url": "https://developer.apple.com/documentation"
        ]))

        #expect(result.success)
        #expect(executor.browserTitle == "developer.apple.com")
        #expect(executor.showInAppBrowser)
    }

    @MainActor
    @Test func execute_shareContent_setsShareSheetState() async {
        let executor = ToolExecutor()
        let result = await executor.execute(ToolCall(name: DeviceToolName.shareContent.rawValue, parameters: [
            "text": "Hello from NeuralEngine"
        ]))
        let payload = Self.parseJSONObject(result.data)

        #expect(result.success)
        #expect(executor.showShareSheet)
        #expect(executor.shareItems.count == 1)
        #expect((executor.shareItems.first as? String) == "Hello from NeuralEngine")
        #expect((payload["status"] as? String) == "share_sheet_opened")
    }

    @MainActor
    @Test func resetComposerState_clearsPendingSMSAndEmailValues() {
        let executor = ToolExecutor()
        executor.pendingSMSTo = "5551234567"
        executor.pendingSMSBody = "Hello"
        executor.showSMSComposer = true
        executor.pendingEmailTo = "hello@example.com"
        executor.pendingEmailSubject = "Subject"
        executor.pendingEmailBody = "Body"
        executor.showEmailComposer = true

        executor.resetSMSComposerState()
        executor.resetEmailComposerState()

        #expect(executor.pendingSMSTo == nil)
        #expect(executor.pendingSMSBody == nil)
        #expect(!executor.showSMSComposer)
        #expect(executor.pendingEmailTo == nil)
        #expect(executor.pendingEmailSubject == nil)
        #expect(executor.pendingEmailBody == nil)
        #expect(!executor.showEmailComposer)
    }
}

private extension ToolCoverageTests {
    static func parseJSONObject(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }

        return object
    }
}
