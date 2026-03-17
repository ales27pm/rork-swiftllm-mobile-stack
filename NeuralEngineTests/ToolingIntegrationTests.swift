import Testing
@testable import NeuralEngine

struct ToolingIntegrationTests {
    @Test func toolPrompt_listsEveryRegisteredTool() {
        let prompt = ToolExecutor.buildToolsPrompt()

        for tool in DeviceToolName.allCases {
            #expect(prompt.contains(tool.rawValue))
            #expect(!tool.description.isEmpty)
            #expect(!tool.parametersSchema.isEmpty)
        }
    }

    @Test func parser_rejectsUnknownToolCalls() {
        let payload = #"<tool_call>{\"name\":\"not_a_real_tool\",\"parameters\":{}}</tool_call>"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.isEmpty)
    }

    @Test func parser_acceptsKnownToolWithArgumentsAlias() {
        let payload = #"<tool_call>{\"name\":\"open_maps\",\"arguments\":{\"latitude\":46.006164,\"longitude\":-73.1645294}}</tool_call>"#
        let calls = ToolCallParser.parse(from: payload)

        #expect(calls.count == 1)
        #expect(calls[0].name == DeviceToolName.openMaps.rawValue)
        #expect((calls[0].parameters["latitude"] as? Double) == 46.006164)
        #expect((calls[0].parameters["longitude"] as? Double) == -73.1645294)
    }
}
