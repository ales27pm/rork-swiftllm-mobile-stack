import Foundation

struct ToolCallParser {
    static let toolCallPattern = #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#
    static let functionCallPattern = #"\[TOOL_CALL\]\s*(\{.*?\})\s*\[/TOOL_CALL\]"#
    static let jsonCallPattern = #"\{"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\}"#

    static func parse(from text: String) -> [ToolCall] {
        var calls: [ToolCall] = []

        let patterns = [toolCallPattern, functionCallPattern]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { continue }
            let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
            for match in matches {
                guard let jsonRange = Range(match.range(at: 1), in: text) else { continue }
                let jsonStr = String(text[jsonRange])
                if let call = parseJSON(jsonStr) {
                    calls.append(call)
                }
            }
        }

        if calls.isEmpty {
            if let regex = try? NSRegularExpression(pattern: jsonCallPattern, options: [.dotMatchesLineSeparators]) {
                let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
                for match in matches {
                    guard let nameRange = Range(match.range(at: 1), in: text),
                          let paramsRange = Range(match.range(at: 2), in: text) else { continue }
                    let name = String(text[nameRange])
                    let paramsStr = String(text[paramsRange])
                    if DeviceToolName(rawValue: name) != nil {
                        let params = parseParams(paramsStr)
                        calls.append(ToolCall(name: name, parameters: params))
                    }
                }
            }
        }

        return calls
    }

    static func stripToolCalls(from text: String) -> String {
        var result = text
        let patterns = [
            #"<tool_call>\s*\{.*?\}\s*</tool_call>"#,
            #"\[TOOL_CALL\]\s*\{.*?\}\s*\[/TOOL_CALL\]"#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { continue }
            result = regex.stringByReplacingMatches(in: result, range: NSRange(result.startIndex..., in: result), withTemplate: "")
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func containsToolCall(_ text: String) -> Bool {
        !parse(from: text).isEmpty
    }

    private static func parseJSON(_ jsonStr: String) -> ToolCall? {
        guard let data = jsonStr.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String,
              DeviceToolName(rawValue: name) != nil else { return nil }

        let params = obj["parameters"] as? [String: Any] ?? obj["arguments"] as? [String: Any] ?? [:]
        return ToolCall(name: name, parameters: params)
    }

    private static func parseParams(_ str: String) -> [String: Any] {
        guard let data = str.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return obj
    }
}
