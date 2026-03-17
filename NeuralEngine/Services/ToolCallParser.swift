import Foundation

struct ToolCallParser {
    static let toolCallPattern = #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#
    static let functionCallPattern = #"\[TOOL_CALL\]\s*(\{.*?\})\s*\[/TOOL_CALL\]"#
    static let batchedToolCallsPattern = #"<tool_calls>\s*(\[.*?\])\s*</tool_calls>"#
    static let jsonCallPattern = #"\{"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\}"#

    static func parse(from text: String) -> [ToolCall] {
        var calls = extractSingleCalls(from: text)
        calls.append(contentsOf: extractBatchedCalls(from: text))

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

        return deduplicated(calls)
    }

    static func stripToolCalls(from text: String) -> String {
        var result = text
        let patterns = [
            #"<tool_call>\s*\{.*?\}\s*</tool_call>"#,
            #"\[TOOL_CALL\]\s*\{.*?\}\s*\[/TOOL_CALL\]"#,
            #"<tool_calls>\s*\[.*?\]\s*</tool_calls>"#
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

    private static func extractSingleCalls(from text: String) -> [ToolCall] {
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

        return calls
    }

    private static func extractBatchedCalls(from text: String) -> [ToolCall] {
        guard let regex = try? NSRegularExpression(pattern: batchedToolCallsPattern, options: [.dotMatchesLineSeparators]) else {
            return []
        }

        var calls: [ToolCall] = []
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        for match in matches {
            guard let jsonRange = Range(match.range(at: 1), in: text) else { continue }
            let jsonStr = String(text[jsonRange])
            guard let data = jsonStr.data(using: .utf8),
                  let payload = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                continue
            }

            for entry in payload {
                guard let name = entry["name"] as? String,
                      DeviceToolName(rawValue: name) != nil else { continue }
                let params = entry["parameters"] as? [String: Any] ?? entry["arguments"] as? [String: Any] ?? [:]
                calls.append(ToolCall(name: name, parameters: params))
            }
        }
        return calls
    }

    private static func deduplicated(_ calls: [ToolCall]) -> [ToolCall] {
        var seen: Set<String> = []
        var unique: [ToolCall] = []

        for call in calls {
            let signature = "\(call.name)::\(serializedParams(call.parameters))"
            guard !seen.contains(signature) else { continue }
            seen.insert(signature)
            unique.append(call)
        }

        return unique
    }

    private static func serializedParams(_ params: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(params),
              let data = try? JSONSerialization.data(withJSONObject: params, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }

    private static func parseParams(_ str: String) -> [String: Any] {
        guard let data = str.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return obj
    }
}
