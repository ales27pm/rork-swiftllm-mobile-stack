import Foundation

nonisolated enum DeviceToolName: String, CaseIterable, Sendable {
    case getLocation = "get_location"
    case getBatteryStatus = "get_battery_status"
    case getNetworkInfo = "get_network_info"
    case getDeviceInfo = "get_device_info"
    case getCurrentTime = "get_current_time"
    case getWeather = "get_weather"
    case getScreenBrightness = "get_screen_brightness"
    case setScreenBrightness = "set_screen_brightness"
    case triggerHaptic = "trigger_haptic"
    case getCalendarEvents = "get_calendar_events"
    case createCalendarEvent = "create_calendar_event"
    case getContacts = "get_contacts"
    case sendSMS = "send_sms"
    case sendEmail = "send_email"
    case shareContent = "share_content"
    case setKeepAwake = "set_keep_awake"
    case getNotificationStatus = "get_notification_status"
    case scheduleNotification = "schedule_notification"
    case takeScreenshot = "take_screenshot"
    case openMaps = "open_maps"
    case webSearch = "web_search"
    case fetchURL = "fetch_url"
    case openURL = "open_url"

    var description: String {
        switch self {
        case .getLocation: return "Get the user's current GPS location (latitude, longitude, address)"
        case .getBatteryStatus: return "Get battery level and charging state"
        case .getNetworkInfo: return "Get network connectivity status (WiFi, cellular, etc.)"
        case .getDeviceInfo: return "Get device model, OS version, and system info"
        case .getCurrentTime: return "Get current date, time, and timezone"
        case .getWeather: return "Get current weather for user's location"
        case .getScreenBrightness: return "Get current screen brightness level (0.0-1.0)"
        case .setScreenBrightness: return "Set screen brightness level. Parameters: {\"level\": 0.0-1.0}"
        case .triggerHaptic: return "Trigger haptic feedback. Parameters: {\"style\": \"light\"|\"medium\"|\"heavy\"|\"success\"|\"warning\"|\"error\"}"
        case .getCalendarEvents: return "Get upcoming calendar events. Parameters: {\"days\": number of days ahead, default 7}"
        case .createCalendarEvent: return "Create a calendar event. Parameters: {\"title\": string, \"startDate\": ISO8601, \"endDate\": ISO8601, \"notes\": string?}"
        case .getContacts: return "Search contacts. Parameters: {\"query\": search string, default returns first 20}"
        case .sendSMS: return "Open SMS composer. Parameters: {\"to\": phone number, \"body\": message text}"
        case .sendEmail: return "Open email composer. Parameters: {\"to\": email, \"subject\": string, \"body\": string}"
        case .shareContent: return "Share text/URL via system share sheet. Parameters: {\"text\": string}"
        case .setKeepAwake: return "Keep screen awake or allow sleep. Parameters: {\"enabled\": true|false}"
        case .getNotificationStatus: return "Get notification permission status"
        case .scheduleNotification: return "Schedule a local notification. Parameters: {\"title\": string, \"body\": string, \"seconds\": delay in seconds}"
        case .takeScreenshot: return "Capture a screenshot of the current screen"
        case .openMaps: return "Open Maps app. Parameters: {\"query\": search string} or {\"latitude\": number, \"longitude\": number}"
        case .webSearch: return "Search the web for information. Parameters: {\"query\": search string}"
        case .fetchURL: return "Fetch and extract text content from a URL. Parameters: {\"url\": string}"
        case .openURL: return "Open a URL in the in-app browser. Parameters: {\"url\": string, \"title\": string (optional)}"
        }
    }

    var parametersSchema: String {
        switch self {
        case .getLocation, .getBatteryStatus, .getNetworkInfo, .getDeviceInfo,
             .getCurrentTime, .getWeather, .getScreenBrightness, .getNotificationStatus, .takeScreenshot:
            return "{}"
        case .setScreenBrightness:
            return "{\"level\": \"number (0.0-1.0)\"}"
        case .triggerHaptic:
            return "{\"style\": \"light|medium|heavy|success|warning|error\"}"
        case .getCalendarEvents:
            return "{\"days\": \"number (optional, default 7)\"}"
        case .createCalendarEvent:
            return "{\"title\": \"string\", \"startDate\": \"ISO8601\", \"endDate\": \"ISO8601\", \"notes\": \"string (optional)\"}"
        case .getContacts:
            return "{\"query\": \"string (optional)\"}"
        case .sendSMS:
            return "{\"to\": \"string\", \"body\": \"string\"}"
        case .sendEmail:
            return "{\"to\": \"string\", \"subject\": \"string\", \"body\": \"string\"}"
        case .shareContent:
            return "{\"text\": \"string\"}"
        case .setKeepAwake:
            return "{\"enabled\": \"boolean\"}"
        case .scheduleNotification:
            return "{\"title\": \"string\", \"body\": \"string\", \"seconds\": \"number\"}"
        case .openMaps:
            return "{\"query\": \"string\"} or {\"latitude\": \"number\", \"longitude\": \"number\"}"
        case .webSearch:
            return "{\"query\": \"string\"}"
        case .fetchURL:
            return "{\"url\": \"string\"}"
        case .openURL:
            return "{\"url\": \"string\", \"title\": \"string (optional)\"}"
        }
    }
}

nonisolated struct ToolCall: Sendable {
    let name: String
    let parameters: [String: Any]

    init(name: String, parameters: [String: Any] = [:]) {
        self.name = name
        self.parameters = parameters
    }
}

nonisolated struct ToolResult: Sendable {
    let toolName: String
    let success: Bool
    let data: String
    let displayIcon: String

    init(toolName: String, success: Bool, data: String, displayIcon: String = "wrench.fill") {
        self.toolName = toolName
        self.success = success
        self.data = data
        self.displayIcon = displayIcon
    }
}
