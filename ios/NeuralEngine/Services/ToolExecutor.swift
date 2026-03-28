import Foundation
import UIKit
import CoreLocation
import EventKit
import Contacts
import MessageUI
import UserNotifications

@MainActor
@Observable
class ToolExecutor: NSObject {
    private let locationManager = CLLocationManager()
    private let eventStore = EKEventStore()
    private let contactStore = CNContactStore()
    private var locationContinuation: CheckedContinuation<CLLocation?, Never>?
    private var pendingShareContent: String?
    var pendingSMSTo: String?
    var pendingSMSBody: String?
    var pendingEmailTo: String?
    var pendingEmailSubject: String?
    var pendingEmailBody: String?
    var showShareSheet: Bool = false
    var showSMSComposer: Bool = false
    var showEmailComposer: Bool = false
    var shareItems: [Any] = []
    var showInAppBrowser: Bool = false
    var browserURL: URL?
    var browserTitle: String = ""
    private let webSearchService = WebSearchService()
    private let weatherKitService = WeatherKitService()

    override init() {
        super.init()
    }

    func execute(_ call: ToolCall) async -> ToolResult {
        guard let tool = DeviceToolName(rawValue: call.name) else {
            return ToolResult(toolName: call.name, success: false, data: "Unknown tool: \(call.name)", displayIcon: "exclamationmark.triangle.fill")
        }

        switch tool {
        case .getLocation:
            return await executeGetLocation()
        case .getBatteryStatus:
            return executeGetBattery()
        case .getNetworkInfo:
            return executeGetNetwork()
        case .getDeviceInfo:
            return executeGetDeviceInfo()
        case .getCurrentTime:
            return executeGetCurrentTime()
        case .getWeather:
            return await executeGetWeather()
        case .getScreenBrightness:
            return executeGetBrightness()
        case .setScreenBrightness:
            return executeSetBrightness(call.parameters)
        case .triggerHaptic:
            return executeTriggerHaptic(call.parameters)
        case .getCalendarEvents:
            return await executeGetCalendarEvents(call.parameters)
        case .createCalendarEvent:
            return await executeCreateCalendarEvent(call.parameters)
        case .getContacts:
            return await executeGetContacts(call.parameters)
        case .sendSMS:
            return executeSendSMS(call.parameters)
        case .sendEmail:
            return executeSendEmail(call.parameters)
        case .shareContent:
            return executeShareContent(call.parameters)
        case .setKeepAwake:
            return executeSetKeepAwake(call.parameters)
        case .getNotificationStatus:
            return await executeGetNotificationStatus()
        case .scheduleNotification:
            return await executeScheduleNotification(call.parameters)
        case .takeScreenshot:
            return executeScreenshot()
        case .openMaps:
            return executeOpenMaps(call.parameters)
        case .webSearch:
            return await executeWebSearch(call.parameters)
        case .fetchURL:
            return await executeFetchURL(call.parameters)
        case .openURL:
            return executeOpenURL(call.parameters)
        }
    }

    private func executeGetLocation() async -> ToolResult {
        let status = locationManager.authorizationStatus
        if status == .notDetermined {
            locationManager.requestWhenInUseAuthorization()
            try? await Task.sleep(for: .seconds(1))
        }

        guard locationManager.authorizationStatus == .authorizedWhenInUse ||
              locationManager.authorizationStatus == .authorizedAlways else {
            return ToolResult(
                toolName: "get_location",
                success: false,
                data: #"{"permissionGranted":false,"error":"location_permission_not_granted"}"#,
                displayIcon: "location.slash.fill"
            )
        }

        guard let location = await requestCurrentLocation(desiredAccuracy: kCLLocationAccuracyBest) else {
            return ToolResult(
                toolName: "get_location",
                success: false,
                data: #"{"permissionGranted":true,"error":"no_location_signal"}"#,
                displayIcon: "location.slash.fill"
            )
        }

        let address = await reverseGeocodedAddress(for: location)
        let confidence = max(0.0, min(1.0, 1 - (location.horizontalAccuracy / 500)))
        let timestamp = ISO8601DateFormatter().string(from: location.timestamp)
        let escapedAddress = address.replacingOccurrences(of: "\"", with: "\\\"")
        let data = """
        {"latitude": \(location.coordinate.latitude), "longitude": \(location.coordinate.longitude), "altitude": \(location.altitude), "accuracy": \(location.horizontalAccuracy), "address": "\(escapedAddress)", "timestamp": "\(timestamp)", "permissionGranted": true, "source": "core_location", "confidence": \(confidence)}
        """
        return ToolResult(toolName: "get_location", success: true, data: data, displayIcon: "location.fill")
    }

    private func executeGetBattery() -> ToolResult {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        let state: String
        switch UIDevice.current.batteryState {
        case .charging: state = "charging"
        case .full: state = "full"
        case .unplugged: state = "unplugged"
        case .unknown: state = "unknown"
        @unknown default: state = "unknown"
        }
        let percentage = level >= 0 ? Int(level * 100) : -1
        let data = "{\"level\": \(percentage), \"state\": \"\(state)\", \"lowPowerMode\": \(ProcessInfo.processInfo.isLowPowerModeEnabled)}"
        return ToolResult(toolName: "get_battery_status", success: true, data: data, displayIcon: "battery.100percent")
    }

    private func executeGetNetwork() -> ToolResult {
        var info: [String: Any] = [:]
        info["available"] = true

        let processInfo = ProcessInfo.processInfo
        info["hostName"] = processInfo.hostName
        info["thermalState"] = "\(processInfo.thermalState.rawValue)"

        let data: String
        if let jsonData = try? JSONSerialization.data(withJSONObject: info),
           let jsonStr = String(data: jsonData, encoding: .utf8) {
            data = jsonStr
        } else {
            data = "{\"available\": true}"
        }
        return ToolResult(toolName: "get_network_info", success: true, data: data, displayIcon: "wifi")
    }

    private func executeGetDeviceInfo() -> ToolResult {
        let device = UIDevice.current
        let processInfo = ProcessInfo.processInfo
        let data = """
        {"model": "\(device.model)", "name": "\(device.name)", "systemName": "\(device.systemName)", "systemVersion": "\(device.systemVersion)", "processorCount": \(processInfo.processorCount), "physicalMemory": \(processInfo.physicalMemory), "activeProcessorCount": \(processInfo.activeProcessorCount)}
        """
        return ToolResult(toolName: "get_device_info", success: true, data: data, displayIcon: "iphone")
    }

    private func executeGetCurrentTime() -> ToolResult {
        let now = Date()
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let readable = now.formatted(date: .complete, time: .complete)
        let tz = TimeZone.current
        let data = "{\"iso8601\": \"\(formatter.string(from: now))\", \"readable\": \"\(readable)\", \"timezone\": \"\(tz.identifier)\", \"utcOffset\": \(tz.secondsFromGMT() / 3600)}"
        return ToolResult(toolName: "get_current_time", success: true, data: data, displayIcon: "clock.fill")
    }

    private func executeGetWeather() async -> ToolResult {
        let status = locationManager.authorizationStatus
        if status == .notDetermined {
            locationManager.requestWhenInUseAuthorization()
            try? await Task.sleep(for: .seconds(1))
        }

        guard locationManager.authorizationStatus == .authorizedWhenInUse ||
              locationManager.authorizationStatus == .authorizedAlways else {
            return ToolResult(
                toolName: "get_weather",
                success: false,
                data: "Location permission is required to load local weather.",
                displayIcon: "cloud.slash.fill"
            )
        }

        guard let location = await requestCurrentLocation(desiredAccuracy: kCLLocationAccuracyKilometer) else {
            return ToolResult(
                toolName: "get_weather",
                success: false,
                data: "Your device could not determine a location for weather yet.",
                displayIcon: "cloud.slash.fill"
            )
        }

        let locationName = await reverseGeocodedAddress(for: location)

        do {
            let snapshot = try await weatherKitService.snapshot(for: location, locationName: locationName)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            guard let data = try? encoder.encode(snapshot), let json = String(data: data, encoding: .utf8) else {
                return ToolResult(toolName: "get_weather", success: false, data: "Weather data could not be encoded.", displayIcon: "cloud.fill")
            }

            return ToolResult(toolName: "get_weather", success: true, data: json, displayIcon: snapshot.current.symbolName)
        } catch {
            return ToolResult(
                toolName: "get_weather",
                success: false,
                data: WeatherKitService.userFacingErrorMessage(for: error),
                displayIcon: "cloud.fill"
            )
        }
    }

    private func executeGetBrightness() -> ToolResult {
        let brightness = UIScreen.main.brightness
        return ToolResult(toolName: "get_screen_brightness", success: true, data: "{\"brightness\": \(brightness)}", displayIcon: "sun.max.fill")
    }

    private func executeSetBrightness(_ params: [String: Any]) -> ToolResult {
        guard let level = params["level"] as? Double else {
            return ToolResult(toolName: "set_screen_brightness", success: false, data: "Missing 'level' parameter (0.0-1.0)", displayIcon: "sun.max.fill")
        }
        let clamped = max(0, min(1, level))
        UIScreen.main.brightness = clamped
        return ToolResult(toolName: "set_screen_brightness", success: true, data: "{\"brightness\": \(clamped)}", displayIcon: "sun.max.fill")
    }

    private func executeTriggerHaptic(_ params: [String: Any]) -> ToolResult {
        let style = params["style"] as? String ?? "medium"
        switch style {
        case "light":
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        case "heavy":
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        case "success":
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        case "warning":
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
        case "error":
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        default:
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        }
        return ToolResult(toolName: "trigger_haptic", success: true, data: "{\"style\": \"\(style)\"}", displayIcon: "hand.tap.fill")
    }

    private func executeGetCalendarEvents(_ params: [String: Any]) async -> ToolResult {
        let days = params["days"] as? Int ?? 7

        do {
            let granted = try await eventStore.requestFullAccessToEvents()
            guard granted else {
                return ToolResult(toolName: "get_calendar_events", success: false, data: "Calendar access not granted", displayIcon: "calendar.badge.exclamationmark")
            }
        } catch {
            return ToolResult(toolName: "get_calendar_events", success: false, data: "Calendar access error: \(error.localizedDescription)", displayIcon: "calendar.badge.exclamationmark")
        }

        let startDate = Date()
        let endDate = Calendar.current.date(byAdding: .day, value: days, to: startDate) ?? startDate
        let predicate = eventStore.predicateForEvents(withStart: startDate, end: endDate, calendars: nil)
        let events = eventStore.events(matching: predicate)

        let eventList = events.prefix(20).map { event -> [String: String] in
            let df = ISO8601DateFormatter()
            return [
                "title": event.title ?? "Untitled",
                "startDate": df.string(from: event.startDate),
                "endDate": df.string(from: event.endDate),
                "location": event.location ?? "",
                "calendar": event.calendar.title
            ]
        }

        guard let jsonData = try? JSONSerialization.data(withJSONObject: eventList),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            return ToolResult(toolName: "get_calendar_events", success: true, data: "[]", displayIcon: "calendar")
        }
        return ToolResult(toolName: "get_calendar_events", success: true, data: jsonStr, displayIcon: "calendar")
    }

    private func executeCreateCalendarEvent(_ params: [String: Any]) async -> ToolResult {
        guard let title = params["title"] as? String,
              let startStr = params["startDate"] as? String,
              let endStr = params["endDate"] as? String else {
            return ToolResult(toolName: "create_calendar_event", success: false, data: "Missing required parameters: title, startDate, endDate", displayIcon: "calendar.badge.plus")
        }

        do {
            let granted = try await eventStore.requestFullAccessToEvents()
            guard granted else {
                return ToolResult(toolName: "create_calendar_event", success: false, data: "Calendar access not granted", displayIcon: "calendar.badge.exclamationmark")
            }
        } catch {
            return ToolResult(toolName: "create_calendar_event", success: false, data: "Calendar access error", displayIcon: "calendar.badge.exclamationmark")
        }

        let df = ISO8601DateFormatter()
        guard let startDate = df.date(from: startStr), let endDate = df.date(from: endStr) else {
            return ToolResult(toolName: "create_calendar_event", success: false, data: "Invalid date format. Use ISO8601.", displayIcon: "calendar.badge.exclamationmark")
        }

        let event = EKEvent(eventStore: eventStore)
        event.title = title
        event.startDate = startDate
        event.endDate = endDate
        event.notes = params["notes"] as? String
        event.calendar = eventStore.defaultCalendarForNewEvents

        do {
            try eventStore.save(event, span: .thisEvent)
            return ToolResult(toolName: "create_calendar_event", success: true, data: "{\"eventId\": \"\(event.eventIdentifier ?? "")\", \"title\": \"\(title)\"}", displayIcon: "calendar.badge.plus")
        } catch {
            return ToolResult(toolName: "create_calendar_event", success: false, data: "Failed to create event: \(error.localizedDescription)", displayIcon: "calendar.badge.exclamationmark")
        }
    }

    private func executeGetContacts(_ params: [String: Any]) async -> ToolResult {
        let status = CNContactStore.authorizationStatus(for: .contacts)
        if status == .notDetermined {
            guard (try? await contactStore.requestAccess(for: .contacts)) == true else {
                return ToolResult(toolName: "get_contacts", success: false, data: "Contacts access not granted", displayIcon: "person.crop.circle.badge.exclamationmark")
            }
        } else if status != .authorized {
            return ToolResult(toolName: "get_contacts", success: false, data: "Contacts access not granted", displayIcon: "person.crop.circle.badge.exclamationmark")
        }

        let keysToFetch: [CNKeyDescriptor] = [
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactPhoneNumbersKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor
        ]

        let query = params["query"] as? String
        let request = CNContactFetchRequest(keysToFetch: keysToFetch)

        if let query, !query.isEmpty {
            request.predicate = CNContact.predicateForContacts(matchingName: query)
        }

        var contacts: [[String: Any]] = []
        do {
            try contactStore.enumerateContacts(with: request) { contact, stop in
                let entry: [String: Any] = [
                    "name": "\(contact.givenName) \(contact.familyName)".trimmingCharacters(in: .whitespaces),
                    "phones": contact.phoneNumbers.map { $0.value.stringValue },
                    "emails": contact.emailAddresses.map { $0.value as String }
                ]
                contacts.append(entry)
                if contacts.count >= 20 { stop.pointee = true }
            }
        } catch {
            return ToolResult(toolName: "get_contacts", success: false, data: "Failed to fetch contacts: \(error.localizedDescription)", displayIcon: "person.crop.circle.badge.exclamationmark")
        }

        guard let jsonData = try? JSONSerialization.data(withJSONObject: contacts),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            return ToolResult(toolName: "get_contacts", success: true, data: "[]", displayIcon: "person.2.fill")
        }
        return ToolResult(toolName: "get_contacts", success: true, data: jsonStr, displayIcon: "person.2.fill")
    }

    private func executeSendSMS(_ params: [String: Any]) -> ToolResult {
        guard let to = params["to"] as? String, let body = params["body"] as? String else {
            return ToolResult(toolName: "send_sms", success: false, data: "Missing 'to' or 'body' parameters", displayIcon: "message.fill")
        }
        guard MFMessageComposeViewController.canSendText() else {
            resetSMSComposerState()
            return ToolResult(toolName: "send_sms", success: false, data: "SMS composer unavailable on this device", displayIcon: "message.fill")
        }
        pendingSMSTo = to
        pendingSMSBody = body
        showSMSComposer = true
        return ToolResult(toolName: "send_sms", success: true, data: "{\"status\": \"composer_opened\", \"to\": \"\(to)\"}", displayIcon: "message.fill")
    }

    private func executeSendEmail(_ params: [String: Any]) -> ToolResult {
        guard let to = params["to"] as? String else {
            return ToolResult(toolName: "send_email", success: false, data: "Missing 'to' parameter", displayIcon: "envelope.fill")
        }
        guard MFMailComposeViewController.canSendMail() else {
            resetEmailComposerState()
            return ToolResult(toolName: "send_email", success: false, data: "Mail composer unavailable on this device", displayIcon: "envelope.fill")
        }
        pendingEmailTo = to
        pendingEmailSubject = params["subject"] as? String ?? ""
        pendingEmailBody = params["body"] as? String ?? ""
        showEmailComposer = true
        return ToolResult(toolName: "send_email", success: true, data: "{\"status\": \"composer_opened\", \"to\": \"\(to)\"}", displayIcon: "envelope.fill")
    }

    func resetSMSComposerState() {
        pendingSMSTo = nil
        pendingSMSBody = nil
        showSMSComposer = false
    }

    func resetEmailComposerState() {
        pendingEmailTo = nil
        pendingEmailSubject = nil
        pendingEmailBody = nil
        showEmailComposer = false
    }

    private func executeShareContent(_ params: [String: Any]) -> ToolResult {
        guard let text = params["text"] as? String else {
            return ToolResult(toolName: "share_content", success: false, data: "Missing 'text' parameter", displayIcon: "square.and.arrow.up")
        }
        shareItems = [text]
        showShareSheet = true
        return ToolResult(toolName: "share_content", success: true, data: "{\"status\": \"share_sheet_opened\"}", displayIcon: "square.and.arrow.up")
    }

    private func executeSetKeepAwake(_ params: [String: Any]) -> ToolResult {
        let enabled = params["enabled"] as? Bool ?? true
        UIApplication.shared.isIdleTimerDisabled = enabled
        return ToolResult(toolName: "set_keep_awake", success: true, data: "{\"keepAwake\": \(enabled)}", displayIcon: enabled ? "eye.fill" : "eye.slash.fill")
    }

    private func executeGetNotificationStatus() async -> ToolResult {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        let status: String
        switch settings.authorizationStatus {
        case .authorized: status = "authorized"
        case .denied: status = "denied"
        case .provisional: status = "provisional"
        case .notDetermined: status = "not_determined"
        case .ephemeral: status = "ephemeral"
        @unknown default: status = "unknown"
        }
        return ToolResult(toolName: "get_notification_status", success: true, data: "{\"status\": \"\(status)\"}", displayIcon: "bell.fill")
    }

    private func executeScheduleNotification(_ params: [String: Any]) async -> ToolResult {
        guard let title = params["title"] as? String, let body = params["body"] as? String else {
            return ToolResult(toolName: "schedule_notification", success: false, data: "Missing 'title' or 'body'", displayIcon: "bell.slash.fill")
        }
        let seconds = params["seconds"] as? Double ?? 5.0

        let center = UNUserNotificationCenter.current()
        let settings = await center.notificationSettings()
        if settings.authorizationStatus == .notDetermined {
            _ = try? await center.requestAuthorization(options: [.alert, .sound, .badge])
        }

        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: max(1, seconds), repeats: false)
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: trigger)

        do {
            try await center.add(request)
            return ToolResult(toolName: "schedule_notification", success: true, data: "{\"scheduled\": true, \"delaySeconds\": \(seconds)}", displayIcon: "bell.badge.fill")
        } catch {
            return ToolResult(toolName: "schedule_notification", success: false, data: "Failed: \(error.localizedDescription)", displayIcon: "bell.slash.fill")
        }
    }

    private func executeScreenshot() -> ToolResult {
        guard let window = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first?.windows.first else {
            return ToolResult(toolName: "take_screenshot", success: false, data: "No active window", displayIcon: "camera.viewfinder")
        }

        let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
        let image = renderer.image { ctx in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }

        if let data = image.pngData() {
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("screenshot_\(Int(Date().timeIntervalSince1970)).png")
            try? data.write(to: tempURL)
            return ToolResult(toolName: "take_screenshot", success: true, data: "{\"saved\": true, \"path\": \"\(tempURL.path)\"}", displayIcon: "camera.viewfinder")
        }
        return ToolResult(toolName: "take_screenshot", success: false, data: "Failed to capture screenshot", displayIcon: "camera.viewfinder")
    }

    private func executeOpenMaps(_ params: [String: Any]) -> ToolResult {
        var urlString: String
        if let query = params["query"] as? String {
            let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
            urlString = "maps://?q=\(encoded)"
        } else if let lat = params["latitude"] as? Double, let lon = params["longitude"] as? Double {
            urlString = "maps://?ll=\(lat),\(lon)"
        } else {
            return ToolResult(toolName: "open_maps", success: false, data: "Missing 'query' or 'latitude'/'longitude'", displayIcon: "map.fill")
        }

        if let url = URL(string: urlString) {
            UIApplication.shared.open(url)
            return ToolResult(toolName: "open_maps", success: true, data: "{\"opened\": true}", displayIcon: "map.fill")
        }
        return ToolResult(toolName: "open_maps", success: false, data: "Invalid maps URL", displayIcon: "map.fill")
    }

    private func executeWebSearch(_ params: [String: Any]) async -> ToolResult {
        guard let query = params["query"] as? String else {
            return ToolResult(toolName: "web_search", success: false, data: "Missing 'query' parameter", displayIcon: "magnifyingglass")
        }

        let results = await webSearchService.search(query: query)
        guard !results.isEmpty else {
            let errorMsg = webSearchService.lastError ?? "No results found"
            return ToolResult(toolName: "web_search", success: false, data: errorMsg, displayIcon: "magnifyingglass")
        }

        let resultList = results.prefix(5).map { result -> [String: String] in
            ["title": result.title, "url": result.url, "snippet": result.snippet]
        }

        guard let jsonData = try? JSONSerialization.data(withJSONObject: resultList),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            return ToolResult(toolName: "web_search", success: true, data: "[]", displayIcon: "magnifyingglass")
        }
        return ToolResult(toolName: "web_search", success: true, data: jsonStr, displayIcon: "globe")
    }

    private func executeFetchURL(_ params: [String: Any]) async -> ToolResult {
        guard let urlString = params["url"] as? String else {
            return ToolResult(toolName: "fetch_url", success: false, data: "Missing 'url' parameter", displayIcon: "doc.text")
        }

        let content = await webSearchService.fetchURL(urlString)
        guard !content.isEmpty else {
            let errorMsg = webSearchService.lastError ?? "Could not fetch content"
            return ToolResult(toolName: "fetch_url", success: false, data: errorMsg, displayIcon: "doc.text")
        }

        return ToolResult(toolName: "fetch_url", success: true, data: content, displayIcon: "doc.text.fill")
    }

    private func executeOpenURL(_ params: [String: Any]) -> ToolResult {
        guard let urlString = params["url"] as? String,
              let url = URL(string: urlString) else {
            return ToolResult(toolName: "open_url", success: false, data: "Missing or invalid 'url' parameter", displayIcon: "globe")
        }
        browserURL = url
        browserTitle = params["title"] as? String ?? url.host ?? "Web Page"
        showInAppBrowser = true
        return ToolResult(toolName: "open_url", success: true, data: "{\"status\": \"browser_opened\", \"url\": \"\(urlString)\"}", displayIcon: "globe")
    }

    private func requestCurrentLocation(desiredAccuracy: CLLocationAccuracy) async -> CLLocation? {
        locationManager.desiredAccuracy = desiredAccuracy
        locationManager.delegate = self

        return await withCheckedContinuation { (continuation: CheckedContinuation<CLLocation?, Never>) in
            self.locationContinuation = continuation
            self.locationManager.requestLocation()
        }
    }

    private func reverseGeocodedAddress(for location: CLLocation) async -> String {
        let geocoder = CLGeocoder()
        if let placemarks = try? await geocoder.reverseGeocodeLocation(location),
           let place = placemarks.first {
            let parts = [place.name, place.locality, place.administrativeArea, place.country].compactMap { $0 }
            if !parts.isEmpty {
                return parts.joined(separator: ", ")
            }
        }

        return "Current Location"
    }

    nonisolated static func buildToolsPrompt() -> String {
        let toolLines = DeviceToolName.allCases.map {
            "- \($0.rawValue): \($0.description) Parameters: \($0.parametersSchema)"
        }.joined(separator: "\n")

        return """

        [Tool Calling Contract]
        You have access to device tools.

        Only emit a tool call when the user explicitly needs device capabilities or external retrieval.
        If no tool is needed, respond normally in plain text.

        Valid tool call formats:
        1) Single call:
        <tool_call>{"name":"tool_name","parameters":{}}</tool_call>

        2) Multiple calls in one turn:
        <tool_calls>[{"name":"tool_a","parameters":{}},{"name":"tool_b","parameters":{}}]</tool_calls>

        Rules:
        - Use ONLY tool names from the list below.
        - Use a JSON object for `parameters` (or `arguments` as an alias if needed).
        - Do not include markdown code fences around tool calls.
        - After receiving tool results, synthesize a concise user-facing answer.

        Available tools:
        \(toolLines)
        """
    }
}

extension ToolExecutor: CLLocationManagerDelegate {
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        Task { @MainActor in
            locationContinuation?.resume(returning: locations.first)
            locationContinuation = nil
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        Task { @MainActor in
            locationContinuation?.resume(returning: nil)
            locationContinuation = nil
        }
    }
}
