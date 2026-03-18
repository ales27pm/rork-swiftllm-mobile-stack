import Foundation
import CoreLocation
import EventKit
import Contacts
import UserNotifications
import Speech
import AVFoundation
import os.log

@MainActor
final class FirstRunPermissionCoordinator: NSObject {
    private enum Constants {
        static let hasRequestedPermissionsKey = "hasRequestedInitialPermissions"
    }

    private let logger: Logger
    private let locationManager = CLLocationManager()
    private let eventStore = EKEventStore()
    private let contactStore = CNContactStore()
    private var locationContinuation: CheckedContinuation<Void, Never>?

    override init() {
        let subsystem = Bundle.main.bundleIdentifier ?? "NeuralEngine"
        logger = Logger(subsystem: subsystem, category: "FirstRunPermissionCoordinator")
        super.init()
        locationManager.delegate = self
    }

    func requestAllPermissionsIfNeeded(using keyValueStore: KeyValueStore) async {
        guard keyValueStore.getBool(Constants.hasRequestedPermissionsKey) != true else {
            logger.debug("Initial permission flow already completed; skipping.")
            return
        }

        logger.info("Starting initial permission request flow.")
        defer {
            keyValueStore.setBool(true, forKey: Constants.hasRequestedPermissionsKey)
            logger.info("Initial permission request flow finished.")
        }

        await requestLocationPermissionIfNeeded()
        await requestCalendarPermissionIfNeeded()
        await requestContactsPermissionIfNeeded()
        await requestNotificationPermissionIfNeeded()
        await requestSpeechPermissionIfNeeded()
        await requestMicrophonePermissionIfNeeded()
        await requestCameraPermissionIfNeeded()
    }

    private func requestLocationPermissionIfNeeded() async {
        let status = locationManager.authorizationStatus
        guard status == .notDetermined else {
            logger.debug("Location permission already resolved: \(String(describing: status.rawValue), privacy: .public)")
            return
        }

        logger.info("Requesting location permission.")
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            locationContinuation = continuation
            locationManager.requestWhenInUseAuthorization()
        }
    }

    private func requestCalendarPermissionIfNeeded() async {
        let status = EKEventStore.authorizationStatus(for: .event)
        guard status == .notDetermined else {
            logger.debug("Calendar permission already resolved: \(String(describing: status.rawValue), privacy: .public)")
            return
        }

        logger.info("Requesting calendar permission.")
        do {
            _ = try await eventStore.requestFullAccessToEvents()
        } catch {
            logger.error("Calendar permission request failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func requestContactsPermissionIfNeeded() async {
        let status = CNContactStore.authorizationStatus(for: .contacts)
        guard status == .notDetermined else {
            logger.debug("Contacts permission already resolved: \(String(describing: status.rawValue), privacy: .public)")
            return
        }

        logger.info("Requesting contacts permission.")
        do {
            _ = try await contactStore.requestAccess(for: .contacts)
        } catch {
            logger.error("Contacts permission request failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func requestNotificationPermissionIfNeeded() async {
        let center = UNUserNotificationCenter.current()
        let settings = await center.notificationSettings()
        guard settings.authorizationStatus == .notDetermined else {
            logger.debug("Notification permission already resolved: \(String(describing: settings.authorizationStatus.rawValue), privacy: .public)")
            return
        }

        logger.info("Requesting notification permission.")
        do {
            _ = try await center.requestAuthorization(options: [.alert, .badge, .sound])
        } catch {
            logger.error("Notification permission request failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func requestSpeechPermissionIfNeeded() async {
        let status = SFSpeechRecognizer.authorizationStatus()
        guard status == .notDetermined else {
            logger.debug("Speech recognition permission already resolved: \(String(describing: status.rawValue), privacy: .public)")
            return
        }

        logger.info("Requesting speech recognition permission.")
        _ = await withCheckedContinuation { (continuation: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { authorizationStatus in
                continuation.resume(returning: authorizationStatus)
            }
        }
    }

    private func requestMicrophonePermissionIfNeeded() async {
        let audioSession = AVAudioSession.sharedInstance()
        switch audioSession.recordPermission {
        case .undetermined:
            logger.info("Requesting microphone permission.")
            _ = await AVAudioApplication.requestRecordPermission()
        case .granted, .denied:
            logger.debug("Microphone permission already resolved.")
        @unknown default:
            logger.error("Unknown microphone permission state encountered.")
        }
    }

    private func requestCameraPermissionIfNeeded() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        guard status == .notDetermined else {
            logger.debug("Camera permission already resolved: \(String(describing: status.rawValue), privacy: .public)")
            return
        }

        logger.info("Requesting camera permission.")
        _ = await AVCaptureDevice.requestAccess(for: .video)
    }
}

extension FirstRunPermissionCoordinator: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            guard let locationContinuation else { return }
            let status = manager.authorizationStatus
            guard status != .notDetermined else { return }
            self.locationContinuation = nil
            locationContinuation.resume()
        }
    }
}
