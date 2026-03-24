import Foundation
import CoreLocation
import UIKit

@Observable
@MainActor
class WeatherViewModel: NSObject, CLLocationManagerDelegate {
    var snapshot: WeatherSnapshot?
    var isLoading: Bool = false
    var errorMessage: String?
    var authorizationStatus: CLAuthorizationStatus

    private let locationManager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private let weatherKitService = WeatherKitService()

    override init() {
        authorizationStatus = locationManager.authorizationStatus
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
    }

    func refreshIfNeeded() {
        guard snapshot == nil, !isLoading else { return }
        refresh()
    }

    func refresh() {
        guard !isLoading else { return }

        errorMessage = nil
        authorizationStatus = locationManager.authorizationStatus

        switch authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            isLoading = true
            locationManager.requestLocation()
        case .notDetermined:
            locationManager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            errorMessage = "Allow location access to load local weather."
        @unknown default:
            errorMessage = "Location access is unavailable right now."
        }
    }

    func openSystemSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    private func loadWeather(for location: CLLocation) async {
        do {
            let locationName = try await resolveLocationName(for: location)
            snapshot = try await weatherKitService.snapshot(for: location, locationName: locationName)
            errorMessage = nil
        } catch {
            errorMessage = WeatherKitService.userFacingErrorMessage(for: error)
        }

        isLoading = false
    }

    private func resolveLocationName(for location: CLLocation) async throws -> String {
        let placemarks = try await geocoder.reverseGeocodeLocation(location)
        guard let placemark = placemarks.first else { return "Current Location" }

        let locality = placemark.locality
        let region = placemark.administrativeArea
        let country = placemark.country
        let candidates: [String?] = [locality, region, country]
        let parts: [String] = candidates.compactMap { value in
            guard let value, !value.isEmpty else { return nil }
            return value
        }

        if parts.isEmpty {
            return placemark.name ?? "Current Location"
        }

        return parts.joined(separator: ", ")
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            authorizationStatus = manager.authorizationStatus
            if manager.authorizationStatus == .authorizedWhenInUse || manager.authorizationStatus == .authorizedAlways {
                refresh()
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else {
            Task { @MainActor in
                errorMessage = "Your device could not determine a location for weather yet."
                isLoading = false
            }
            return
        }

        Task { @MainActor in
            await loadWeather(for: location)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        Task { @MainActor in
            errorMessage = "Your device could not determine a location for weather yet."
            isLoading = false
        }
    }
}
