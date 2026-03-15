import SwiftUI
import MapKit
import CoreLocation

@Observable
class MapViewModel: NSObject, CLLocationManagerDelegate {
    var cameraPosition: MapCameraPosition = .automatic
    var searchText: String = ""
    var searchResults: [MKMapItem] = []
    var selectedPlace: MKMapItem?
    var showPlaceDetail: Bool = false
    var mapStyle: MapStyleOption = .standard
    var isSearching: Bool = false
    var userLocation: CLLocation?
    var savedPlaces: [MKMapItem] = []
    var route: MKRoute?
    var showDirections: Bool = false
    var lookAroundScene: MKLookAroundScene?

    private let locationManager = CLLocationManager()
    private let completer = MKLocalSearchCompleter()
    var completionResults: [MKLocalSearchCompletion] = []
    private var completerDelegate: SearchCompleterDelegate?

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest

        let delegate = SearchCompleterDelegate { [weak self] results in
            self?.completionResults = results
        }
        completerDelegate = delegate
        completer.delegate = delegate
        completer.resultTypes = [.address, .pointOfInterest]
    }

    func requestLocationPermission() {
        if locationManager.authorizationStatus == .notDetermined {
            locationManager.requestWhenInUseAuthorization()
        } else if locationManager.authorizationStatus == .authorizedWhenInUse ||
                    locationManager.authorizationStatus == .authorizedAlways {
            locationManager.requestLocation()
        }
    }

    func centerOnUser() {
        cameraPosition = .userLocation(fallback: .automatic)
    }

    func search() async {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            searchResults = []
            return
        }

        isSearching = true
        defer { isSearching = false }

        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = trimmed
        if let location = userLocation {
            request.region = MKCoordinateRegion(
                center: location.coordinate,
                latitudinalMeters: 50000,
                longitudinalMeters: 50000
            )
        }

        do {
            let search = MKLocalSearch(request: request)
            let response = try await search.start()
            searchResults = response.mapItems
            if let first = response.mapItems.first {
                cameraPosition = .region(MKCoordinateRegion(
                    center: first.placemark.coordinate,
                    latitudinalMeters: 5000,
                    longitudinalMeters: 5000
                ))
            }
        } catch {
            searchResults = []
        }
    }

    func searchFromCompletion(_ completion: MKLocalSearchCompletion) async {
        let request = MKLocalSearch.Request(completion: completion)
        do {
            let search = MKLocalSearch(request: request)
            let response = try await search.start()
            searchResults = response.mapItems
            searchText = completion.title
            completionResults = []
            if let first = response.mapItems.first {
                selectPlace(first)
            }
        } catch {}
    }

    func updateCompletions() {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count >= 2 {
            completer.queryFragment = trimmed
        } else {
            completionResults = []
        }
    }

    func selectPlace(_ item: MKMapItem) {
        selectedPlace = item
        showPlaceDetail = true
        cameraPosition = .region(MKCoordinateRegion(
            center: item.placemark.coordinate,
            latitudinalMeters: 2000,
            longitudinalMeters: 2000
        ))
        Task { await fetchLookAround(for: item) }
    }

    func dismissPlaceDetail() {
        showPlaceDetail = false
        selectedPlace = nil
        lookAroundScene = nil
        route = nil
        showDirections = false
    }

    func getDirections(to destination: MKMapItem) async {
        guard let userLocation else { return }
        let request = MKDirections.Request()
        request.source = MKMapItem(placemark: MKPlacemark(coordinate: userLocation.coordinate))
        request.destination = destination
        request.transportType = .automobile

        do {
            let directions = MKDirections(request: request)
            let response = try await directions.calculate()
            route = response.routes.first
            showDirections = true
            if let route {
                cameraPosition = .rect(route.polyline.boundingMapRect)
            }
        } catch {}
    }

    func fetchLookAround(for item: MKMapItem) async {
        let request = MKLookAroundSceneRequest(coordinate: item.placemark.coordinate)
        lookAroundScene = try? await request.scene
    }

    func cycleMapStyle() {
        switch mapStyle {
        case .standard: mapStyle = .imagery
        case .imagery: mapStyle = .hybrid
        case .hybrid: mapStyle = .standard
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        Task { @MainActor in
            userLocation = locations.last
        }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            if manager.authorizationStatus == .authorizedWhenInUse || manager.authorizationStatus == .authorizedAlways {
                manager.requestLocation()
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}
}

nonisolated enum MapStyleOption: String, CaseIterable, Sendable {
    case standard
    case imagery
    case hybrid

    var label: String {
        switch self {
        case .standard: return "Standard"
        case .imagery: return "Satellite"
        case .hybrid: return "Hybrid"
        }
    }

    var icon: String {
        switch self {
        case .standard: return "map"
        case .imagery: return "globe.americas"
        case .hybrid: return "square.stack.3d.up"
        }
    }
}

private class SearchCompleterDelegate: NSObject, MKLocalSearchCompleterDelegate {
    let onUpdate: @Sendable ([MKLocalSearchCompletion]) -> Void

    init(onUpdate: @escaping @Sendable ([MKLocalSearchCompletion]) -> Void) {
        self.onUpdate = onUpdate
    }

    func completerDidUpdateResults(_ completer: MKLocalSearchCompleter) {
        onUpdate(completer.results)
    }

    func completer(_ completer: MKLocalSearchCompleter, didFailWithError error: Error) {}
}
