import SwiftUI
import MapKit

struct MapView: View {
    @State var viewModel = MapViewModel()
    @State private var isSearchExpanded: Bool = false
    @FocusState private var isSearchFocused: Bool

    var body: some View {
        ZStack(alignment: .top) {
            mapContent

            VStack(spacing: 0) {
                searchBar
                if isSearchExpanded && !viewModel.completionResults.isEmpty {
                    completionsList
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)

            VStack {
                Spacer()
                HStack {
                    Spacer()
                    controlButtons
                }
                .padding(.trailing, 16)
                .padding(.bottom, viewModel.showPlaceDetail ? 280 : 24)
            }
        }
        .sheet(isPresented: $viewModel.showPlaceDetail) {
            if let place = viewModel.selectedPlace {
                PlaceDetailSheet(
                    place: place,
                    lookAroundScene: viewModel.lookAroundScene,
                    route: viewModel.route,
                    showDirections: viewModel.showDirections,
                    onGetDirections: {
                        Task { await viewModel.getDirections(to: place) }
                    },
                    onDismiss: { viewModel.dismissPlaceDetail() },
                    onOpenInMaps: {
                        place.openInMaps(launchOptions: [
                            MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving
                        ])
                    }
                )
                .presentationDetents([.fraction(0.4), .large])
                .presentationDragIndicator(.visible)
                .presentationBackgroundInteraction(.enabled(upThrough: .fraction(0.4)))
                .presentationContentInteraction(.scrolls)
            }
        }
        .onAppear {
            viewModel.requestLocationPermission()
        }
    }

    private var mapContent: some View {
        Map(position: $viewModel.cameraPosition, selection: Binding<MKMapItem?>(
            get: { viewModel.selectedPlace },
            set: { if let item = $0 { viewModel.selectPlace(item) } }
        )) {
            UserAnnotation()

            ForEach(viewModel.searchResults, id: \.self) { item in
                Marker(
                    item.name ?? "Location",
                    systemImage: categoryIcon(for: item),
                    coordinate: item.placemark.coordinate
                )
                .tint(categoryColor(for: item))
                .tag(item)
            }

            if let route = viewModel.route {
                MapPolyline(route.polyline)
                    .stroke(.blue, lineWidth: 5)
            }
        }
        .mapStyle(resolvedMapStyle)
        .mapControls {
            MapCompass()
            MapScaleView()
        }
        .ignoresSafeArea(edges: .top)
    }

    private var resolvedMapStyle: MapStyle {
        switch viewModel.mapStyle {
        case .standard: return .standard(elevation: .realistic)
        case .imagery: return .imagery(elevation: .realistic)
        case .hybrid: return .hybrid(elevation: .realistic)
        }
    }

    private var searchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .font(.system(size: 16, weight: .medium))

            TextField("Search places", text: $viewModel.searchText)
                .font(.body)
                .focused($isSearchFocused)
                .submitLabel(.search)
                .onSubmit {
                    Task { await viewModel.search() }
                    isSearchExpanded = false
                    isSearchFocused = false
                }
                .onChange(of: viewModel.searchText) { _, _ in
                    viewModel.updateCompletions()
                    isSearchExpanded = true
                }
                .onChange(of: isSearchFocused) { _, focused in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isSearchExpanded = focused
                    }
                }

            if !viewModel.searchText.isEmpty {
                Button {
                    viewModel.searchText = ""
                    viewModel.searchResults = []
                    viewModel.completionResults = []
                    isSearchFocused = false
                    isSearchExpanded = false
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
            }

            if isSearchExpanded {
                Button("Cancel") {
                    isSearchFocused = false
                    isSearchExpanded = false
                }
                .font(.subheadline)
                .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .background(.ultraThinMaterial)
        .clipShape(.rect(cornerRadius: 14))
        .shadow(color: .black.opacity(0.12), radius: 8, y: 4)
    }

    private var completionsList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(viewModel.completionResults.prefix(6), id: \.self) { completion in
                    Button {
                        Task { await viewModel.searchFromCompletion(completion) }
                        isSearchFocused = false
                        isSearchExpanded = false
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: "mappin.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.red)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(completion.title)
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)
                                if !completion.subtitle.isEmpty {
                                    Text(completion.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            }
                            Spacer()
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                    }
                    .contentShape(Rectangle())
                }
            }
        }
        .frame(maxHeight: 260)
        .background(.ultraThinMaterial)
        .clipShape(.rect(cornerRadius: 14))
        .shadow(color: .black.opacity(0.1), radius: 6, y: 3)
        .transition(.opacity.combined(with: .move(edge: .top)))
    }

    private var controlButtons: some View {
        VStack(spacing: 12) {
            Button {
                viewModel.centerOnUser()
            } label: {
                Image(systemName: "location.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
                    .shadow(color: .black.opacity(0.12), radius: 4, y: 2)
            }

            Button {
                viewModel.cycleMapStyle()
            } label: {
                Image(systemName: viewModel.mapStyle.icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
                    .shadow(color: .black.opacity(0.12), radius: 4, y: 2)
            }
        }
    }

    private func categoryIcon(for item: MKMapItem) -> String {
        guard let category = item.pointOfInterestCategory else { return "mappin" }
        switch category {
        case .restaurant, .cafe, .bakery: return "fork.knife"
        case .hotel: return "bed.double"
        case .gasStation, .evCharger: return "fuelpump"
        case .hospital, .pharmacy: return "cross.case"
        case .school, .university, .library: return "graduationcap"
        case .store: return "bag"
        case .parking: return "p.circle"
        case .park, .nationalPark: return "leaf"
        case .airport: return "airplane"
        case .museum: return "building.columns"
        case .theater, .movieTheater: return "theatermasks"
        case .fitnessCenter: return "dumbbell"
        default: return "mappin"
        }
    }

    private func categoryColor(for item: MKMapItem) -> Color {
        guard let category = item.pointOfInterestCategory else { return .red }
        switch category {
        case .restaurant, .cafe, .bakery: return .orange
        case .hotel: return .purple
        case .hospital, .pharmacy: return .red
        case .park, .nationalPark: return .green
        case .store: return .blue
        case .gasStation, .evCharger: return .green
        default: return .red
        }
    }
}

struct PlaceDetailSheet: View {
    let place: MKMapItem
    let lookAroundScene: MKLookAroundScene?
    let route: MKRoute?
    let showDirections: Bool
    let onGetDirections: () -> Void
    let onDismiss: () -> Void
    let onOpenInMaps: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top, spacing: 14) {
                    ZStack {
                        Circle()
                            .fill(Color.red.opacity(0.12))
                            .frame(width: 50, height: 50)
                        Image(systemName: "mappin.circle.fill")
                            .font(.title)
                            .foregroundStyle(.red)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text(place.name ?? "Unknown Place")
                            .font(.title3.weight(.bold))
                        if let category = place.pointOfInterestCategory?.rawValue {
                            Text(formatCategory(category))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                }

                if let scene = lookAroundScene {
                    LookAroundPreview(initialScene: scene)
                        .frame(height: 180)
                        .clipShape(.rect(cornerRadius: 14))
                }

                if let address = formattedAddress {
                    Label {
                        Text(address)
                            .font(.subheadline)
                    } icon: {
                        Image(systemName: "location.fill")
                            .foregroundStyle(.blue)
                    }
                }

                if let phone = place.phoneNumber {
                    Label {
                        Text(phone)
                            .font(.subheadline)
                    } icon: {
                        Image(systemName: "phone.fill")
                            .foregroundStyle(.green)
                    }
                }

                if let url = place.url {
                    Label {
                        Link(url.host ?? url.absoluteString, destination: url)
                            .font(.subheadline)
                    } icon: {
                        Image(systemName: "globe")
                            .foregroundStyle(.blue)
                    }
                }

                if let route, showDirections {
                    HStack(spacing: 8) {
                        Image(systemName: "car.fill")
                            .foregroundStyle(.blue)
                        Text(formattedDistance(route.distance))
                            .font(.subheadline.weight(.medium))
                        Text("·")
                            .foregroundStyle(.secondary)
                        Text(formattedTime(route.expectedTravelTime))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(.rect(cornerRadius: 12))
                }

                HStack(spacing: 12) {
                    Button {
                        onGetDirections()
                    } label: {
                        Label("Directions", systemImage: "arrow.triangle.turn.up.right.diamond.fill")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        onOpenInMaps()
                    } label: {
                        Label("Open in Maps", systemImage: "map.fill")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(20)
        }
    }

    private var formattedAddress: String? {
        let placemark = place.placemark
        let parts = [
            placemark.subThoroughfare,
            placemark.thoroughfare,
            placemark.locality,
            placemark.administrativeArea,
            placemark.postalCode
        ].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }

    private func formatCategory(_ raw: String) -> String {
        raw.replacingOccurrences(of: "MKPOICategory", with: "")
            .replacingOccurrences(of: "([a-z])([A-Z])", with: "$1 $2", options: .regularExpression)
    }

    private func formattedDistance(_ meters: Double) -> String {
        let formatter = MKDistanceFormatter()
        formatter.unitStyle = .abbreviated
        return formatter.string(fromDistance: meters)
    }

    private func formattedTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds / 60)
        if minutes < 60 { return "\(minutes) min" }
        let hours = minutes / 60
        let remaining = minutes % 60
        return "\(hours)h \(remaining)m"
    }
}
