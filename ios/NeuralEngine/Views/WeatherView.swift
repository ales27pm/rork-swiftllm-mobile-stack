import SwiftUI
import CoreLocation

struct WeatherView: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var viewModel: WeatherViewModel = WeatherViewModel()

    private let detailColumns: [GridItem] = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12)
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                content
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Weather")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    viewModel.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(viewModel.isLoading)
            }
        }
        .task {
            viewModel.refreshIfNeeded()
        }
        .refreshable {
            viewModel.refresh()
        }
    }

    @ViewBuilder
    private var content: some View {
        if let snapshot = viewModel.snapshot {
            currentConditionsCard(snapshot: snapshot)
            hourlyForecastSection(snapshot: snapshot)
            dailyForecastSection(snapshot: snapshot)
            detailGrid(snapshot: snapshot)
            attributionSection(snapshot: snapshot)

            if let errorMessage = viewModel.errorMessage {
                inlineErrorCard(message: errorMessage)
            }
        } else if viewModel.isLoading {
            loadingState
        } else {
            emptyState
        }
    }

    private var loadingState: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text("Loading local weather…")
                .font(.headline)
            Text("Fetching current conditions, hourly forecast, and the upcoming outlook for your location.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("Weather Unavailable", systemImage: "cloud.slash")
        } description: {
            Text(viewModel.errorMessage ?? "Weather data could not be loaded yet.")
        } actions: {
            Button("Try Again") {
                viewModel.refresh()
            }
            if viewModel.authorizationStatus == .denied || viewModel.authorizationStatus == .restricted {
                Button("Open Settings") {
                    viewModel.openSystemSettings()
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 60)
    }

    private func currentConditionsCard(snapshot: WeatherSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(snapshot.locationName)
                        .font(.title2.weight(.semibold))
                    Text(snapshot.current.temperature)
                        .font(.system(size: 56, weight: .thin, design: .default))
                    Text(snapshot.current.conditionDescription)
                        .font(.headline)
                        .foregroundStyle(.secondary)
                    Text("Feels like \(snapshot.current.feelsLike)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 12)

                Image(systemName: snapshot.current.symbolName)
                    .symbolRenderingMode(.multicolor)
                    .font(.system(size: 54))
                    .accessibilityHidden(true)
            }

            Divider()

            HStack(spacing: 16) {
                weatherMetaPill(title: "Updated", value: snapshot.updatedAt.formatted(.dateTime.hour().minute()))
                weatherMetaPill(title: "Coordinates", value: coordinateString(snapshot: snapshot))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
        .background(
            LinearGradient(
                colors: [Color.blue.opacity(0.18), Color.cyan.opacity(0.10), Color(.secondarySystemGroupedBackground)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: .rect(cornerRadius: 24)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 24)
                .strokeBorder(Color.white.opacity(colorScheme == .dark ? 0.08 : 0.35), lineWidth: 1)
        }
    }

    private func hourlyForecastSection(snapshot: WeatherSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle(title: "Next 12 Hours", systemImage: "clock")

            ScrollView(.horizontal) {
                HStack(spacing: 12) {
                    ForEach(snapshot.hourly) { hour in
                        VStack(spacing: 10) {
                            Text(hour.date.formatted(.dateTime.hour()))
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.secondary)
                            Image(systemName: hour.symbolName)
                                .symbolRenderingMode(.multicolor)
                                .font(.title3)
                            Text(hour.temperature)
                                .font(.headline)
                            Text(hour.precipitationChance)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .frame(width: 74)
                        .padding(.vertical, 14)
                        .background(.ultraThinMaterial, in: .rect(cornerRadius: 18))
                    }
                }
            }
            .contentMargins(.horizontal, 0)
            .scrollIndicators(.hidden)
        }
    }

    private func dailyForecastSection(snapshot: WeatherSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle(title: "5-Day Outlook", systemImage: "calendar")

            VStack(spacing: 0) {
                ForEach(Array(snapshot.daily.enumerated()), id: \.element.id) { index, day in
                    HStack(spacing: 12) {
                        Text(day.date.formatted(.dateTime.weekday(.abbreviated)))
                            .font(.headline)
                            .frame(width: 44, alignment: .leading)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(day.conditionDescription)
                                .font(.subheadline)
                            Text("H \(day.highTemperature) · L \(day.lowTemperature)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .padding(.vertical, 12)

                    if index < snapshot.daily.count - 1 {
                        Divider()
                    }
                }
            }
            .padding(.horizontal, 16)
            .background(.ultraThinMaterial, in: .rect(cornerRadius: 20))
        }
    }

    private func detailGrid(snapshot: WeatherSnapshot) -> some View {
        LazyVGrid(columns: detailColumns, spacing: 12) {
            weatherDetailCard(title: "Humidity", value: snapshot.current.humidity, systemImage: "humidity.fill")
            weatherDetailCard(title: "Wind", value: snapshot.current.wind, systemImage: "wind")
            weatherDetailCard(title: "UV Index", value: snapshot.current.uvIndex, systemImage: "sun.max.fill")
            weatherDetailCard(title: "Visibility", value: snapshot.current.visibility, systemImage: "eye.fill")
            weatherDetailCard(title: "Cloud Cover", value: snapshot.current.cloudCover, systemImage: "cloud.fill")
            weatherDetailCard(title: "Daylight", value: snapshot.current.isDaylight ? "Yes" : "No", systemImage: snapshot.current.isDaylight ? "sun.max.fill" : "moon.fill")
        }
    }

    private func attributionSection(snapshot: WeatherSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle(title: "Source", systemImage: "apple.logo")

            HStack(alignment: .center, spacing: 14) {
                if let markURL = currentAttributionMarkURL(for: snapshot.attribution) {
                    AsyncImage(url: markURL) { phase in
                        if let image = phase.image {
                            image
                                .resizable()
                                .scaledToFit()
                                .frame(width: 108, height: 26)
                        } else {
                            ProgressView()
                                .frame(width: 108, height: 26)
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(snapshot.attribution.notice)
                        .font(.subheadline)
                    if let legalURL = URL(string: snapshot.attribution.legalPageURL) {
                        Link("Legal Attribution", destination: legalURL)
                            .font(.caption.weight(.semibold))
                    }
                }

                Spacer()
            }
            .padding(16)
            .background(.ultraThinMaterial, in: .rect(cornerRadius: 20))
        }
    }

    private func inlineErrorCard(message: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(16)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 18))
    }

    private func weatherMetaPill(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: .capsule)
    }

    private func weatherDetailCard(title: String, value: String, systemImage: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 20))
    }

    private func sectionTitle(title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.headline)
    }

    private func currentAttributionMarkURL(for attribution: WeatherSnapshot.Attribution) -> URL? {
        let urlString: String?
        switch colorScheme {
        case .dark:
            urlString = attribution.combinedMarkDarkURL ?? attribution.combinedMarkLightURL
        case .light:
            urlString = attribution.combinedMarkLightURL ?? attribution.combinedMarkDarkURL
        @unknown default:
            urlString = attribution.combinedMarkLightURL ?? attribution.combinedMarkDarkURL
        }

        guard let urlString else { return nil }
        return URL(string: urlString)
    }

    private func coordinateString(snapshot: WeatherSnapshot) -> String {
        "\(String(format: "%.2f", snapshot.latitude)), \(String(format: "%.2f", snapshot.longitude))"
    }
}
