import Foundation
import CoreLocation
import WeatherKit

nonisolated final class WeatherKitService: Sendable {
    func snapshot(for location: CLLocation, locationName: String) async throws -> WeatherSnapshot {
        let (current, hourly, daily) = try await WeatherService.shared.weather(
            for: location,
            including: .current,
            .hourly,
            .daily
        )

        let attribution = try? await WeatherService.shared.attribution

        return WeatherSnapshot(
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude,
            locationName: locationName,
            updatedAt: Date(),
            current: WeatherSnapshot.CurrentConditions(
                temperature: current.temperature.formatted(),
                feelsLike: current.apparentTemperature.formatted(),
                conditionDescription: current.condition.description,
                symbolName: current.symbolName,
                humidity: Self.percentageString(from: current.humidity),
                wind: Self.windSummary(speed: current.wind.speed, direction: current.wind.direction),
                uvIndex: String(current.uvIndex.value),
                visibility: current.visibility.formatted(),
                cloudCover: Self.percentageString(from: current.cloudCover),
                isDaylight: current.isDaylight
            ),
            hourly: Array(hourly.forecast.prefix(12)).map {
                WeatherSnapshot.HourlyForecastEntry(
                    date: $0.date,
                    temperature: $0.temperature.formatted(),
                    symbolName: $0.symbolName,
                    precipitationChance: Self.percentageString(from: $0.precipitationChance)
                )
            },
            daily: Array(daily.forecast.prefix(5)).map {
                WeatherSnapshot.DailyForecastEntry(
                    date: $0.date,
                    highTemperature: $0.highTemperature.formatted(),
                    lowTemperature: $0.lowTemperature.formatted(),
                    conditionDescription: $0.condition.description
                )
            },
            attribution: WeatherSnapshot.Attribution(
                notice: "Weather data provided by Apple Weather.",
                legalPageURL: attribution?.legalPageURL.absoluteString ?? "https://weather.apple.com",
                combinedMarkLightURL: attribution?.combinedMarkLightURL.absoluteString,
                combinedMarkDarkURL: attribution?.combinedMarkDarkURL.absoluteString
            )
        )
    }

    static func userFacingErrorMessage(for error: Error) -> String {
        if let weatherError = error as? WeatherError {
            switch weatherError {
            case .permissionDenied:
                return "WeatherKit is not available for this build yet. Enable the WeatherKit capability for the app identifier and try again."
            case .unknown:
                return "Apple Weather is temporarily unavailable. Try again in a moment."
            @unknown default:
                return error.localizedDescription
            }
        }

        return error.localizedDescription
    }

    private static func percentageString(from value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    private static func windSummary(speed: Measurement<UnitSpeed>, direction: Measurement<UnitAngle>) -> String {
        let degrees = direction.converted(to: .degrees).value
        return "\(compassDirection(from: degrees)) \(speed.formatted())"
    }

    private static func compassDirection(from degrees: Double) -> String {
        let directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        let normalized = degrees.truncatingRemainder(dividingBy: 360)
        let positive = normalized >= 0 ? normalized : normalized + 360
        let index = Int((positive / 45.0).rounded()) % directions.count
        return directions[index]
    }
}
