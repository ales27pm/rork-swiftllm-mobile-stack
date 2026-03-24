import Foundation

nonisolated struct WeatherSnapshot: Codable, Sendable {
    let latitude: Double
    let longitude: Double
    let locationName: String
    let updatedAt: Date
    let current: CurrentConditions
    let hourly: [HourlyForecastEntry]
    let daily: [DailyForecastEntry]
    let attribution: Attribution

    nonisolated struct CurrentConditions: Codable, Sendable {
        let temperature: String
        let feelsLike: String
        let conditionDescription: String
        let symbolName: String
        let humidity: String
        let wind: String
        let uvIndex: String
        let visibility: String
        let cloudCover: String
        let isDaylight: Bool
    }

    nonisolated struct HourlyForecastEntry: Codable, Sendable, Identifiable {
        let date: Date
        let temperature: String
        let symbolName: String
        let precipitationChance: String

        var id: Date { date }
    }

    nonisolated struct DailyForecastEntry: Codable, Sendable, Identifiable {
        let date: Date
        let highTemperature: String
        let lowTemperature: String
        let conditionDescription: String

        var id: Date { date }
    }

    nonisolated struct Attribution: Codable, Sendable {
        let notice: String
        let legalPageURL: String
        let combinedMarkLightURL: String?
        let combinedMarkDarkURL: String?
    }
}
