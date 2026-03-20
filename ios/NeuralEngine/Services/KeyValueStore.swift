import Foundation

@MainActor
@Observable
class KeyValueStore {
    private let defaults: UserDefaults
    private let suiteName: String?

    init(suiteName: String? = nil) {
        self.suiteName = suiteName
        self.defaults = suiteName.flatMap { UserDefaults(suiteName: $0) } ?? .standard
    }

    func getString(_ key: String) -> String? {
        defaults.string(forKey: key)
    }

    func setString(_ value: String, forKey key: String) {
        defaults.set(value, forKey: key)
    }

    func getInt(_ key: String) -> Int? {
        defaults.object(forKey: key) != nil ? defaults.integer(forKey: key) : nil
    }

    func setInt(_ value: Int, forKey key: String) {
        defaults.set(value, forKey: key)
    }

    func getDouble(_ key: String) -> Double? {
        defaults.object(forKey: key) != nil ? defaults.double(forKey: key) : nil
    }

    func setDouble(_ value: Double, forKey key: String) {
        defaults.set(value, forKey: key)
    }

    func getBool(_ key: String) -> Bool? {
        defaults.object(forKey: key) != nil ? defaults.bool(forKey: key) : nil
    }

    func setBool(_ value: Bool, forKey key: String) {
        defaults.set(value, forKey: key)
    }

    func getData(_ key: String) -> Data? {
        defaults.data(forKey: key)
    }

    func setData(_ value: Data, forKey key: String) {
        defaults.set(value, forKey: key)
    }

    func getCodable<T: Decodable & Sendable>(_ type: T.Type, forKey key: String) -> T? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    func setCodable<T: Encodable & Sendable>(_ value: T, forKey key: String) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        defaults.set(data, forKey: key)
    }

    func remove(_ key: String) {
        defaults.removeObject(forKey: key)
    }

    func has(_ key: String) -> Bool {
        defaults.object(forKey: key) != nil
    }

    func allKeys() -> [String] {
        Array(defaults.dictionaryRepresentation().keys)
    }

    func removeAll() {
        guard let suiteName else {
            let keys = defaults.dictionaryRepresentation().keys
            for key in keys {
                defaults.removeObject(forKey: key)
            }
            return
        }
        defaults.removePersistentDomain(forName: suiteName)
    }

    func multiGet(_ keys: [String]) -> [String: Any] {
        var result: [String: Any] = [:]
        for key in keys {
            if let value = defaults.object(forKey: key) {
                result[key] = value
            }
        }
        return result
    }

    func multiSet(_ pairs: [String: Any]) {
        for (key, value) in pairs {
            defaults.set(value, forKey: key)
        }
    }

    func multiRemove(_ keys: [String]) {
        for key in keys {
            defaults.removeObject(forKey: key)
        }
    }
}
