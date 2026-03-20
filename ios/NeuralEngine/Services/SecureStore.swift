import Foundation
import Security
import LocalAuthentication

nonisolated final class SecureStore: Sendable {
    private let service: String
    private let accessGroup: String?

    init(service: String = "com.neuralengine.securestore", accessGroup: String? = nil) {
        self.service = service
        self.accessGroup = accessGroup
    }

    func getString(_ key: String) -> String? {
        guard let data = getData(key) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func setString(_ value: String, forKey key: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }
        return setData(data, forKey: key)
    }

    func getData(_ key: String) -> Data? {
        var query = baseQuery(key: key)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess else { return nil }
        return result as? Data
    }

    func setData(_ value: Data, forKey key: String) -> Bool {
        delete(key)

        var query = baseQuery(key: key)
        query[kSecValueData as String] = value
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock

        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    @discardableResult
    func delete(_ key: String) -> Bool {
        let query = baseQuery(key: key)
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    func has(_ key: String) -> Bool {
        var query = baseQuery(key: key)
        query[kSecReturnData as String] = false

        let status = SecItemCopyMatching(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    func allKeys() -> [String] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let items = result as? [[String: Any]] else { return [] }

        return items.compactMap { $0[kSecAttrAccount as String] as? String }
    }

    func removeAll() -> Bool {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }

        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    func getCodable<T: Decodable & Sendable>(_ type: T.Type, forKey key: String) -> T? {
        guard let data = getData(key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    func setCodable<T: Encodable & Sendable>(_ value: T, forKey key: String) -> Bool {
        guard let data = try? JSONEncoder().encode(value) else { return false }
        return setData(data, forKey: key)
    }

    func setStringWithBiometric(_ value: String, forKey key: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }
        delete(key)

        let access = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .userPresence,
            nil
        )

        var query = baseQuery(key: key)
        query[kSecValueData as String] = data
        if let access {
            query[kSecAttrAccessControl as String] = access
        } else {
            query[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        }

        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    func getStringWithBiometric(_ key: String) -> Data? {
        let prompt = "Authenticate to access secure data"
        let authenticationContext = LAContext()
        authenticationContext.localizedReason = prompt

        var query = baseQuery(key: key)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecUseOperationPrompt as String] = prompt
        query[kSecUseAuthenticationContext as String] = authenticationContext

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess else { return nil }
        return result as? Data
    }

    func rotateKey(oldKey: String, newKey: String) -> Bool {
        guard let data = getData(oldKey) else { return false }
        let stored = setData(data, forKey: newKey)
        if stored {
            delete(oldKey)
        }
        return stored
    }

    func audit() -> SecureStoreAudit {
        let keys = allKeys()
        var totalSize: Int = 0
        for key in keys {
            if let data = getData(key) {
                totalSize += data.count
            }
        }
        return SecureStoreAudit(
            keyCount: keys.count,
            totalSizeBytes: totalSize,
            keys: keys,
            timestamp: Date()
        )
    }

    private func baseQuery(key: String) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        return query
    }
}

nonisolated struct SecureStoreAudit: Sendable {
    let keyCount: Int
    let totalSizeBytes: Int
    let keys: [String]
    let timestamp: Date
}
