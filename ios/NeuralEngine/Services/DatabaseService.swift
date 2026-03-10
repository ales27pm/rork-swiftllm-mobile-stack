import Foundation
import SQLite3

nonisolated final class DatabaseService: Sendable {
    private let dbPath: String
    private let queue: DispatchQueue

    init(name: String = "neuralengine.sqlite3") {
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.dbPath = documentsDir.appendingPathComponent(name).path
        self.queue = DispatchQueue(label: "com.neuralengine.database.\(name)")
    }

    private func openDB() -> OpaquePointer? {
        var db: OpaquePointer?
        guard sqlite3_open(dbPath, &db) == SQLITE_OK else {
            sqlite3_close(db)
            return nil
        }
        sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nil, nil, nil)
        sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nil, nil, nil)
        return db
    }

    func execute(_ sql: String, params: [Any] = []) -> Bool {
        queue.sync {
            guard let db = openDB() else { return false }
            defer { sqlite3_close(db) }

            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                sqlite3_finalize(stmt)
                return false
            }
            defer { sqlite3_finalize(stmt) }

            bindParams(stmt: stmt, params: params)

            let result = sqlite3_step(stmt)
            return result == SQLITE_DONE || result == SQLITE_ROW
        }
    }

    func query(_ sql: String, params: [Any] = []) -> [[String: Any]] {
        queue.sync {
            guard let db = openDB() else { return [] }
            defer { sqlite3_close(db) }

            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                sqlite3_finalize(stmt)
                return []
            }
            defer { sqlite3_finalize(stmt) }

            bindParams(stmt: stmt, params: params)

            var rows: [[String: Any]] = []
            let columnCount = sqlite3_column_count(stmt)

            while sqlite3_step(stmt) == SQLITE_ROW {
                var row: [String: Any] = [:]
                for i in 0..<columnCount {
                    let name = String(cString: sqlite3_column_name(stmt, i))
                    switch sqlite3_column_type(stmt, i) {
                    case SQLITE_INTEGER:
                        row[name] = sqlite3_column_int64(stmt, i)
                    case SQLITE_FLOAT:
                        row[name] = sqlite3_column_double(stmt, i)
                    case SQLITE_TEXT:
                        if let cStr = sqlite3_column_text(stmt, i) {
                            row[name] = String(cString: cStr)
                        }
                    case SQLITE_BLOB:
                        if let ptr = sqlite3_column_blob(stmt, i) {
                            let size = sqlite3_column_bytes(stmt, i)
                            row[name] = Data(bytes: ptr, count: Int(size))
                        }
                    case SQLITE_NULL:
                        row[name] = NSNull()
                    default:
                        break
                    }
                }
                rows.append(row)
            }

            return rows
        }
    }

    func queryScalar(_ sql: String, params: [Any] = []) -> Any? {
        let rows = query(sql, params: params)
        return rows.first?.values.first
    }

    func transaction(_ operations: () -> Bool) -> Bool {
        queue.sync {
            guard let db = openDB() else { return false }
            defer { sqlite3_close(db) }

            sqlite3_exec(db, "BEGIN TRANSACTION;", nil, nil, nil)
            let success = operations()
            if success {
                sqlite3_exec(db, "COMMIT;", nil, nil, nil)
            } else {
                sqlite3_exec(db, "ROLLBACK;", nil, nil, nil)
            }
            return success
        }
    }

    func tableExists(_ tableName: String) -> Bool {
        let rows = query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            params: [tableName]
        )
        return !rows.isEmpty
    }

    func deleteDatabase() -> Bool {
        queue.sync {
            let fm = FileManager.default
            do {
                if fm.fileExists(atPath: dbPath) {
                    try fm.removeItem(atPath: dbPath)
                }
                let walPath = dbPath + "-wal"
                if fm.fileExists(atPath: walPath) {
                    try fm.removeItem(atPath: walPath)
                }
                let shmPath = dbPath + "-shm"
                if fm.fileExists(atPath: shmPath) {
                    try fm.removeItem(atPath: shmPath)
                }
                return true
            } catch {
                return false
            }
        }
    }

    private func bindParams(stmt: OpaquePointer?, params: [Any]) {
        for (index, param) in params.enumerated() {
            let i = Int32(index + 1)
            switch param {
            case let value as String:
                sqlite3_bind_text(stmt, i, (value as NSString).utf8String, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
            case let value as Int:
                sqlite3_bind_int64(stmt, i, Int64(value))
            case let value as Int64:
                sqlite3_bind_int64(stmt, i, value)
            case let value as Double:
                sqlite3_bind_double(stmt, i, value)
            case let value as Data:
                value.withUnsafeBytes { ptr in
                    sqlite3_bind_blob(stmt, i, ptr.baseAddress, Int32(value.count), unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                }
            case is NSNull:
                sqlite3_bind_null(stmt, i)
            case let value as Bool:
                sqlite3_bind_int(stmt, i, value ? 1 : 0)
            default:
                sqlite3_bind_text(stmt, i, ("\(param)" as NSString).utf8String, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
            }
        }
    }
}
