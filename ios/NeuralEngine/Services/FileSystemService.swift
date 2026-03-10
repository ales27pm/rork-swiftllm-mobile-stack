import Foundation

nonisolated final class FileSystemService: Sendable {
    private let fm = FileManager.default

    var documentsDirectory: URL {
        fm.urls(for: .documentDirectory, in: .userDomainMask).first!
    }

    var cachesDirectory: URL {
        fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
    }

    var temporaryDirectory: URL {
        fm.temporaryDirectory
    }

    var appSupportDirectory: URL {
        let url = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    func readString(at path: URL) -> String? {
        try? String(contentsOf: path, encoding: .utf8)
    }

    func writeString(_ content: String, to path: URL) -> Bool {
        do {
            let dir = path.deletingLastPathComponent()
            if !fm.fileExists(atPath: dir.path) {
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            try content.write(to: path, atomically: true, encoding: .utf8)
            return true
        } catch {
            return false
        }
    }

    func readData(at path: URL) -> Data? {
        try? Data(contentsOf: path)
    }

    func writeData(_ data: Data, to path: URL) -> Bool {
        do {
            let dir = path.deletingLastPathComponent()
            if !fm.fileExists(atPath: dir.path) {
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            try data.write(to: path, options: .atomic)
            return true
        } catch {
            return false
        }
    }

    func exists(at path: URL) -> Bool {
        fm.fileExists(atPath: path.path)
    }

    func isDirectory(at path: URL) -> Bool {
        var isDir: ObjCBool = false
        fm.fileExists(atPath: path.path, isDirectory: &isDir)
        return isDir.boolValue
    }

    func delete(at path: URL) -> Bool {
        do {
            try fm.removeItem(at: path)
            return true
        } catch {
            return false
        }
    }

    func createDirectory(at path: URL) -> Bool {
        do {
            try fm.createDirectory(at: path, withIntermediateDirectories: true)
            return true
        } catch {
            return false
        }
    }

    func listContents(of directory: URL) -> [URL] {
        (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey])) ?? []
    }

    func copy(from source: URL, to destination: URL) -> Bool {
        do {
            let dir = destination.deletingLastPathComponent()
            if !fm.fileExists(atPath: dir.path) {
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.copyItem(at: source, to: destination)
            return true
        } catch {
            return false
        }
    }

    func move(from source: URL, to destination: URL) -> Bool {
        do {
            let dir = destination.deletingLastPathComponent()
            if !fm.fileExists(atPath: dir.path) {
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: source, to: destination)
            return true
        } catch {
            return false
        }
    }

    func fileSize(at path: URL) -> Int64? {
        guard let attrs = try? fm.attributesOfItem(atPath: path.path) else { return nil }
        return attrs[.size] as? Int64
    }

    func modificationDate(at path: URL) -> Date? {
        guard let attrs = try? fm.attributesOfItem(atPath: path.path) else { return nil }
        return attrs[.modificationDate] as? Date
    }

    func availableDiskSpace() -> Int64? {
        guard let attrs = try? fm.attributesOfFileSystem(forPath: documentsDirectory.path) else { return nil }
        return attrs[.systemFreeSize] as? Int64
    }

    func totalDiskSpace() -> Int64? {
        guard let attrs = try? fm.attributesOfFileSystem(forPath: documentsDirectory.path) else { return nil }
        return attrs[.systemSize] as? Int64
    }

    func directorySize(at path: URL) -> Int64 {
        guard let enumerator = fm.enumerator(at: path, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
            return 0
        }
        var totalSize: Int64 = 0
        while let url = enumerator.nextObject() as? URL {
            if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(size)
            }
        }
        return totalSize
    }

    func clearCaches() -> Bool {
        let contents = listContents(of: cachesDirectory)
        var success = true
        for url in contents {
            if !delete(at: url) { success = false }
        }
        return success
    }

    func clearTemporary() -> Bool {
        let contents = listContents(of: temporaryDirectory)
        var success = true
        for url in contents {
            if !delete(at: url) { success = false }
        }
        return success
    }
}
