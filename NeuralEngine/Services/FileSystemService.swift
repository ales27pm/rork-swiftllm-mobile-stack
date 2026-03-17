import Foundation
import CryptoKit

// Rork: Mark this nonisolated class as `@unchecked Sendable` because it holds a
// non-Sendable FileManager instance. Swift 6 treats non-Sendable stored
// properties in Sendable types as an error. Using `@unchecked Sendable`
// indicates we've manually audited thread-safety and are opting out of the
// compiler's Sendable checking for this type.

nonisolated final class FileSystemService: @unchecked Sendable {
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

    var modelStorageDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent("Models", isDirectory: true)
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
        excludeFromBackup(url)
        return url
    }

    var tokenizerStorageDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent("Tokenizers", isDirectory: true)
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
        excludeFromBackup(url)
        return url
    }

    var modelMetadataDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent("ModelMeta", isDirectory: true)
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    func excludeFromBackup(_ url: URL) {
        var mutableURL = url
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try? mutableURL.setResourceValues(resourceValues)
    }

    func isExcludedFromBackup(_ url: URL) -> Bool {
        guard let values = try? url.resourceValues(forKeys: [.isExcludedFromBackupKey]) else { return false }
        return values.isExcludedFromBackup ?? false
    }

    func ensureModelStorageReady() {
        let _ = modelStorageDirectory
        let _ = tokenizerStorageDirectory
        let _ = modelMetadataDirectory
    }

    func modelPath(forModelID id: String) -> URL {
        modelMetadataDirectory.appendingPathComponent("model_path_\(id).txt")
    }

    func tokenizerPath(forModelID id: String) -> URL {
        modelMetadataDirectory.appendingPathComponent("tokenizer_path_\(id).txt")
    }

    func saveModelPath(_ url: URL, forModelID id: String) throws {
        try url.path.write(to: modelPath(forModelID: id), atomically: true, encoding: .utf8)
    }

    func saveTokenizerPath(_ url: URL, forModelID id: String) throws {
        try url.path.write(to: tokenizerPath(forModelID: id), atomically: true, encoding: .utf8)
    }

    func loadModelPath(forModelID id: String) -> URL? {
        guard let path = try? String(contentsOf: modelPath(forModelID: id), encoding: .utf8) else { return nil }
        let url = URL(fileURLWithPath: path)
        return fm.fileExists(atPath: url.path) ? url : nil
    }

    func loadTokenizerPath(forModelID id: String) -> URL? {
        guard let path = try? String(contentsOf: tokenizerPath(forModelID: id), encoding: .utf8) else { return nil }
        let url = URL(fileURLWithPath: path)
        return fm.fileExists(atPath: url.path) ? url : nil
    }

    func deleteModelAssets(forModelID id: String) {
        try? fm.removeItem(at: modelPath(forModelID: id))
        try? fm.removeItem(at: tokenizerPath(forModelID: id))
    }

    func modelStorageUsageBytes() -> Int64 {
        directorySize(at: modelStorageDirectory) + directorySize(at: tokenizerStorageDirectory)
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

    func computeSHA256(for url: URL) -> String? {
        let isDir = isDirectory(at: url)
        if isDir {
            return computeDirectorySHA256(at: url)
        }
        guard let data = try? Data(contentsOf: url) else { return nil }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    func computeStreamingSHA256(for url: URL) -> String? {
        guard let stream = InputStream(url: url) else { return nil }
        stream.open()
        defer { stream.close() }

        var hasher = SHA256()
        let bufferSize = 1_048_576
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        while stream.hasBytesAvailable {
            let bytesRead = stream.read(buffer, maxLength: bufferSize)
            if bytesRead < 0 { return nil }
            if bytesRead == 0 { break }
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: buffer, count: bytesRead))
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func computeDirectorySHA256(at directory: URL) -> String? {
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        var hasher = SHA256()
        var fileURLs: [URL] = []

        while let url = enumerator.nextObject() as? URL {
            let isFile = (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
            if isFile {
                fileURLs.append(url)
            }
        }

        fileURLs.sort { $0.path < $1.path }

        for fileURL in fileURLs {
            let relativePath = fileURL.path.replacingOccurrences(of: directory.path, with: "")
            if let pathData = relativePath.data(using: .utf8) {
                hasher.update(data: pathData)
            }
            guard let data = try? Data(contentsOf: fileURL) else { continue }
            hasher.update(data: data)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    func verifyModelIntegrity(at url: URL, format: String) -> AssetIntegrityResult {
        guard fm.fileExists(atPath: url.path) else {
            return .missing
        }

        switch format {
        case "mlpackage":
            let manifest = url.appendingPathComponent("Manifest.json")
            guard fm.fileExists(atPath: manifest.path) else {
                return .corrupted("Missing Manifest.json")
            }
            if let manifestData = try? Data(contentsOf: manifest),
               let manifestJSON = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any] {
                let hasItems = manifestJSON["itemInfoEntries"] != nil || manifestJSON["rootModelIdentifier"] != nil
                if !hasItems {
                    return .corrupted("Manifest.json is malformed — missing expected keys")
                }
            }
            let dataDir = url.appendingPathComponent("Data")
            guard fm.fileExists(atPath: dataDir.path) else {
                return .corrupted("Missing Data directory")
            }
            let dataContents = listContents(of: dataDir)
            if dataContents.isEmpty {
                return .corrupted("Data directory is empty — incomplete download")
            }
            return .intact

        case "mlmodelc":
            guard isDirectory(at: url) else {
                return .corrupted("Expected directory for compiled model")
            }
            let contents = listContents(of: url)
            if contents.isEmpty {
                return .corrupted("Compiled model directory is empty")
            }
            let hasModelFile = contents.contains { $0.pathExtension == "espresso" || $0.lastPathComponent == "model.mil" || $0.lastPathComponent == "coremldata.bin" || $0.pathExtension == "bin" }
            if !hasModelFile {
                return .corrupted("No valid model data files found")
            }
            let metadataIndicators = ["metadata.json", "coremldata.bin", "model.mil"]
            let presentMetadata = contents.filter { metadataIndicators.contains($0.lastPathComponent) }
            if presentMetadata.isEmpty {
                return .corrupted("Compiled model metadata stripped by OS — recompilation required")
            }
            let totalSize = directorySize(at: url)
            if totalSize < 1024 {
                return .corrupted("Compiled model directory too small — likely stripped or truncated")
            }
            return .intact

        case "gguf":
            guard let size = fileSize(at: url), size > 1024 else {
                return .corrupted("GGUF file too small — likely truncated")
            }
            if let data = try? Data(contentsOf: url, options: .mappedIfSafe) {
                let magic: [UInt8] = [0x47, 0x47, 0x55, 0x46]
                let header = Array(data.prefix(4))
                if header != magic {
                    return .corrupted("Invalid GGUF magic bytes")
                }
            }
            return .intact

        default:
            return .intact
        }
    }

    func verifyIntegrity(for manifest: ModelManifest, at url: URL) -> AssetIntegrityResult {
        guard fm.fileExists(atPath: url.path) else {
            return .missing
        }

        let structuralResult = verifyModelIntegrity(at: url, format: manifest.format.rawValue)
        guard structuralResult.isValid else {
            return structuralResult
        }

        guard !manifest.checksum.isEmpty else {
            return .intact
        }

        let actual: String?
        if isDirectory(at: url) {
            actual = computeDirectorySHA256(at: url)
        } else {
            actual = computeStreamingSHA256(for: url)
        }

        guard let actual else {
            return .corrupted("Unable to compute SHA-256 hash")
        }

        if actual != manifest.checksum {
            return .checksumMismatch(expected: manifest.checksum, actual: actual)
        }

        return .intact
    }

    func checksumPath(forModelID id: String) -> URL {
        modelMetadataDirectory.appendingPathComponent("checksum_\(id).txt")
    }

    func saveChecksum(_ hash: String, forModelID id: String) {
        _ = writeString(hash, to: checksumPath(forModelID: id))
    }

    func loadChecksum(forModelID id: String) -> String? {
        readString(at: checksumPath(forModelID: id))
    }
}

nonisolated enum AssetIntegrityResult: Sendable, Equatable {
    case intact
    case missing
    case corrupted(String)
    case checksumMismatch(expected: String, actual: String)

    var isValid: Bool {
        if case .intact = self { return true }
        return false
    }

    var description: String {
        switch self {
        case .intact: return "Integrity verified"
        case .missing: return "Asset not found"
        case .corrupted(let reason): return "Corrupted: \(reason)"
        case .checksumMismatch(let expected, let actual): return "Checksum mismatch: expected \(expected.prefix(12))… got \(actual.prefix(12))…"
        }
    }
}
