import SwiftUI

@Observable
@MainActor
class MemoryViewModel {
    let memoryService: MemoryService

    var searchText: String = ""
    var selectedCategory: MemoryCategory? = nil
    var sortByImportance: Bool = false

    init(memoryService: MemoryService) {
        self.memoryService = memoryService
    }

    var filteredMemories: [MemoryEntry] {
        var result = memoryService.memories

        if let cat = selectedCategory {
            result = result.filter { $0.category == cat }
        }

        if !searchText.isEmpty {
            let results = memoryService.searchMemories(query: searchText, maxResults: 50)
            let ids = Set(results.map(\.memory.id))
            result = result.filter { ids.contains($0.id) }
        }

        if sortByImportance {
            result.sort { $0.importance > $1.importance }
        }

        return result
    }

    var memoryCounts: [MemoryCategory: Int] {
        var counts: [MemoryCategory: Int] = [:]
        for m in memoryService.memories {
            counts[m.category, default: 0] += 1
        }
        return counts
    }

    var totalLinks: Int {
        memoryService.associativeLinks.count
    }

    func deleteMemory(_ id: String) {
        memoryService.deleteMemory(id)
    }

    func clearAll() {
        memoryService.clearAllMemories()
    }

    func addManualMemory(content: String, category: MemoryCategory, importance: Int) {
        let memory = MemoryEntry(
            content: content,
            category: category,
            importance: importance,
            source: .manual
        )
        memoryService.addMemory(memory)
    }
}
