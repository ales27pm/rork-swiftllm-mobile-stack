import Foundation

actor KVCacheArena {
    private var pages: [KVPage] = []
    private var freePageIndices: [Int] = []
    private let pageSize: Int
    private let layerCount: Int

    init(pageSize: Int = 128, layerCount: Int = 32) {
        self.pageSize = pageSize
        self.layerCount = layerCount
    }

    func allocatePage(tokenStart: Int, tokenCount: Int) -> KVPage {
        if let freeIndex = freePageIndices.popLast() {
            var page = pages[freeIndex]
            page = KVPage(
                id: page.id,
                tokenStart: tokenStart,
                tokenCount: tokenCount,
                layerCount: layerCount,
                pageSize: pageSize,
                isActive: true
            )
            pages[freeIndex] = page
            return page
        }

        let page = KVPage(
            tokenStart: tokenStart,
            tokenCount: tokenCount,
            layerCount: layerCount,
            pageSize: pageSize,
            isActive: true
        )
        pages.append(page)
        return page
    }

    func freePage(at index: Int) {
        guard index < pages.count else { return }
        pages[index].isActive = false
        freePageIndices.append(index)
    }

    func reset() {
        pages.removeAll(keepingCapacity: true)
        freePageIndices.removeAll(keepingCapacity: true)
    }

    var activePageCount: Int {
        pages.filter(\.isActive).count
    }

    var totalPageCount: Int {
        pages.count
    }

    var estimatedMemoryBytes: Int64 {
        Int64(pages.filter(\.isActive).reduce(0) { $0 + $1.memoryEstimateBytes })
    }
}
