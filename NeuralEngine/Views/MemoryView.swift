import SwiftUI

struct MemoryView: View {
    @Bindable var viewModel: MemoryViewModel
    @State private var showClearConfirmation: Bool = false
    @State private var showAddMemory: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    statsCard
                    categoryFilter
                    memoryList
                }
                .padding(.vertical, 12)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Memory")
            .navigationBarTitleDisplayMode(.large)
            .searchable(text: $viewModel.searchText, prompt: "Search memories")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("Add Memory", systemImage: "plus") {
                            showAddMemory = true
                        }
                        Toggle("Sort by Importance", isOn: $viewModel.sortByImportance)
                        if !viewModel.memoryService.memories.isEmpty {
                            Button("Clear All", systemImage: "trash", role: .destructive) {
                                showClearConfirmation = true
                            }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .alert("Clear All Memories", isPresented: $showClearConfirmation) {
                Button("Delete All", role: .destructive) {
                    withAnimation { viewModel.clearAll() }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will permanently delete all stored memories and associative links.")
            }
            .sheet(isPresented: $showAddMemory) {
                AddMemorySheet(viewModel: viewModel)
            }
        }
    }

    private var statsCard: some View {
        HStack(spacing: 0) {
            statItem(value: "\(viewModel.memoryService.memories.count)", label: "Memories", icon: "brain.fill", color: .purple)
            Divider().frame(height: 36)
            statItem(value: "\(viewModel.totalLinks)", label: "Links", icon: "link", color: .blue)
            Divider().frame(height: 36)
            statItem(value: "\(viewModel.memoryCounts.count)", label: "Categories", icon: "tag.fill", color: .orange)
        }
        .padding(.vertical, 14)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 14))
        .padding(.horizontal, 16)
    }

    private func statItem(value: String, label: String, icon: String, color: Color) -> some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundStyle(color)
                Text(value)
                    .font(.title3.bold().monospacedDigit())
            }
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private var categoryFilter: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                filterChip(label: "All", isSelected: viewModel.selectedCategory == nil) {
                    withAnimation(.snappy) { viewModel.selectedCategory = nil }
                }
                ForEach(MemoryCategory.allCases, id: \.self) { cat in
                    let count = viewModel.memoryCounts[cat] ?? 0
                    filterChip(
                        label: "\(cat.rawValue.capitalized) (\(count))",
                        isSelected: viewModel.selectedCategory == cat
                    ) {
                        withAnimation(.snappy) { viewModel.selectedCategory = cat }
                    }
                }
            }
        }
        .contentMargins(.horizontal, 16)
        .scrollIndicators(.hidden)
    }

    private func filterChip(label: String, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.caption.weight(.medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(isSelected ? AnyShapeStyle(.blue) : AnyShapeStyle(Color(.tertiarySystemBackground)))
                .foregroundStyle(isSelected ? .white : .primary)
                .clipShape(Capsule())
        }
    }

    private var memoryList: some View {
        LazyVStack(spacing: 10) {
            if viewModel.filteredMemories.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "brain")
                        .font(.system(size: 36))
                        .foregroundStyle(.quaternary)
                    Text("No memories yet")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Text("Memories are automatically extracted from conversations")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)
            } else {
                ForEach(viewModel.filteredMemories) { memory in
                    MemoryCardView(memory: memory) {
                        withAnimation(.snappy) {
                            viewModel.deleteMemory(memory.id)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 16)
    }
}

struct MemoryCardView: View {
    let memory: MemoryEntry
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                categoryBadge
                Spacer()
                importanceStars
            }

            Text(memory.content)
                .font(.subheadline)
                .lineLimit(4)

            if !memory.keywords.isEmpty {
                ScrollView(.horizontal) {
                    HStack(spacing: 6) {
                        ForEach(memory.keywords, id: \.self) { keyword in
                            Text(keyword)
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color(.tertiarySystemBackground))
                                .clipShape(Capsule())
                        }
                    }
                }
                .scrollIndicators(.hidden)
            }

            HStack(spacing: 12) {
                Label("\(memory.accessCount)x", systemImage: "arrow.counterclockwise")
                Label(Date(timeIntervalSince1970: memory.timestamp / 1000).formatted(.relative(presentation: .named)), systemImage: "clock")
                Spacer()
                decayIndicator
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 14))
        .contextMenu {
            Button("Copy", systemImage: "doc.on.doc") {
                UIPasteboard.general.string = memory.content
            }
            Button("Delete", systemImage: "trash", role: .destructive) {
                onDelete()
            }
        }
    }

    private var categoryBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: categoryIcon)
                .font(.caption2)
            Text(memory.category.rawValue.capitalized)
                .font(.caption2.bold())
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(categoryColor.opacity(0.15))
        .foregroundStyle(categoryColor)
        .clipShape(Capsule())
    }

    private var categoryIcon: String {
        switch memory.category {
        case .preference: return "heart.fill"
        case .fact: return "book.fill"
        case .context: return "bubble.left.fill"
        case .instruction: return "list.bullet"
        case .emotion: return "face.smiling.fill"
        case .skill: return "wrench.fill"
        }
    }

    private var categoryColor: Color {
        switch memory.category {
        case .preference: return .pink
        case .fact: return .blue
        case .context: return .green
        case .instruction: return .orange
        case .emotion: return .purple
        case .skill: return .cyan
        }
    }

    private var importanceStars: some View {
        HStack(spacing: 2) {
            ForEach(1...5, id: \.self) { i in
                Image(systemName: i <= memory.importance ? "star.fill" : "star")
                    .font(.system(size: 8))
                    .foregroundStyle(i <= memory.importance ? Color.yellow : Color(.quaternaryLabel))
            }
        }
    }

    private var decayIndicator: some View {
        HStack(spacing: 3) {
            Circle()
                .fill(memory.decay > 0.7 ? .green : memory.decay > 0.3 ? .yellow : .red)
                .frame(width: 5, height: 5)
            Text("\(Int(memory.decay * 100))%")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
    }
}

struct AddMemorySheet: View {
    let viewModel: MemoryViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var content: String = ""
    @State private var category: MemoryCategory = .context
    @State private var importance: Int = 3

    var body: some View {
        NavigationStack {
            Form {
                Section("Content") {
                    TextEditor(text: $content)
                        .frame(minHeight: 100)
                }

                Section("Category") {
                    Picker("Category", selection: $category) {
                        ForEach(MemoryCategory.allCases, id: \.self) { cat in
                            Text(cat.rawValue.capitalized).tag(cat)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section("Importance") {
                    Picker("Importance", selection: $importance) {
                        ForEach(1...5, id: \.self) { i in
                            Text("\(i)").tag(i)
                        }
                    }
                    .pickerStyle(.segmented)
                }
            }
            .navigationTitle("Add Memory")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                        viewModel.addManualMemory(content: content, category: category, importance: importance)
                        dismiss()
                    }
                    .disabled(content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
    }
}
