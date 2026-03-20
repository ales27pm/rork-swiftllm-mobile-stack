import SwiftUI

struct SystemPromptTemplatesView: View {
    @Bindable var chatViewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTemplateId: String = "default"
    @State private var showCustomEditor: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    activePersonaCard
                    templatesGrid
                    customSection
                }
                .padding(.vertical, 16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Personas")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                selectedTemplateId = identifyActiveTemplate()
            }
        }
    }

    private var activePersonaCard: some View {
        let template = SystemPromptTemplate.builtIn.first { $0.id == selectedTemplateId } ?? SystemPromptTemplate.builtIn[0]
        return VStack(spacing: 10) {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(templateColor(template.accentColorName).opacity(0.15))
                        .frame(width: 48, height: 48)
                    Image(systemName: template.icon)
                        .font(.title3)
                        .foregroundStyle(templateColor(template.accentColorName))
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Active Persona")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.green)
                    }
                    Text(template.name)
                        .font(.headline)
                }

                Spacer()
            }

            Text(template.description)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(.rect(cornerRadius: 16))
        .padding(.horizontal, 16)
    }

    private var templatesGrid: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Choose Persona")
                .font(.subheadline.bold())
                .padding(.horizontal, 16)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(SystemPromptTemplate.builtIn) { template in
                    templateCard(template)
                }
            }
            .padding(.horizontal, 16)
        }
    }

    private func templateCard(_ template: SystemPromptTemplate) -> some View {
        let isSelected = selectedTemplateId == template.id
        let color = templateColor(template.accentColorName)

        return Button {
            withAnimation(.spring(duration: 0.3)) {
                selectedTemplateId = template.id
                chatViewModel.systemPrompt = template.prompt
                chatViewModel.saveSettings()
            }
        } label: {
            VStack(spacing: 10) {
                ZStack {
                    Circle()
                        .fill(color.opacity(isSelected ? 0.2 : 0.08))
                        .frame(width: 44, height: 44)
                    Image(systemName: template.icon)
                        .font(.system(size: 18))
                        .foregroundStyle(color)
                }

                VStack(spacing: 3) {
                    Text(template.name)
                        .font(.caption.bold())
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    Text(template.description)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .padding(.horizontal, 8)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(.rect(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(isSelected ? color : .clear, lineWidth: 2)
            )
        }
    }

    private var customSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Custom Prompt")
                .font(.subheadline.bold())
                .padding(.horizontal, 16)

            Button {
                showCustomEditor = true
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "square.and.pencil")
                        .font(.title3)
                        .foregroundStyle(.blue)
                        .frame(width: 36)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Edit System Prompt")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.primary)
                        Text(String(chatViewModel.systemPrompt.prefix(80)) + (chatViewModel.systemPrompt.count > 80 ? "..." : ""))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(.quaternary)
                }
                .padding(14)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(.rect(cornerRadius: 14))
            }
            .padding(.horizontal, 16)
        }
        .sheet(isPresented: $showCustomEditor) {
            CustomPromptEditor(chatViewModel: chatViewModel)
        }
    }

    private func identifyActiveTemplate() -> String {
        let current = chatViewModel.systemPrompt
        for template in SystemPromptTemplate.builtIn {
            if template.prompt == current {
                return template.id
            }
        }
        return "default"
    }

    private func templateColor(_ name: String) -> Color {
        switch name {
        case "blue": return .blue
        case "green": return .green
        case "purple": return .purple
        case "orange": return .orange
        case "teal": return .teal
        case "red": return .red
        default: return .blue
        }
    }
}

struct CustomPromptEditor: View {
    @Bindable var chatViewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var draftPrompt: String = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                TextEditor(text: $draftPrompt)
                    .font(.subheadline)
                    .padding(16)

                Divider()

                HStack(spacing: 12) {
                    Text("\(draftPrompt.count) characters")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Spacer()
                    Button("Reset to Default") {
                        draftPrompt = SystemPromptTemplate.builtIn[0].prompt
                    }
                    .font(.caption)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color(.secondarySystemBackground))
            }
            .navigationTitle("System Prompt")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        chatViewModel.systemPrompt = draftPrompt
                        chatViewModel.saveSettings()
                        dismiss()
                    }
                }
            }
            .onAppear {
                draftPrompt = chatViewModel.systemPrompt
            }
        }
    }
}
