import SwiftUI
import MessageUI

struct MessageComposerView: UIViewControllerRepresentable {
    @Bindable var toolExecutor: ToolExecutor

    func makeCoordinator() -> Coordinator {
        Coordinator(toolExecutor: toolExecutor)
    }

    func makeUIViewController(context: Context) -> MFMessageComposeViewController {
        let controller = MFMessageComposeViewController()
        controller.messageComposeDelegate = context.coordinator
        controller.recipients = toolExecutor.pendingSMSTo.map { [$0] }
        controller.body = toolExecutor.pendingSMSBody
        return controller
    }

    func updateUIViewController(_ uiViewController: MFMessageComposeViewController, context: Context) {
        uiViewController.recipients = toolExecutor.pendingSMSTo.map { [$0] }
        uiViewController.body = toolExecutor.pendingSMSBody
    }

    final class Coordinator: NSObject, @preconcurrency MFMessageComposeViewControllerDelegate {
        private let toolExecutor: ToolExecutor

        init(toolExecutor: ToolExecutor) {
            self.toolExecutor = toolExecutor
        }

        @MainActor
        func messageComposeViewController(_ controller: MFMessageComposeViewController, didFinishWith result: MessageComposeResult) {
            toolExecutor.resetSMSComposerState()
            controller.dismiss(animated: true)
        }
    }
}

struct MailComposerView: UIViewControllerRepresentable {
    @Bindable var toolExecutor: ToolExecutor

    func makeCoordinator() -> Coordinator {
        Coordinator(toolExecutor: toolExecutor)
    }

    func makeUIViewController(context: Context) -> MFMailComposeViewController {
        let controller = MFMailComposeViewController()
        controller.mailComposeDelegate = context.coordinator
        controller.setToRecipients(toolExecutor.pendingEmailTo.map { [$0] } ?? [])
        controller.setSubject(toolExecutor.pendingEmailSubject ?? "")
        controller.setMessageBody(toolExecutor.pendingEmailBody ?? "", isHTML: false)
        return controller
    }

    func updateUIViewController(_ uiViewController: MFMailComposeViewController, context: Context) {
        uiViewController.setToRecipients(toolExecutor.pendingEmailTo.map { [$0] } ?? [])
        uiViewController.setSubject(toolExecutor.pendingEmailSubject ?? "")
        uiViewController.setMessageBody(toolExecutor.pendingEmailBody ?? "", isHTML: false)
    }

    final class Coordinator: NSObject, @preconcurrency MFMailComposeViewControllerDelegate {
        private let toolExecutor: ToolExecutor

        init(toolExecutor: ToolExecutor) {
            self.toolExecutor = toolExecutor
        }

        @MainActor
        func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
            toolExecutor.resetEmailComposerState()
            controller.dismiss(animated: true)
        }
    }
}
