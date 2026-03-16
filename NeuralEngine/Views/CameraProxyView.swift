import SwiftUI
import AVFoundation

struct CameraProxyView: View {
    let onCapture: (UIImage) -> Void

    var body: some View {
        Group {
            #if targetEnvironment(simulator)
            CameraUnavailablePlaceholder()
            #else
            if AVCaptureDevice.default(for: .video) != nil {
                CameraCaptureView(onCapture: onCapture)
            } else {
                CameraUnavailablePlaceholder()
            }
            #endif
        }
    }
}

struct CameraUnavailablePlaceholder: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "camera.fill")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Camera Preview")
                .font(.title2)
                .fontWeight(.semibold)
            Text("Install this app on your device\nvia the Rork App to use the camera.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }
}

struct CameraCaptureView: UIViewControllerRepresentable {
    let onCapture: (UIImage) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onCapture: onCapture)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onCapture: (UIImage) -> Void

        init(onCapture: @escaping (UIImage) -> Void) {
            self.onCapture = onCapture
        }

        nonisolated func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            picker.dismiss(animated: true)
            let onCapture = self.onCapture
            if let image = info[.originalImage] as? UIImage {
                Task { @MainActor in
                    onCapture(image)
                }
            }
        }

        nonisolated func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
