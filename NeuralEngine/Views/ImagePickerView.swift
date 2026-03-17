import SwiftUI
import PhotosUI

struct ImagePickerView: UIViewControllerRepresentable {
    let onPick: (UIImage) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.selectionLimit = 1
        config.filter = .images
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let onPick: (UIImage) -> Void

        init(onPick: @escaping (UIImage) -> Void) {
            self.onPick = onPick
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else { return }

            let onPick = self.onPick
            provider.loadObject(ofClass: UIImage.self) { image, _ in
                guard let uiImage = image as? UIImage else { return }
                Task { @MainActor in
                    onPick(uiImage)
                }
            }
        }
    }
}
