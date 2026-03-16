import SwiftUI
import WebKit

struct InAppBrowserView: View {
    let url: URL
    let title: String
    @Environment(\.dismiss) private var dismiss
    @State private var isLoading: Bool = true
    @State private var progress: Double = 0
    @State private var canGoBack: Bool = false
    @State private var canGoForward: Bool = false
    @State private var currentURL: URL?
    @State private var webViewProxy = WebViewProxy()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if isLoading {
                    ProgressView(value: progress)
                        .tint(.blue)
                        .scaleEffect(y: 1.5)
                }

                WebViewRepresentable(
                    url: url,
                    proxy: webViewProxy,
                    isLoading: $isLoading,
                    progress: $progress,
                    canGoBack: $canGoBack,
                    canGoForward: $canGoForward,
                    currentURL: $currentURL
                )
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 1) {
                        Text(title)
                            .font(.caption.bold())
                            .lineLimit(1)

                        Text(currentURL?.host ?? url.host ?? "")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }

                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 16) {
                        Button {
                            webViewProxy.goBack()
                        } label: {
                            Image(systemName: "chevron.left")
                                .font(.subheadline.weight(.semibold))
                        }
                        .disabled(!canGoBack)

                        Button {
                            webViewProxy.goForward()
                        } label: {
                            Image(systemName: "chevron.right")
                                .font(.subheadline.weight(.semibold))
                        }
                        .disabled(!canGoForward)

                        if isLoading {
                            Button {
                                webViewProxy.stopLoading()
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.subheadline.weight(.semibold))
                            }
                        } else {
                            Button {
                                webViewProxy.reload()
                            } label: {
                                Image(systemName: "arrow.clockwise")
                                    .font(.subheadline.weight(.semibold))
                            }
                        }
                    }
                }

                ToolbarItem(placement: .bottomBar) {
                    HStack {
                        ShareLink(item: currentURL ?? url) {
                            Image(systemName: "square.and.arrow.up")
                        }

                        Spacer()

                        Button {
                            if let shareURL = currentURL ?? Optional(url) {
                                UIApplication.shared.open(shareURL)
                            }
                        } label: {
                            Image(systemName: "safari")
                        }
                    }
                }
            }
        }
    }
}

@Observable
class WebViewProxy {
    var webView: WKWebView?

    func goBack() { webView?.goBack() }
    func goForward() { webView?.goForward() }
    func reload() { webView?.reload() }
    func stopLoading() { webView?.stopLoading() }
}

struct WebViewRepresentable: UIViewRepresentable {
    let url: URL
    let proxy: WebViewProxy
    @Binding var isLoading: Bool
    @Binding var progress: Double
    @Binding var canGoBack: Bool
    @Binding var canGoForward: Bool
    @Binding var currentURL: URL?

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.allowsBackForwardNavigationGestures = true

        context.coordinator.progressObservation = webView.observe(\.estimatedProgress) { view, _ in
            Task { @MainActor in
                self.progress = view.estimatedProgress
            }
        }
        context.coordinator.loadingObservation = webView.observe(\.isLoading) { view, _ in
            Task { @MainActor in
                self.isLoading = view.isLoading
                self.canGoBack = view.canGoBack
                self.canGoForward = view.canGoForward
                self.currentURL = view.url
            }
        }

        proxy.webView = webView
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    class Coordinator: NSObject, WKNavigationDelegate {
        let parent: WebViewRepresentable
        var progressObservation: NSKeyValueObservation?
        var loadingObservation: NSKeyValueObservation?

        init(_ parent: WebViewRepresentable) {
            self.parent = parent
        }

        nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            Task { @MainActor in
                parent.canGoBack = webView.canGoBack
                parent.canGoForward = webView.canGoForward
                parent.currentURL = webView.url
            }
        }
    }
}
