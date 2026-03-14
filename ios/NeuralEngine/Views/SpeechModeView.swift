import SwiftUI
import Combine

struct SpeechModeView: View {
    @Bindable var viewModel: SpeechViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var wavePhase: Double = 0
    @State private var pulseScale: CGFloat = 1.0
    @State private var glowOpacity: Double = 0.3
    @State private var orbRotation: Double = 0

    private let timer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            backgroundLayer
            VStack(spacing: 0) {
                topBar
                Spacer()
                orbSection
                Spacer()
                transcriptArea
                bottomControls
            }
            .padding(.bottom, 20)
        }
        .task {
            await viewModel.requestPermissions()
        }
        .onReceive(timer) { _ in
            viewModel.updateAudioLevel()
        }
    }

    private var backgroundLayer: some View {
        Rectangle()
            .fill(backgroundGradient)
            .ignoresSafeArea()
            .animation(.easeInOut(duration: 1.5), value: viewModel.state)
    }

    private var backgroundGradient: LinearGradient {
        switch viewModel.state {
        case .idle:
            LinearGradient(colors: [.black, Color(white: 0.08), .black], startPoint: .top, endPoint: .bottom)
        case .listening:
            LinearGradient(colors: [Color(red: 0, green: 0.05, blue: 0.2), Color(red: 0.05, green: 0.15, blue: 0.4), Color(red: 0, green: 0.05, blue: 0.15)], startPoint: .top, endPoint: .bottom)
        case .processing:
            LinearGradient(colors: [Color(red: 0.1, green: 0, blue: 0.2), Color(red: 0.2, green: 0.05, blue: 0.35), Color(red: 0.08, green: 0, blue: 0.15)], startPoint: .top, endPoint: .bottom)
        case .speaking:
            LinearGradient(colors: [Color(red: 0, green: 0.1, blue: 0.08), Color(red: 0.02, green: 0.22, blue: 0.15), Color(red: 0, green: 0.08, blue: 0.05)], startPoint: .top, endPoint: .bottom)
        }
    }

    private var topBar: some View {
        HStack {
            Button {
                viewModel.stopConversation()
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.7))
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
            }

            Spacer()

            VStack(spacing: 2) {
                Text(stateLabel)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                Text("Speech Mode")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
            }

            Spacer()

            Color.clear.frame(width: 36, height: 36)
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
    }

    private var stateLabel: String {
        switch viewModel.state {
        case .idle: "Ready"
        case .listening: "Listening..."
        case .processing: "Thinking..."
        case .speaking: "Speaking..."
        }
    }

    private var orbSection: some View {
        OrbCanvasView(
            phase: wavePhase,
            state: viewModel.state,
            orbRotation: orbRotation,
            glowOpacity: glowOpacity,
            pulseScale: pulseScale,
            audioLevel: viewModel.audioLevel
        )
        .frame(width: 240, height: 240)
        .onAppear {
            withAnimation(.linear(duration: 20).repeatForever(autoreverses: false)) {
                orbRotation = 360
            }
            withAnimation(.easeInOut(duration: 2).repeatForever(autoreverses: true)) {
                glowOpacity = 0.6
            }
        }
        .onChange(of: viewModel.state) { _, newState in
            withAnimation(.spring(response: 0.5, dampingFraction: 0.6)) {
                switch newState {
                case .listening: pulseScale = 1.1
                case .speaking: pulseScale = 1.05
                default: pulseScale = 1.0
                }
            }
        }
        .onChange(of: viewModel.audioLevel) { _, level in
            wavePhase += Double(level) * 0.3 + 0.02
        }
    }

    private var transcriptArea: some View {
        VStack(spacing: 12) {
            if !viewModel.displayText.isEmpty {
                Text(viewModel.displayText)
                    .font(.body)
                    .foregroundStyle(.white.opacity(0.9))
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                    .padding(.horizontal, 32)
            }

            if viewModel.state == .speaking, !viewModel.responseText.isEmpty {
                Text(viewModel.responseText)
                    .font(.callout)
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                    .lineLimit(4)
                    .padding(.horizontal, 32)
            }

            if let error = viewModel.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 32)
            }
        }
        .frame(minHeight: 100)
        .animation(.easeInOut(duration: 0.3), value: viewModel.displayText)
        .animation(.easeInOut(duration: 0.3), value: viewModel.responseText)
    }

    private var bottomControls: some View {
        HStack(spacing: 40) {
            if viewModel.state == .speaking {
                Button {
                    viewModel.synthesisService.stop()
                    viewModel.startListening()
                } label: {
                    secondaryButtonLabel(icon: "forward.fill", text: "Skip")
                }
            }

            Button {
                handleMainButton()
            } label: {
                VStack(spacing: 6) {
                    Circle()
                        .fill(mainButtonColor.gradient)
                        .frame(width: 72, height: 72)
                        .shadow(color: mainButtonColor.opacity(0.4), radius: 12)
                        .overlay {
                            Image(systemName: mainButtonIconName)
                                .font(.title.weight(.semibold))
                                .foregroundStyle(.white)
                        }
                    Text(mainButtonText)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .sensoryFeedback(.impact(weight: .medium), trigger: viewModel.state)

            if viewModel.state != .idle {
                Button {
                    viewModel.stopConversation()
                } label: {
                    secondaryButtonLabel(icon: "stop.fill", text: "End")
                }
            }
        }
        .padding(.bottom, 20)
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: viewModel.state)
    }

    private func secondaryButtonLabel(icon: String, text: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.white.opacity(0.7))
                .frame(width: 48, height: 48)
                .background(.ultraThinMaterial)
                .clipShape(Circle())
            Text(text)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.4))
        }
    }

    private var mainButtonIconName: String {
        switch viewModel.state {
        case .idle: "mic.fill"
        case .listening: "mic.slash.fill"
        case .processing: "ellipsis"
        case .speaking: "mic.fill"
        }
    }

    private var mainButtonText: String {
        switch viewModel.state {
        case .idle: "Tap to Speak"
        case .listening: "Listening"
        case .processing: "Processing"
        case .speaking: "Interrupt"
        }
    }

    private var mainButtonColor: Color {
        switch viewModel.state {
        case .idle: .blue
        case .listening: .red
        case .processing: .purple
        case .speaking: .blue
        }
    }

    private func handleMainButton() {
        switch viewModel.state {
        case .idle:
            viewModel.startConversation()
        case .listening:
            viewModel.recognitionService.stopListening()
            let transcript = viewModel.recognitionService.transcript
            if !transcript.isEmpty {
                viewModel.displayText = transcript
                viewModel.state = .processing
                viewModel.startConversation()
            } else {
                viewModel.state = .idle
            }
        case .processing:
            break
        case .speaking:
            viewModel.synthesisService.stop()
            viewModel.startListening()
        }
    }
}

private struct OrbCanvasView: View {
    let phase: Double
    let state: SpeechModeState
    let orbRotation: Double
    let glowOpacity: Double
    let pulseScale: CGFloat
    let audioLevel: Float

    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let ringColor = ringResolvedColor

            for ringIndex in (0..<3).reversed() {
                let baseSize: CGFloat = [140, 180, 220][ringIndex]
                let radius = baseSize / 2
                let ringPhase = phase + Double(ringIndex) * 0.8
                let amp = orbAmplitude(for: ringIndex)
                let ringOpacity = 0.3 - Double(ringIndex) * 0.08
                let rotation = orbRotation + Double(ringIndex) * 30

                var path = Path()
                let pointCount = 180
                for i in 0...pointCount {
                    let angle = (Double(i) / Double(pointCount)) * 2 * .pi
                    let wave = sin(angle * Double(4 + ringIndex) + ringPhase) * amp * radius
                    let r = radius + wave
                    let rotRad = rotation * .pi / 180
                    let cosA = cos(angle + rotRad)
                    let sinA = sin(angle + rotRad)
                    let x = center.x + r * cosA
                    let y = center.y + r * sinA
                    if i == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                path.closeSubpath()

                context.stroke(path, with: .color(ringColor.opacity(ringOpacity)), lineWidth: 1.5)
            }

            let orbRadius: CGFloat = 60 * pulseScale
            let glowColor = orbGlowResolvedColor

            var orbPath = Path()
            orbPath.addEllipse(in: CGRect(
                x: center.x - orbRadius,
                y: center.y - orbRadius,
                width: orbRadius * 2,
                height: orbRadius * 2
            ))

            context.fill(orbPath, with: .radialGradient(
                Gradient(colors: orbCenterColors),
                center: center,
                startRadius: 0,
                endRadius: orbRadius
            ))

            context.drawLayer { ctx in
                ctx.addFilter(.blur(radius: 40))
                var glowPath = Path()
                glowPath.addEllipse(in: CGRect(
                    x: center.x - orbRadius,
                    y: center.y - orbRadius,
                    width: orbRadius * 2,
                    height: orbRadius * 2
                ))
                ctx.fill(glowPath, with: .color(glowColor.opacity(glowOpacity)))
            }
        }
        .overlay {
            Image(systemName: stateIcon)
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(.white)
        }
    }

    private var stateIcon: String {
        switch state {
        case .idle: "waveform"
        case .listening: "ear.fill"
        case .processing: "brain.filled.head.profile"
        case .speaking: "mouth.fill"
        }
    }

    private var ringResolvedColor: Color {
        switch state {
        case .idle: .gray
        case .listening: .cyan
        case .processing: .purple
        case .speaking: .mint
        }
    }

    private var orbGlowResolvedColor: Color {
        switch state {
        case .idle: .white
        case .listening: .blue
        case .processing: .purple
        case .speaking: .green
        }
    }

    private var orbCenterColors: [Color] {
        switch state {
        case .idle: [.white.opacity(0.15), .white.opacity(0.02)]
        case .listening: [.blue.opacity(0.4), .blue.opacity(0.05)]
        case .processing: [.purple.opacity(0.4), .purple.opacity(0.05)]
        case .speaking: [.green.opacity(0.3), .green.opacity(0.05)]
        }
    }

    private func orbAmplitude(for ring: Int) -> Double {
        let base: Double
        switch state {
        case .idle: base = 0.02
        case .listening: base = Double(audioLevel) * 0.15 + 0.03
        case .processing: base = 0.06
        case .speaking: base = 0.08
        }
        return base * (1.0 - Double(ring) * 0.2)
    }
}
