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
            contentLayer
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
            return LinearGradient(colors: [.black, Color(white: 0.08), .black], startPoint: .top, endPoint: .bottom)
        case .listening:
            return LinearGradient(colors: [Color(red: 0, green: 0.05, blue: 0.2), Color(red: 0.05, green: 0.15, blue: 0.4), Color(red: 0, green: 0.05, blue: 0.15)], startPoint: .top, endPoint: .bottom)
        case .processing:
            return LinearGradient(colors: [Color(red: 0.1, green: 0, blue: 0.2), Color(red: 0.2, green: 0.05, blue: 0.35), Color(red: 0.08, green: 0, blue: 0.15)], startPoint: .top, endPoint: .bottom)
        case .speaking:
            return LinearGradient(colors: [Color(red: 0, green: 0.1, blue: 0.08), Color(red: 0.02, green: 0.22, blue: 0.15), Color(red: 0, green: 0.08, blue: 0.05)], startPoint: .top, endPoint: .bottom)
        }
    }

    private var contentLayer: some View {
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

    private var topBar: some View {
        HStack {
            closeButton
            Spacer()
            statusLabel
            Spacer()
            Color.clear.frame(width: 36, height: 36)
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
    }

    private var closeButton: some View {
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
    }

    private var statusLabel: some View {
        VStack(spacing: 2) {
            Text(stateLabel)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
            Text("Speech Mode")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.5))
        }
    }

    private var stateLabel: String {
        switch viewModel.state {
        case .idle: return "Ready"
        case .listening: return "Listening..."
        case .processing: return "Thinking..."
        case .speaking: return "Speaking..."
        }
    }

    private var orbSection: some View {
        ZStack {
            orbRing0
            orbRing1
            orbRing2
            orbCenter
            orbStateIcon
        }
        .onAppear { startOrbAnimations() }
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

    private func startOrbAnimations() {
        withAnimation(.linear(duration: 20).repeatForever(autoreverses: false)) {
            orbRotation = 360
        }
        withAnimation(.easeInOut(duration: 2).repeatForever(autoreverses: true)) {
            glowOpacity = 0.6
        }
    }

    private var orbRing0: some View {
        makeRing(index: 0, size: 140)
    }

    private var orbRing1: some View {
        makeRing(index: 1, size: 180)
    }

    private var orbRing2: some View {
        makeRing(index: 2, size: 220)
    }

    private func makeRing(index: Int, size: CGFloat) -> some View {
        let ringPhase = wavePhase + Double(index) * 0.8
        let amp = orbAmplitude(for: index)
        let ringOpacity = 0.3 - Double(index) * 0.08
        let rotation = orbRotation + Double(index) * 30
        return WaveRing(phase: ringPhase, amplitude: amp, ringIndex: index)
            .stroke(ringColor.opacity(ringOpacity), lineWidth: 1.5)
            .frame(width: size, height: size)
            .rotationEffect(.degrees(rotation))
    }

    private var ringColor: Color {
        switch viewModel.state {
        case .idle: return .gray
        case .listening: return .cyan
        case .processing: return .purple
        case .speaking: return .mint
        }
    }

    private var orbCenter: some View {
        let glowColor = orbGlowColor
        let shadow1 = glowColor.opacity(glowOpacity)
        let shadow2 = glowColor.opacity(glowOpacity * 0.5)
        return Circle()
            .fill(orbFill)
            .frame(width: 120, height: 120)
            .scaleEffect(pulseScale)
            .shadow(color: shadow1, radius: 40)
            .shadow(color: shadow2, radius: 80)
    }

    private var orbFill: RadialGradient {
        RadialGradient(colors: orbCenterColors, center: .center, startRadius: 0, endRadius: 60)
    }

    private var orbCenterColors: [Color] {
        switch viewModel.state {
        case .idle: return [.white.opacity(0.15), .white.opacity(0.02)]
        case .listening: return [.blue.opacity(0.4), .blue.opacity(0.05)]
        case .processing: return [.purple.opacity(0.4), .purple.opacity(0.05)]
        case .speaking: return [.green.opacity(0.3), .green.opacity(0.05)]
        }
    }

    private var orbGlowColor: Color {
        switch viewModel.state {
        case .idle: return .white
        case .listening: return .blue
        case .processing: return .purple
        case .speaking: return .green
        }
    }

    private var orbStateIcon: some View {
        Image(systemName: stateIcon)
            .font(.system(size: 32, weight: .light))
            .foregroundStyle(.white)
    }

    private var stateIcon: String {
        switch viewModel.state {
        case .idle: return "waveform"
        case .listening: return "ear.fill"
        case .processing: return "brain.filled.head.profile"
        case .speaking: return "mouth.fill"
        }
    }

    private func orbAmplitude(for ring: Int) -> Double {
        let base: Double
        switch viewModel.state {
        case .idle: base = 0.02
        case .listening: base = Double(viewModel.audioLevel) * 0.15 + 0.03
        case .processing: base = 0.06
        case .speaking: base = 0.08
        }
        return base * (1.0 - Double(ring) * 0.2)
    }

    private var transcriptArea: some View {
        VStack(spacing: 12) {
            userTranscriptText
            responseTranscriptText
            errorText
        }
        .frame(minHeight: 100)
        .animation(.easeInOut(duration: 0.3), value: viewModel.displayText)
        .animation(.easeInOut(duration: 0.3), value: viewModel.responseText)
    }

    @ViewBuilder
    private var userTranscriptText: some View {
        if !viewModel.displayText.isEmpty {
            Text(viewModel.displayText)
                .font(.body)
                .foregroundStyle(.white.opacity(0.9))
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .padding(.horizontal, 32)
        }
    }

    @ViewBuilder
    private var responseTranscriptText: some View {
        if viewModel.state == .speaking, !viewModel.responseText.isEmpty {
            Text(viewModel.responseText)
                .font(.callout)
                .foregroundStyle(.white.opacity(0.6))
                .multilineTextAlignment(.center)
                .lineLimit(4)
                .padding(.horizontal, 32)
        }
    }

    @ViewBuilder
    private var errorText: some View {
        if let error = viewModel.errorMessage {
            Label(error, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .padding(.horizontal, 32)
        }
    }

    private var bottomControls: some View {
        HStack(spacing: 40) {
            skipButton
            mainButton
            endButton
        }
        .padding(.bottom, 20)
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: viewModel.state)
    }

    @ViewBuilder
    private var skipButton: some View {
        if viewModel.state == .speaking {
            Button {
                viewModel.synthesisService.stop()
                viewModel.startListening()
            } label: {
                secondaryButtonLabel(icon: "forward.fill", text: "Skip")
            }
        }
    }

    @ViewBuilder
    private var endButton: some View {
        if viewModel.state != .idle {
            Button {
                viewModel.stopConversation()
            } label: {
                secondaryButtonLabel(icon: "stop.fill", text: "End")
            }
        }
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

    private var mainButton: some View {
        Button {
            handleMainButton()
        } label: {
            mainButtonLabel
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: viewModel.state)
    }

    private var mainButtonLabel: some View {
        VStack(spacing: 6) {
            ZStack {
                Circle()
                    .fill(mainButtonColor.gradient)
                    .frame(width: 72, height: 72)
                    .shadow(color: mainButtonColor.opacity(0.4), radius: 12)
                Image(systemName: mainButtonIconName)
                    .font(.title.weight(.semibold))
                    .foregroundStyle(.white)
            }
            Text(mainButtonText)
                .font(.caption.weight(.medium))
                .foregroundStyle(.white.opacity(0.6))
        }
    }

    private var mainButtonIconName: String {
        switch viewModel.state {
        case .idle: return "mic.fill"
        case .listening: return "mic.slash.fill"
        case .processing: return "ellipsis"
        case .speaking: return "mic.fill"
        }
    }

    private var mainButtonText: String {
        switch viewModel.state {
        case .idle: return "Tap to Speak"
        case .listening: return "Listening"
        case .processing: return "Processing"
        case .speaking: return "Interrupt"
        }
    }

    private var mainButtonColor: Color {
        switch viewModel.state {
        case .idle: return .blue
        case .listening: return .red
        case .processing: return .purple
        case .speaking: return .blue
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

struct WaveRing: Shape {
    var phase: Double
    var amplitude: Double
    var ringIndex: Int

    var animatableData: AnimatablePair<Double, Double> {
        get { AnimatablePair(phase, amplitude) }
        set {
            phase = newValue.first
            amplitude = newValue.second
        }
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2
        let pointCount = 180

        for i in 0...pointCount {
            let angle = (Double(i) / Double(pointCount)) * 2 * .pi
            let wave = sin(angle * Double(4 + ringIndex) + phase) * amplitude * radius
            let r = radius + wave
            let x = center.x + r * cos(angle)
            let y = center.y + r * sin(angle)
            let point = CGPoint(x: x, y: y)
            if i == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        path.closeSubpath()
        return path
    }
}
