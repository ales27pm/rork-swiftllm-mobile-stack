import SwiftUI
import Combine

struct SpeechModeView: View {
    @Bindable var viewModel: SpeechViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var wavePhase: Double = 0
    @State private var pulseScale: CGFloat = 1.0
    @State private var glowOpacity: Double = 0.3
    @State private var orbRotation: Double = 0
    @State private var showTranscript: Bool = false

    private let timer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            backgroundLayer

            VStack(spacing: 0) {
                topBar
                    .padding(.top, 8)

                if showTranscript {
                    transcriptList
                } else {
                    Spacer()
                    orbSection
                    Spacer()
                    liveCaption
                }

                bottomControls
                    .padding(.bottom, 20)
            }
        }
        .task {
            await viewModel.requestPermissions()
        }
        .onReceive(timer) { _ in
            viewModel.updateAudioLevel()
        }
        .statusBarHidden()
    }

    private var backgroundLayer: some View {
        Rectangle()
            .fill(backgroundGradient)
            .ignoresSafeArea()
            .animation(.easeInOut(duration: 1.2), value: viewModel.state)
    }

    private var backgroundGradient: some ShapeStyle {
        switch viewModel.state {
        case .idle:
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [0.5, 0], [1, 0],
                    [0, 0.5], [0.5, 0.5], [1, 0.5],
                    [0, 1], [0.5, 1], [1, 1]
                ],
                colors: [
                    .black, Color(white: 0.05), .black,
                    Color(white: 0.03), Color(white: 0.08), Color(white: 0.03),
                    .black, Color(white: 0.04), .black
                ]
            )
        case .listening:
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [0.5, 0], [1, 0],
                    [0, 0.5], [0.5, 0.5], [1, 0.5],
                    [0, 1], [0.5, 1], [1, 1]
                ],
                colors: [
                    Color(red: 0, green: 0.02, blue: 0.12), Color(red: 0, green: 0.05, blue: 0.18), Color(red: 0, green: 0.02, blue: 0.12),
                    Color(red: 0.02, green: 0.08, blue: 0.25), Color(red: 0.05, green: 0.15, blue: 0.4), Color(red: 0.02, green: 0.08, blue: 0.25),
                    Color(red: 0, green: 0.03, blue: 0.1), Color(red: 0, green: 0.06, blue: 0.15), Color(red: 0, green: 0.03, blue: 0.1)
                ]
            )
        case .processing:
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [0.5, 0], [1, 0],
                    [0, 0.5], [0.5, 0.5], [1, 0.5],
                    [0, 1], [0.5, 1], [1, 1]
                ],
                colors: [
                    Color(red: 0.06, green: 0, blue: 0.12), Color(red: 0.1, green: 0, blue: 0.18), Color(red: 0.06, green: 0, blue: 0.12),
                    Color(red: 0.12, green: 0.02, blue: 0.25), Color(red: 0.2, green: 0.05, blue: 0.35), Color(red: 0.12, green: 0.02, blue: 0.25),
                    Color(red: 0.04, green: 0, blue: 0.08), Color(red: 0.08, green: 0, blue: 0.15), Color(red: 0.04, green: 0, blue: 0.08)
                ]
            )
        case .speaking:
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [0.5, 0], [1, 0],
                    [0, 0.5], [0.5, 0.5], [1, 0.5],
                    [0, 1], [0.5, 1], [1, 1]
                ],
                colors: [
                    Color(red: 0, green: 0.06, blue: 0.04), Color(red: 0, green: 0.08, blue: 0.06), Color(red: 0, green: 0.06, blue: 0.04),
                    Color(red: 0, green: 0.14, blue: 0.08), Color(red: 0.02, green: 0.22, blue: 0.15), Color(red: 0, green: 0.14, blue: 0.08),
                    Color(red: 0, green: 0.04, blue: 0.03), Color(red: 0, green: 0.08, blue: 0.05), Color(red: 0, green: 0.04, blue: 0.03)
                ]
            )
        }
    }

    private var topBar: some View {
        HStack(alignment: .center) {
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
                HStack(spacing: 6) {
                    Circle()
                        .fill(stateIndicatorColor)
                        .frame(width: 6, height: 6)
                        .shadow(color: stateIndicatorColor.opacity(0.6), radius: 3)

                    Text(stateLabel)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                }

                if viewModel.sessionDuration > 0 {
                    Text(viewModel.formattedDuration)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.4))
                }
            }

            Spacer()

            Button {
                withAnimation(.spring(response: 0.3)) {
                    showTranscript.toggle()
                }
            } label: {
                Image(systemName: showTranscript ? "waveform" : "text.bubble")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.7))
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
            }
        }
        .padding(.horizontal, 20)
    }

    private var stateIndicatorColor: Color {
        switch viewModel.state {
        case .idle: .gray
        case .listening: .cyan
        case .processing: .purple
        case .speaking: .green
        }
    }

    private var stateLabel: String {
        switch viewModel.state {
        case .idle: "Ready"
        case .listening: "Listening"
        case .processing: "Thinking"
        case .speaking: "Speaking"
        }
    }

    private var orbSection: some View {
        OrbCanvasView(
            phase: wavePhase,
            state: viewModel.state,
            orbRotation: orbRotation,
            glowOpacity: glowOpacity,
            pulseScale: pulseScale,
            audioLevel: viewModel.audioLevel,
            speechProgress: viewModel.synthesisService.progress
        )
        .frame(width: 260, height: 260)
        .onAppear {
            withAnimation(.linear(duration: 25).repeatForever(autoreverses: false)) {
                orbRotation = 360
            }
            withAnimation(.easeInOut(duration: 2.5).repeatForever(autoreverses: true)) {
                glowOpacity = 0.6
            }
        }
        .onChange(of: viewModel.state) { _, newState in
            withAnimation(.spring(response: 0.4, dampingFraction: 0.65)) {
                switch newState {
                case .listening: pulseScale = 1.12
                case .speaking: pulseScale = 1.06
                case .processing: pulseScale = 0.95
                default: pulseScale = 1.0
                }
            }
        }
        .onChange(of: viewModel.audioLevel) { _, level in
            wavePhase += Double(level) * 0.4 + 0.015
        }
        .sensoryFeedback(.impact(weight: .light, intensity: 0.4), trigger: viewModel.state)
    }

    private var liveCaption: some View {
        VStack(spacing: 16) {
            if !viewModel.displayText.isEmpty {
                Text(viewModel.displayText)
                    .font(.title3.weight(.medium))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                    .padding(.horizontal, 32)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            if viewModel.state == .speaking, !viewModel.responseText.isEmpty {
                VStack(spacing: 8) {
                    Text(viewModel.synthesisService.spokenText.isEmpty
                         ? viewModel.responseText
                         : viewModel.synthesisService.spokenText)
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.6))
                        .multilineTextAlignment(.center)
                        .lineLimit(4)
                        .padding(.horizontal, 32)

                    SpeechProgressBar(progress: viewModel.synthesisService.progress)
                        .padding(.horizontal, 48)
                }
            }

            if let error = viewModel.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 32)
            }
        }
        .frame(minHeight: 120)
        .animation(.easeInOut(duration: 0.25), value: viewModel.displayText)
        .animation(.easeInOut(duration: 0.25), value: viewModel.responseText)
    }

    private var transcriptList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.conversationTranscript) { entry in
                        TranscriptBubble(entry: entry)
                            .id(entry.id)
                    }

                    if viewModel.state == .listening && !viewModel.displayText.isEmpty {
                        TranscriptBubble(
                            entry: VoiceTranscriptEntry(
                                role: .user,
                                text: viewModel.displayText + "..."
                            )
                        )
                        .opacity(0.6)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: viewModel.conversationTranscript.count) { _, _ in
                if let last = viewModel.conversationTranscript.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var bottomControls: some View {
        HStack(spacing: 32) {
            if viewModel.state == .speaking {
                Button {
                    viewModel.interruptAndListen()
                } label: {
                    secondaryButton(icon: "forward.fill", text: "Skip")
                }
                .transition(.scale.combined(with: .opacity))
            } else if viewModel.state != .idle {
                Button {
                    viewModel.stopConversation()
                } label: {
                    secondaryButton(icon: "stop.fill", text: "End")
                }
                .transition(.scale.combined(with: .opacity))
            } else {
                Color.clear.frame(width: 48, height: 48)
            }

            Button {
                handleMainButton()
            } label: {
                ZStack {
                    Circle()
                        .fill(mainButtonColor.gradient)
                        .frame(width: 76, height: 76)
                        .shadow(color: mainButtonColor.opacity(0.5), radius: 16, y: 4)

                    if viewModel.state == .processing {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(1.2)
                    } else {
                        Image(systemName: mainButtonIconName)
                            .font(.title.weight(.semibold))
                            .foregroundStyle(.white)
                            .contentTransition(.symbolEffect(.replace))
                    }
                }
            }
            .sensoryFeedback(.impact(weight: .medium), trigger: viewModel.state)

            if viewModel.state != .idle {
                Button {
                    viewModel.isAutoListenEnabled.toggle()
                } label: {
                    secondaryButton(
                        icon: viewModel.isAutoListenEnabled ? "arrow.triangle.2.circlepath" : "arrow.triangle.2.circlepath.circle",
                        text: viewModel.isAutoListenEnabled ? "Auto" : "Manual"
                    )
                }
                .transition(.scale.combined(with: .opacity))
            } else {
                Color.clear.frame(width: 48, height: 48)
            }
        }
        .padding(.bottom, 16)
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: viewModel.state)
    }

    private func secondaryButton(icon: String, text: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.white.opacity(0.8))
                .frame(width: 48, height: 48)
                .background(.ultraThinMaterial)
                .clipShape(Circle())
            Text(text)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.white.opacity(0.5))
        }
    }

    private var mainButtonIconName: String {
        switch viewModel.state {
        case .idle: "mic.fill"
        case .listening: "mic.badge.xmark"
        case .processing: "ellipsis"
        case .speaking: "mic.fill"
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
            if !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                viewModel.displayText = transcript
                viewModel.state = .processing
                viewModel.startConversation()
            } else {
                viewModel.state = .idle
            }
        case .processing:
            break
        case .speaking:
            viewModel.interruptAndListen()
        }
    }
}

private struct SpeechProgressBar: View {
    let progress: Double

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(.white.opacity(0.1))
                    .frame(height: 3)

                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [.green.opacity(0.6), .mint.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: geometry.size.width * progress, height: 3)
                    .animation(.linear(duration: 0.15), value: progress)
            }
        }
        .frame(height: 3)
    }
}

private struct TranscriptBubble: View {
    let entry: VoiceTranscriptEntry

    var body: some View {
        HStack {
            if entry.role == .user { Spacer(minLength: 48) }

            VStack(alignment: entry.role == .user ? .trailing : .leading, spacing: 4) {
                Text(entry.text)
                    .font(.subheadline)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(bubbleBackground)
                    .clipShape(.rect(cornerRadius: 18))

                Text(entry.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.3))
            }

            if entry.role == .assistant { Spacer(minLength: 48) }
        }
    }

    private var bubbleBackground: some ShapeStyle {
        entry.role == .user
            ? AnyShapeStyle(Color.blue.opacity(0.3))
            : AnyShapeStyle(Color.white.opacity(0.1))
    }
}

private struct OrbCanvasView: View {
    let phase: Double
    let state: SpeechModeState
    let orbRotation: Double
    let glowOpacity: Double
    let pulseScale: CGFloat
    let audioLevel: Float
    let speechProgress: Double

    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let ringColor = ringResolvedColor

            for ringIndex in (0..<4).reversed() {
                let baseSize: CGFloat = [120, 155, 190, 225][ringIndex]
                let radius = baseSize / 2
                let ringPhase = phase + Double(ringIndex) * 0.7
                let amp = orbAmplitude(for: ringIndex)
                let ringOpacity = 0.25 - Double(ringIndex) * 0.05
                let rotation = orbRotation + Double(ringIndex) * 25

                var path = Path()
                let pointCount = 200
                for i in 0...pointCount {
                    let angle = (Double(i) / Double(pointCount)) * 2 * .pi
                    let wave1 = sin(angle * Double(3 + ringIndex) + ringPhase) * amp * radius
                    let wave2 = sin(angle * Double(5 + ringIndex) - ringPhase * 0.7) * amp * radius * 0.3
                    let r = radius + wave1 + wave2
                    let rotRad = rotation * .pi / 180
                    let x = center.x + r * cos(angle + rotRad)
                    let y = center.y + r * sin(angle + rotRad)
                    if i == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                path.closeSubpath()
                context.stroke(path, with: .color(ringColor.opacity(ringOpacity)), lineWidth: 1.2)
            }

            let orbRadius: CGFloat = 55 * pulseScale
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
                ctx.addFilter(.blur(radius: 45))
                var glowPath = Path()
                glowPath.addEllipse(in: CGRect(
                    x: center.x - orbRadius * 1.3,
                    y: center.y - orbRadius * 1.3,
                    width: orbRadius * 2.6,
                    height: orbRadius * 2.6
                ))
                ctx.fill(glowPath, with: .color(glowColor.opacity(glowOpacity * 0.7)))
            }

            if state == .speaking && speechProgress > 0 {
                let progressRadius = orbRadius + 12
                var progressPath = Path()
                progressPath.addArc(
                    center: center,
                    radius: progressRadius,
                    startAngle: .degrees(-90),
                    endAngle: .degrees(-90 + 360 * speechProgress),
                    clockwise: false
                )
                context.stroke(
                    progressPath,
                    with: .color(.mint.opacity(0.5)),
                    lineWidth: 2.5
                )
            }
        }
        .overlay {
            Image(systemName: stateIcon)
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(.white.opacity(0.9))
                .contentTransition(.symbolEffect(.replace))
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
        case .idle: [.white.opacity(0.12), .white.opacity(0.02)]
        case .listening: [.blue.opacity(0.45), .cyan.opacity(0.05)]
        case .processing: [.purple.opacity(0.45), .indigo.opacity(0.05)]
        case .speaking: [.green.opacity(0.35), .mint.opacity(0.05)]
        }
    }

    private func orbAmplitude(for ring: Int) -> Double {
        let base: Double
        switch state {
        case .idle: base = 0.015
        case .listening: base = Double(audioLevel) * 0.18 + 0.025
        case .processing: base = 0.05 + sin(phase * 2) * 0.02
        case .speaking: base = 0.06 + speechProgress * 0.04
        }
        return base * (1.0 - Double(ring) * 0.15)
    }
}
