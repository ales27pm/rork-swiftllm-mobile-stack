import Foundation
import simd

protocol ShapeGrammarSolver: Sendable {
    func solve(configuration: LSystemConfiguration) -> [SIMD3<Float>]
}

protocol ReactionDiffusionSolver: Sendable {
    func step(state: GrayScottState) -> GrayScottState
}

nonisolated struct ScalarField2D: Sendable {
    let width: Int
    let height: Int
    private(set) var values: [Float]

    init(width: Int, height: Int, fill: Float = 0) {
        precondition(width > 0 && height > 0, "ScalarField2D dimensions must be > 0")
        self.width = width
        self.height = height
        self.values = Array(repeating: fill, count: width * height)
    }

    init(width: Int, height: Int, values: [Float]) {
        precondition(width > 0 && height > 0, "ScalarField2D dimensions must be > 0")
        precondition(values.count == width * height, "ScalarField2D values must match dimensions")
        self.width = width
        self.height = height
        self.values = values
    }

    subscript(x: Int, y: Int) -> Float {
        get { values[index(x, y)] }
        set { values[index(x, y)] = newValue }
    }

    func index(_ x: Int, _ y: Int) -> Int {
        y * width + x
    }

    func wrappedValue(x: Int, y: Int) -> Float {
        let wrappedX = (x % width + width) % width
        let wrappedY = (y % height + height) % height
        return values[index(wrappedX, wrappedY)]
    }
}

nonisolated struct GrayScottState: Sendable {
    var u: ScalarField2D
    var v: ScalarField2D

    init(u: ScalarField2D, v: ScalarField2D) {
        precondition(u.width == v.width && u.height == v.height, "U and V fields must share dimensions")
        self.u = u
        self.v = v
    }
}

nonisolated struct LaplacianStencil: Sendable {
    let center: Float
    let cardinal: Float
    let diagonal: Float

    static let grayScottDefault = LaplacianStencil(center: -1, cardinal: 0.2, diagonal: 0.05)
}

nonisolated struct GrayScottSolver: ReactionDiffusionSolver {
    let du: Float
    let dv: Float
    let feedRate: Float
    let killRate: Float
    let dt: Float
    let stencil: LaplacianStencil

    init(
        du: Float,
        dv: Float,
        feedRate: Float,
        killRate: Float,
        dt: Float = 1,
        stencil: LaplacianStencil = .grayScottDefault
    ) {
        self.du = du
        self.dv = dv
        self.feedRate = feedRate
        self.killRate = killRate
        self.dt = dt
        self.stencil = stencil
    }

    func step(state: GrayScottState) -> GrayScottState {
        var nextU = state.u
        var nextV = state.v

        for y in 0..<state.u.height {
            for x in 0..<state.u.width {
                let u = state.u[x, y]
                let v = state.v[x, y]
                let lapU = laplacian(for: state.u, x: x, y: y)
                let lapV = laplacian(for: state.v, x: x, y: y)

                let uvv = u * v * v
                let duDt = du * lapU - uvv + feedRate * (1 - u)
                let dvDt = dv * lapV + uvv - (feedRate + killRate) * v

                nextU[x, y] = min(max(u + duDt * dt, 0), 1)
                nextV[x, y] = min(max(v + dvDt * dt, 0), 1)
            }
        }

        return GrayScottState(u: nextU, v: nextV)
    }

    private func laplacian(for field: ScalarField2D, x: Int, y: Int) -> Float {
        let center = field.wrappedValue(x: x, y: y) * stencil.center
        let cardinals = (
            field.wrappedValue(x: x - 1, y: y) +
            field.wrappedValue(x: x + 1, y: y) +
            field.wrappedValue(x: x, y: y - 1) +
            field.wrappedValue(x: x, y: y + 1)
        ) * stencil.cardinal

        let diagonals = (
            field.wrappedValue(x: x - 1, y: y - 1) +
            field.wrappedValue(x: x + 1, y: y - 1) +
            field.wrappedValue(x: x - 1, y: y + 1) +
            field.wrappedValue(x: x + 1, y: y + 1)
        ) * stencil.diagonal

        return center + cardinals + diagonals
    }
}

nonisolated struct IsolineSegment: Sendable {
    let start: SIMD2<Float>
    let end: SIMD2<Float>
}

nonisolated struct MarchingSquaresExtractor: Sendable {
    let isoValue: Float

    private static let caseTable: [Int: [(Int, Int)]] = [
        0: [],
        1: [(3, 0)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(1, 2)],
        5: [(3, 2), (0, 1)],
        6: [(0, 2)],
        7: [(3, 2)],
        8: [(2, 3)],
        9: [(0, 2)],
        10: [(2, 1), (3, 0)],
        11: [(2, 1)],
        12: [(1, 3)],
        13: [(0, 1)],
        14: [(3, 0)],
        15: []
    ]

    func extract(field: ScalarField2D) -> [IsolineSegment] {
        guard field.width > 1, field.height > 1 else { return [] }

        var segments: [IsolineSegment] = []
        segments.reserveCapacity((field.width - 1) * (field.height - 1))

        for y in 0..<(field.height - 1) {
            for x in 0..<(field.width - 1) {
                let bl = field[x, y]
                let br = field[x + 1, y]
                let tr = field[x + 1, y + 1]
                let tl = field[x, y + 1]

                let cellCase =
                    (bl >= isoValue ? 1 : 0) |
                    (br >= isoValue ? 2 : 0) |
                    (tr >= isoValue ? 4 : 0) |
                    (tl >= isoValue ? 8 : 0)

                guard let edgePairs = Self.caseTable[cellCase], !edgePairs.isEmpty else { continue }

                let corners = [
                    SIMD2<Float>(Float(x), Float(y)),
                    SIMD2<Float>(Float(x + 1), Float(y)),
                    SIMD2<Float>(Float(x + 1), Float(y + 1)),
                    SIMD2<Float>(Float(x), Float(y + 1))
                ]
                let values = [bl, br, tr, tl]

                for (edgeA, edgeB) in edgePairs {
                    let p0 = interpolatedPoint(edge: edgeA, corners: corners, values: values)
                    let p1 = interpolatedPoint(edge: edgeB, corners: corners, values: values)
                    segments.append(IsolineSegment(start: p0, end: p1))
                }
            }
        }

        return segments
    }

    private func interpolatedPoint(edge: Int, corners: [SIMD2<Float>], values: [Float]) -> SIMD2<Float> {
        let (i0, i1): (Int, Int)
        switch edge {
        case 0: (i0, i1) = (0, 1)
        case 1: (i0, i1) = (1, 2)
        case 2: (i0, i1) = (3, 2)
        default: (i0, i1) = (0, 3)
        }

        let v0 = values[i0]
        let v1 = values[i1]
        let t: Float
        if abs(v1 - v0) < 1e-6 {
            t = 0.5
        } else {
            t = min(max((isoValue - v0) / (v1 - v0), 0), 1)
        }

        return corners[i0] + (corners[i1] - corners[i0]) * t
    }
}

nonisolated struct LSystemConfiguration: Sendable {
    let axiom: String
    let rules: [Character: [String]]
    let iterations: Int
    let angleRadians: Float
    let stepLength: Float
    let turnJitterRadians: Float
    let seed: UInt64

    init(
        axiom: String,
        rules: [Character: [String]],
        iterations: Int,
        angleRadians: Float,
        stepLength: Float = 1,
        turnJitterRadians: Float = 0,
        seed: UInt64 = 0
    ) {
        self.axiom = axiom
        self.rules = rules
        self.iterations = max(0, iterations)
        self.angleRadians = angleRadians
        self.stepLength = stepLength
        self.turnJitterRadians = max(0, turnJitterRadians)
        self.seed = seed
    }
}

nonisolated struct LSystemShapeGrammarSolver: ShapeGrammarSolver {
    func solve(configuration: LSystemConfiguration) -> [SIMD3<Float>] {
        let commands = rewrite(configuration: configuration)
        return interpret(commands: commands, configuration: configuration)
    }

    private func rewrite(configuration: LSystemConfiguration) -> String {
        var current = configuration.axiom
        var rng = SeededGenerator(state: configuration.seed)

        for _ in 0..<configuration.iterations {
            var next = String()
            next.reserveCapacity(current.count * 2)

            for symbol in current {
                if let options = configuration.rules[symbol], !options.isEmpty {
                    let index = Int(rng.nextUInt32()) % options.count
                    next.append(contentsOf: options[index])
                } else {
                    next.append(symbol)
                }
            }

            current = next
        }

        return current
    }

    private func interpret(commands: String, configuration: LSystemConfiguration) -> [SIMD3<Float>] {
        struct TurtleState {
            var position: SIMD2<Float>
            var heading: Float
        }

        var rng = SeededGenerator(state: configuration.seed ^ 0x9E3779B97F4A7C15)
        var turtle = TurtleState(position: SIMD2<Float>(0, 0), heading: .pi / 2)
        var stack: [TurtleState] = []
        var points: [SIMD3<Float>] = [SIMD3<Float>(0, 0, 0)]
        points.reserveCapacity(commands.count + 1)

        for symbol in commands {
            switch symbol {
            case "F":
                let direction = SIMD2<Float>(cos(turtle.heading), sin(turtle.heading))
                turtle.position += direction * configuration.stepLength
                points.append(SIMD3<Float>(turtle.position.x, turtle.position.y, 0))
            case "f":
                let direction = SIMD2<Float>(cos(turtle.heading), sin(turtle.heading))
                turtle.position += direction * configuration.stepLength
            case "+":
                turtle.heading += configuration.angleRadians + jitter(configuration: configuration, rng: &rng)
            case "-":
                turtle.heading -= configuration.angleRadians + jitter(configuration: configuration, rng: &rng)
            case "[":
                stack.append(turtle)
            case "]":
                if let restored = stack.popLast() {
                    turtle = restored
                    points.append(SIMD3<Float>(turtle.position.x, turtle.position.y, 0))
                }
            default:
                continue
            }
        }

        return points
    }

    private func jitter(configuration: LSystemConfiguration, rng: inout SeededGenerator) -> Float {
        guard configuration.turnJitterRadians > 0 else { return 0 }
        return rng.nextSignedUnitFloat() * configuration.turnJitterRadians
    }
}

private struct SeededGenerator: Sendable {
    private(set) var state: UInt64

    mutating func nextUInt32() -> UInt32 {
        state = 6364136223846793005 &* state &+ 1442695040888963407
        return UInt32(truncatingIfNeeded: state >> 32)
    }

    mutating func nextUnitFloat() -> Float {
        Float(nextUInt32()) / Float(UInt32.max)
    }

    mutating func nextSignedUnitFloat() -> Float {
        nextUnitFloat() * 2 - 1
    }
}

nonisolated struct ProceduralContext: Sendable {
    let solver: String
    let iterationCount: Int
    let outputDimensions: Int
    let timestamp: Date

    init(solver: String, iterationCount: Int, outputDimensions: Int) {
        self.solver = solver
        self.iterationCount = iterationCount
        self.outputDimensions = outputDimensions
        self.timestamp = Date()
    }
}
