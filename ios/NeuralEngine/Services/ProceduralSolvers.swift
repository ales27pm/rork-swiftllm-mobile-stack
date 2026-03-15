import Foundation
import simd

protocol ShapeGrammarSolver: Sendable {
    func solve(rules: [String], iterations: Int) -> [SIMD3<Float>]
}

protocol ReactionDiffusionSolver: Sendable {
    func step(grid: [[Float]], feed: Float, kill: Float) -> [[Float]]
}

nonisolated struct DefaultShapeGrammarSolver: ShapeGrammarSolver {
    func solve(rules: [String], iterations: Int) -> [SIMD3<Float>] {
        var points: [SIMD3<Float>] = [SIMD3<Float>(0, 0, 0)]
        var direction = SIMD3<Float>(0, 1, 0)
        let angleStep: Float = .pi / 6

        for iteration in 0..<iterations {
            var expanded: [SIMD3<Float>] = []
            for rule in rules {
                for point in points {
                    switch rule {
                    case "F":
                        let next = point + direction * Float(1.0 / Float(iteration + 1))
                        expanded.append(next)
                    case "+":
                        let cosA = cos(angleStep)
                        let sinA = sin(angleStep)
                        direction = SIMD3<Float>(
                            direction.x * cosA - direction.z * sinA,
                            direction.y,
                            direction.x * sinA + direction.z * cosA
                        )
                    case "-":
                        let cosA = cos(-angleStep)
                        let sinA = sin(-angleStep)
                        direction = SIMD3<Float>(
                            direction.x * cosA - direction.z * sinA,
                            direction.y,
                            direction.x * sinA + direction.z * cosA
                        )
                    default:
                        expanded.append(point)
                    }
                }
            }
            if !expanded.isEmpty {
                points.append(contentsOf: expanded)
            }
        }
        return points
    }
}

nonisolated struct DefaultReactionDiffusionSolver: ReactionDiffusionSolver {
    func step(grid: [[Float]], feed: Float, kill: Float) -> [[Float]] {
        let rows = grid.count
        guard rows > 2 else { return grid }
        let cols = grid[0].count
        guard cols > 2 else { return grid }

        let diffusionA: Float = 1.0
        let diffusionB: Float = 0.5
        let dt: Float = 1.0

        var result = grid

        for i in 1..<(rows - 1) {
            for j in 1..<(cols - 1) {
                let a = grid[i][j]
                let laplacian =
                    grid[i - 1][j] + grid[i + 1][j] +
                    grid[i][j - 1] + grid[i][j + 1] -
                    4.0 * a

                let reaction = -a * a * a + feed * (1.0 - a) - kill * a
                let diffusion = diffusionA * laplacian + diffusionB * laplacian * 0.5
                result[i][j] = a + (reaction + diffusion) * dt
                result[i][j] = max(0, min(1, result[i][j]))
            }
        }
        return result
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
