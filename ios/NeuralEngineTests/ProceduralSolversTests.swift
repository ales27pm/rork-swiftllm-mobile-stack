import Testing
import simd
@testable import NeuralEngine

struct ProceduralSolversTests {

    @Test func grayScott_step_matchesGoldenValuesForDeterministicState() {
        let width = 5
        let height = 5
        var u = ScalarField2D(width: width, height: height, fill: 1)
        var v = ScalarField2D(width: width, height: height, fill: 0)

        for y in 2...3 {
            for x in 2...3 {
                u[x, y] = 0.5
                v[x, y] = 0.25
            }
        }

        let solver = GrayScottSolver(du: 0.16, dv: 0.08, feedRate: 0.06, killRate: 0.062)
        let next = solver.step(state: GrayScottState(u: u, v: v))

        let expectedU: [Float] = [
            1.0, 1.0, 0.99, 0.99, 1.0,
            1.0, 0.975, 0.85, 0.85, 0.975,
            0.99, 0.85, 0.46000004, 0.46000004, 0.85,
            0.99, 0.85, 0.46000004, 0.46000004, 0.85,
            1.0, 0.975, 0.85, 0.85, 0.975
        ]

        let expectedV: [Float] = [
            0.0, 0.0, 0.005, 0.005, 0.0,
            0.0, 0.0125, 0.037499998, 0.037499998, 0.0125,
            0.005, 0.037499998, 0.26075003, 0.26075003, 0.037499998,
            0.005, 0.037499998, 0.26075003, 0.26075003, 0.037499998,
            0.0, 0.0125, 0.037499998, 0.037499998, 0.0125
        ]

        #expect(next.u.values.count == expectedU.count)
        #expect(next.v.values.count == expectedV.count)
        for i in 0..<expectedU.count {
            #expect(abs(next.u.values[i] - expectedU[i]) < 0.0001)
            #expect(abs(next.v.values[i] - expectedV[i]) < 0.0001)
        }
    }

    @Test func marchingSquares_caseTableCoversAll16Cases() {
        let extractor = MarchingSquaresExtractor(isoValue: 0.5)
        let expectedSegmentCounts = [0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0]

        for cellCase in 0...15 {
            let bl: Float = (cellCase & 1) == 0 ? 0 : 1
            let br: Float = (cellCase & 2) == 0 ? 0 : 1
            let tr: Float = (cellCase & 4) == 0 ? 0 : 1
            let tl: Float = (cellCase & 8) == 0 ? 0 : 1
            let field = ScalarField2D(width: 2, height: 2, values: [bl, br, tl, tr])

            let segments = extractor.extract(field: field)
            #expect(segments.count == expectedSegmentCounts[cellCase])

            for segment in segments {
                #expect(segment.start.x >= 0 && segment.start.x <= 1)
                #expect(segment.start.y >= 0 && segment.start.y <= 1)
                #expect(segment.end.x >= 0 && segment.end.x <= 1)
                #expect(segment.end.y >= 0 && segment.end.y <= 1)
            }
        }
    }

    @Test func lSystem_solverMatchesGoldenPolylineForSeededNoise() {
        let solver = LSystemShapeGrammarSolver()
        let config = LSystemConfiguration(
            axiom: "F",
            rules: [
                "F": ["F[+F]F[-F]F"]
            ],
            iterations: 2,
            angleRadians: .pi / 6,
            stepLength: 1,
            turnJitterRadians: 0.1,
            seed: 42
        )

        let points = solver.solve(configuration: config)

        #expect(points.count == 34)
        let golden: [SIMD3<Float>] = [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(-0.05083492, 0.99870723, 0),
            SIMD3<Float>(0.3525554, 1.913734, 0),
            SIMD3<Float>(-0.05083492, 0.99870723, 0),
            SIMD3<Float>(-0.10166984, 1.9974145, 0),
            SIMD3<Float>(0.26765248, 2.9267044, 0),
            SIMD3<Float>(-0.10166984, 1.9974145, 0),
            SIMD3<Float>(-0.07993234, 2.9971783, 0),
            SIMD3<Float>(-0.30191642, 3.9722202, 0),
            SIMD3<Float>(-0.07993234, 2.9971783, 0)
        ]

        for (index, expected) in golden.enumerated() {
            #expect(simd_distance(points[index], expected) < 0.0001)
        }
    }
}
