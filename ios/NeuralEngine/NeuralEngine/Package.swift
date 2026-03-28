// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "NeuralEngine",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "NeuralEngine",
            targets: ["NeuralEngine"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.0.0"),
        .package(url: "https://github.com/mattt/llama.swift.git", from: "2.8216.0"),
    ],
    targets: [
        .target(
            name: "NeuralEngine",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "LlamaSwift", package: "llama.swift"),
            ],
            path: "Sources/NeuralEngine",
            resources: [
                .process("Assets.xcassets"),
                .process("Resources"),
                .process("en.lproj"),
                .process("fr-CA.lproj"),
            ],
            swiftSettings: [
                .unsafeFlags([
                    "-swift-version", "5",
                    "-Xfrontend", "-strict-concurrency=minimal",
                ]),
            ],
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
    ]
)
