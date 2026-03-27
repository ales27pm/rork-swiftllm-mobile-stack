# iOS Signing / Archive Troubleshooting

If you see errors like:

- `exportArchive Automatic signing cannot update bundle identifier ...`
- `exportArchive No profiles for '...' were found`

it usually means the project is trying to archive with a bundle identifier or team that your Apple Developer account cannot provision.

## What changed in this repo

The Xcode project now uses configurable build settings:

- `APP_BUNDLE_IDENTIFIER` (default: `com.yourorg.neuralengine`)
- `DEVELOPMENT_TEAM` is intentionally left blank in project settings

This lets you pass your own values from Xcode or `xcodebuild` without editing the project each time.

## Fix in Xcode

1. Open `ios/NeuralEngine.xcodeproj`.
2. Select the `NeuralEngine` target → **Signing & Capabilities**.
3. Pick your Team.
4. Set a unique Bundle Identifier (reverse-DNS you own), for example:
   - `com.yourname.neuralengine`

Xcode will generate profiles automatically when using an account with signing permissions.

## Fix in CI / CLI (`xcodebuild`)

Pass both settings explicitly when archiving/exporting:

```bash
xcodebuild \
  -project ios/NeuralEngine.xcodeproj \
  -scheme NeuralEngine \
  -configuration Release \
  -destination generic/platform=iOS \
  -archivePath build/NeuralEngine.xcarchive \
  APP_BUNDLE_IDENTIFIER=com.yourname.neuralengine \
  DEVELOPMENT_TEAM=YOURTEAMID \
  archive
```

If exporting an `.xcarchive`, use matching values in your `ExportOptions.plist` (`teamID`, signing style, method).

## Notes

- Bundle identifiers must be globally unique within your Apple Developer account.
- If you're using a free Apple ID, export/distribution options are limited compared to a paid Developer Program account.
