# FaceTrap

FaceTrap is an Android face-recognition demo using InsightFace `buffalo_l`: SCRFD detects and aligns a face, ArcFace produces a 512-dimensional embedding, and cosine similarity selects the closest enrolled identity.

## Prepare the model assets

The small reference embeddings are included. The ONNX files are downloaded locally because `w600k_r50.onnx` exceeds GitHub's normal file-size limit.

From the repository root, run:

```bash
python3 -m pip install insightface onnxruntime opencv-python numpy
python3 build_face_references.py --android-assets app/src/main/assets
```

This downloads InsightFace `buffalo_l`, rebuilds the three reference embeddings from `dataset/synthetic`, and copies these required files into `app/src/main/assets`:

```text
det_10g.onnx
w600k_r50.onnx
you_ref.npy
teammate_ref.npy
professor_ref.npy
```

## Run the Android app

1. Open the repository in Android Studio and use JDK 17.
2. Allow Gradle to sync and select an Android device or emulator (API 26 or newer).
3. For an emulator, set the front camera to `webcam0`, cold boot it, and close other applications using the webcam.
4. Run the `app` configuration and grant camera permission.

Recognized students display `Hey Students`, the professor displays `Hi Sir`, and unmatched faces display `Unknown`. Diagnostic scores are available in Logcat with `tag:FaceTrap`.

Command-line build:

```bash
./gradlew assembleDebug
```

The assignment brief is included in [`Computer_Systems_Security__CS402M.pdf`](Computer_Systems_Security__CS402M.pdf).
