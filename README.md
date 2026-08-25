# FaceTrap

FaceTrap is the Android submission for IIT Tirupati CS402M's biometric-trigger assignment. It performs fully on-device face recognition and exposes two paths:

- **Path A — enrolled student:** display `Authentication successful` and append a timestamped app-private audit event.
- **Path B — target professor:** when calibrated confidence is strictly greater than `0.85`, open a full-screen, reversible availability-impact simulation.

The Path B payload is a safe simulator, not malware. It contains no encryption, deletion, downloader, persistence, privilege escalation, network access, shared-storage access, or broad storage permission. It operates only on fictional decoys created inside the app's private directory.

## Architecture

```text
CameraX frame
    -> SCRFD face detection and five-point alignment
    -> ArcFace 512-D normalized embedding
    -> cosine comparison with three train-only reference centroids
       -> student: authentication success + audit log
       -> professor: logistic calibration -> confidence > 0.85
          -> private decoy availability marker + manifest + incident UI
```

SCRFD runs at `640x640` with detection threshold `0.50`. ArcFace uses aligned `112x112` crops. Ordinary identity acceptance uses cosine `>=0.30`. The professor cosine is mapped to confidence with validation-fitted logistic calibration before applying the assignment's strict `>0.85` rule.

## Repository layout

```text
app/                          Android application and unit tests
dataset/references/           Three original reference portraits
dataset/synthetic/            500 generated portraits per identity
dataset/splits/               Deterministic 80/10/10 CSV manifests
reference_embeddings/         Train-only ArcFace centroids
report/                       NeurIPS source, PDF, metrics, and plots
generate_class_A.py           Original Class A diffusion generator
generate_class_B.py           Original Class B diffusion generator
prepare_dataset_splits.py     Seeded, identity-stratified split creation
evaluate_face_model.py        Optional held-out evaluation and plotting
```

The Android application does not run Python or load the image dataset. At runtime it uses only the packaged ONNX models and three `.npy` reference vectors.

## Run with Android Studio

Requirements: Ubuntu 22.04, Android Studio, JDK 17, Android SDK, and an emulator.

1. Open this repository in Android Studio and allow Gradle sync to complete.
2. Select JDK 17 under **Settings → Build Tools → Gradle → Gradle JDK**.
3. In **Device Manager**, create or start a Pixel emulator. API 34 or newer is suitable.
4. To use the Ubuntu webcam, stop the emulator, edit it, open **Advanced Settings**, and set **Front Camera → Webcam0**. Cold boot it.
5. Select the `app` run configuration and the emulator, then click **Run ▶**.
6. Grant camera permission inside the emulator.

Expected behavior:

- Enrolled student: `Authentication successful`.
- Unknown face: `Unknown`; Path B remains inactive.
- Professor below calibrated confidence `0.85`: threshold message only.
- Professor above `0.85`: `SIMULATION ONLY` incident screen, one-hour display timer, and manifest.
- Entering `RESET-DEMO` removes the marker and manifest and restores the simulated read path.

If a professor is recognized but the incident screen does not reopen, a previous simulation marker is probably active. Use `RESET-DEMO`, or clear the emulator's state through **Settings → Apps → FaceTrap → Storage & cache → Clear storage**, and then relaunch the app.

## Verify the project

Run from the repository root:

```bash
./gradlew testDebugUnitTest assembleDebug lintDebug
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`. The automated tests verify the strict threshold, manifest creation, byte-for-byte decoy invariance, blocked demo reads, calibration, and deterministic reset.

Static safety check:

```bash
rg -n "uses-permission|Cipher|SecretKey|HttpURLConnection|OkHttp|Socket|Environment.getExternalStorage|MANAGE_EXTERNAL_STORAGE" app/src/main
```

Expected output is the camera permission only. For the debuggable emulator build, inspect evidence with:

```bash
adb shell run-as com.facetrap cat files/auth_audit.log
adb shell run-as com.facetrap cat files/availability_demo/ransomware_manifest.txt
```

Verified results:

- Android unit tests: 4 passed, 0 failed.
- APK assembly: passed.
- Android lint: 0 errors.
- Dataset integrity tests: 3 passed.
- Held-out validation/test identity accuracy: `1.000` on the synthetic benchmark.
- Target test precision, recall, F1, and ROC AUC: `1.000`; false-positive rate: `0.000`.
- Emulator Path B: cosine `0.7626` calibrated to `0.9676`; three decoys recorded as simulated unavailable.

These synthetic results demonstrate pipeline consistency, not real-world biometric validity.

## Dataset and evaluation

The dataset contains 500 images for each of two Class A identities and one Class B professor identity. `prepare_dataset_splits.py` records a deterministic seed-402 partition without copying or modifying images:

| Split | Per identity | Total |
|---|---:|---:|
| Train | 400 | 1,200 |
| Validation | 50 | 150 |
| Test | 50 | 150 |

The `.npy` files packaged in the app are centroids of training images only. Validation and test images are excluded from enrollment.

The metrics and plots are already present. Reproduction is optional and is not required to run Android:

```bash
python3 prepare_dataset_splits.py
python3 -m venv .venv
.venv/bin/pip install -r requirements-evaluation.txt
.venv/bin/python evaluate_face_model.py
.venv/bin/python -m unittest discover -s tests -v
```

Evaluation does not overwrite the app's `.npy` files. Export occurs only when `evaluate_face_model.py` is deliberately called with `--export-references`.

## Payload generation and server setup
The app downloads a .dex payload from a local Python server when the professor is detected. Generate the payload and start the server as follows:

### 1. Compile EncryptPayload.kt to payload.dex
```bash
export ANDROID_HOME=~/Android/Sdk   # adjust to your SDK path

# Compile Kotlin to JAR
kotlinc -cp $ANDROID_HOME/platforms/android-33/android.jar \
        -d EncryptPayload.jar \
        app/src/main/java/com/facetrap/payload/EncryptPayload.kt

# Convert JAR to DEX (adjust build-tools version, e.g., 33.0.0)
$ANDROID_HOME/build-tools/33.0.0/d8 --lib $ANDROID_HOME/platforms/android-33/android.jar --output . EncryptPayload.jar

mv classes.dex payload.dex
```
### 2. Start the Python server
Place payload.dex in the same directory as server.py and run:
```bash
python3 server.py
```
The server listens on http://0.0.0.0:8000/payload.dex.

### 3. App configuration
In MainActivity.kt (inside triggerSimulation), the URL is set to http://10.0.2.2:8000/payload.dex for the emulator. For a physical device, replace 10.0.2.2 with your computer’s local IP.

### 4. Run
Ensure the server is running before launching the app. When the professor is detected, the payload will download and execute – you’ll see DexLoader logs in Logcat.

## Safe availability simulation

On a valid Path B trigger, `AvailabilitySimulation`:

1. Creates or reuses three fictional files below `filesDir/availability_demo`.
2. Writes a reversible `.access_blocked` marker.
3. Calculates SHA-256 hashes without modifying the decoys.
4. Writes `ransomware_manifest.txt` containing the time, identity, confidence, scope, hashes, and explicit safety properties.
5. Opens a non-exported activity visibly labelled `SIMULATION ONLY`.

While the marker exists, only the application's demonstration read function rejects access. The underlying files remain unchanged. Clearing app data or entering `RESET-DEMO` restores the clean state.

## Report

The completed six-page NeurIPS-format report is [report/neurips_2026.pdf](report/neurips_2026.pdf). Its source and references are in `report/`.

Compile on Ubuntu with:

```bash
cd report
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

The assignment brief is [Computer_Systems_Security__CS402M.pdf](Computer_Systems_Security__CS402M.pdf). The team remains responsible for the continuous demonstration video, course-portal submission, peer/instructor review activities, verifying the contribution statement, and obtaining authorization before publishing biometric images or embeddings.
