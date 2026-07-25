# Persistent Robust HID Calibration Design

## Context

The runtime is a DLL with a stable C API. It must not understand accounts, own a UI, or decide
when caller settings have changed. The caller decides when to recalibrate.

Production testing established four separate facts:

- the RP2350 protocol-v2 path and all four HID directions work;
- one full run produced a valid 24-sample profile with quality 0.678 and X/Y gains near
  1.4 counts per pixel;
- whole-frame phase correlation can intermittently reject one large horizontal level even when
  the camera moved;
- a later run saw no measurable camera motion at any discovery count, which is indistinguishable
  from an unfocused/non-receptive game or a visually unmeasurable scene.

The valid profile was lost when the process later failed while opening ONNX Runtime because
calibration is currently memory-only. Repeating calibration on every startup unnecessarily
couples HID readiness, game focus, DXGI image quality, and inference initialization.

## Goals

- Persist one versioned HID calibration profile at a path supplied through the C API.
- Load and install a valid existing profile without moving the mouse.
- Let the caller explicitly decide when to call the existing calibration API again.
- Make calibration tolerate isolated bad frames, one unusable direction, and an unusable high
  level without increasing beyond the discovered command.
- Distinguish “no coherent visual movement” from a normal fit rejection in diagnostics.
- Never replace a valid in-memory or on-disk profile with a failed or partially saved result.
- Keep the calibration-only probe ceiling at 2048 and normal runtime output at 120.

## Non-goals

- No account identifiers, profile lists, profile selection policy, or automatic account
  detection.
- No UI, foreground-window switching, synthetic focus changes, or game-memory inspection.
- No attempt to fabricate a profile when the game accepts no HID input or the screen contains
  no measurable visual structure.
- No change to the model, inference backend, firmware, RP2350 SDK, CUDA, TensorRT, or ORT
  versions.

## C API Contract

The existing `VaHidCalibrationProfile` remains ABI-compatible. Add:

```c
VA_API int32_t va_set_hid_calibration_path(
    VaRuntime* runtime,
    const char* calibration_path
);

VA_API int32_t va_get_hid_calibration(
    VaRuntime* runtime,
    VaHidCalibrationProfile* profile
);
```

`va_set_hid_calibration_path` has transactional semantics:

- a non-null, non-empty path is required;
- if the file exists, parse and fully validate it before changing runtime state;
- if the file is valid, install it and remember the path;
- if the file does not exist, remember the path and clear any profile associated with the
  previous path;
- if the file exists but is corrupt or incompatible, return `-1` and leave the previous path
  and in-memory profile unchanged.

`va_get_hid_calibration` always initializes the output structure. It returns `0`; `valid=0`
means no profile is installed.

The existing `va_calibrate_hid` remains the caller-controlled recalibration operation. On
success it:

1. produces and validates a candidate;
2. atomically saves it when a persistence path is configured;
3. only after the save succeeds, installs it in runtime memory;
4. fills the caller’s output structure.

For backward compatibility, calibration without a configured path remains memory-only. A failed
recalibration keeps the previous profile and file untouched.

Python bindings expose matching `set_hid_calibration_path()` and
`get_hid_calibration()` methods but contain no account policy.

## Persistent File

Use OpenCV `FileStorage` JSON support, already available through the linked OpenCV core library.
The document is versioned:

```json
{
  "schema_version": 1,
  "frame_width": 1920,
  "frame_height": 1080,
  "x_shift_px": [8.05, 23.53, 46.58],
  "x_counts_per_pixel": [1.37, 1.40, 1.42],
  "y_shift_px": [7.93, 24.20, 47.15],
  "y_counts_per_pixel": [1.39, 1.41, 1.42],
  "deadzone_px": 1.0,
  "max_step": 120,
  "noise_px": 0.009,
  "quality": 0.678,
  "accepted_samples": 24
}
```

Loading requires:

- schema version 1;
- positive frame dimensions;
- finite, positive, strictly increasing shift knots;
- finite nonzero signed gains with consistent sign per axis;
- deadzone in `[0, 8]`;
- max step in `[1, 120]`;
- finite noise and quality, quality at least the fitter threshold;
- at least twelve accepted movement samples.

Saving writes a temporary sibling file, closes and reopens it for validation, then replaces the
destination with a Windows write-through atomic rename. A failed write or validation removes
the temporary file and preserves the old destination.

## Robust Visual Shift Estimator

The current estimator phase-correlates one center 640×640 crop. Static crosshair/weapon pixels,
uniform geometry, perspective change, and newly exposed borders can make that global result lock
near zero even when the camera moved.

The robust estimator uses several independent image regions:

1. convert both frames to grayscale float;
2. consider overlapping tiles in the upper 70% of the gameplay frame;
3. exclude the lower weapon/HUD area and the small central crosshair region;
4. calculate gradient energy and ignore low-texture tiles;
5. apply a Hanning window and phase-correlate each remaining tile;
6. reject non-finite and low-response candidates;
7. choose the largest internally consistent candidate group;
8. return the median X/Y shift and median response only when at least two textured tiles agree.

Agreement uses a bounded absolute/relative tolerance so normal perspective variation is allowed,
while zero-lock and unrelated tile motion are rejected. The estimator reports an explicit
invalid result instead of returning a plausible-looking zero.

The existing simple estimator remains available for its current unit-level callers. Calibration
uses the robust estimator.

## Calibration State Machine

### Discovery

Discovery stays balanced: every outward command is followed by its exact inverse before another
probe.

For each count, collect up to three round trips instead of trusting one frame pair. Accept a
discovery count when at least one round trip has a coherent robust outward estimate meeting the
main-shift and cross-axis limits. Low response does not drive proportional scaling; unreliable
counts increase only through the existing bounded discovery ladder.

If the complete ladder reaches 2048 without coherent movement, wait briefly and repeat one
bounded discovery sweep. If the second sweep also has no coherent movement, return a specific
diagnostic:

```text
HID calibration input not ready: axis=x no coherent visual movement through 2048 counts
```

No profile is modified. The DLL cannot safely calibrate without observable motion, so it must
report this condition rather than inventing data.

### Final levels

Start from the axis-specific `8/24/48` target plan. Each level performs two signed round trips,
which provide four independently measured legs.

A level is usable when at least one round trip provides coherent outward and return measurements
for both signed buckets. If a planned middle or high level is unusable:

1. retry the same count once;
2. calculate the integer midpoint between the previous accepted count and the current count;
3. try the smaller distinct count;
4. repeat for at most four distinct downward counts.

Fallback counts must remain strictly greater than the previous accepted level and never exceed
the original planned count. The low level retries at the same count but cannot shrink below the
minimum measurable range.

Log every fallback:

```text
level_fallback axis=x level=2 from_counts=66 to_counts=49 reason=no_coherent_shift
```

After all levels are selected, log the actual counts:

```text
sample_levels axis=x counts=11,33,49
```

The fitter continues to enforce signed agreement, cross-axis rejection, three strictly
increasing knots, and minimum quality.

## Failure and Old-Profile Behavior

Calibration operates on a candidate detached from `runtime->options.hid_calibration`.

- Capture, HID, fit, or persistence failure returns an error and leaves the installed profile
  unchanged.
- A corrupt configured file never clears a valid current profile.
- A missing newly selected path intentionally clears the profile from the previous path because
  the caller selected a different persistence target.
- A process crash during save leaves either the complete old file or the complete new file.
- No calibration failure opens the live session or arms physical output.

## Runtime Data Flow

Typical caller flow:

```c
VaRuntime* runtime = va_create();
va_set_hid_calibration_path(runtime, "hid-calibration.json");

VaHidCalibrationProfile profile;
va_get_hid_calibration(runtime, &profile);
if (!profile.valid) {
    va_calibrate_hid(runtime, 0, 0, &profile);
}

// Caller may invoke va_calibrate_hid again whenever its own settings change.
va_open_dxgi(runtime, 0, 0, 0);
```

The DLL does not decide whether a valid cached profile corresponds to an account. That decision
belongs entirely to the caller.

## Testing

Tests cover:

- robust estimation recovers a synthetic translation when a large static overlay and one blank
  tile would mislead a whole-frame estimate;
- two or more agreeing textured tiles are required;
- discovery retries an unreliable measurement without proportional explosion;
- discovery performs a second bounded sweep before returning input-not-ready;
- high-level fallback is strictly downward, bounded, and remains above the previous level;
- one failed high count can select a smaller valid third knot;
- no fallback path can increase counts or change runtime max step;
- JSON profile save/load round-trips every field;
- corrupt, truncated, incompatible, NaN, non-monotonic, and over-120 profiles are rejected;
- failed saves and failed recalibration preserve the previous installed profile and destination;
- missing paths clear the old in-memory profile only after the new path selection succeeds;
- new C API functions handle null runtime, null path, null output, missing file, valid file, and
  corrupt file;
- existing algorithm, C API, Python wrapper, package, and safety tests remain green.

## Packaging and Acceptance

The incremental package may contain the changed DLL, CLI, import library, C header, Python
binding, example, documentation, and manifest. It must not contain changed models, CUDA, cuDNN,
TensorRT, ORT, cache, or firmware files.

Production acceptance has two stages:

1. configure an empty calibration path, calibrate once, and verify a valid local file is written;
2. restart the process, set the same path, verify `va_get_hid_calibration` reports the saved
   profile, and open DXGI without any calibration movement.

An explicit caller-triggered recalibration must replace the file only after a new valid profile
is fully measured and validated.
