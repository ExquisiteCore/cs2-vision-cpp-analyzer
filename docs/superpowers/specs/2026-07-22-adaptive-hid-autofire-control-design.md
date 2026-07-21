# Adaptive HID Calibration and Python-Controlled Autofire Design

## Goal

Make the DLL usable as a fully automatic runtime controlled by another Python
program. The Python host decides when movement output and automatic firing are
enabled. The DLL continues to own capture, detection, target selection, aiming,
and click timing.

The same change adds startup HID calibration that adapts to the active account's
resolution, in-game sensitivity, axis direction, and common nonlinear pointer
response. Head targets remain preferred, but body targets may also trigger an
aggressive single-frame shot when the crosshair is inside a valid hit region.

## Confirmed Product Behavior

- Python is the control plane; no UI or internal hotkey is added.
- Python calls calibration after the game has entered a stable playable view.
- Calibration runs once per startup and remains in memory for that runtime.
- Python separately controls movement output and automatic firing.
- A head target is preferred. A body target is an allowed lower-priority
  fallback.
- A valid head or torso hit region may trigger on the first qualifying frame.
- Existing C API entry points remain binary-compatible.
- Movement and firing remain disabled on a newly created runtime until Python
  explicitly enables them.

## Non-Goals

- No GUI, overlay, internal keyboard hook, weapon recognition, recoil model, or
  game-memory integration.
- No continuous online gain learning while tracking a moving target. Target
  motion and player input make that feedback ambiguous.
- No promise to reproduce an arbitrary acceleration curve exactly. Three
  measured operating points provide a bounded approximation.
- The one-click production environment test remains a dry-run and never arms
  HID output.

## Chosen Approach

Use a per-axis, signed, three-level calibration curve. Calibration sends small,
medium, and large positive and negative HID movements, observes the resulting
DXGI visual shift, rejects inconsistent samples, and stores the accepted curve
inside `VaRuntime`.

This is preferred over:

- One scalar gain: simple, but it cannot represent different X/Y response,
  inverted axes, or nonlinear acceleration.
- Continuous runtime learning: potentially adaptive, but it cannot reliably
  distinguish camera response from enemy movement, player movement, recoil, or
  deliberate host input.

## Control Model

Two independent runtime gates are defined:

1. `output_enabled` controls normal runtime HID emission. When false, no planned
   aim movement or click reaches the RP2350. The explicit, blocking
   `va_calibrate_hid` operation is the sole exception: it sends only its bounded
   test movements and always ends with `stop_all`.
2. `fire_enabled` controls automatic click planning. When false, aiming may
   continue but `click_left` is not planned.

Both must be true for a physical automatic click. `VaRuntimeAction.click_left`
continues to represent a planned click when firing is enabled, even if physical
output is currently disabled. This permits safe dry diagnostics.

The expected Python sequence is:

```python
runtime.set_hid_port("COM3")
profile = runtime.calibrate_hid(adapter=0, output=0)
runtime.open_dxgi(adapter=0, output=0, player_side="ct", dry_run=False)
runtime.set_fire_policy(
    body_enabled=True,
    head_confidence=0.35,
    body_confidence=0.45,
    cooldown_frames=3,
)
runtime.set_output_enabled(True)
runtime.set_fire_enabled(True)
```

The host may stop firing without stopping aim movement, or stop all physical
output immediately:

```python
runtime.set_fire_enabled(False)
runtime.set_output_enabled(False)
```

Disabling physical output calls `stop_all()` on the HID client before returning.

## C API Additions

The existing API is preserved. The following fixed-layout C structure and
functions are added to `vision_runtime_c_api.h`:

```c
#define VA_HID_CALIBRATION_LEVELS 3

typedef struct VaHidCalibrationProfile {
    int32_t schema_version;
    int32_t valid;
    int32_t frame_width;
    int32_t frame_height;
    float x_shift_px[VA_HID_CALIBRATION_LEVELS];
    float x_counts_per_pixel[VA_HID_CALIBRATION_LEVELS];
    float y_shift_px[VA_HID_CALIBRATION_LEVELS];
    float y_counts_per_pixel[VA_HID_CALIBRATION_LEVELS];
    float deadzone_px;
    int32_t max_step;
    float noise_px;
    float quality;
    int32_t accepted_samples;
} VaHidCalibrationProfile;

VA_API int32_t va_calibrate_hid(
    VaRuntime* runtime,
    int32_t adapter,
    int32_t output,
    VaHidCalibrationProfile* profile
);

VA_API int32_t va_set_fire_enabled(VaRuntime* runtime, int32_t enabled);

VA_API int32_t va_set_fire_policy(
    VaRuntime* runtime,
    int32_t body_enabled,
    float head_confidence,
    float body_confidence,
    int32_t cooldown_frames
);
```

`va_calibrate_hid` requires a configured HID port and a closed runtime session.
It updates the runtime's in-memory movement profile and also returns the profile
to the caller. A null profile pointer, open session, missing HID port, invalid
DXGI output, or rejected calibration returns `-1` with details from
`va_last_error`.

`va_set_output_enabled`, already exported by the DLL, is added to every wrapper
and example. It remains the only API that can arm physical output.

`va_set_fire_enabled` may be called before or after opening a session. A live
session observes the change on the next processed frame. Disabling it clears the
click cooldown.

`va_set_fire_policy` validates confidence values in `[0, 1]` and a non-negative
cooldown. It may be called before or after opening; a live controller receives
the updated policy without reopening the model.

The existing scalar `va_set_hid_tuning` remains available as an uncalibrated
fallback for old consumers. A successful calibration takes precedence over the
scalar gain until the runtime is destroyed or recalibrated.

## Python API Additions

The `ctypes` binding adds the matching C structure and function signatures. A
frozen Python `HidCalibrationProfile` dataclass exposes tuples for the X/Y curve,
dimensions, deadzone, maximum step, noise, quality, and sample count.

`VisionRuntime` adds:

```python
def set_output_enabled(self, enabled: bool) -> None: ...
def set_fire_enabled(self, enabled: bool) -> None: ...
def set_fire_policy(
    self,
    *,
    body_enabled: bool = True,
    head_confidence: float = 0.35,
    body_confidence: float = 0.45,
    cooldown_frames: int = 3,
) -> None: ...
def calibrate_hid(
    self,
    *,
    adapter: int = 0,
    output: int = 0,
) -> HidCalibrationProfile: ...
```

The wrapper does not automatically arm movement or firing. The external Python
program owns those state transitions explicitly.

## Calibration Procedure

### Preconditions

- The game is in a stable playable view rather than a menu or loading screen.
- The RP2350 HID port is configured and responsive.
- The requested DXGI adapter/output is available.
- No runtime session is currently open.
- Firing is forced off and physical output begins disabled.

### Sampling

The initial movement levels are 16, 40, and 80 HID counts. Each level is sampled
twice on both axes and in both directions. When a probe is below the measurable
shift threshold or too large for reliable correlation, it is retried once with
an adjusted count bounded to 8–120. Positive and negative commands are paired so
the net requested count returns to zero. With a 100 ms settling period, the
complete calibration normally takes approximately three to five seconds.

Before movement, several stationary frames estimate visual noise. Each movement
sample records:

- requested `counts_dx` and `counts_dy`;
- measured X/Y visual shift;
- phase-correlation response;
- main-axis and cross-axis motion;
- source dimensions and timestamp.

The signed gain formula is:

```text
counts_per_pixel = -requested_counts / measured_visual_shift
```

The negative sign converts the observed scene shift into the HID direction that
reduces a target's screen-space offset. A negative fitted Y gain therefore
supports an inverted Y axis without a separate special case.

### Acceptance Rules

A sample is accepted only when:

- phase-correlation response is at least `0.15`;
- main-axis movement is at least `max(1.5 px, 3 * noise_px)`;
- cross-axis movement is no more than 35% of main-axis movement;
- the value is finite and produces a nonzero signed gain.

For each axis and level, positive and negative median gain magnitudes must agree
within 40%. Every curve level must contain an accepted pair. The overall quality
score combines phase response, directional consistency, and signal-to-noise and
must be at least `0.55`.

Any rejection calls `stop_all`, discards the partial profile, leaves output and
firing disabled, and returns a diagnostic error. Python can wait for a more
stable scene and retry.

### Runtime Mapping

Each axis stores three knots: measured absolute visual shift and signed
counts-per-pixel. For an aim error:

- interpolate between adjacent knots by absolute pixel error;
- use the small gain below the first knot;
- use the large gain above the last knot;
- multiply by the signed error and round to an HID count;
- clamp to the calibrated `max_step`.

The deadzone is derived from stationary noise as
`clamp(ceil(3 * noise_px), 1, 8)`. The calibrated maximum step remains 120 counts
for the fixed 16/40/80 sampling profile. If the three gains are effectively
equal, interpolation naturally behaves like a linear scalar calibration.

Because DXGI and ROI offsets are measured in physical captured pixels, rerunning
calibration after startup covers resolution changes. Pointer acceleration is
included when it affects the game's input path; games using Raw Input naturally
produce a curve dominated by in-game sensitivity instead.

## Target Selection and Fire Policy

Matched head/body detections continue to collapse to the head detection. Head
tracks use a `0.65` score multiplier versus `1.0` for body tracks. This makes a
comparable head target win while allowing a much closer body target to remain a
practical fallback.

Firing no longer depends on a fixed 18-pixel radius, three locked frames, or the
tracking stability state. Lock state remains available as telemetry.

The current frame center is tested against a class-specific region:

- Head: the full detected head rectangle.
- Body: the torso rectangle covering 20%–80% of body width and 10%–70% of body
  height.

A head may request a click on the first frame where the crosshair is inside the
head region and confidence is at least `0.35`. A body may request a click on the
first frame where the crosshair is inside the torso region, body firing is
enabled, and confidence is at least `0.45`.

The default cooldown is three processed frames. A click requires:

```text
fire_enabled
AND confidence threshold passed
AND crosshair inside class hit region
AND cooldown available
```

Physical emission additionally requires `output_enabled`. Setting
`fire_enabled=false` clears the cooldown so re-enabling does not inherit stale
state.

Player-side filtering remains unchanged: `ct` targets T classes and `t` targets
CT classes. `unknown` remains invalid for live output.

## Threading and State

Calibration is a blocking, pre-open operation and must not run concurrently with
frame processing. Output enablement remains protected by `HidActionSender`'s
mutex. Fire enablement and the active fire policy are updated through a
controller synchronization boundary so a Python control thread may change them
between frames without a data race.

Closing a session disables firing, calls `stop_all`, and closes DXGI, but keeps
the accepted profile in `VaRuntime` so the same handle may reopen without another
calibration. Destroying the runtime discards the profile.

## Packaging and Examples

The portable package adds:

- `python/cs2_vision_runtime/` containing the stdlib-only `ctypes` wrapper;
- a no-UI Python example showing calibration, open, arm, process, disarm, and
  cleanup;
- documentation of the required Python call order and returned calibration
  fields.

The example requires an explicit live-output command-line acknowledgement before
calling `set_output_enabled(True)`. The package's one-click environment test and
DXGI diagnostic scripts remain dry-run only.

No Python interpreter is bundled because the intended host is an existing Python
program. The wrapper itself has no third-party Python dependency.

## Error Handling

- Every new C API validates null pointers and ranges and reports through
  `va_last_error`.
- Calibration always attempts `stop_all` during unwinding.
- A failed calibration never installs a partial curve.
- Enabling output without an open live HID session may update the requested
  state, but cannot emit anything until a non-dry DXGI session is opened.
- Firing enabled without physical output may produce diagnostic planned-click
  values but no HID event.
- Changing fire policy during an open session takes effect at the next frame.
- Existing consumers that never call the new APIs remain safely disarmed.

## Testing

### C++ algorithm tests

- fit signed X/Y gains from synthetic positive and negative shifts;
- preserve inverted Y sign;
- reject low response, excessive noise, cross-axis movement, and inconsistent
  opposite-direction samples;
- interpolate small/medium/large curves and clamp output;
- retain scalar-gain fallback when no profile is installed;
- prefer comparable head tracks while allowing a substantially closer body;
- trigger a head click in one qualifying frame;
- trigger a body click only inside the torso region and only when enabled;
- enforce per-class confidence and cooldown;
- ensure fire disable clears cooldown;
- ensure output disable still stops the fake HID client immediately.

### C API tests

- validate new argument ranges and null profile pointers;
- verify fire enable and policy updates reach an open runtime controller;
- verify calibration profile fields use a stable fixed layout;
- preserve all existing API calls and return semantics.

### Python tests

- bind `va_set_output_enabled`, calibration, fire enable, and fire policy;
- convert the C calibration profile into the Python dataclass;
- verify the wrapper forwards the exact call order and values;
- verify failures surface `va_last_error`;
- verify the live example always disarms in `finally`.

### Build and package verification

- run Python tests, xmake algorithm tests, xmake C API tests, and CMake/CTest;
- run package safety tests and rebuild the portable ZIP;
- extract the ZIP into a clean directory and repeat static validation;
- run a three-frame dry OpenCV smoke test;
- confirm automatic package scripts never arm HID output.

Real calibration, TensorRT inference, RP2350 movement, and firing acceptance must
be completed on the GTX 1080 Ti production machine with a stable CS2 scene. The
acceptance log must include calibration quality, all curve knots, FPS, detection
count, target class, planned movement, fire state, and click count.
