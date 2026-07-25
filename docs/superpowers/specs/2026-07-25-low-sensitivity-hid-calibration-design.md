# Low-Sensitivity HID Calibration Design

## Context

The adaptive HID calibration currently starts from fixed movement counts `16`, `40`, and
`80`. A probe that is too small or too large is retried once, but the adjusted count is
clamped to `8..120`. The same value, `120`, is also the normal runtime movement clamp.

Production testing on the GTX 1080 Ti host established the following boundary:

- the RP2350 moves the Windows pointer in all four directions;
- the CS2 camera visibly moves during the calibration sequence;
- a single `80`-count in-game probe is not visibly measurable;
- a single `1000`-count in-game probe moves the camera clearly;
- calibration samples at `120` counts usually report only about two pixels with phase
  responses below `0.15`, so the fitter rejects every level.

The runtime path, firmware, serial protocol, and CS2 input path therefore work. The defect is
that calibration probe range is incorrectly coupled to the much smaller runtime movement
clamp.

## Goals

- Automatically calibrate accounts whose sensitivity requires more than `120` HID counts for
  a reliable visual measurement.
- Limit calibration probes to `2048` counts per axis.
- Keep normal automatic-aim movement limited to `120` counts per processed frame.
- Preserve exact requested-count return pairs for every successful outward probe.
- Preserve the existing three-knot signed X/Y profile, inverted-axis support, C API, Python API,
  and RP2350 protocol.
- Produce diagnostics that distinguish an exhausted low-sensitivity probe from an unsuitable
  scene or directional camera limit.

## Non-goals

- No firmware, RP2350 SDK, model, ONNX Runtime, TensorRT, CUDA, or Python API changes.
- No UI or manual sensitivity input.
- No persistent calibration profile; the accepted profile remains startup memory only.
- No increase to the normal runtime `max_step` of `120`.

## Selected Approach

Use an axis-level discovery probe before collecting the existing repeated samples. Discovery
finds one reliable counts-per-pixel estimate for X and one for Y. It then derives three
axis-specific, strictly increasing count levels. This avoids independently climbing from a
small count for every direction and repeat.

Alternatives rejected:

1. Retrying every sample independently is a smaller local edit but repeats the same count
   ladder up to 24 times and makes low-sensitivity calibration unnecessarily slow.
2. A user-supplied multiplier is simple but violates the requirement that different accounts
   calibrate automatically.

## Constants and Separation of Limits

Calibration owns constants separate from runtime control:

```text
discovery_start_counts = 16
calibration_probe_max_counts = 2048
discovery_max_attempts = 8
discovery_target_shift_px = 8
minimum_phase_response = 0.15
runtime_max_step = 120
```

`calibration_probe_max_counts` is used only while calibration is disarmed. The profile returned
to `AimController` continues to contain `max_step=120`.

## Discovery Data Flow

Discovery runs once per axis after the stationary-noise samples:

1. Capture a baseline frame.
2. Send a positive-axis probe at the current absolute count.
3. Wait for the configured settling period and capture the moved frame.
4. Best-effort send the exact inverse count, wait, and capture the returned baseline before
   evaluating or attempting another count.
5. Measure main-axis shift, cross-axis shift, and phase response.
6. Accept the discovery result only when:
   - values are finite;
   - phase response is at least `0.15`;
   - main-axis shift is at least `max(4 px, 3 * noise_px)`;
   - main-axis shift is no more than 25% of the shorter frame dimension;
   - cross-axis shift is no more than 35% of main-axis shift.
7. If the signal is too small or its response is too low, increase the absolute count. A
   reliable-but-small measurement scales proportionally toward eight pixels; an unreliable
   measurement doubles the count. The next count must be strictly larger and is capped at
   `2048`.
8. If a shift is too large, reduce the count proportionally toward eight pixels.
9. Try at most eight distinct counts. Stop with a diagnostic error if no reliable result exists
   after eight attempts or at the `2048` limit. A calculated count already attempted is an
   exhausted search rather than an invitation to loop.

If capture or measurement throws after an outward command, the probe helper attempts the exact
inverse command before propagating the original failure. `stop_all` remains the final cleanup,
but it is not treated as an inverse relative movement.

## Deriving Three Sampling Levels

For each axis, discovery estimates the absolute gain:

```text
counts_per_pixel = discovery_counts / discovery_shift_px
```

The largest reachable target shift is:

```text
high_px = min(80, 2048 / counts_per_pixel)
```

The three desired shifts are compressed when the original `8,32,80` targets cannot fit:

```text
low_px  = min(8,  high_px / 4)
mid_px  = min(32, high_px / 2)
high_px = high_px
```

The desired shifts are converted back to integer counts. Rounding must preserve three strictly
increasing counts in `8..2048`. Discovery fails explicitly if the reachable range cannot supply
three distinct levels whose smallest expected shift meets the existing measurable-shift
threshold.

This produces the original scale for ordinary sensitivity and a compressed but measurable
curve for low sensitivity. For example, a discovered gain near `60 counts/px` yields levels
near `480`, `1024`, and `2048`, while the runtime clamp remains `120`.

## Final Sampling and Fitting

The existing sampling structure remains:

- two axes;
- three levels per axis;
- two repeats;
- positive and negative directions;
- an exact inverse movement and returned baseline after every sample.

The only change is that each axis uses its discovered count levels instead of fixed
`16,40,80`. Samples retain their actual signed counts. Existing fit rules remain authoritative:

- response at least `0.15`;
- measurable main-axis shift;
- cross-axis shift no more than 35%;
- positive/negative gain agreement within 40%;
- all three levels present on both axes;
- overall quality at least `0.55`;
- strictly increasing measured shift knots.

Final sampling retains one bounded corrective retry. A too-small sample is increased toward its
derived target without exceeding `2048`; a too-large sample is reduced. A low-response sample
whose magnitude is otherwise measurable is repeated once at the same count, because its
magnitude is not trustworthy enough to rescale. After the retry, the sample is recorded as-is
and the fitter decides whether it is acceptable. Final sampling never starts another discovery
ladder.

Consequently, discovery cannot make a bad scene, a pitch limit, or inconsistent directions
silently pass.

## Diagnostics

Calibration logs add parseable records for every discovery attempt and final level selection:

```text
probe_discovery axis=x attempt=0 counts=16 shift_px=... cross_px=... response=...
probe_levels axis=x counts=480,1024,2048 probe_max=2048
```

An exhausted probe reports its axis, limit, last shift, and last response. The final `fit` and
curve records remain unchanged. The opening calibration record reports both
`probe_max_counts=2048` and `runtime_max_step=120` so the two limits cannot be confused during
support.

## Safety and Error Handling

- Calibration still requires explicit live-output permission from Python.
- Firing remains disabled throughout calibration.
- Every completed outward probe is paired with its exact inverse count.
- The probe limit is below the signed 16-bit SDK limit; firmware may safely split it into HID
  report-sized chunks.
- A failed discovery installs no profile and never opens the live runtime session.
- Exceptions attempt inverse movement, then `stop_all`, then release DXGI in the existing
  cleanup path.
- Normal runtime output remains clamped to `120`, regardless of calibration probe counts.

## Testing

TDD coverage will include:

- an unreliable small probe increases rather than stopping at `120`;
- proportional discovery can select a count above `120` and never above `2048`;
- ordinary sensitivity derives three ordered levels without unnecessarily reaching the cap;
- low sensitivity derives three ordered levels that may end at `2048`;
- an insufficient reachable range is rejected instead of producing duplicate curve knots;
- invalid and non-finite observations are rejected;
- profile fitting still accepts signed axes and inverted Y;
- the returned profile and runtime controller remain limited to `120`;
- all existing algorithm and C API tests continue to pass.

The production acceptance test is the existing Python command. Success requires
`fit valid=1`, a nonzero quality, three nonzero X/Y curve knots, and transition to
`DXGI 已打开` on the low-sensitivity account.

## Packaging

The incremental package will replace only the native runtime artifacts and documentation needed
for this fix. It will not replace the model, TensorRT/CUDA environment, RP2350 firmware, or Python
wrapper unless build verification shows an ABI-visible change, which this design does not make.
