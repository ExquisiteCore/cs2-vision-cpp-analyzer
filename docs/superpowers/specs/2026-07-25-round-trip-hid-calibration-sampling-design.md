# Round-Trip HID Calibration Sampling Design

## Context

Production testing after the low-sensitivity discovery update showed that discovery itself is
working. With CS2 sensitivity `2.52`, a 16-count X probe measured about 11.8 pixels and produced
ordinary-sensitivity levels near `11,43,108`; Y produced similar results.

Final sampling then exposed a separate estimator failure. A positive 43-count X sample was
reported as roughly 0.06 pixels with a marginal phase response, while the matching negative
43-count sample measured roughly 30.7 pixels. The retry logic treated the near-zero estimate as
real and scaled the next command to 2048 counts. Those large commands produced visible camera
swings and unusable samples. The fitter correctly rejected the incomplete signed curves.

The firmware, serial protocol, HID direction handling, discovery, and runtime clamp are not the
cause. The defect is that final sampling trusts one scene-dependent phase-correlation result
enough to change the command by orders of magnitude.

## Goals

- Prevent final sampling from escalating a normal account to a 2048-count command after one bad
  visual estimate.
- Retain automatic discovery up to 2048 counts for genuinely low-sensitivity accounts.
- Obtain both signed samples from real HID commands while preserving exact inverse movement.
- Reduce the visual displacement requested from ordinary accounts so the center-crop phase
  estimator remains in a more reliable range.
- Keep the accepted runtime profile limited to `max_step=120`.
- Preserve the C API, Python API, RP2350 firmware, dependency versions, and model.

## Non-goals

- No manual sensitivity input.
- No firmware or RP2350 SDK change.
- No optical-flow or feature-matching subsystem in this patch.
- No persistence of calibration profiles.
- No reduction of the calibration-only discovery ceiling below 2048.

## Selected Approach

Final calibration uses balanced round trips instead of treating each sign as an independent
outward probe. A round trip sends one signed count, captures the outward frame, sends the exact
inverse count, and captures the returned frame. It produces two samples:

1. baseline to outward frame, attributed to the outward command;
2. outward frame to returned frame, attributed to the inverse command.

With two repeats, the first round trip starts in the positive direction and the second starts in
the negative direction. If phase correlation is scene-dependent in one direction, the opposite
round trip can still provide a reliable outward and return pair for both signed fitter buckets.

Final sampling uses the discovered count levels as authoritative. It does not rescale a final
sample from its measured displacement and therefore cannot jump from 43 to 2048 after a bad
estimate. Unreliable samples are recorded with their real response and are rejected by the
existing fitter.

## Sampling Levels

The ordinary-sensitivity target shifts change from `8,32,80` pixels to `8,24,48` pixels. For a
gain of 2 counts per pixel this produces `16,48,96` counts. The smaller middle and high targets
reduce perspective change, newly exposed borders, and competition from static weapon/HUD pixels
inside the center crop.

Low-sensitivity compression remains bounded by the same 2048-count ceiling:

```text
high_px = min(48, 2048 / counts_per_pixel)
low_px  = min(8, high_px / 4)
mid_px  = min(24, high_px / 2)
```

For the previously designed 60 counts-per-pixel case, the reachable high shift is about 34
pixels, so the levels remain `480,1024,2048`. The change therefore improves ordinary accounts
without removing the low-sensitivity range.

## Data Flow

1. Capture stationary-noise samples.
2. Discover one reliable positive probe independently for X and Y, exactly as in the current
   implementation. Discovery alone may increase to 2048.
3. Derive three axis-specific levels with `8,24,48` as the unconstrained targets.
4. For every axis and level:
   - repeat 0 sends the positive count and its exact negative inverse;
   - repeat 1 sends the negative count and its exact positive inverse;
   - estimate and log both legs;
   - append two samples with the actual signed counts.
5. Fit the unchanged signed three-knot profile.
6. Accept only if the existing response, cross-axis, signed-consistency, curve, and quality gates
   pass.
7. Return a profile whose runtime `max_step` is still 120.

## Diagnostics

Each movement record keeps the existing `sample type=move` prefix and adds:

```text
leg=outward
leg=return
```

The opening record continues to report:

```text
probe_max_counts=2048 runtime_max_step=120
```

`probe_retry` is removed from final sampling because final measurements no longer change command
counts. Discovery diagnostics and exhaustion errors remain unchanged.

## Error Handling and Safety

- A successful outward send is always paired with its exact inverse before the helper returns.
- If capture fails after the outward send, the helper makes a best-effort inverse send before
  propagating the error.
- If the return capture fails, the inverse has already been sent; cleanup still calls
  `stop_all`.
- Final sampling never sends a count outside its discovered three-level plan.
- Discovery remains disarmed and firing remains disabled.
- Any missing sign or level still rejects the profile rather than silently installing a partial
  curve.

## Testing

Unit tests will cover:

- ordinary gain derives `16,48,96`;
- 60 counts-per-pixel still derives `480,1024,2048`;
- a pure round-trip sample builder assigns outward and inverse counts to the correct signed
  buckets for X and Y;
- returned visual shifts are preserved rather than synthesized from the outward result;
- final sampling has no count-rescaling path;
- the adaptive fitter can build a valid profile when one round-trip direction is rejected but
  the opposite round trip supplies reliable outward and return samples;
- runtime `max_step` remains 120;
- all algorithm, C API, and package safety tests continue to pass.

## Packaging and Acceptance

The second incremental ZIP replaces only changed native artifacts, documentation, and its
manifest. It does not include a model, ORT, CUDA, cuDNN, TensorRT, cache, Python interpreter, or
firmware.

Production acceptance uses the same COM4 Python command. Success requires:

```text
probe_levels axis=x counts=...
probe_levels axis=y counts=...
fit valid=1
标定完成 quality=...
DXGI 已打开
```

For the sensitivity-2.52 account, final movement samples must remain at the logged discovered
levels and must not contain a final-sampling jump to 2048 unless 2048 is itself one of that
axis's three discovered levels.
