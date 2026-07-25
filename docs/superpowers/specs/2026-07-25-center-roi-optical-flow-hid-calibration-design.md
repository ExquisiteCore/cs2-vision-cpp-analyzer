# Center-ROI Optical-Flow HID Calibration Design

## Context

The current HID calibration measures a single before/after frame with phase
correlation. Static screen-space HUD can dominate that measurement, while a
real 3D camera turn is not a rigid whole-image translation. The result is a
coherent near-zero estimate even though the game camera visibly moved. Raising
the probe through 2048 counts makes the failure disruptive without adding
useful evidence.

The RP2350 serial path, protocol-v2 ACK path, positive and negative HID motion,
DXGI capture, and profile persistence have already been verified separately.
This redesign replaces only visual movement measurement and bounded probe
discovery.

## Decision

Measure the movement of game-world texture near the screen center. The
crosshair itself remains fixed at the screen center, so the estimator tracks
the background around it rather than the crosshair pixels.

Use a centered ROI, mask a small crosshair box, find distributed scene
features, track them with pyramidal Lucas-Kanade optical flow, reject outliers,
and report the median inlier displacement. Use short frame bursts after each
HID command and select the best valid measurement. Every outward command is
paired with an exact inverse command.

## Alternatives Considered

1. **Center-ROI multi-frame optical flow — selected.** It directly measures
   local scene motion, tolerates perspective and new edge content, and ignores
   most HUD without requiring game-specific HUD templates.
2. **Calculate from caller-supplied CS2 sensitivity/FOV.** This is simpler but
   is configuration conversion rather than visual calibration and adds caller
   inputs that can be stale.
3. **Reuse a previously saved profile.** This is valid for unchanged settings
   but does not solve explicit recalibration after settings change.

## Measurement Pipeline

1. Capture a stable baseline burst. Stability is judged only inside the
   center ROI; animated HUD outside it is irrelevant.
2. Use a centered ROI no larger than 640 by 480 pixels. Mask the central
   96-by-96 crosshair area.
3. Detect features with `goodFeaturesToTrack`. Require features in at least
   three spatial cells so one object or one edge cannot determine the result.
4. Send one bounded relative HID command and wait for the firmware/SDK command
   to complete.
5. Capture up to six post-command frames. For each frame, run
   `calcOpticalFlowPyrLK`, keep valid forward tracks, and use robust median/MAD
   filtering to reject outliers.
6. Select the candidate with the most inliers and lowest residual spread.
   Report median X/Y displacement, inlier count, cell coverage, and spread.
7. Send the exact inverse HID command in the existing exception-safe cleanup
   path and repeat the burst measurement for the return leg.
8. Accept a round trip only when outward and return measurements are both
   reliable, primarily aligned to the requested axis, and opposite in sign.

The estimator never tracks the drawn crosshair and never treats static HUD
agreement as camera movement.

## Bounded Discovery and Fitting

- Probe counts are limited to `16, 32, 64, 120`; calibration never emits a
  movement larger than the runtime `max_step=120`.
- The first count producing a reliable center displacement of at least four
  pixels seeds the existing three-level curve plan.
- Formal levels remain increasing and at or below 120 counts.
- Each level uses two balanced round trips with alternating outward signs.
- The existing adaptive profile fitter, quality checks, C ABI structure, and
  atomic JSON persistence remain unchanged.
- If 120 counts cannot produce a reliable measurement, stop with an actionable
  error asking the caller to present a stable, textured game-world surface.
  Do not continue moving the view.

## Diagnostics

Each discovery and formal sample logs:

- requested axis and signed counts;
- selected post-command frame index;
- detected feature count, valid track count, inlier count, and spatial cells;
- median X/Y shift and residual spread;
- the concrete rejection reason when unusable.

The final error must distinguish insufficient texture, unstable frames,
cross-axis motion, and asymmetric return movement.

## API and Ownership

No account, UI, or game-setting logic enters the DLL. Existing APIs remain:

- the caller selects the calibration path;
- the caller requests explicit recalibration;
- the DLL measures, validates, installs, and atomically saves the profile;
- the caller decides when settings/account changes require recalibration.

## Testing

Tests must be written before production changes and cover:

1. center translation with a fixed crosshair;
2. perspective-warped camera motion with static radar/score/kill-feed overlays;
3. animated overlay pixels outside the center ROI;
4. insufficient texture and insufficient spatial feature coverage;
5. dominant cross-axis motion rejection;
6. opposite, symmetric outward/return validation;
7. a hard assertion that every generated probe is at most 120 counts;
8. unchanged profile fitting, C API persistence, and 84-byte profile ABI.

Release packaging must use the current worktree's CMake Release output. Before
publishing, inspect the packaged DLL export table and instantiate
`VisionRuntime` through the packaged Python wrapper so stale-parent build
artifacts cannot pass verification again.
