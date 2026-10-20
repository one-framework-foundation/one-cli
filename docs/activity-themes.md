# Activity themes

The persistent `one` terminal defaults to `braid-stream`. Choose at startup:

```sh
one --theme braid-stream
```

Switch during typing or model work with `/theme equalizer` and Enter. Bare `/theme` randomly selects a different preset from `patternflow-port`, `orb-pulse`, `scan-line`, `equalizer`, and `braid-stream`.
Switches are consumed locally, including while a model turn is busy. They never
enter the model conversation or change prompt/tool schemas. Inside a human answer
gate the text remains an answer, so an answer cannot be swallowed as a UI command.
The environment alternative is `ONE_CLI_THEME`. Selection lasts for this session.

Typing excites the animation; submission adds a fading birth ring in the spawn
theme. Waiting, tool execution, and streamed model chunks drive a smoothed activity
envelope. Idle motion stays quieter. The displayed status stays separate from
animation intensity: the animation is not a percentage, cost estimate or token meter.

Local detached task launches attach their actual task object to the surface. Up to
two panels show together with birth/death fades. Process exit is observed even
between user turns; before task result parsing the status says `exited <code>`,
not that the task succeeded. Remote host jobs are not observed by this UI.

The five presets are terminal renderings of five animation styles. The
`patternflow-port` name refers to the Patternflow style; it is not a complete
port. Old motion fades to zero rather than being cut off. Theme switching uses critically damped numeric parameters, preserving position and
velocity even on rapid switches. Scan motion and equalizer
heights are continuous fields before ASCII quantization. Final character changes
are necessarily discrete. Rendering targets 30 fps with elapsed-time advancement;
terminal fonts, SSH latency and display refresh still affect perceived smoothness.

`ONE_REDUCED_MOTION=1` uses a static motif while status remains live. Small terminal
layouts prioritize transcript/input. Plain output, legacy chat, and static task
cards keep their existing rendering paths. Legacy banner theme names remain valid;
the persistent surface uses braid-stream when given a legacy name.


## Parametric activity fields

`/theme braid-stream:0.7+orb-pulse:0.3` selects a weighted blend. All presets
share numeric shape weights and current/ripple parameters in `PRESETS`.
Bare `/theme` excludes the current named preset. Invalid blends leave state intact.
Selections stay local to the session; an upgraded one-cli takes effect when `one` is restarted.

Typing and paste seed theme-native disturbances at the input position; submission
makes a stronger gesture. Patternflow forms compact curls, orb sends radial pulses,
scan emits short-lived warm embers from its tracer head without distorting its
path, equalizer plucks vertical columns, and braid sends
opposing disturbances along its strands. There is no shared typing ripple overlay.
Shape weights blend these gestures continuously alongside the base shapes.
Thinking circulates, streaming creates travelling ribbons, tools stir paired eddies,
and answer waits breathe, scaled by the preset's numeric parameters.
These are analytic fluid-like fields, not a Navier–Stokes solver. Birth/death use
quintic windows; status and theme parameters retain velocity when retargeted.
Reduced motion keeps a static field. ASCII quantization remains discrete.

## Animated colour

The top banner and detached task motifs use per-cell colour currents instead of
`A_DIM` monochrome. Each preset has two RGB endpoints in the existing damped
parameter space, so weighted blends and rapid theme changes also blend colour.
Native typing gestures add local warmth, streaming lifts cyan, and tools add
amber. Colour motion follows the integrated geometry clock. The 256-colour
terminal adapter groups adjacent equal colours into runs; it does not change
terminal palette entries. Smaller palettes retain status accents, `NO_COLOR`
retains monochrome, and reduced motion keeps the colour field static.

Colours are computed as RGB fields; a 256-colour terminal shows them quantized
to its 216-colour cube.
