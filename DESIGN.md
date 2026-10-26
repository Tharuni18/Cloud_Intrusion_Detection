---
name: Neural Defense Telemetry
colors:
  surface: '#0d131f'
  surface-dim: '#0d131f'
  surface-bright: '#333946'
  surface-container-lowest: '#080e1a'
  surface-container-low: '#161c28'
  surface-container: '#1a202c'
  surface-container-high: '#242a37'
  surface-container-highest: '#2f3542'
  on-surface: '#dde2f4'
  on-surface-variant: '#cfc2d6'
  inverse-surface: '#dde2f4'
  inverse-on-surface: '#2a303d'
  outline: '#988d9f'
  outline-variant: '#4d4354'
  surface-tint: '#ddb7ff'
  primary: '#ddb7ff'
  on-primary: '#490080'
  primary-container: '#b76dff'
  on-primary-container: '#400071'
  inverse-primary: '#842bd2'
  secondary: '#dcfdff'
  on-secondary: '#00373a'
  secondary-container: '#00f1fd'
  on-secondary-container: '#006a6f'
  tertiary: '#ffb3b6'
  on-tertiary: '#68001a'
  tertiary-container: '#ff5168'
  on-tertiary-container: '#5b0015'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#f0dbff'
  primary-fixed-dim: '#ddb7ff'
  on-primary-fixed: '#2c0051'
  on-primary-fixed-variant: '#6900b3'
  secondary-fixed: '#6ff6ff'
  secondary-fixed-dim: '#00dce6'
  on-secondary-fixed: '#002022'
  on-secondary-fixed-variant: '#004f53'
  tertiary-fixed: '#ffdada'
  tertiary-fixed-dim: '#ffb3b6'
  on-tertiary-fixed: '#40000c'
  on-tertiary-fixed-variant: '#920028'
  background: '#0d131f'
  on-background: '#dde2f4'
  surface-variant: '#2f3542'
typography:
  display-lg:
    fontFamily: Sora
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 38px
    letterSpacing: -0.02em
  display-md:
    fontFamily: Sora
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-lg:
    fontFamily: Sora
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Sora
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  title-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 22px
    letterSpacing: 0em
  body-lg:
    fontFamily: Geist
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: 0em
  body-md:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Geist
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.05em
  telemetry-num-lg:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.03em
  telemetry-num-sm:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 0.75rem
  gutter-mobile: 0.5rem
  margin: 1rem
  margin-compact: 0.75rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style
The design system establishes an ultra-refined, mission-critical neural command interface engineered for elite SecOps operators and machine learning infrastructure engineers. It bridges high-density cyber defense diagnostics with intuitive telemetry consumption on mobile viewports.

The visual style fuses technical precision minimalism with high-contrast tactical luminescence:
- **Atmospheric Obsidian:** Deep, low-noise dark surfaces minimize visual fatigue during critical operational incidents.
- **Luminescent Telemetry:** Targeted neural violet and electric cyan focal points isolate anomalous data patterns and automated agent pipelines.
- **Tactical Utility:** Zero decorative clutter. Every pixel, glowing border, and data node conveys immediate operational state, latency status, or threat vector classification.

## Colors
The color architecture relies on a pure dark mode foundation tailored for OLED mobile displays, maintaining precise WCAG 2.1 AAA contrast ratios across data streams.

### Surface Hierarchy
- **Canvas Base (`#0A0D14`):** The absolute background canvas providing total contrast containment.
- **Surface Level 1 (`#101622`):** Primary card panels, container blocks, and bottom sheets.
- **Surface Level 2 (`#182234`):** Elevated inner wells, interactive rows, code inputs, and nested metric chips.
- **Surface Stroke (`rgba(255, 255, 255, 0.08)`): Crisp boundary definition for glass panels.

### Brand & Neural Accents
- **Primary Violet (`#A855F7` / `#8A2BE2`):** Reserved for unsupervised ML autoencoders, heuristic models, primary interactive nodes, and system-level neural activity.
- **Telemetry Cyan (`#00F2FE` / `#4FACFE`):** High-velocity network packets, data ingestion streams, and live telemetry tracking.

### Semantic Threat Matrix
- **Critical Ruby (`#FF3B5C`):** Breach confirmations, critical zero-day heuristics, model collapse alerts.
- **High Alert Orange (`#FF8A00`):** Threat escalations, anomalous tensor shifts, throttling.
- **Warning Amber (`#FFB800`):** High memory pressure, latency jitter, untrusted cert handshakes.
- **Healthy Emerald (`#00E676`):** Nominal baseline, secure ingress/egress, verified pipeline synchronization.

## Typography
Typography balances geometric authority with dense technical readability. 

- **Headlines (Sora):** Sora delivers geometric gravitas, engineered structure, and high legibility across executive dashboards and modular section titles.
- **Body & Telemetry (Geist):** Geist brings clean horizontal proportions and tabular spacing optimization essential for real-time telemetry metrics, terminal outputs, and complex ML confidence percentages.

Numeric values across data visualizations, sparkline anchors, and real-time counters must utilize tabular figure styles (`font-variant-numeric: tabular-nums`) to prevent jitter during live data re-renders.

## Layout & Spacing
The layout model is optimized for mobile touch ergonomics while supporting compact, split-panel diagnostic monitoring.

### Architecture & Grids
- **Mobile Grid:** 4-column fluid layout on viewports under 600px, expanding to 8 columns on foldables and tablets. 
- **Density System:** Baseline 4px and 8px rhythmic increments. Tighter spacing (`space-xs` to `space-md`) governs related telemetry groups, ensuring vital metric sets remain above the fold.
- **Safe Ergonomics:** Standard margin is set at 16px (`1rem`), with critical controls strictly positioned within the lower two-thirds of the viewport for single-handed emergency triage.

## Elevation & Depth
Elevation is expressed through tonal luminosity and edge clarity rather than heavy directional drop shadows.

- **Level 0 (Canvas):** Pure `#0A0D14`. Ground plane for navigation anchors and non-card backgrounds.
- **Level 1 (Substrate Panels):** `#101622` with a 1px frosted glass boundary (`rgba(255, 255, 255, 0.08)`). Soft ambient backlight blur (`backdrop-filter: blur(12px)`).
- **Level 2 (Active Nodes & Modals):** `#182234` paired with subtle localized glow halos corresponding to operational status:
  - Neural Core: `box-shadow: 0 0 16px -4px rgba(168, 85, 247, 0.25)`
  - Active Telemetry: `box-shadow: 0 0 16px -4px rgba(0, 242, 254, 0.25)`
  - Critical Intercept: `box-shadow: 0 0 20px -2px rgba(255, 59, 92, 0.35)`
- **Depth Inversion:** Inset micro-wells (`background: rgba(10, 13, 20, 0.75); border: 1px solid rgba(255, 255, 255, 0.04)`) host sparklines, logs, and numeric feeds.

## Shapes
The design system applies a disciplined roundedness level of `2` (`0.5rem` / `8px` baseline).

- **Standard Cards & Modules:** `8px` corner radius. Maintains a tight, technical cadence without harsh needle-sharp corners or overly casual pill geometry.
- **Nested Containers & Metric Wells:** `6px` radius to maintain concentric nested balance within 8px parent containers.
- **Status Badges & Pill Toggles:** `16px` to `9999px` strictly for atomic chips, status states, and live state tags to visually distinguish metadata from operational cards.

## Components

### Buttons & Interactive Triggers
- **Primary Action (Neural Deploy):** Background `#A855F7`, label `#FFFFFF` in Geist Semibold, 44px minimum touch target height. Subtle ambient border highlight (`1px solid rgba(255, 255, 255, 0.2)`).
- **Secondary Action (Subsystem Query):** Background `#182234`, border `1px solid rgba(255, 255, 255, 0.12)`, text `#E2E8F0`. Hover/active shifts border to `#00F2FE`.
- **Destructive/Neutralize Trigger:** Background `rgba(255, 59, 92, 0.12)`, border `1px solid #FF3B5C`, text `#FF3B5C`.

### Status Pills & Glow Badges
- **Structure:** Height 22px, padding 2px 8px, font Geist 10px Bold (`label-sm`), uppercase with `0.05em` tracking.
- **Variants:**
  - *Autoencoder Synchronized:* Violet text, violet hairline border, 6px pulsing dot indicator.
  - *Telemetry Stream Active:* Cyan text, cyan hairline border, static luminous dot.
  - *Threat Intercepted:* Ruby text, ruby background at 15% opacity, rapid breathing glow.

### Telemetry Cards & Sparklines
- **Container:** `#101622` background, 1px frosted glass border, padding `space-md` (12px).
- **Data Architecture:** Metric label top-left (`label-md`), differential delta badge top-right (`+14.2%` in emerald or `-4.1%` in ruby), monolithic readout (`telemetry-num-lg`) center-left.
- **Sparklines:** Vector SVG paths (1.5px stroke width) terminating in a glowing end-node circle (3px diameter). Under-chart gradient fill decaying from 20% opacity to 0% at the base axis.

### Lists & Activity Feeds
- **Log Items:** Stacked vertical layout with 1px border dividers (`rgba(255, 255, 255, 0.04)`). Monospace timestamp in Geist muted grey (`#64748B`), threat severity icon, payload string in Geist Regular.
- **Selection State:** Row background transitions to `rgba(168, 85, 247, 0.08)` with a 2px left border strip in `#A855F7`.

### Form Controls & Filter Toggles
- **Inputs:** Dark field `#0A0D14`, hairline border `1px solid rgba(255, 255, 255, 0.1)`. Focus state switches border to `#00F2FE` with a 2px diffused cyan glow.
- **Checkboxes & Radios:** 18px dimensions, 4px inner corner radius. Active state fills with `#A855F7` carrying a micro-inset white pip or glyph.