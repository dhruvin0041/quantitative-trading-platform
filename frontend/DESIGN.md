# Design System: Hydra Terminal

## 1. Visual Theme & Atmosphere
A restrained, institutional-grade quantitative trading interface. The atmosphere is clinical, exact, and highly readable—fusion of Bloomberg rigor with Stripe's polished modernism and Apple's minimalism. Surfaces are crisp white with layered soft grays to establish depth via subtle glassmorphism and diffused shadows. Density is "Cockpit Dense" (8/10) to maximize data visibility without feeling cluttered. Motion is "Fluid CSS" (6/10), prioritizing performance and clarity over excessive animation.

## 2. Color Palette & Roles
- **Canvas White** (#FFFFFF) — Primary background surface.
- **Subtle Surface** (#F9FAFB) — Secondary panels, cards, and secondary structural containers.
- **Deep Charcoal Ink** (#0F172A) — Primary text, maximum contrast for readability.
- **Muted Steel** (#64748B) — Secondary text, labels, axis ticks, and metadata.
- **Whisper Border** (rgba(15, 23, 42, 0.08)) — Structural lines, delicate dividers, and un-focused input borders.
- **Signal Buy (Emerald)** (#10B981) — Explicitly for BUY signals, positive PnL, up-ticks.
- **Signal Sell (Rose)** (#F43F5E) — Explicitly for SELL signals, negative PnL, down-ticks, and critical warnings.
- **Signal Hold (Amber)** (#F59E0B) — HOLD states, neutral transitions, warnings.
- **Analytics Blue (Indigo)** (#6366F1) — Primary accent for charts, analytical highlights, active tabs, and informational elements.

## 3. Typography Rules
- **Display/Headers:** `Geist Sans` — Track-tight, controlled scale. Hierarchy through weight (SemiBold/Bold) rather than massive size.
- **Body:** `Geist Sans` — Clean, functional.
- **Mono:** `Geist Mono` — Mandatory for ALL financial numbers, tickers, timestamps, and tabular data. Tabular lining must be enabled.
- **Banned:** Generic serifs, Inter for premium headers.

## 4. Component Stylings
- **Buttons:** Sharp or subtly rounded (4px). Primary buttons use Analytics Blue fill. Hover states use slight opacity shift or subtle scale. No heavy drop shadows.
- **Cards:** Crisp boundaries. Soft, diffused "whisper" shadows (`shadow-sm` or `shadow-md` tinted with the background hue). High density layouts should prefer border dividers instead of isolated cards when stacking data.
- **Inputs & Controls:** Minimalist. Unfocused states have Whisper Borders. Focused states use a sharp 1px or 2px Analytics Blue ring.
- **Data Tables:** Dense, with Mono font for values. Zebra striping or subtle hover backgrounds for row legibility.
- **Loaders:** Skeletal shimmer matching the exact layout dimensions.

## 5. Layout Principles
- Grid-first architecture. 
- The Hero header is horizontal and information-dense (Ticker, Signal, Confidence, Update time).
- No flexbox percentage hacks; use CSS Grid for predictable alignment of panels (Chart, Intelligence, Risk, Portfolio).
- Multi-column complex layouts should collapse cleanly to single-column on mobile.
- Use generous internal padding within panels to prevent visual suffocation.

## 6. Motion & Interaction
- Smooth 150-300ms transitions for all hover states.
- Data updates should flash softly or animate counters.
- Slide-in from bottom or fade-in for panel mounting.
- No layout-shifting transforms on hover.

## 7. Anti-Patterns (Banned)
- No pure black (#000000) for backgrounds or text.
- No heavy, unrealistic drop shadows.
- No emojis for icons (use Lucide/SVG).
- No animated or bouncing arrows.
- No "3 equal cards" generic layouts where data demands complex grids.
- No rounded-pill badges if a sharp structural tag fits the institutional vibe better.
