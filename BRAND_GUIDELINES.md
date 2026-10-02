# AutoScale Labs — Brand Guidelines (v1.0)

Visual reference: open `brand/brand-board.html`. Logo files are in `brand/`.

## 1. Positioning in one line
**AutoScale Labs builds the Autonomous Capability Layer (ACL): production AI agents, validated end-to-end, deployed in 4–8 weeks.**

Target feel: a quiet, expensive deep-tech lab (WorldQuant Foundry register), not a SaaS landing page. Restraint signals price.

## 2. Archetype and voice
**The Clever Friend & World-Class Systems Architect.** Grounded, direct, precise.

| Do | Don't |
|---|---|
| State the mechanism ("PydanticAI validates every output against a schema") | Use adjectives instead of mechanisms ("powerful AI") |
| Give numbers with their assumptions | Give numbers with no source |
| Short sentences. Active voice. | "Revolutionize", "unleash", "supercharge", "seamless" |
| Admit limits ("We don't do X") | Promise everything |

Sample headline: *Agents that run your operations. Not demos.*

## 3. Adaptive logo — "Morph-Mark"
The mark simplifies as space shrinks, so it stays legible everywhere.

| Tier | Name | Use | Min size | File |
|---|---|---|---|---|
| 1 | **Sovereign Core** — concentric rings, 8 nodes, animated cyan core | Hero, decks, large surfaces | 96px | `brand/sovereign-core.svg` |
| 2 | **Neural Monolith** — square bracket + delta + cyan data dot | Header/nav, email signature | 24px | `brand/neural-monolith.svg` |
| 3 | **High-Contrast Delta** — hairline bracket on obsidian tile | Favicon, mobile, app icon | 16px | `brand/delta-favicon.svg` |

Rules: clear space = height of the cyan dot ×2 on all sides. Never recolor the cyan dot. Never stretch, outline, or add effects. Each tier-1 and tier-2 mark has a `-light.svg` variant (obsidian ink, darker cyan) for alabaster surfaces; the default files are for dark surfaces.

Wordmark: "AutoScale Labs" in Space Grotesk Bold, sentence case.

## 4. Color (60 / 30 / 10)
| Role | Name | Hex | Share |
|---|---|---|---|
| Base | Obsidian Midnight | `#111112` (panels `#1A1A1B`) | 60% |
| Type / warmth | Warm Alabaster | `#F8F4F0` | 30% |
| Live signal | Electric Cyan | `#38F3FF` | 10% |
| Structure | Weathered Clay | `#8B735B` | hairlines only |

Rules
- **Cyan means live.** Telemetry, active agent, primary CTA. Nothing decorative.
- **Clay is for borders, hairlines and coordinate markers.** Contrast against obsidian is about 4.2:1, which is under the 4.5:1 body-text standard, so do not use it for small body copy.
- Alabaster on obsidian is about 17:1; cyan on obsidian is about 14:1. Both pass.
- Dark-mode first. Light surfaces use alabaster background, obsidian text, and a darker cyan (`#0A9FAA`) for any cyan text.

```css
:root{
  --obsidian:#111112; --panel:#1A1A1B; --alabaster:#F8F4F0;
  --cyan:#38F3FF; --clay:#8B735B;
}
```

## 5. Typography
| Role | Face | Weight | Use |
|---|---|---|---|
| Headlines | Space Grotesk | 700 | H1–H3, stat numbers |
| Body | Inter | 400 / 500 | Paragraphs, UI |
| Editorial | Cormorant Garamond | Italic 500 | Pull quotes, founder note, one per page max |
| Data | JetBrains Mono | 500 | Telemetry, logs, ROI tables, coordinates |

All four are open-source (Google Fonts), so no licence cost. Load only the weights listed. Body 16px/1.6; headlines tight (-0.02em, 1.05 line height).

## 6. Texture and layout language
- Fine physical grain overlay at about 7% opacity (see board).
- Cartesian grid, `+` crosshairs and `[0xFA]`-style coordinate labels in clay, as section markers.
- Hairline (1px) clay borders, square corners. No drop shadows.
- Motion: one slow cyan pulse on the core mark. Respect `prefers-reduced-motion`.

## 7. Anti-AI-slop rules (hard "no")
1. No purple/blue glow orbs or gradient-mesh backgrounds.
2. No stock 3D robot hands, glowing brains, or holographic UI.
3. No emoji in headings. No "AI-powered" as a headline.
4. No fake logos, fake testimonials, or invented metrics. Every number on the site must have a source or an explicit "assumption" tag.

## 8. Open decisions for the founder
- **Name check:** "AutoScale Labs" is used as given. Run a trademark and domain search before launch.
- **Real proof:** the site needs at least one named or anonymized case with measured results before the pricing page can carry weight.
- **Photography:** none yet. Recommendation: team and workspace only, no stock.
