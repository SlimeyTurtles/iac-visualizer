# CLAUDE.md — Agent Guide

This is a visualization site for **connectionist semantic-memory models**. It started as a single Interactive Activation and Competition (IAC) visualizer and grew into a small gallery of supervised + unsupervised models. Read this before making changes so you don't repeat past missteps.

## What this project is, in one paragraph

A static-ish Node/Express site (`server.js`) that serves a set of HTML pages from [public/](public/). Each page is a standalone D3.js visualization of one cognitive model. **Most models run entirely in the browser** — pure JS, no server roundtrip. The original IAC model is the lone exception: it still has a Python backend in [server-python/](server-python/) that the server `spawn`s on demand. There is no build step, no framework, no transpiler — just `npm start` and HTML.

## Quick orient

| Path | What lives there |
|---|---|
| [server.js](server.js) | Express server + Python bridge for IAC |
| [public/](public/) | All HTML pages, [theme.css](public/theme.css), [nav.js](public/nav.js), inlined data |
| [server-python/](server-python/) | `iac.py` + `iac_bridge.py` + `jets_sharks.csv` (the *only* live Python) |
| [scripts/](scripts/) | Dev/research tools (hyperparameter sweeps, probes) — **not** runtime |
| [archive/](archive/) | Dead code kept for reference: old Rumelhart/Unsupervised Python implementations |
| [docs/](docs/) | How-tos. Start with [docs/adding-a-model.md](docs/adding-a-model.md) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | One-page system diagram |

## Conventions you need to know

### Shared design system

Every page links [public/theme.css](public/theme.css). It defines design tokens as CSS custom properties (`--accent`, `--bg-panel`, `--text-muted`, etc.) and the shared component styles (sidebar, tooltip, info-card, episode-list, btn-primary, etc.). **Don't hardcode colors.** Use `var(--accent)`, `var(--text-muted)`, `var(--success)`, etc.

Pages that need a different accent override at the top of their inline `<style>`:
```css
:root {
    --accent: #8b5cf6;          /* Rumelhart / autoencoder use purple */
    --accent-dark: #7c3aed;
    --accent-glow: rgba(139, 92, 246, 0.2);
    --accent-soft: #a78bfa;
}
```

Page-specific selectors go in the inline `<style>` after the override. **Don't add another `<style>` block on top of theme.css — keep one per page**, and use it only for what's truly unique to that page.

### Shared nav

[public/nav.js](public/nav.js) injects a floating menu button on every page that lets users switch between models. To add a new page, append it to the `PAGES` array there. Also handles the data-arrow-step convention for number inputs (lets you set custom arrow-key step sizes on `<input type="number">`).

### Data is inlined

Page datasets that need to ship to the browser are embedded as JS string literals (see [public/sensory_data.js](public/sensory_data.js)). The page parses the CSV at load time. This avoids an extra fetch and keeps the page self-contained. **Don't add `/data/*.csv` HTTP routes for new pages** unless you have a strong reason.

### Models are usually client-side JS

When this project started, models ran in Python and the browser fetched results. **That pattern is dead** — the only model still using it is IAC. New models should be implemented in JavaScript directly inside the page (D3 for viz, plain JS for the model class). Look at the `UnsupervisedModel` class in [public/sensory_dropout.html](public/sensory_dropout.html) for the canonical pattern: a class that owns `templates`/`weights` + `step()`/`train()` methods, called from page event handlers.

**Why this matters**: spawning Python per request adds latency, requires server changes for every model, and forces users to install torch. Doing it client-side keeps the project zero-install for end users.

### URL surface

URLs are flat: `/iac.html`, `/sensory_dropout.html`, etc. Don't move pages into subdirectories under `public/` — it breaks the URL and the nav.

## What lives outside `public/`

### `server.js`

Only two live endpoints:
- `GET /network-info` — IAC initial network state (spawns `iac_bridge.py info`)
- `POST /run-simulation` — runs IAC for N steps (spawns `iac_bridge.py run ...`)

Static `public/` is served on `/`. **If you need a new endpoint, think hard first**: can it be client-side? If you genuinely need server-side compute, use the `runPythonBridge` helper.

### `server-python/`

`iac.py` — IACModel class (PyTorch). `iac_bridge.py` — JSON-over-stdio wrapper that `server.js` spawns. `jets_sharks.csv` — the example dataset. The bridge resolves the CSV path via `__file__`, so it works regardless of `cwd`.

### `scripts/`

[scripts/sweep_sensory.js](scripts/sweep_sensory.js) — runs the JS UnsupervisedModel headless over a hyperparameter grid to find configs that cluster the 8 concepts cleanly. Loads `public/sensory_data.js` directly. Was used to choose the current Hierarchy Emergence ART defaults (v=0.95, lr=0.05, gain=1, inh=0, passes=2).

[scripts/probe_extreme.js](scripts/probe_extreme.js) — companion probe for narrow parameter regions. Both are dev-only; not served, not imported by the site.

### `archive/`

Old Python implementations of Rumelhart and Unsupervised models, plus the `synthetic_sensory_dropout.xlsx` source for sensory_data.js and `Algorithmv5.docx` design notes. **Nothing here is called at runtime.** Keep for reference; delete freely if it gets in the way.

## How to add a new model page

See [docs/adding-a-model.md](docs/adding-a-model.md) for the full recipe. Short version: copy `public/sensory_dropout.html` as your template, swap the model class, add the page to `public/nav.js` and `public/index.html`, done.

## Things NOT to do

- **Don't add new Python bridges for new models.** Run them in JS in the browser. The Rumelhart/Unsupervised Python bridges were removed when the JS ports replaced them — adding new Python is reversing that lesson.
- **Don't hardcode colors.** Use the CSS custom properties from [theme.css](public/theme.css). If a new accent is needed, override `--accent` family at the top of the page, don't paste hex codes inline.
- **Don't introduce a build step / framework** (Next.js, Vite, React, Tailwind, etc.) without asking. We evaluated this and chose vanilla HTML + theme.css for performance and simplicity. The cost of a build pipeline is not worth it for ~8 pages.
- **Don't fetch CSVs from `public/data/`.** Inline them as JS string literals (see [public/sensory_data.js](public/sensory_data.js)).
- **Don't move pages out of `public/`.** URL stability matters and nav.js assumes flat paths.
- **Don't write `<style>` blocks that duplicate theme.css rules.** If you're tempted to redefine `.tooltip` or `.btn-primary`, you're working against the system — fix theme.css instead.

## Running it

```bash
npm install   # one-time
npm start     # http://localhost:3000
```

If you change a page, just refresh the browser — there's no build, and caching headers are disabled in `server.js` for dev.

## Past decisions to know

- **Rumelhart and Unsupervised used to be Python-backed.** They were ported to client-side JS. The Python files are in `archive/` for reference.
- **Hierarchy Emergence ART uses normalized Euclidean per-pool similarity, not cosine.** Each pool's similarity is `1 - ‖x_p - t_p‖² / num_features_in_pool ∈ [0, 1]`. Templates are NOT unit-normalized — they store raw EWMA of input vectors so magnitude differences (e.g. salmon brighter than sunfish) are preserved. The earlier cosine-based version couldn't split sunfish/salmon because they share color *direction* even though they differ in *brightness*. Per-pool normalization makes the metric scale-invariant across pools, so a 5-feature pool can't outweigh a 3-feature pool.
- **Defaults for Hierarchy Emergence ART**: `vigilance=0.977, lr=0.08, gain=1, inh=0, passes=1`. This produces exactly 8 hubs, one per concept, 8/8 perfect (all 160 samples in their own species' pure hub). Lower vigilance still produces emergent superordinate hubs ("fish" merging sunfish+salmon, "bird" merging robin+canary) — the model can demonstrate both behaviors. See `scripts/sweep_sensory.js` and `scripts/find_8hubs.js`.
- **CSS was deduplicated into `theme.css` in May 2026.** Each page was ~1500 lines, ~80% of which was repeated. Now each page links `theme.css` and keeps only its unique selectors. Don't undo this.
