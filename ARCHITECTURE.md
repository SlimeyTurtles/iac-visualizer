# Architecture

One-page overview of how the pieces fit together.

## System diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (the user)                         │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│   │ index.html   │  │ iac.html     │  │ All other model pages    │  │
│   │  (landing)   │  │ (uses Python)│  │ (pure client-side JS)    │  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬──────────┬────────┘  │
│          │                 │                 │          │           │
│          │  All pages link theme.css + nav.js (shared design + nav) │
│          │                 │                 │          │           │
│          └────────┬────────┘                 │          │           │
│                   │                          │          │           │
│                   │  HTTP                    │          │           │
└───────────────────┼──────────────────────────┼──────────┼───────────┘
                    │                          │          │
                    ▼                          │          │
┌─────────────────────────────────┐            │          │
│        server.js  (Express)     │            │          │
│                                 │            │          │
│  static serve   ◄────────────── ┼────────────┘          │
│  /  →  public/                  │  (HTML, JS, CSS)      │
│                                 │                       │
│  GET /network-info     ─────────┼──┐                    │
│  POST /run-simulation  ─────────┼──┤                    │
│                                 │  │  spawn python3     │
└─────────────────────────────────┘  │                    │
                                     ▼                    │
                          ┌───────────────────────┐       │
                          │ server-python/        │       │
                          │   iac_bridge.py       │       │
                          │   iac.py (PyTorch)    │       │
                          │   jets_sharks.csv     │       │
                          └───────────────────────┘       │
                                                          │
                                                          │
                                            in-page data: │
                                            public/sensory_data.js
                                            inlined CSV as JS string
                                            parsed at page load
```

## Two model patterns

### Pattern A — client-side JS (the default for new models)

Used by: rumelhart.html, unsupervised.html, unsupervised_csv.html, unsupervised_autoencoder.html, sensory_dropout.html, hierarchical_art.html.

The model is a JavaScript class defined inside the page's `<script>` block. The page parses its dataset from an inlined JS string (e.g. [public/sensory_data.js](public/sensory_data.js)), constructs the model, trains it in a loop, and feeds intermediate state to D3 for visualization. The server only serves the static HTML file — no roundtrips.

**Pros**: zero install for end users, no latency, no Python dependency, easier to debug in DevTools.
**Cons**: training has to fit in browser memory and a reasonable time budget. For these toy datasets (≤200 samples), that's never been a problem.

### Pattern B — Python backend (only IAC)

Used by: iac.html.

The page calls `GET /network-info` to fetch the network topology, then `POST /run-simulation` with selected nodes and params. `server.js` `spawn`s `python3 iac_bridge.py info` (or `run`), pipes stdout JSON back, returns it. The bridge loads `jets_sharks.csv`, instantiates the PyTorch `IACModel`, runs N steps, and prints the activation history.

**Why this still exists**: IAC was the original model and the JS port is just busywork that no one has gotten around to. If you find yourself touching this, consider porting it — see `scripts/sweep_sensory.js` for an example of how to load the same model headless from JS for verification.

## File responsibilities

| Layer | Files | Responsibility |
|---|---|---|
| Entry point | [server.js](server.js) | Express server, static dir, IAC endpoints |
| Site shell | [public/index.html](public/index.html), [public/nav.js](public/nav.js), [public/theme.css](public/theme.css) | Landing page, floating nav, design tokens |
| Model pages | [public/](public/) (`*.html`) | One file per model: HTML + inline CSS overrides + inline JS for model + D3 viz |
| Inlined data | [public/sensory_data.js](public/sensory_data.js) | Datasets that the pages parse client-side |
| Backend (legacy) | [server-python/](server-python/) | PyTorch IACModel + JSON-over-stdio bridge |
| Dev tooling | [scripts/](scripts/) | Hyperparameter sweeps, probes — runs from a terminal, not from the web |
| Reference | [archive/](archive/) | Old Python implementations of models that have been ported to JS |

## Data flow per page (snapshots)

**`iac.html`**: page loads → `fetch('/network-info')` → server spawns `iac_bridge.py info` → returns JSON network → page renders. User selects nodes + presses Run → `fetch('/run-simulation', ...)` → server spawns `iac_bridge.py run [...]` → returns step-by-step activation history → page animates.

**`sensory_dropout.html`** (Hierarchy Emergence ART): page loads → parses `window.SENSORY_DROPOUT_CSV` from `sensory_data.js` → renders empty network. User presses Train → in-page `UnsupervisedModel` trains over N passes → page renders hub structure + PCA projection.

**`hierarchical_art.html`** (Forced Hierarchical ART): same as above but with 3 parallel ART layers at increasing vigilance.

**`rumelhart.html`** (Backprop semantic memory): builds a small dataset client-side, runs autograd-style backprop in JS, animates weight changes.

**`unsupervised_autoencoder.html`**: client-side encoder-decoder network; uses purple accent override.

## When to update what

| Change | Touch |
|---|---|
| Add a new model | New file in `public/`, register in `nav.js` + `index.html`. See [docs/adding-a-model.md](docs/adding-a-model.md). |
| Tweak site-wide colors | [public/theme.css](public/theme.css) (`--bg-*`, `--text-*`, `--accent`, etc.) |
| Tweak a single page's accent | `:root { --accent: ... }` block at the top of that page's inline `<style>` |
| Add a sidebar/tooltip component | Add it to [theme.css](public/theme.css) if multiple pages need it; otherwise inline on the page |
| Tune IAC model behavior | [server-python/iac.py](server-python/iac.py) |
| Tune any other model | The JS class inside that page's `<script>` block |
| Add a new endpoint | [server.js](server.js) — but read CLAUDE.md first, you probably don't need one |
