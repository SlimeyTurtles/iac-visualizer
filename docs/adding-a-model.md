# Adding a new model page

The fastest way to add a new visualization is to copy an existing page and modify it. This recipe assumes a **client-side JS** model (see [ARCHITECTURE.md](../ARCHITECTURE.md) for why).

## 1. Pick a template

Match your model type to the closest existing page:

| If your model is… | Copy |
|---|---|
| Unsupervised, vigilance-based clustering | [public/sensory_dropout.html](../public/sensory_dropout.html) |
| Multi-layer / multi-vigilance ART | [public/hierarchical_art.html](../public/hierarchical_art.html) |
| Supervised backprop with hidden layers | [public/rumelhart.html](../public/rumelhart.html) |
| CSV-driven with user-loaded data | [public/unsupervised_csv.html](../public/unsupervised_csv.html) |
| Encoder/decoder autoencoder | [public/unsupervised_autoencoder.html](../public/unsupervised_autoencoder.html) |

## 2. Copy it and rename

```bash
cp public/sensory_dropout.html public/your_model.html
```

## 3. Update the page identity

In your new file:

- `<title>` → your model's name
- `<h2>` in the sidebar → same
- `<div class="subtitle">` → one-sentence description
- Any inline `<style>` rules — keep page-specific ones, drop the rest (everything in [theme.css](../public/theme.css) is already shared)
- If you want a non-orange accent, add at the top of `<style>`:
  ```css
  :root {
      --accent: #38bdf8;          /* your color */
      --accent-dark: #0284c7;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --accent-soft: #7dd3fc;
  }
  ```

## 4. Swap the model class

Find the existing model class in the `<script>` block (e.g. `UnsupervisedModel`). Replace it with yours. Keep the same shape: a constructor that takes hyperparameters + data, a `train()` or `step()` method, accessors for current state. The page already wires UI events → model methods → D3 updates; if your method names match the template's, you barely need to touch the UI code.

## 5. Provide data

Two choices:

**(a)** Inline a CSV in a `public/your_data.js` file, mirroring [public/sensory_data.js](../public/sensory_data.js):
```js
window.YOUR_DATASET_CSV = `Name,Feature1,Feature2,...
sample1,...
...`;
```
Add `<script src="/your_data.js"></script>` to your HTML. This is the preferred path — keeps the page self-contained, no extra HTTP calls.

**(b)** If the user loads data themselves (file picker, paste box), copy the parsing pattern from `unsupervised_csv.html`.

## 6. Register the page in the nav and home

[public/nav.js](../public/nav.js) — add to the `PAGES` array:
```js
{ href: '/your_model.html', label: 'Your Model Name', section: 'Supervised' /* or Unsupervised */ },
```

[public/index.html](../public/index.html) — add a card to the appropriate `.card-grid`:
```html
<a href="/your_model.html" class="model-link your-css-class">
    <h3>Your Model Name</h3>
    <p>One-sentence description.</p>
</a>
```
And give the new CSS class an accent at the bottom of index.html's `<style>`:
```css
.your-css-class { --accent: #38bdf8; }
```

## 7. Update the server console output (optional)

[server.js](../server.js) prints the available pages on startup. Add your line there if you want it listed.

## 8. Run and verify

```bash
npm start
```

Open `http://localhost:3000/your_model.html`. The floating nav menu in the top-right should now include your page.

## Don't add server endpoints

If you find yourself reaching for `app.get('/your-model-info', ...)`: stop. Run the model in the browser. See the "Things NOT to do" section in [CLAUDE.md](../CLAUDE.md). The Rumelhart and Unsupervised pages used to have Python bridges and they were removed because they were unnecessary overhead.

## Tuning defaults

If your model has hyperparameters, pick sensible defaults that produce a *visible* result in 1–2 training passes. Users should see something happen when they hit TRAIN without first touching any sliders.

For the Hierarchy Emergence ART, we ran a hyperparameter sweep (see [scripts/sweep_sensory.js](../scripts/sweep_sensory.js)) to find the configuration that produces emergent superordinate hubs. You can adapt that script for your model.

## Component patterns to reuse

These classes already exist in [theme.css](../public/theme.css) — use them rather than inventing:

| Class | What it gives you |
|---|---|
| `#sidebar` | Left rail with form controls, scrollable |
| `.subtitle` | Page tagline under the h2 |
| `.section-header` | Small accent-colored section divider |
| `.param-group` | Label + input pair, vertical |
| `.params-grid` | 2-column grid for compact param layouts |
| `.btn-primary` | Big accent-gradient action button |
| `.scrubber-section` | Step slider + playback buttons housing |
| `.episode-list` + `.episode-btn` | Scrollable bank of selectable items |
| `.info-card` | Top-left floating info panel |
| `.tooltip`, `.tooltip-header`, `.tooltip-content` | Hover tooltips |
| `.legend` | Bottom-right activation gradient legend |
| `.status` (with `.running`/`.ready`/`.error` modifier) | Tiny status line in the sidebar |
| `.dataset-info` | Small "what's in this dataset" bubble |
| `.hub-list`, `.hub-item` | Sidebar list of trained hubs/clusters |
| `.pool-similarity-bar` | Per-pool similarity bar w/ vigilance threshold marker |
| `.reset-view-btn` | "Reset view" affordance in the top-right of the viz |

If your page needs something genuinely new and reusable, add it to [theme.css](../public/theme.css), not the page.
