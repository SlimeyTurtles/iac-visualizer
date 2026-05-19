# Semantic Network Models

Interactive visualizations of classic and modern connectionist models for concept learning and semantic memory.

## Run it

```bash
npm install
npm start
```

Open <http://localhost:3000>.

## What's here

- **IAC Visualizer** — Interactive Activation and Competition (localist semantic memory)
- **Rumelhart Model** — Backprop-trained concept-property semantic memory
- **Concept Learning (Default / CSV)** — Vigilance-based ART clustering
- **Hierarchy Emergence ART** — Flat ART where a superordinate "fish" hub emerges organically
- **Forced Hierarchical ART** — Three parallel ART layers at increasing vigilance (coarse → fine)
- **Unsupervised Autoencoder** — Encoder/decoder over the same dataset

## Requirements

- Node.js
- Python 3 with PyTorch + pandas (only needed for the IAC page; everything else is pure JS)

## Project layout

| Folder | What |
|---|---|
| [public/](public/) | HTML pages, shared CSS/JS |
| [server-python/](server-python/) | Python backend for IAC only |
| [scripts/](scripts/) | Dev tools (hyperparameter sweeps) |
| [archive/](archive/) | Old code kept for reference |
| [docs/](docs/) | How-to guides |

## Deployment

Auto-deploys to EC2 on every push to `main` via GitHub Actions ([.github/workflows/deploy.yml](.github/workflows/deploy.yml)). See [docs/deploy-setup.md](docs/deploy-setup.md) for the one-time EC2 + GitHub-secrets setup.

## For contributors (including AI agents)

Read [CLAUDE.md](CLAUDE.md) for project conventions and [ARCHITECTURE.md](ARCHITECTURE.md) for the system overview. To add a new model, see [docs/adding-a-model.md](docs/adding-a-model.md).
