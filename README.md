# Entity-Based Topic Clustering Tool

A **100% client-side** knowledge graph and topic clustering tool that extracts named entities from your page titles and meta descriptions, resolves them to Wikidata, and visualizes the relationships — all running in your browser's RAM.

**Live demo:** [https://client-side-kg.pages.dev](https://client-side-kg.pages.dev)

---

## What It Does

1. **Paste** your website's titles and meta descriptions (supports thousands of pages)
2. **ML models run locally** in your browser to extract named entities (people, companies, locations) and compute semantic embeddings
3. **Wikidata resolution** maps each entity to its canonical knowledge graph ID (QID)
4. **Three visualization modes** let you explore the relationships

No server, no API key, no data ever leaves your machine (except public Wikidata lookups for entity names).

---

## Visualization Modes

| Mode | Best For |
|---|---|
| **2D Network Graph** | Seeing entity co-occurrence and semantic similarity edges between pages |
| **3D t-SNE Map** | Exploring how pages and entities cluster in semantic space (rotatable 3D scatter) |
| **SEO Topic Clusters** | Interactive treemap showing which entities are your content pillars and which pages belong to each |

---

## Tech Stack

Everything loads via CDN. Zero build step required.

| Layer | Library | Purpose |
|---|---|---|
| NER | [`Xenova/distilbert-base-multilingual-cased-ner-hrl`](https://huggingface.co/Xenova/distilbert-base-multilingual-cased-ner-hrl) via [Transformers.js](https://huggingface.co/docs/transformers.js) | Extract person / org / location entities from text |
| Embeddings | [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) via Transformers.js | 384-dim sentence vectors for similarity and t-SNE |
| Entity Resolution | [Wikidata API](https://www.wikidata.org/w/api.php) | Map entity names to canonical QIDs with descriptions |
| 2D Graph | [vis-network](https://visjs.github.io/vis-network/) | Force-directed entity relationship graph |
| 3D Scatter | [Plotly.js](https://plotly.com/javascript/) + [tsne-js](https://github.com/nicktran/tsne-js) | Interactive 3D t-SNE visualization |
| Treemap | [Plotly.js](https://plotly.com/javascript/) | Hierarchical topic cluster view |
| Styling | [Tailwind CSS](https://tailwindcss.com/) | UI framework |

---

## Use Cases

- **Topic Clustering** — group pages by the entities they share
- **Topical Authority Mapping** — see which entities you cover deeply vs. thinly
- **Internal Linking Opportunities** — pages sharing entities should link to each other
- **Cannibalization Detection** — pages with very high similarity (>0.85) may compete for the same queries
- **Entity-Based Site Architecture** — organize your content around knowledge graph entities instead of keywords

---

## Input Format

Supports two formats:

**Title/Description pairs** (recommended):
```
title: Apple iPhone 15 Pro Review & Specs
desc: Our in-depth review of the Apple iPhone 15 Pro camera, USB-C, and performance.

title: Tesla Model 3 Long Range Price & Range 2024
desc: The definitive guide to the 2024 Tesla Model 3 Long Range.
```

**Plain text** (one page per line):
```
Apple iPhone 15 Pro Review & Specs
Tesla Model 3 Long Range Price & Range 2024
```

---

## Running Locally

```bash
# Clone
git clone https://github.com/metehan777/entity-topic-cluster.git
cd entity-topic-cluster

# Serve the public/ folder (any static server works)
npx serve public

# Open http://localhost:3000
```

No `npm install` needed for the frontend — everything loads from CDN.

---

## Deploying to Cloudflare Pages

```bash
# Install wrangler
npm install -g wrangler

# Login
wrangler login

# Deploy
wrangler pages deploy public --project-name="your-project-name"
```

---

## Cloudflare Worker (Optional)

The repo also includes a standalone **Cloudflare Worker** (`src/index.ts`) that provides a REST API for entity resolution:

```bash
# Install dependencies
npm install

# Deploy the worker
wrangler deploy
```

**POST `/resolve`** — send NER output, get Wikidata QIDs back:
```json
{
  "entities": [
    {"text": "Barack Obama", "label": "PER"},
    {"text": "Google", "label": "ORG"}
  ],
  "language": "en",
  "limit": 3
}
```

---

## Project Structure

```
├── public/
│   ├── index.html          # Main UI
│   ├── app.js              # Core logic (ML, Wikidata, graph rendering)
│   └── sample-data.js      # 60-page preset dataset
├── src/
│   └── index.ts            # Cloudflare Worker REST API (optional)
├── wrangler.toml            # Cloudflare config
├── package.json
└── tsconfig.json
```

---

## Contributing

Contributions welcome. Some ideas:

- [ ] **Content Gap Analysis** — query Wikidata SPARQL for related entities your site doesn't cover
- [ ] **CSV/TSV import** — bulk import from Screaming Frog, Ahrefs, or GSC exports
- [ ] **Entity type filtering** — toggle PER / ORG / LOC visibility in the graph
- [ ] **Cluster labeling** — auto-generate topic cluster names from entity groups
- [ ] **URL column support** — associate each title/desc with its URL for linking recommendations
- [ ] **Web Worker inference** — move ML to a background thread to keep the UI responsive on large datasets

---

## License

MIT
