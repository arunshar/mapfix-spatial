# Distortion-Aware MapFix

MapFix Spatial is an interactive MVP for correcting distorted geospatial coordinates and rendering a cleaner projection.

Live GitHub Pages demo:

```text
https://arunshar.github.io/mapfix-spatial/
```

The browser includes a deterministic fallback correction engine, and `server.py` adds a live OpenAI backend:

- `/api/correct` computes the MapFix correction server-side and asks GPT for a concise geospatial analysis.
- `/api/render-map` calls the Image API to render a clean corrected map projection preview.
- `OPENAI_API_KEY` stays on the server and is never exposed to browser code.

## Run locally

From this folder:

```bash
OPENAI_API_KEY=your_key_here python server.py
```

Then open:

```text
http://127.0.0.1:4173
```

Without `OPENAI_API_KEY`, the page still runs as a local deterministic demo and the UI will show that the key is missing.

Optional model overrides:

```bash
OPENAI_TEXT_MODEL=gpt-5.5
OPENAI_IMAGE_MODEL=gpt-image-1
OPENAI_IMAGE_SIZE=1536x1024
OPENAI_IMAGE_QUALITY=medium
```

## Deploy

For the live backend, deploy this folder as a Python/Docker web service. The included `Dockerfile` and `render.yaml` are ready for Render-style Docker hosting.

Required environment variable:

- `OPENAI_API_KEY`

The static-only files can still be deployed to Vercel, Netlify, Cloudflare Pages, or GitHub Pages, but GPT/Image API features require `server.py` or an equivalent serverless API.

## Files

- `index.html` - app shell
- `styles.css` - responsive dashboard styling
- `app.js` - coordinate distortion, correction, rendering, metrics, and PNG export
- `server.py` - OpenAI-backed API and static-file server
- `Dockerfile` - container entrypoint for hosted backend deploys
