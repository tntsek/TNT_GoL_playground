# TNT_GoL_playground

Conway's Game of Life playground with multiple geometries:

- Square
- Triangle
- Rhombus
- Penrose
- Einstein
- Hex
- Trihex
- Octagon
- Voronoi

The project now uses `uv` for Python environment management, and it includes a browser-friendly static web app in `web/index.html`.

## Desktop App

Install dependencies:

```bash
uv sync
```

Run the original pygame desktop app:

```bash
uv run python conway_life.py
```

## Browser App

Serve the static site locally:

```bash
cd web
uv run python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The browser app supports painting, play/pause, stepping, randomization, image import, PNG export, and the alternate tilings above.
