# SpaceFlow project website

**[Visit the website](https://spaceflow3d.github.io/)**

Project page for **SpaceFlow: Locally Controllable 3D Generation**. This repository contains the static website, its research assets, and the static build/validation scripts.

## Features

- Author links and official CVG, Gradient Spaces, ETH Zürich, and Stanford logos; the latest supplied 4K teaser with a TL;DR, a visible abstract, and an overview before the examples.
- Caveat Bold branding with the paper's pink-to-orange gradient.
- All 36 interactive input/result comparisons in a horizontally scrolling main-page carousel, with synchronized cameras and GLB downloads. The first six examples include part-tracked local appearance labels. Models rotate slowly by default, yield to dragging, and respect reduced-motion preferences. The standalone gallery remains available.
- An animated paper teaser with all six input geometries and six matching outputs. Drag either model to rotate its pair. Annotation arrows follow the relevant input parts and hide when occluded.
- Supplementary primitive-to-part routing figures, the method overview, and an interactive results chart.
- Light theme by default, a persistent light/dark toggle, keyboard controls, reduced-motion support, pause/resume, and original-figure fallbacks.

All six teaser examples have matching 3D input/output pairs. The image-conditioned cactus from the supplementary material replaces the lamp in the animated version; the original paper figure remains available with the figure toggle. No inference service is required.

## Local preview

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/>. Use an HTTP server so the browser can load the 3D assets.

## Build and publish

```sh
python3 scripts/build.py
python3 scripts/validate.py --dist
```

The build creates `dist/` from an explicit website allowlist. Validation checks local resources, anchors, gallery policy, model hashes, embedded glTF dependencies, and GitHub's per-file size limit. All fonts, models, and rendering libraries are hosted with the site; runtime asset paths work under the `/spaceflow/` subdirectory.

GitHub Pages publishes the root of the `main` branch. Run the build and validation commands before pushing an update; a push to `main` updates the public website. The `.nojekyll` file keeps the site as plain static files.

## Assets and attribution

Research content, figures, video, and model outputs belong to their respective authors. No additional research-content license is granted by this repository. Font and renderer license notices are included with the corresponding assets. Three.js 0.180.0 is vendored locally under its MIT license. Model source provenance and SHA-256 hashes are recorded in `assets/teaser/provenance.json` and `assets/gallery-data.js`.

Official logo source URLs are recorded in `assets/logos/sources.json`.

The publication identifier, final citation metadata, and method-code link will be added when available.
