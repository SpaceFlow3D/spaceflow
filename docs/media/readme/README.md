# README graphics

These small, static assets reproduce the visual identity of the
[SpaceFlow project website](https://spaceflow3d.github.io/spaceflow/) in GitHub's README.
Light and dark variants are selected with GitHub's supported `<picture>` markup.

## Lettering

- The SpaceFlow wordmark uses **Caveat Bold** and the website's pink-to-gold
  gradient (`#ff66c4` to `#ffc259`). Its complete ink bounds include the final
  letter's overhang.
- The subtitle uses **Manrope SemiBold**; the resource buttons use **DM Sans
  SemiBold**, matching the website's font families.
- Lettering is converted to SVG paths. No font installation, external font
  request, script, or runtime dependency is required to display the header.

The source fonts are the website's `Caveat-Bold.ttf`, `manrope-2.ttf`, and
`dm-sans-2.ttf`, available under `https://spaceflow3d.github.io/spaceflow/assets/fonts/`.
They use the SIL Open Font License: [Caveat](https://github.com/googlefonts/caveat),
[Manrope](https://github.com/sharanda/manrope), and
[DM Sans](https://github.com/googlefonts/dm-fonts).
The generated assets contain outlined lettering, not distributed font files.

## Research group and institution logos

The logos use the same source artwork as the project website:

- [Computer Vision and Geometry Group (CVG)](https://cvg.ethz.ch/assets/images/logo/cvg-logo.svg),
  copied unchanged from the website's `assets/logos/cvg.svg`. The transparent gold artwork is used in both themes.
- [Gradient Spaces Lab](https://gradientspaces.stanford.edu/sites/g/files/sbiybj27761/files/media/image/logo.001_0.png),
  also available as the website's `assets/logos/gradient-spaces.png`. The SVG wrappers embed the original PNG unchanged, trim its surrounding whitespace with a viewBox, and use the website's white-matte removal filter. The dark variant applies the website's inversion and hue rotation so the wordmark remains readable without a white background.
- [ETH Zürich](https://ethz.ch/etc/designs/ethz/img/header/ethz_logo_black.svg),
  also available as the website's `assets/logos/ethz.svg`.
- [Stanford University red block S with tree](https://identity.stanford.edu/wp-content/uploads/sites/3/2020/07/block-s-right.png),
  also available as the website's `assets/logos/stanford-s.png`. The original transparent artwork and colors are retained in both themes.
- Microsoft supplies [gray-text artwork](https://news.microsoft.com/microsoft-logo_rgb_c-gray-2/) and [white-text artwork](https://news.microsoft.com/microsoft-logo_rgb_c-wht/),
  stored as `microsoft-light.png` and `microsoft-dark.png` with their original transparency and colors.

The ETH Zürich dark variant changes the ink color while retaining the source logo geometry. Stanford keeps the supplied red block S with tree in both themes. Microsoft uses its supplied gray- and white-text variants.
Institution logos remain the respective institutions' trademarks.
Author names, affiliations, contribution marks, and profile links are taken from
the project website. The simple resource icons were drawn for this README.
