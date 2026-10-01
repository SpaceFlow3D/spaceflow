"""Build an allowlisted, dependency-free static site for GitHub Pages."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'dist'
FILES=('index.html','gallery.html','style.css','site.js','gallery.css','gallery-setup.js','gallery-accessibility.js')
# Only the generated output directory is replaced; original research files stay intact.
if DEST.exists(): shutil.rmtree(DEST)
DEST.mkdir()
for name in FILES: shutil.copy2(ROOT/name,DEST/name)
shutil.copytree(ROOT/'assets',DEST/'assets')
(DEST/'.nojekyll').touch()
print(f'Built {len(list(DEST.rglob("*")))} paths in dist/.')
