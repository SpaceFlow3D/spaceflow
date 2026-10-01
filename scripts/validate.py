"""Static integrity checks for local links, compressed models, and release files."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json,gzip,hashlib,struct,re,sys
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'dist' if '--dist' in sys.argv else ROOT
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set();self.errors=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:
   if a['id'] in self.ids:self.errors.append('duplicate id: '+a['id'])
   self.ids.add(a['id'])
  for k in ('src','href','poster'):
   if a.get(k):self.links.append(a[k])
  if tag=='img' and not a.get('alt'):self.errors.append('image missing alt')
  if tag=='iframe' and not a.get('title'):self.errors.append('iframe missing title')
parsers={}
for file in ['index.html','gallery.html']:
 p=Parser();p.feed((BASE/file).read_text());parsers[file]=p;assert not p.errors,p.errors
# The standalone source disallowed external files; the extracted gallery must
# explicitly allow its same-origin renderer, styles, fonts, textures, and models.
gallery_html=(BASE/'gallery.html').read_text()
csp_match=re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"',gallery_html)
assert csp_match,'missing gallery CSP'
directives={parts[0]:set(parts[1:]) for directive in csp_match.group(1).split(';') if (parts:=directive.split())}
for kind in ('script-src','style-src','font-src','img-src','connect-src'):
 assert "'self'" in directives.get(kind,set()),f'CSP blocks same-origin {kind}'
count=0
for name,p in parsers.items():
 for link in p.links:
  u=urlsplit(link)
  if u.scheme or link.startswith('//'):continue
  if u.path:
   assert not u.path.startswith('/'),f'absolute path incompatible with repository Pages: {link}'
   path=BASE/unquote(u.path);assert path.is_file(),f'missing {link} in {name}'
  if u.fragment and (not u.path or u.path.endswith('.html')):
   target=parsers.get(u.path or name)
   if target:assert u.fragment in target.ids,f'missing anchor {link}'
  count+=1
for file in ['style.css','gallery.css']:
 for url in re.findall(r'url\([\'\"]?([^\)\'\"]+)',(BASE/file).read_text()):
  if not url.startswith(('data:','http','#')):assert (BASE/url).is_file(),f'missing CSS asset: {url}'
p=json.loads((BASE/'assets/gallery-data.js').read_text().split('=',1)[1].rstrip(';'))
assert len(p['cards'])==36 and len(p['assets'])==72
for key,a in p['assets'].items():
 data=(BASE/a['url']).read_bytes();raw=gzip.decompress(data)
 assert len(raw)==a['rawBytes'] and hashlib.sha256(raw).hexdigest()==a['sha256'],key
 magic,version,length=struct.unpack_from('<4sII',raw)
 assert magic==b'glTF' and version==2 and length==len(raw),key
 jsonlen,kind=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+jsonlen])
 for section in ('buffers','images'):
  for obj in gltf.get(section,[]):assert not obj.get('uri') or obj['uri'].startswith('data:'),f'external model dependency {key}'
provenance=json.loads((BASE/'assets/teaser/provenance.json').read_text())
for name,metadata in provenance['models'].items():
 raw=(BASE/'assets/teaser'/name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==metadata['sha256'],name
 magic,version,length=struct.unpack_from('<4sII',raw)
 assert magic==b'glTF' and version==2 and length==len(raw),name
 jsonlen,kind=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+jsonlen])
 for section in ('buffers','images'):
  for obj in gltf.get(section,[]):assert not obj.get('uri') or obj['uri'].startswith('data:'),f'external teaser dependency {name}'
assert (BASE/'assets/teaser/figure-plate.webp').is_file()
# Include vendored ES-module imports in static release validation.
for module in [BASE/'assets/teaser/viewer.js',*(BASE/'assets/vendor/three').glob('*.js')]:
 for relative in re.findall(r"from\s*['\"](\.[^'\"]+)['\"]",module.read_text()):
  assert (module.parent/urlsplit(relative).path).is_file(),f'missing module {relative} in {module.name}'
for f in BASE.rglob('*'):
 if f.is_file() and '.git' not in f.parts and f.stat().st_size>=100*1024*1024:raise AssertionError(f'100 MiB file: {f}')
assert 'shape:\'\'' not in (BASE/'site.js').read_text()
print(f'PASS: {count} local references, image/iframe labels, unique anchors, {72+len(provenance['models'])} SHA-256-verified self-contained GLBs (72 gallery + {len(provenance['models'])} teaser), 36 scenes, and GitHub per-file size limit.')
