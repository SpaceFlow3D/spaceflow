'use strict';
const viewer=document.getElementById('hero-viewer');
document.getElementById('reset-view').addEventListener('click',()=>viewer.contentDocument?.getElementById('resetButton')?.click());
for(const [id,delta] of [['carousel-prev',-1],['carousel-next',1]])document.getElementById(id).addEventListener('click',()=>viewer.contentWindow?.postMessage({type:'spaceflow-carousel-step',delta},location.origin));
window.addEventListener('message',event=>{
 if(event.origin!==location.origin||event.source!==viewer.contentWindow||event.data?.type!=='spaceflow-carousel')return;
 const {index,total}=event.data;document.getElementById('carousel-position').textContent=total?`${index} / ${total}`:'No matches';
 document.getElementById('carousel-prev').disabled=index<=1;document.getElementById('carousel-next').disabled=index>=total;
});
const steps = [
 ['THE INPUT','The input consists of editable geometric parts, each with a local control level and an optional text or image appearance cue. Deformable superquadrics provide a compact representation for specifying these parts.'],
 ['LOCAL GEOMETRIC GUIDANCE','During structure generation, conditioned flow preserves stronger geometric constraints in designated regions while allowing the generative prior to complete weakly constrained regions. This makes the fidelity–freedom trade-off local rather than uniform across the object.'],
 ['PART-SPECIFIC APPEARANCE','PartField features segment the generated structure into semantic regions, which are matched to the input primitives. Routed cross-attention sends each part only its assigned text or image cue; part-aware guidance encourages coherent appearance within each region.']
];
document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => {
 document.querySelectorAll('[data-step]').forEach(item => {const active=item===button;item.classList.toggle('active',active);item.setAttribute('aria-pressed',String(active));});
 const [label,description]=steps[Number(button.dataset.step)];document.getElementById('method-label').textContent=label;document.getElementById('method-description').textContent=description;
}));
const metrics = {
 overall:{values:[75.9,68.7],ci:['65.7, 83.8','58.1, 77.6'],description:'Overall preference balances fidelity to the input and realism.'},
 pgf:{values:[78.3,66.3],ci:['68.3, 85.8','55.6, 75.5'],description:'Prompt and geometric fidelity measures high-control shape adherence together with prompt-driven completion in low-control regions.'},
 realism:{values:[43.4,85.5],ci:['33.2, 54.1','76.4, 91.5'],description:'Realism measures how plausible the object looks, independently of input adherence. Against low uniform guidance, the confidence interval includes 50%; there is no clear preference on this criterion.'}
};
document.getElementById('metric-select').addEventListener('change', event => {
 const m=metrics[event.target.value];['low','high'].forEach((key,index)=>{document.getElementById('value-'+key).textContent=m.values[index]+'%';document.getElementById('bar-'+key).style.width=m.values[index]+'%';document.getElementById('ci-'+key).textContent='95% CI ['+m.ci[index]+']';});
 document.querySelector('.bar-chart').setAttribute('aria-label',event.target.selectedOptions[0].text+': SpaceFlow wins '+m.values[0]+' percent against low uniform guidance and '+m.values[1]+' percent against high uniform guidance.');
 document.getElementById('metric-note').textContent=m.description+' These are VLM-judge results over 83 assets, aggregated by majority vote across three passes; intervals are 95% Wilson confidence intervals.';
});
document.getElementById('copy-citation').addEventListener('click', async () => {
 const status=document.getElementById('copy-status'), text=document.getElementById('bibtex').textContent;
 try {await navigator.clipboard.writeText(text);status.textContent='BibTeX copied to clipboard.';}
 catch {const range=document.createRange();range.selectNodeContents(document.getElementById('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';}
});
// Pausing after scrolling away saves work without starting media unexpectedly.
const video=document.getElementById('demo-video');
new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)video.pause();else if(!matchMedia('(prefers-reduced-motion: reduce)').matches)video.play().catch(()=>{});},{threshold:.15}).observe(video);
document.addEventListener('visibilitychange',()=>{if(document.hidden)video.pause();});

let viewerVisible = true;
const sendViewerVisibility = () => viewer.contentWindow?.postMessage({type:'spaceflow-visibility',visible:viewerVisible && !document.hidden},location.origin);
new IntersectionObserver(entries=>{viewerVisible=entries[0].isIntersecting;sendViewerVisibility();},{rootMargin:'100px'}).observe(viewer);
viewer.addEventListener('load',sendViewerVisibility);
document.addEventListener('visibilitychange',sendViewerVisibility);

// Keep the image usable while the two exact GLBs and the 3D renderer load.
const animatedTeaser = document.getElementById('animated-teaser');
if (animatedTeaser) {
 const teaserObserver = new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) return;
  teaserObserver.disconnect();
  import('./assets/teaser/viewer.js?v=20260909-1').then(module => module.mountTeaser(animatedTeaser)).catch(error => {
   document.getElementById('teaser-status').textContent = 'Original figure · 3D unavailable';
   console.error('Unable to initialize the elephant teaser:', error);
  });
 }, {rootMargin:'200px'});
 teaserObserver.observe(animatedTeaser);
}

// Mirror the gallery's rotation control without restarting its models.
const heroRotate = document.getElementById('hero-rotate');
let rotationObserver;
function syncRotationControl() {
 const control = viewer.contentDocument?.getElementById('autoRotateButton');
 if (!control) return;
 const rotating = control.getAttribute('aria-pressed') === 'true';
 heroRotate.setAttribute('aria-pressed', String(rotating));
 heroRotate.textContent = rotating ? 'Pause rotation' : 'Resume rotation';
}
viewer.addEventListener('load', () => {
 rotationObserver?.disconnect();
 const control = viewer.contentDocument?.getElementById('autoRotateButton');
 if (!control) return;
 rotationObserver = new MutationObserver(syncRotationControl);
 rotationObserver.observe(control, {attributes:true, attributeFilter:['aria-pressed']});
 syncRotationControl();
});
heroRotate.addEventListener('click', () => viewer.contentDocument?.getElementById('autoRotateButton')?.click());
syncRotationControl();
