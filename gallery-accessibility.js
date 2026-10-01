'use strict';
// Keep the original gallery controls, and add focus containment and motion support.
const modal = document.getElementById('inspectModal');
modal.addEventListener('keydown', event => {
  if (event.key !== 'Tab') return;
  const nodes = [...modal.querySelectorAll('button:not([disabled]), [tabindex="0"]')].filter(node => node.offsetParent !== null);
  const first = nodes[0], last = nodes[nodes.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
});
const motion = matchMedia('(prefers-reduced-motion: reduce)');
motion.addEventListener('change', event => {
  const rotate = document.getElementById('autoRotateButton');
  if (event.matches && rotate.getAttribute('aria-pressed') === 'true') rotate.click();
});
const search = document.getElementById('searchInput');
const empty = document.createElement('p');
empty.textContent = 'No matching examples. Try a different object, material, or part.';
empty.hidden = true; empty.setAttribute('role', 'status');
document.getElementById('gallery').after(empty);
search.addEventListener('input', () => { empty.hidden = !![...document.querySelectorAll('.result-card')].find(card => !card.hidden); });

window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== window.parent) return;
  if (event.data?.type === 'spaceflow-visibility') window.__SPACEFLOW_VISIBLE__ = !!event.data.visible;
});
