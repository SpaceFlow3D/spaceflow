// Shared by the project page and gallery; run before styles to avoid a theme flash.
(() => {
  const root = document.documentElement;
  const key = 'spaceflow-color-theme';
  const system = matchMedia('(prefers-color-scheme: dark)');
  let preference;
  try { preference = localStorage.getItem(key); } catch {}
  const valid = value => value === 'dark' || value === 'light';
  if (!valid(preference)) preference = null;
  const preferred = () => preference || 'light';
  function render() {
    const dark = root.dataset.theme === 'dark';
    const button = document.getElementById('theme-toggle');
    if (button) {
      button.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} mode`);
      button.setAttribute('aria-pressed', String(dark));
      button.querySelector('.theme-label').textContent = dark ? 'Light mode' : 'Dark mode';
    }
    const galleryButton = document.getElementById('themeButton');
    if (galleryButton) {
      galleryButton.classList.toggle('active', !dark);
      galleryButton.setAttribute('aria-label', `Use ${dark ? 'light' : 'dark'} theme`);
      galleryButton.title = galleryButton.getAttribute('aria-label');
      galleryButton.querySelector('.theme-label').textContent = dark ? 'Light theme' : 'Dark theme';
      galleryButton.querySelector('.theme-icon').textContent = dark ? '☀' : '☾';
    }
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#17141e' : '#fcfcfc');
    document.getElementById('hero-viewer')?.contentWindow?.postMessage({type:'spaceflow-theme', theme:root.dataset.theme}, location.origin);
  }
  function apply(theme) {
    if (!valid(theme)) return;
    if (root.dataset.theme !== theme) root.dataset.theme = theme;
    render();
  }
  function choose(theme) {
    preference = theme;
    try { localStorage.setItem(key, theme); } catch {}
    apply(theme);
  }
  apply(preferred());
  new MutationObserver(render).observe(root, {attributes:true, attributeFilter:['data-theme']});
  document.addEventListener('DOMContentLoaded', () => {
    render();
    document.getElementById('theme-toggle')?.addEventListener('click', () => choose(root.dataset.theme === 'dark' ? 'light' : 'dark'));
    // The existing gallery control sets its own theme before this listener runs.
    document.getElementById('themeButton')?.addEventListener('click', () => choose(root.dataset.theme));
    document.getElementById('hero-viewer')?.addEventListener('load', render);
  });
  system.addEventListener('change', () => { if (!preference) apply(preferred()); });
  window.addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    preference = valid(event.newValue) ? event.newValue : null;
    apply(preferred());
  });
  window.addEventListener('message', event => {
    if (event.origin === location.origin && event.source === window.parent && event.data?.type === 'spaceflow-theme') apply(event.data.theme);
  });
})();
