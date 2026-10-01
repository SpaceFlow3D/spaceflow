'use strict';
const params = new URLSearchParams(location.search);
if (params.get('embed') === '1') {
  document.documentElement.classList.add('embedded');
  const selected = window.__GALLERY_PAYLOAD__.cards.find(card => card.id === params.get('scene')) || window.__GALLERY_PAYLOAD__.cards.find(card => card.id === 'scene-009');
  window.__GALLERY_PAYLOAD__.cards = [selected];
  window.__GALLERY_PAYLOAD__.sceneCount = 1;
  window.__GALLERY_PAYLOAD__.assets = Object.fromEntries([selected.inputKey, selected.outputKey].map(key => [key, window.__GALLERY_PAYLOAD__.assets[key]]));
}

if (params.get('carousel') === '1') {
 document.documentElement.classList.add('carousel');
 const order=['scene-009','scene-012','scene-011','scene-017','scene-033','scene-035'];
 window.__GALLERY_PAYLOAD__.cards.sort((a,b)=>(order.includes(a.id)?order.indexOf(a.id):100+a.index)-(order.includes(b.id)?order.indexOf(b.id):100+b.index));
}
