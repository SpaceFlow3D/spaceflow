(() => {
 if(!document.documentElement.classList.contains('carousel'))return;
 const gallery=document.getElementById('gallery'),search=document.getElementById('searchInput');
 gallery.tabIndex=0;gallery.setAttribute('role','region');gallery.setAttribute('aria-label','SpaceFlow examples carousel');
 const cards=[...gallery.querySelectorAll('.result-card')];
 for(const card of cards){
  const data=window.__GALLERY_PAYLOAD__.cards.find(c=>c.id===card.dataset.cardId);
  const header=document.createElement('div');header.className='carousel-card-heading';
  header.append(card.querySelector('.card-heading-row'));
  const prompt=document.createElement('p');prompt.className='carousel-shape';prompt.textContent='“'+data.shape+'”';header.append(prompt);card.prepend(header);
  const shape=card.querySelector('.prompt-list>div');shape.remove();
 }
 let previous='';
 const visible=()=>cards.filter(c=>!c.hidden);
 const position=()=>{
  const list=visible(),left=gallery.getBoundingClientRect().left+parseFloat(getComputedStyle(gallery).paddingLeft);
  let index=0,best=Infinity;list.forEach((c,i)=>{const distance=Math.abs(c.getBoundingClientRect().left-left);if(distance<best){best=distance;index=i;}});
  return {list,index};
 };
 function report(){const {list,index}=position(),state={type:'spaceflow-carousel',index:list.length?index+1:0,total:list.length};const key=JSON.stringify(state);if(key!==previous){previous=key;window.parent.postMessage(state,location.origin);}}
 function go(delta){const {list,index}=position();const target=list[Math.max(0,Math.min(list.length-1,index+delta))];if(!target)return;const left=target.getBoundingClientRect().left-gallery.getBoundingClientRect().left-parseFloat(getComputedStyle(gallery).paddingLeft);gallery.scrollBy({left,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}
 gallery.addEventListener('keydown',event=>{if(event.target!==gallery||!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();go(event.key==='ArrowLeft'?-1:1);});
 gallery.addEventListener('scroll',report,{passive:true});new ResizeObserver(report).observe(gallery);
 search.addEventListener('input',()=>{gallery.scrollLeft=0;report();});
 window.addEventListener('message',event=>{
  if(event.origin!==location.origin||event.source!==parent)return;
  if(event.data?.type==='spaceflow-carousel-step')go(event.data.delta);
  if(event.data?.type==='spaceflow-carousel-search'){search.value=String(event.data.query||'');search.dispatchEvent(new Event('input',{bubbles:true}));}
 });
 window.__CAROUSEL__={go,report};report();
})();
