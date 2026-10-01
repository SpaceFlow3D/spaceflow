
// Only the six embedded comparisons receive editorial callouts. The full gallery
// retains its compact cards and the same underlying camera and asset renderer.
if (document.documentElement.matches('.embedded,.carousel')) {
 const {Box3, Vector2, Vector3, Raycaster}=await import('./vendor/three/three.module.js');
 const renderers=[];
 for(const card of window.__GALLERY_PAYLOAD__.cards){
 const spec=window.__FEATURED_SCENES__?.[card.id];
 if (spec) {
  const element=document.querySelector(`[data-card-id="${card.id}"]`);
  const pane=element.querySelector('.viewer-input'), output=element.querySelector('.viewer-output');
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  svg.classList.add('featured-leaders');svg.setAttribute('aria-hidden','true');
  svg.innerHTML=`<defs><marker id="featured-tip-${card.id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="currentColor" stroke-width="1.5"/></marker></defs>`;
  pane.append(svg);
  const names={'face screen':'Face screen','nose cone, left fin, right fin':'Nose cone + fins','left main wing, right main wing':'Main wings','left runner, right runner':'Runners','left solar panel, right solar panel':'Solar panels','dish antenna':'Dish antenna','front axle and wheel pair, rear axle and wheel pair':'Wheels'};
  const labels=spec.cues.map((cue,i)=>{
   const label=document.createElement('div');label.className='featured-cue'+(i?' cue-secondary':'');
   const part=document.createElement('span');part.className='cue-part';part.textContent=names[cue.label]||cue.label;
   const prompt=document.createElement('strong');prompt.textContent='“'+cue.prompt+'”';
   const level=document.createElement('span');level.className='cue-control';
   const levels=[...new Set(cue.parts.map(id=>spec.parts[id].controlLevel))];
   const high=levels.length===1&&levels[0]==='high';label.dataset.control=high?'high':'low';
   level.textContent=high?'High geometric control':'Low geometric control';
   label.append(part,prompt,level);label.title=cue.label;pane.append(label);
   const paths=cue.parts.map(id=>{const path=document.createElementNS(ns,'path');path.setAttribute('marker-end',`url(#featured-tip-${card.id})`);svg.append(path);return {id,path};});
   return {cue,label,paths};
  });
  const overall=document.createElement('div');overall.className='featured-overall';overall.textContent='Overall appearance: '+spec.global;output.append(overall);
  const raycaster=new Raycaster(),ndc=new Vector3();let cachedScene,meshes,centers,announced=false;
  const state={scene:card.id,frames:0,targets:{}};
  (window.__CAROUSEL_ANNOTATIONS__||={})[card.id]=state;
  if(document.documentElement.classList.contains('embedded'))window.__FEATURED_ANNOTATIONS__=state;
  renderers.push(()=>{
   const bounds=pane.getBoundingClientRect();if(bounds.right<0||bounds.left>innerWidth||bounds.bottom<0||bounds.top>innerHeight)return;
   const context=window.__GALLERY_TEST_API__?.getAnnotationContext();if(!context)return;
   const view=context.views.find(v=>v.element===pane),asset=context.assets.get(card.inputKey),result=context.assets.get(card.outputKey);
   if(!view||asset?.status!=='ready'||result?.status!=='ready')return;
   if(!announced){if(document.documentElement.classList.contains('embedded'))window.parent.postMessage({type:'spaceflow-featured-ready',scene:card.id},location.origin);announced=true;}
   if(asset.scene!==cachedScene){cachedScene=asset.scene;meshes=[];asset.scene.traverse(m=>{if(m.isMesh)meshes.push(m)});centers=meshes.map(m=>new Box3().setFromObject(m).getCenter(new Vector3()));}
   const box=pane.getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
   for(const {label,paths} of labels){
    const r=label.getBoundingClientRect(),bottom=label.classList.contains('cue-secondary');
    const sx=r.left-box.left+r.width*.65,sy=bottom?r.top-box.top:r.bottom-box.top;
    for(const {id,path} of paths){
     if(!centers[id])continue;
     ndc.copy(centers[id]).project(view.camera);raycaster.setFromCamera(new Vector2(ndc.x,ndc.y),view.camera);
     const hit=raycaster.intersectObjects(meshes,false)[0],ex=(ndc.x+1)*box.width/2,ey=(1-ndc.y)*box.height/2;
     const visible=hit?.object===meshes[id]&&ex>0&&ex<box.width&&ey>0&&ey<box.height;
     path.style.visibility=visible?'visible':'hidden';state.targets[id]={visible,x:ex,y:ey};
     if(visible)path.setAttribute('d',`M${sx} ${sy} Q${sx} ${(sy+ey)/2} ${ex} ${ey}`);
    }
   }
   state.frames++;
  });
 }
 }
 window.__SPACEFLOW_ANNOTATE__=()=>renderers.forEach(render=>render());
}
