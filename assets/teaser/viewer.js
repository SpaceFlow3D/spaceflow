import * as THREE from '../vendor/three/three.module.js';
import {GLTFLoader} from '../vendor/three/GLTFLoader.js';
import {RoomEnvironment} from '../vendor/three/RoomEnvironment.js';
import {FIGURE,EXAMPLES} from './layout.js?v=20260906-4';

export async function mountTeaser(figure) {
 const stage=figure.querySelector('.teaser-stage'), image=figure.querySelector('#teaser-image');
 const play=figure.querySelector('#teaser-play'), toggle=figure.querySelector('#teaser-static'), status=figure.querySelector('#teaser-status');
 const original=image.src, originalAlt=image.alt, animatedAlt='Local control examples: chair, trophy, figurine, balloon, elephant, and image-conditioned cactus, with rotating input and output assets.', background=new URL('./figure-plate.webp',import.meta.url).href;
 const motion=matchMedia('(prefers-reduced-motion: reduce)'), pitch=.29;
 const pairs=EXAMPLES.map(example=>({...example,views:[],angle:example.angle}));
 const slots=[],views=[],leaders=[],fallbacks=[];
 let renderer,environment,visible=false,paused=motion.matches,staticView=false,raf=0,last=0,pointer=null,failed=false;
 const frameState={ready:false,frames:0,visible:false,paused,staticView:false,pairs:{},annotations:{},unavailable:[]};
 figure.teaserState=frameState;
 const svgNS='http://www.w3.org/2000/svg',svg=document.createElementNS(svgNS,'svg');
 svg.setAttribute('viewBox',`0 0 ${FIGURE.width} ${FIGURE.height}`);svg.classList.add('teaser-annotations');svg.setAttribute('aria-hidden','true');
 svg.innerHTML='<defs><marker id="teaser-arrow-tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#3f4144" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>';
 const setControls=()=>{
  play.textContent=paused?'Rotate assets':'Pause rotation';play.setAttribute('aria-pressed',String(!paused));
  toggle.textContent=staticView?'Show 3D assets':'Original figure';toggle.setAttribute('aria-pressed',String(staticView));
  play.disabled=staticView;frameState.paused=paused;frameState.staticView=staticView;
 };
 function requestRender(){if(!raf&&!failed)raf=requestAnimationFrame(render);}
 const raycaster=new THREE.Raycaster(),ndc=new THREE.Vector3();
 function updateLeader(leader) {
  const {view,spec,path}=leader,[left,top,width,height]=view.rect;
  let best=null;
  // Cast through each candidate part's center: the first surface must belong
  // to that part. Hidden parts get no arrow, avoiding a pointer to an occluder.
  for(const target of leader.targets){
   ndc.copy(target.center).project(view.camera);
   raycaster.setFromCamera(new THREE.Vector2(ndc.x,ndc.y),view.camera);
   const hit=raycaster.intersectObjects(view.meshes,false)[0];
   if(!hit||hit.object!==target.mesh)continue;
   const end=[left+(ndc.x+1)*width/2,top+(1-ndc.y)*height/2];
   const distance=Math.hypot(end[0]-spec.start[0],end[1]-spec.start[1]);
   if(!best||distance<best.distance)best={end,distance,part:target.mesh.name};
  }
  path.style.visibility=best?'visible':'hidden';
  if(!best){frameState.annotations[spec.id]={visible:false};return;}
  const [sx,sy]=spec.start,[ex,ey]=best.end,dx=ex-sx,dy=ey-sy;
  // Keep the label end fixed and bend the short leader away from the text.
  const bend=Math.min(130,Math.hypot(dx,dy)*.2),cx=(sx+ex)/2-dy/(best.distance||1)*bend,cy=(sy+ey)/2+dx/(best.distance||1)*bend;
  path.setAttribute('d',`M ${sx} ${sy} Q ${cx} ${cy} ${ex} ${ey}`);
  frameState.annotations[spec.id]={visible:true,end:best.end,part:best.part};
 }
 function render(time=performance.now()) {
  raf=0;
  if(failed||!frameState.ready||!visible||document.hidden||staticView)return;
  // Cap automatic animation near 30 fps: twelve models share one WebGL context.
  if(last&&!paused&&!pointer&&time-last<30){raf=requestAnimationFrame(render);return;}
  const delta=Math.min((time-(last||time))/1000,.1);last=time;
  if(!paused&&!pointer)pairs.forEach(pair=>pair.angle+=delta*.30);
  const bounds=stage.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);
  if(renderer.getPixelRatio()!==ratio)renderer.setPixelRatio(ratio);
  const size=renderer.getSize(new THREE.Vector2());
  if(size.x!==Math.round(bounds.width)||size.y!==Math.round(bounds.height))renderer.setSize(Math.round(bounds.width),Math.round(bounds.height),false);
  renderer.setScissorTest(false);renderer.clear();renderer.setScissorTest(true);
  for(const view of views){
   const [left,top,width,height]=view.rect,w=width/FIGURE.width*bounds.width,h=height/FIGURE.height*bounds.height;
   const x=left/FIGURE.width*bounds.width,y=bounds.height-(top+height)/FIGURE.height*bounds.height;
   const aspect=w/h,extent=Math.max(view.verticalHalf,view.horizontalHalf/aspect)*1.04,angle=view.pair.angle;
   const camera=view.camera;camera.left=-extent*aspect;camera.right=extent*aspect;camera.top=extent;camera.bottom=-extent;
   camera.position.set(5*Math.sin(angle)*Math.cos(pitch),5*Math.sin(pitch),5*Math.cos(angle)*Math.cos(pitch));camera.lookAt(0,0,0);camera.updateProjectionMatrix();camera.updateMatrixWorld();
   renderer.setViewport(x,y,w,h);renderer.setScissor(x,y,w,h);renderer.render(view.scene,camera);
  }
  renderer.setScissorTest(false);
  leaders.forEach(updateLeader);
  for(const pair of pairs)frameState.pairs[pair.id]={angle:pair.angle,views:pair.views.length};
  frameState.frames++;
  if(!paused&&!pointer)raf=requestAnimationFrame(render);
 }
 function fail(error){
  failed=true;cancelAnimationFrame(raf);image.src=original;image.alt=originalAlt;figure.classList.add('is-static');slots.forEach(s=>s.hidden=true);
  play.disabled=true;toggle.disabled=true;status.textContent='Original figure · 3D unavailable';frameState.ready=false;console.error('Animated teaser:',error);
 }
 try {
  renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});
  renderer.setClearColor(0,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;renderer.autoClear=false;renderer.domElement.setAttribute('aria-hidden','true');
  const generator=new THREE.PMREMGenerator(renderer),room=new RoomEnvironment();environment=generator.fromScene(room,.04);room.dispose();generator.dispose();
  const loader=new GLTFLoader();
  const loaded=(await Promise.all(pairs.map(async pair=>{
   try{return await Promise.all((pair.roles||['input','output']).map(async role=>({pair,role,gltf:await loader.loadAsync(new URL(`${pair.id}-${role}.glb`,import.meta.url).href)})));}
   catch(error){
    frameState.unavailable.push(pair.id);
    const [x,y,w,h]=pair.panel,patch=document.createElement('div'),snapshot=document.createElement('img');
    patch.className='teaser-fallback';patch.style.cssText=`left:${x/FIGURE.width*100}%;top:${y/FIGURE.height*100}%;width:${w/FIGURE.width*100}%;height:${h/FIGURE.height*100}%`;
    snapshot.src=original;snapshot.alt=`${pair.label}, original paper figure`;
    snapshot.style.cssText=`width:${FIGURE.width/w*100}%;left:${-x/w*100}%;top:${-y/h*100}%`;
    if(pair.id==='cactus'){snapshot.src=new URL('./cactus-poster.png',import.meta.url).href;snapshot.alt='Image-conditioned cactus input and SpaceFlow result';snapshot.style.cssText='width:100%;height:100%;left:0;top:0';}
    patch.append(snapshot);fallbacks.push(patch);return [];
   }
  }))).flat();
  if(!loaded.length)throw new Error('No teaser models could be loaded');
  for(const {pair,role,gltf} of loaded){
   const scene=new THREE.Scene();scene.environment=environment.texture;scene.add(new THREE.HemisphereLight(0xffffff,0xa7a4a0,.8));
   const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(-3,5,4);scene.add(light);
   const object=gltf.scene;
   if(role==='input')object.traverse(mesh=>{
    const colors=mesh.geometry?.getAttribute('color');if(!colors)return;
    // The input exporter writes the UI's sRGB swatches into COLOR_0.
    const linear=new Float32Array(colors.count*3),color=new THREE.Color();
    for(let i=0;i<colors.count;i++){color.setRGB(colors.getX(i),colors.getY(i),colors.getZ(i),THREE.SRGBColorSpace);color.toArray(linear,i*3);}
    mesh.geometry.setAttribute('color',new THREE.BufferAttribute(linear,3));
   });
   object.rotation.x=-Math.PI/2;object.updateMatrixWorld(true);
   const bounds=new THREE.Box3().setFromObject(object);object.position.sub(bounds.getCenter(new THREE.Vector3()));scene.add(object);object.updateMatrixWorld(true);
   let horizontalHalf=0,verticalHalf=0;const vertex=new THREE.Vector3(),meshes=[];
   object.traverse(mesh=>{
    if(!mesh.isMesh)return;meshes.push(mesh);const positions=mesh.geometry.getAttribute('position');
    for(let i=0;i<positions.count;i++){
     vertex.fromBufferAttribute(positions,i).applyMatrix4(mesh.matrixWorld);const radius=Math.hypot(vertex.x,vertex.z);
     horizontalHalf=Math.max(horizontalHalf,radius);verticalHalf=Math.max(verticalHalf,Math.abs(vertex.y)*Math.cos(pitch)+radius*Math.sin(pitch));
    }
   });
   const rect=pair.slots[role==='input'?0:1],camera=new THREE.OrthographicCamera(-1,1,1,-1,.01,50);
   const view={pair,role,rect,scene,camera,meshes,horizontalHalf,verticalHalf};views.push(view);pair.views.push(view);
   const slot=document.createElement('div');slot.className=`teaser-slot teaser-${pair.id}-${role}`;slot.tabIndex=0;slot.setAttribute('role','img');slot.setAttribute('aria-label',`${pair.label}, ${role==='input'?'conditioning geometry':'SpaceFlow output'}. Drag or use arrow keys to rotate this pair. Press R to reset.`);
   slot.style.cssText=`left:${rect[0]/FIGURE.width*100}%;top:${rect[1]/FIGURE.height*100}%;width:${rect[2]/FIGURE.width*100}%;height:${rect[3]/FIGURE.height*100}%`;slots.push(slot);view.slot=slot;
   if(role==='input')for(const spec of pair.annotations){
    const path=document.createElementNS(svgNS,'path');path.classList.add('teaser-leader');path.dataset.annotation=spec.id;path.setAttribute('marker-end','url(#teaser-arrow-tip)');svg.append(path);
    const targets=spec.parts.map(id=>{const mesh=meshes.find(m=>m.name===`superquadric_${id}`);if(!mesh)throw new Error(`Missing ${pair.id} part ${id}`);return {mesh,center:new THREE.Box3().setFromObject(mesh).getCenter(new THREE.Vector3())};});
    leaders.push({view,spec,path,targets});
   }
  }
  const plate=new Image();plate.src=background;await plate.decode();
  // Native figure panel: replace the unavailable lamp with the supplied cactus example.
  const replacement=document.createElement('div');replacement.className='teaser-replacement';
  replacement.style.cssText=`left:${7984/FIGURE.width*100}%;top:${3815/FIGURE.height*100}%;width:${3558/FIGURE.width*100}%;height:${2378/FIGURE.height*100}%`;
  replacement.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3558 2378" width="100%" height="100%" aria-hidden="true"><rect width="3558" height="2378" rx="25" fill="#f4f9f9"/><image href="${new URL('./cactus-tree.png',import.meta.url).href}" x="1600" y="50" width="470" height="520" preserveAspectRatio="xMidYMid meet"/><image href="${new URL('./cactus-pot.jpg',import.meta.url).href}" x="30" y="1070" width="430" height="590" preserveAspectRatio="xMidYMid meet"/><path d="M1770 1300 H2200 m-70 -35 70 35 -70 35" fill="none" stroke="#34383b" stroke-width="15"/><text x="1779" y="2210" text-anchor="middle" font-family="Arial,sans-serif" font-size="225" fill="#141719">“a potted cactus”</text></svg>`;
  stage.append(replacement,...fallbacks,renderer.domElement,svg,...slots);image.src=background;image.alt=animatedAlt;frameState.ready=true;
  status.textContent=frameState.unavailable.length?'Drag to rotate · some examples shown as original figures':'Drag either asset to rotate its input/output pair';play.disabled=false;toggle.disabled=false;setControls();
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();fail(new Error('WebGL context lost'));});
  const visibility=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;frameState.visible=visible;last=0;if(visible)requestRender();else{cancelAnimationFrame(raf);raf=0;}},{threshold:.05});visibility.observe(stage);
  new ResizeObserver(requestRender).observe(stage);
  document.addEventListener('visibilitychange',()=>{last=0;if(document.hidden){cancelAnimationFrame(raf);raf=0;}else requestRender();});
  play.addEventListener('click',()=>{paused=!paused;setControls();last=0;requestRender();});
  toggle.addEventListener('click',()=>{staticView=!staticView;figure.classList.toggle('is-static',staticView);image.src=staticView?original:background;image.alt=staticView?originalAlt:animatedAlt;setControls();last=0;requestRender();});
  motion.addEventListener('change',event=>{if(event.matches){paused=true;setControls();requestRender();}});
  for(const view of views){const {slot,pair}=view;
   slot.addEventListener('pointerdown',event=>{if(event.pointerType==='mouse'&&event.button!==0)return;pointer={id:event.pointerId,x:event.clientX,angle:pair.angle,pair};slot.setPointerCapture(event.pointerId);});
   slot.addEventListener('pointermove',event=>{if(pointer?.id!==event.pointerId)return;pointer.pair.angle=pointer.angle+(event.clientX-pointer.x)*.018;requestRender();});
   const end=event=>{if(pointer?.id===event.pointerId){pointer=null;last=0;requestRender();}};
   slot.addEventListener('pointerup',end);slot.addEventListener('pointercancel',end);slot.addEventListener('lostpointercapture',end);
   slot.addEventListener('keydown',event=>{if(event.key==='ArrowLeft')pair.angle-=.15;else if(event.key==='ArrowRight')pair.angle+=.15;else if(event.key.toLowerCase()==='r')pair.angle=EXAMPLES.find(e=>e.id===pair.id).angle;else return;event.preventDefault();requestRender();});
  }
 }catch(error){fail(error);environment?.dispose();renderer?.dispose();}
}
