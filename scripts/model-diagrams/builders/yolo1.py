"""Draw the bundled Darknet cfg graphs for YOLO1; reused by sibling cfg families.

Each cfg layer is visible, with index-preserving routes and expanded primitive
blocks. Layout uses source-layer columns and labeled tensor continuations only
where a tensor crosses columns; it does not invent a shared tiny/base topology.
"""
from pathlib import Path
import argparse, os, sys, json, subprocess, importlib.util, types, math
WEBSITE=Path(__file__).resolve().parents[3]
FAMILIES={
 'yolo1':('yolov1','YOLOv1',{'t':('yolov1-tiny',448),'b':('yolov1',448)},20),
 'yolo2':('yolov2','YOLOv2',{'t':('yolov2-tiny',416),'b':('yolov2',608)},80),
 'yolo3':('yolov3','YOLOv3',{'t':('yolov3-tiny',416),'b':('yolov3',416),'spp':('yolov3-spp',608)},80),
 'yolo4':('yolov4','YOLOv4',{'t':('yolov4-tiny',416),'b':('yolov4',608)},80),
}
def load_source(src):
 package=types.ModuleType('diagram_darknet');package.__path__=[str(src/'libreyolo/models/darknet')];sys.modules['diagram_darknet']=package
 loaded={}
 for name in ['cfg','blocks','net']:
  spec=importlib.util.spec_from_file_location('diagram_darknet.'+name,src/f'libreyolo/models/darknet/{name}.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);loaded[name]=mod
 return loaded

def main(family='yolo1'):
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve()
 scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 import torch
 torch.set_num_threads(4);mods=load_source(src);slug,title,variants,nc=FAMILIES[family]
 rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit']
 out=WEBSITE/f'public/diagrams/models/{slug}';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/f'scripts/model-diagrams/evidence/{family}.json'
 evidence=json.loads(ep.read_text()) if args.diagram_only else dict(family=family,source_revision=rev,source_files=[f'libreyolo/models/{family}/model.py','libreyolo/models/darknet/net.py','libreyolo/models/darknet/blocks.py','libreyolo/postprocess/darknet_yolo.py'],variants={})
 views=[]
 for size,(cfgname,H) in variants.items():
  netopts,cfg=mods['cfg'].load_bundled_cfg(cfgname);netopts.update(width=str(H),height=str(H))
  mode='meta' if family=='yolo1' and size=='b' else 'cpu'
  if args.diagram_only:mode=evidence['variants'][size]['verification']
  with torch.device('meta' if args.diagram_only else mode):model=mods['net'].DarknetNet(netopts,cfg,num_classes=nc).eval()
  if not args.diagram_only:
   hooks={}
   def hook(idx):
    def record(mod,ins,outs):hooks[str(idx)]={'input':list(ins[0].shape),'output':list(outs.shape)}
    return record
   for i,mod in enumerate(model.layers):mod.register_forward_hook(hook(i))
   with torch.inference_mode():preds=model(torch.zeros(1,3,H,H,device=mode))
   actual=[list(p.shape) for p in preds]
  else:hooks=evidence['variants'][size]['module_shapes'];actual=evidence['variants'][size]['raw_outputs']
  # Module outputs are observed. Route/shortcut shapes are derived from their
  # actual input tensors and checked against the next executed operation.
  shapes={-1:[1,3,H,H]};sources={};labels={};kind={};blocks={}
  for i,(layer,mod,meta) in enumerate(zip(cfg,model.layers,model._layer_meta)):
   t=layer.type;sources[i]=meta['indices'] if t=='route' else ([i-1,meta['from']] if t=='shortcut' else [i-1])
   if str(i) in hooks:shapes[i]=hooks[str(i)]['output']
   elif t=='route':
    sh=shapes[sources[i][0]].copy();sh[1]=sum(shapes[j][1] for j in sources[i])//meta['groups'];shapes[i]=sh
    assert all(shapes[j][2:]==shapes[sources[i][0]][2:] for j in sources[i])
   elif t=='shortcut':
    assert shapes[i-1]==shapes[meta['from']];shapes[i]=shapes[i-1].copy()
   else:shapes[i]=shapes[i-1].copy()
   block='';k='plain'
   if t=='convolutional':
    act=str(layer.get('activation','linear'));bn=bool(layer.get_int('batch_normalize',0));l=f'Conv {layer.get_int("size",1)}×{layer.get_int("size",1)}, s={layer.get_int("stride",1)}'
    l+=' + norm' if bn else '';l+=' + '+act if act!='linear' else '';k='conv';block='DarknetConv'
   elif t=='route':l=('Concat' if len(sources[i])>1 else 'Route')+(f'; split {meta["groups"]}, take group {meta["group_id"]}' if meta['groups']>1 else '');k='concat' if len(sources[i])>1 else 'split'
   elif t=='shortcut':l='Add residual';k='plain'
   elif t=='maxpool':l=f'MaxPool2d {layer.get_int("size",2)}×{layer.get_int("size",2)}, s={layer.get_int("stride",2)}';k='pool';block='DarknetMaxPool'
   elif t=='upsample':l=f'Nearest upsample ×{layer.get_int("stride",2)}';k='norm'
   elif t=='reorg':l=f'Reorg, stride {layer.get_int("stride",2)}';k='split';block='Reorg'
   elif t=='local':l=f'Local {layer.get_int("size",1)}×{layer.get_int("size",1)}, '+str(layer.get('activation','linear'));k='linear';block='DarknetLocal'
   elif t=='connected':l='Flatten + Linear';k='linear';block='DarknetConnected'
   elif t=='dropout':l='Dropout (identity in eval)'
   else:l='Raw '+t+' head';k='activation'
   labels[i]=l;kind[i]=k;blocks[i]=block
  observed_heads=[shapes[x.layer_index] for x in model.detections];assert observed_heads==actual,(observed_heads,actual)
  cols=math.ceil(len(cfg)/28);rows=math.ceil(len(cfg)/cols);PW=650;graphh=120+rows*98;defcount=2+int(family=='yolo1')+int(any(x.type=='reorg' for x in cfg))+int(family=='yolo1' and size=='b')+int(any(x.get('activation')=='mish' for x in cfg));defh=defcount*415
  mainh=max(graphh,defh);W=(cols+1)*PW+80;W=max(W,1380);top=240;decodey=top+mainh+35;HEIGHT=decodey+620
  d=Diagram(f'{title}-{size.upper()}',f'Detection; {H} × {H} RGB; {nc} classes; batch 1; unfused eval. Every cfg layer is shown.',width=W,height=HEIGHT,revision=rev,source_label=f'darknet/cfgs/{cfgname}.cfg; darknet/net.py; darknet/blocks.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/darknet/cfgs/{cfgname}.cfg',logo=WEBSITE/'public/icon-128.png')
  d.text(50,218,'L = cfg layer index. Matching F labels continue the same tensor across columns. Full routes remain visible.',14)
  panels=[];coords={};cross=set()
  for col in range(cols):
   lo=col*rows;hi=min(len(cfg),(col+1)*rows)
   pan=d.panel('layers'+str(col),f'Layers {lo} to {hi-1}',40+col*PW,top,PW-24,graphh)
   panels.append(pan)
   for i in range(lo,hi):
    y=80+(i-lo)*98;coords[i]=(col,y)
    detail=' × '.join(map(str,shapes[i][1:]))
    pan.box('L'+str(i),140,y,370,'L'+str(i)+' '+labels[i],detail=detail,kind=kind[i],block_type=blocks[i],h=68,font_size=15,description=f'{cfgname}.cfg layer {i}: '+labels[i]+'; inputs '+str(sources[i])+'; output '+str(shapes[i]))
   if col==0:pan.text(140,59,f'Input: 3 × {H} × {H}',15);pan.wire([(325,63),(325,80)],start='input',end='L0')
  # Local branches use interval-colored right corridors. Cross-column edges
  # use explicit labels at both ends to avoid a wide web over unrelated nodes.
  for col,pan in enumerate(panels):
   intervals=[]
   for dest,srcs in sources.items():
    if coords[dest][0]!=col:continue
    for j,origin in enumerate(srcs):
     if origin<0:continue
     if coords[origin][0]==col and origin!=dest-1:intervals.append((origin,dest,j))
   lanes=[];assigned={}
   for origin,dest,j in sorted(intervals,key=lambda x:(x[0],x[1])):
    lane=next((k for k,last in enumerate(lanes) if last<origin),len(lanes))
    if lane==len(lanes):lanes.append(dest)
    else:lanes[lane]=dest
    assigned[origin,dest,j]=lane
   assert len(lanes)<=4,(family,size,col,lanes)
   for dest,srcs in sources.items():
    if coords[dest][0]!=col:continue
    dy=coords[dest][1];nonseq=[o for o in srcs if o!=dest-1];nonseqposition=0
    for j,origin in enumerate(srcs):
     if origin<0:continue
     if coords[origin][0]!=col:
      off=16+20*nonseqposition;nonseqposition+=1
      pan.text(16,dy+off+5,f'F{origin}',13,weight=600);pan.wire([(82,dy+off),(140,dy+off)],start='L'+str(origin),end='L'+str(dest));cross.add(origin)
     elif origin==dest-1:
      pan.connect('L'+str(origin),'L'+str(dest))
     else:
      sy=coords[origin][1];lane=assigned[origin,dest,j];lx=534+lane*24;off=16+20*nonseqposition;nonseqposition+=1
      pan.wire([(510,sy+34),(lx,sy+34),(lx,dy+off),(510,dy+off)],start='L'+str(origin),end='L'+str(dest))
  for origin in cross:
   col,y=coords[origin];pan=panels[col]
   # Continuation tag is placed under the source box, off the sequential stem.
   pan.wire([(140,y+34),(102,y+34)],start='L'+str(origin),end='continuation'+str(origin));pan.text(20,y+39,f'F{origin}',13,weight=600)
  # Visible primitive definitions occupy the right column.
  DX=40+cols*PW;DY=top
  def panel(id,label,kind='conv'):
   nonlocal DY
   p=d.panel(id,label,DX,DY,PW-24,390,kind=kind,dashed=True,block_type=id);DY+=415;return p
  def b(p,id,x,y,w,l,dt='',k='plain',h=44):return p.box(id,x,y,w,l,detail=dt,kind=k,h=h,font_size=14,description=l+': '+dt)
  def seq(p,ids):
   for a,z in zip(ids,ids[1:]):p.connect(a,z)
  p=panel('DarknetConv','Convolution block')
  b(p,'cv',155,65,340,'Conv2d','k, stride and channels from layer label','conv2d');b(p,'bn',155,145,340,'Darknet normalization','Only layers labeled + norm','norm');b(p,'act',155,225,340,'Activation','leaky: LeakyReLU(0.1); linear: identity','activation');seq(p,['cv','bn','act']);p.text(20,330,'Norm layers: bias=False. Without norm: bias=True.',15);p.text(20,360,'Mish uses its separate definition when present.',15)
  p=panel('DarknetMaxPool','Normalization and pooling','pool')
  for i,(id,l,dt,k) in enumerate([('sub','Subtract running mean','x - mean','plain'),('divide','Divide by standard deviation','sqrt(running variance) + 0.000001','norm'),('affine','Multiply scale, add bias','Learned channel-wise affine transform','linear')]):b(p,id,110,60+i*73,405,l,dt,k)
  seq(p,['sub','divide','affine']);p.text(20,310,'MaxPool2d: p = floor((k - 1) / 2).',15);p.text(20,340,'k=2, s=1: pad right/bottom with negative infinity.',14);p.text(20,367,'That pool preserves its input spatial size.',14)
  if family=='yolo1':
   p=panel('DarknetConnected','Fully connected head','bottleneck');fc=next(x for x in model.layers if isinstance(x,mods['blocks'].DarknetConnected))
   for i,(id,l,dt,k) in enumerate([('flat','Flatten C, H, W',f'{fc.in_features:,} features; channel-major','plain'),('fc','Linear',f'{fc.in_features:,} inputs; {fc.out_features:,} outputs; bias','linear'),('fcact','Activation','Identity for the final prediction vector','activation')]):b(p,id,105,65+i*90,410,l,dt,k)
   seq(p,['flat','fc','fcact'])
  if family=='yolo1' and size=='b':
   p=panel('DarknetLocal','Locally connected layer','bottleneck');loc=next(x for x in model.layers if isinstance(x,mods['blocks'].DarknetLocal));locations=loc.out_h*loc.out_w;patch=loc.in_channels*loc.size*loc.size
   for i,(id,l,dt,k) in enumerate([('unfold','Unfold image patches',f'kernel {loc.size}; padding {loc.pad}; stride {loc.stride}','plain'),('matmul','Per-location matrix multiply',f'{locations} separate banks; {loc.out_channels} × {patch} each','linear'),('bias','Add per-location bias',f'{loc.out_channels} × {locations} biases','linear'),('localact','Reshape + LeakyReLU(0.1)',f'{loc.out_channels} × {loc.out_h} × {loc.out_w}','activation')]):b(p,id,95,58+i*76,430,l,dt,k)
   seq(p,['unfold','matmul','bias','localact'])
  if any(x.type=='reorg' for x in cfg):
   p=panel('Reorg','Reorg (Darknet channel ordering)','aggregate');ri=next(i for i,x in enumerate(cfg) if x.type=='reorg');_,c,hh,ww=shapes[ri-1];ss=cfg[ri].get_int('stride',2)
   ops=[('rv1','Reshape + transpose axes 3, 4',f'1 × {c} × {hh//ss} × {ss} × {ww//ss} × {ss}'),('rv2','Reshape + transpose axes 2, 3',f'1 × {c} × {(hh//ss)*(ww//ss)} × {ss*ss}'),('rv3','Reshape + transpose axes 1, 2',f'1 × {c} × {ss*ss} × {hh//ss} × {ww//ss}'),('rv4','Reshape contiguous result',' × '.join(map(str,shapes[ri])))]
   for i,(id,l,dt) in enumerate(ops):b(p,id,70,55+i*78,480,l,dt,'split')
   seq(p,[x[0] for x in ops])
  if any(x.get('activation')=='mish' for x in cfg):
   p=panel('Mish','Mish activation','conv');b(p,'mi',180,60,300,'Input');b(p,'sp',180,140,300,'Softplus',k='activation');b(p,'tanh',180,220,300,'Tanh',k='activation');b(p,'mul',180,310,300,'Elementwise multiply','Input × tanh(softplus(input))','activation');seq(p,['mi','sp','tanh','mul']);p.connect('mi','mul',from_port='left',to_port='left',via=[(60,82),(60,332)])
  # Decode is a separate, fully visible graph after raw neural-network output.
  p=d.panel('decode','Detection decoding (postprocessing)',40,decodey,W-80,510,kind='plain')
  rawtext='; '.join(' × '.join(map(str,shapes[sp.layer_index][1:])) for sp in model.detections)
  p.text(25,62,'Raw head outputs: '+rawtext,16)
  small=family=='yolo1';rw=(W-160)/3;xs=[20+i*(rw+20) for i in range(3)];mid=(W-80)/2
  b(p,'raw',mid-260,95,520,'Raw prediction tensor','Field selection is explicit in the three branches','split',55)
  if small:
   specs=[('xy','Select xy; add grid; divide by 7','Centers scaled to 448 pixels','plain'),('wh','Select wh; square dimensions','Scale dimensions to 448 pixels','activation'),('score','Select confidence and class values','Multiply; 20 scores for each of 98 boxes','linear')]
  else:
   specs=[('xy','Select xy; sigmoid center offsets','Grid + offsets; multiply by per-head stride','activation'),('wh','Select wh; exponentiate logits','Multiply by anchor widths and heights','activation'),('score','Select objectness and class logits','Sigmoid objectness × '+('softmax classes' if family=='yolo2' else 'sigmoid classes'),'linear')]
  for j,(id,l,dt,k) in enumerate(specs):b(p,id,xs[j],215,rw,l,dt,k,55)
  p.wire([(mid-90,150),(mid-90,175),(xs[0]+rw/2,175),(xs[0]+rw/2,215)],start='raw',end='xy')
  p.wire([(mid,150),(mid,215)],start='raw',end='wh')
  p.wire([(mid+90,150),(mid+90,190),(xs[2]+rw/2,190),(xs[2]+rw/2,215)],start='raw',end='score')
  b(p,'boxjoin',xs[0]+rw/2,355,rw,'Stack cx, cy, width, height','Convert to corner coordinates','concat',55)
  p.wire([(xs[0]+rw/2,270),(xs[0]+rw/2,305),(xs[0]+rw/2+60,305),(xs[0]+rw/2+60,355)],start='xy',end='boxjoin')
  p.wire([(xs[1]+rw/2,270),(xs[1]+rw/2,330),(xs[0]+rw*1.5-60,330),(xs[0]+rw*1.5-60,355)],start='wh',end='boxjoin')
  b(p,'filter',xs[2],355,rw,'Confidence filter + class NMS','Then invert resize to the original image','pool',55)
  p.connect('score','filter');p.connect('boxjoin','filter',from_port='right',to_port='left')
  if not small:
   p.text(25,422,'Anchors by raw head (width, height): '+ '; '.join(str(sp.anchors) for sp in model.detections),14)
   p.text(25,449,'Strides: '+', '.join(str(sp.stride) for sp in model.detections)+'. '+('Region anchors are in grid cells, so width/height also multiply by stride.' if family=='yolo2' else 'YOLO anchors are in input pixels.'),14)
   p.text(25,475,'Center scale_x_y per head: '+', '.join(str(sp.scale_x_y) for sp in model.detections)+'. Offset = sigmoid(raw) × scale - (scale - 1)/2.',14)
  else:p.text(25,432,'VOC 20 classes. No anchors. Shared class values per 7 × 7 cell; 2 boxes per cell. Stretch preprocessing.',16)
  path=out/f'{size}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL)
  routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True))
  evidence['variants'][size]=dict(cfg=f'libreyolo/models/darknet/cfgs/{cfgname}.cfg',input=[1,3,H,H],classes=nc,verification=mode,module_shapes=hooks,all_layer_shapes=shapes,raw_outputs=actual,routes=routes,limitations='Meta-only forward for large local/FC architecture; no numerical inference' if mode=='meta' else 'Random-weight CPU inference; no pretrained accuracy claim')
  views.append(dict(id=size,label=title+' '+size.upper(),task='detect',size=size,kind='concrete',svg=f'/diagrams/models/{slug}/{size}.svg',html=f'/diagrams/models/{slug}/{size}.html',input=f'1 × 3 × {H} × {H}',verification=mode))
  print(f'{family} {size}: {len(cfg)} layers, {mode}, route findings {routes["total_findings"]}',flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family=family,slug=slug,title=title,source_revision=rev,default_view='b',views=views),indent=2)+'\n')
 evidence.update(weights='No external weights or downloads',family_view='Tiny, base and SPP cfgs have different topology; no multiplier-only family view is asserted.',visual='Parent performs browser/PNG QA',reproduce=f'python scripts/model-diagrams/builders/{family}.py --source /path/to/libreyolo')
 ep.write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
