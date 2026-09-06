"""Concrete YOLOX n/t/s/m/l/x and regular-convolution family diagram."""
from pathlib import Path
import argparse,os,sys,json,subprocess
from yolo9 import box,seq,fanin
WEBSITE=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.yolox.nn import LibreYOLOXModel
 import torch
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/yolox';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/yolox.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];evidence=json.loads(ep.read_text()) if args.diagram_only else dict(family='yolox',source_revision=rev,source_files=['libreyolo/models/yolox/nn.py','libreyolo/postprocess/yolox.py'],shapes={});views=[]
 cfgs=LibreYOLOXModel.CONFIGS
 for size in [*cfgs,'family-regular']:
  symbolic=size.startswith('family');sz='s' if symbolic else size;cfg=cfgs[sz];H=416 if sz in 'nt' else 640;C=int(64*cfg['width']);D=max(round(3*cfg['depth']),1);depthwise=cfg['depthwise'];unit='DWConv' if depthwise else 'BaseConv';N=sum((H//s)**2 for s in [8,16,32]);hC=4*C
  if not symbolic and not args.diagram_only:
   model=LibreYOLOXModel(sz,nb_classes=80).eval();obs={}
   def hook(name):
    def record(mod,ins,output):
     def shape(v):
      if isinstance(v,torch.Tensor):return list(v.shape)
      if isinstance(v,(tuple,list)):return [shape(x) for x in v]
      if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
      return None
     obs[name]={'input':shape(ins),'output':shape(output)}
    return record
   for name,mod in model.named_modules():
    if name and (name.count('.')<=2 or name.startswith(('head.','backbone.backbone.dark'))):mod.register_forward_hook(hook(name))
   with torch.inference_mode():pred=model(torch.zeros(1,3,H,H))
   assert [list(x.shape) for x in pred]==[[1,85,H//s,H//s] for s in [8,16,32]];evidence['shapes'][size]=obs
  d=Diagram('YOLOX regular family' if symbolic else 'YOLOX-'+sz.upper(),('Detection; symbolic C and D values below; variable input H.' if symbolic else f'Detection; {H} × {H} RGB; 80 classes; batch 1.')+' Unfused PyTorch eval; raw box offsets and sigmoid scores.',width=2340,height=(3350 if depthwise else 2750)+(470 if symbolic else 0),revision=rev,source_label='models/yolox/nn.py; postprocess/yolox.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/yolox/nn.py',logo=WEBSITE/'public/icon-128.png')
  b=d.panel('backbone','CSPDarknet backbone',40,230,440,1100);n=d.panel('neck','YOLOPAFPN',510,230,780,1100);h=d.panel('head','YOLOXHead (three independent scales)',1320,230,980,1100)
  def c(k):return (('C' if k==1 else str(k)+'C') if symbolic else str(k*C))
  def sh(k,div):return c(k)+' × '+(f'H/{div} × H/{div}' if symbolic else f'{H//div} × {H//div}')
  stage=[('image','Input',('3 × H × H' if symbolic else f'3 × {H} × {H}'),'plain',''),('focus','Focus',sh(1,2),'aggregate','Focus'),('d2',unit+' 3×3, s=2',sh(2,4),'conv',unit),('csp2','CSPLayer, residual enabled',sh(2,4)+('; n=D' if symbolic else f'; n={D}'),'aggregate','CSPLayer'),('d3',unit+' 3×3, s=2',sh(4,8),'conv',unit),('B3','CSPLayer (B3)',sh(4,8)+('; n=3D' if symbolic else f'; n={3*D}'),'aggregate','CSPLayer'),('d4',unit+' 3×3, s=2',sh(8,16),'conv',unit),('B4','CSPLayer (B4)',sh(8,16)+('; n=3D' if symbolic else f'; n={3*D}'),'aggregate','CSPLayer'),('d5',unit+' 3×3, s=2',sh(16,32),'conv',unit),('spp','SPPBottleneck',sh(16,32),'spp','SPPBottleneck'),('B5','CSPLayer (B5), no residual',sh(16,32)+('; n=D' if symbolic else f'; n={D}'),'aggregate','CSPLayer')]
  for i,(id,l,dt,k,bl) in enumerate(stage):box(b,id,50,70+i*86,340,l,dt,k,bl)
  seq(b,[x[0] for x in stage]);b.text(20,1045,'B3, B4 and B5 continue to the neck.',14)
  td=[('lat','BaseConv 1×1',8,32,'conv','BaseConv'),('up4','Nearest upsample ×2',8,16,'norm',''),('cat4','Concat with B4',16,16,'concat',''),('N4','CSPLayer (no residual)',8,16,'aggregate','CSPLayer'),('red','BaseConv 1×1',4,16,'conv','BaseConv'),('up3','Nearest upsample ×2',4,8,'norm',''),('cat3','Concat with B3',8,8,'concat',''),('P3','CSPLayer (P3)',4,8,'aggregate','CSPLayer')]
  bu=[('down4',unit+' 3×3, s=2',4,16,'conv',unit),('catr','Concat with red',8,16,'concat',''),('P4','CSPLayer (P4)',8,16,'aggregate','CSPLayer'),('down5',unit+' 3×3, s=2',8,32,'conv',unit),('catl','Concat with lat',16,32,'concat',''),('P5','CSPLayer (P5)',16,32,'aggregate','CSPLayer')]
  for x,stages in [(65,td),(475,bu)]:
   for i,(id,l,mul,div,k,bl) in enumerate(stages):box(n,id,x,90+i*94,275,l,sh(mul,div)+(('; n=D' if symbolic else f'; n={D}') if bl=='CSPLayer' else ''),k,bl)
   seq(n,[v[0] for v in stages])
  sy=n.port('P3','right')[1];n.wire([(340,sy),(397,sy),(397,58),(612.5,58),(612.5,90)],start='P3',end='down4')
  for dest,source in [('lat','B5'),('cat4','B4'),('cat3','B3'),('catr','red'),('catl','lat')]:
   px,py=n.port(dest,'left');n.text(px-48,py-11,source,12,weight=600);n.wire([(px-28,py),(px,py)],start=source,end=dest)
  for id in ['lat','red']:
   px,py=n.port(id,'right');n.wire([(px,py),(px+30,py)],start=id,end='tensor-'+id);n.text(px+4,py-10,id,12)
  n.text(25,1020,'Neck bottlenecks do not add residuals.',15);n.text(25,1050,'Matching B/lat/red labels identify tensor continuations.',14)
  # One explicitly drawn repeated unit, instantiated at each listed scale.
  h.text(25,63,'Execute this graph separately for P3, P4 and P5.',15);box(h,'hin',350,90,280,'One scale feature','P3 / P4 / P5 dimensions below');box(h,'stem',350,170,280,'BaseConv 1×1',c(4)+' output channels','conv','BaseConv');h.connect('hin','stem')
  for prefix,x in [('cls',40),('reg',590)]:
   for j in range(2):box(h,prefix+str(j),x,280+j*82,330,unit+' 3×3',c(4)+' channels','conv',unit)
   h.connect('stem',prefix+'0',via=[(490,240),(x+165,240)]);h.connect(prefix+'0',prefix+'1')
  box(h,'clsout',40,455,330,'Conv2d 1×1','80 logits; bias=True','conv2d');box(h,'clssig',40,535,330,'Sigmoid','80 class probabilities','activation');seq(h,['cls1','clsout','clssig'])
  box(h,'boxout',545,455,175,'Conv2d 1×1','4 box offsets','conv2d');box(h,'objout',760,455,175,'Conv2d 1×1','1 objectness logit','conv2d');box(h,'objsig',760,535,175,'Sigmoid','1 probability','activation');h.wire([(755,408),(755,430),(632.5,430),(632.5,455)],start='reg1',end='boxout');h.wire([(755,408),(755,430),(847.5,430),(847.5,455)],start='reg1',end='objout');h.connect('objout','objsig')
  box(h,'hout',180,695,620,'Concat box offsets, objectness, class probabilities','85 channels per location','concat')
  h.wire([(205,581),(205,625),(265,625),(265,695)],start='clssig',end='hout');h.wire([(632.5,501),(632.5,650),(490,650),(490,695)],start='boxout',end='hout');h.wire([(847.5,581),(847.5,674),(715,674),(715,695)],start='objsig',end='hout')
  h.text(25,795,'Scale        feature channels        square grid        raw output channels',15,weight=600)
  for j,(lv,mul) in enumerate([(3,4),(4,8),(5,16)]):
   for xx,txt in zip([25,180,470,740],[f'P{lv}',c(mul),f'H/{2**lv}' if symbolic else str(H//2**lv),'85']):h.text(xx,832+j*36,txt,15)
  h.text(25,980,'Convolutions in different scales have independent weights.',15);h.text(25,1015,'No DFL bins. Objectness and class probabilities are separate.',15)
  definitions=['BaseConv','Focus','CSPLayer','Bottleneck','SPPBottleneck','Decode']+(['DWConv'] if depthwise else [])
  for i,name in enumerate(definitions):
   row=i//3;col=i%3;p=d.panel('def'+name,name,40+col*770,1370+row*585,740,555,kind='aggregate' if name in ['Focus','CSPLayer'] else 'pool' if name in ['SPPBottleneck','Decode'] else 'conv',dashed=True,block_type=name)
   def bb(id,y,l,dt='',kind='plain',x=190,w=380,bl=''):return box(p,name+id,x,y,w,l,dt,kind,bl,h=43)
   def sq(ids):seq(p,[name+x for x in ids])
   if name=='BaseConv':
    for id,y,l,dt,k in [('conv',85,'Conv2d','k, stride, groups, channels from occurrence; no bias','conv2d'),('bn',210,'BatchNorm2d','eps=0.001; momentum=0.03','norm'),('silu',335,'SiLU','x × sigmoid(x)','activation')]:bb(id,y,l,dt,k)
    sq(['conv','bn','silu']);p.text(25,495,'Padding = (k - 1) / 2 for the odd kernels shown.',15)
   elif name=='DWConv':
    bb('dw',100,'BaseConv depthwise 3×3','groups=input channels; spatial stride from occurrence','conv',bl='BaseConv');bb('pw',270,'BaseConv pointwise 1×1','groups=1; target output channels','conv',bl='BaseConv');sq(['dw','pw']);p.text(25,450,'Nano uses this unit for the marked 3×3 paths.',15)
   elif name=='Focus':
    bb('input',60,'Input','3 × H × H' if symbolic else f'3 × {H} × {H}')
    for j,(id,l) in enumerate([('tl','Even row, even col'),('bl','Odd row, even col'),('tr','Even row, odd col'),('br','Odd row, odd col')]):
     xx=20+j*180;box(p,name+id,xx,190,160,l,'3 channels','split',h=43);p.wire([(230+j*95,103),(230+j*95,130+j*12),(xx+80,130+j*12),(xx+80,190)],start=name+'input',end=name+id)
    bb('cat',340,'Concat in TL, BL, TR, BR order','12 × H/2 × H/2' if symbolic else f'12 × {H//2} × {H//2}','concat')
    for j,id in enumerate(['tl','bl','tr','br']):
     xx=100+j*180;port=220+j*100;yy=258+j*24;p.wire([(xx,233),(xx,yy),(port,yy),(port,340)],start=name+id,end=name+'cat')
    bb('conv',450,'BaseConv 3×3',c(1)+' channels','conv',bl='BaseConv');sq(['cat','conv'])
   elif name=='CSPLayer':
    bb('in',60,'Input','Width and repeats from occurrence')
    bb('left',150,'BaseConv 1×1',('Q/2' if symbolic else ' / '.join(str(C*k) for k in [1,2,4,8]))+' hidden channels','conv',x=25,w=320,bl='BaseConv');bb('right',150,'BaseConv 1×1','Same hidden width','conv',x=395,w=320,bl='BaseConv');p.connect(name+'in',name+'left',via=[(380,125),(185,125)]);p.connect(name+'in',name+'right',via=[(380,125),(555,125)])
    bb('bottle',260,'Bottleneck repeated n times','n is printed on each backbone/neck block','bottleneck',x=25,w=320,bl='Bottleneck');p.connect(name+'left',name+'bottle');bb('cat',390,'Concat',('Q channels' if symbolic else ' / '.join(str(C*k) for k in [2,4,8,16])+' channels'),'concat');p.wire([(185,303),(185,345),(260,345),(260,390)],start=name+'bottle',end=name+'cat');p.wire([(555,193),(555,369),(500,369),(500,390)],start=name+'right',end=name+'cat');bb('out',475,'BaseConv 1×1','Output width Q of the occurrence','conv',bl='BaseConv');sq(['cat','out'])
   elif name=='Bottleneck':
    bb('in',60,'Input','Hidden width of parent CSPLayer');bb('c1',160,'BaseConv 1×1','Width unchanged','conv',bl='BaseConv');bb('c2',260,unit+' 3×3','Width unchanged','conv',bl=unit);p.sum(name+'add',380,410);sq(['in','c1','c2']);p.connect(name+'c2',name+'add');p.connect(name+'in',name+'add',from_port='left',via=[(65,81.5),(65,410)],to_port='left');p.text(25,497,'Residual only in dark2, dark3 and dark4; otherwise output conv2.',15)
   elif name=='SPPBottleneck':
    bb('in',55,'BaseConv 1×1',c(8)+' channels','conv',bl='BaseConv')
    for j,k in enumerate([5,9,13]):box(p,name+'p'+str(j),145+j*190,185,175,f'MaxPool2d {k}×{k}',f's=1; p={k//2}','pool',h=43)
    for j in range(3):p.wire([(280+j*90,98),(280+j*90,125+j*12),(232.5+j*190,125+j*12),(232.5+j*190,185)],start=name+'in',end=name+'p'+str(j))
    bb('cat',415,'Concat input and three parallel pools',c(32)+' channels','concat',x=235,w=470);p.wire([(190,76.5),(45,76.5),(45,395),(260,395),(260,415)],start=name+'in',end=name+'cat')
    for j in range(3):
     sx=232.5+j*190;px=390+j*130;yy=300+j*24;p.wire([(sx,228),(sx,yy),(px,yy),(px,415)],start=name+'p'+str(j),end=name+'cat')
    bb('out',490,'BaseConv 1×1',c(16)+' channels','conv',bl='BaseConv',x=235,w=470);sq(['cat','out'])
   elif name=='Decode':
    bb('raw',55,'Flatten and concatenate three scales',('1 × N × 85' if symbolic else f'1 × {N:,} × 85'),'concat')
    bb('xy',175,'Add zero-based grid; multiply stride','x/y offsets have no sigmoid','plain',x=20,w=340);bb('wh',175,'Exp width/height; multiply stride','Strides: 8, 16, 32','activation',x=400,w=320);p.connect(name+'raw',name+'xy',via=[(380,130),(190,130)]);p.connect(name+'raw',name+'wh',via=[(380,130),(560,130)])
    bb('boxes',310,'Join cx, cy, width, height','Convert to corner coordinates','concat');p.wire([(190,218),(190,265),(265,265),(265,310)],start=name+'xy',end=name+'boxes');p.wire([(560,218),(560,289),(495,289),(495,310)],start=name+'wh',end=name+'boxes');bb('nms',435,'Objectness × class scores; filter; NMS','Postprocessing occurs after raw network outputs','pool');sq(['boxes','nms']);p.text(25,520,'No anchor templates or half-cell grid offset.',15)
  # Numeric part/repeat mapping remains explicit even inside shared insets.
  d.text(50,3185 if depthwise else 2585,('C = stem channels, D = base repeats, H = input size, Q = CSPLayer output channels; n = its printed repeat count.' if symbolic else f'Hidden CSP widths: {C}, {2*C}, {4*C}, {8*C}. Backbone repeats: {D}, {3*D}, {3*D}, {D}; neck repeats: {D}.'),15)
  if symbolic:
   p=d.panel('familyvalues','Regular-convolution variant values (Nano has a separate DWConv graph)',40,2630,2260,455)
   columns=[25,210,405,650,895,1130,1390,1670,1990];headers=['Size','Input H','C','B3 / P3','B4 / P4','B5 / P5','D','3D','Head width']
   for x,label in zip(columns,headers):p.text(x,60,label,16,weight=600)
   for i,v in enumerate(['t','s','m','l','x']):
    cf=cfgs[v];cc=int(cf['width']*64);dd=max(round(cf['depth']*3),1);values=[v.upper(),416 if v=='t' else 640,cc,4*cc,8*cc,16*cc,dd,3*dd,4*cc]
    for x,value in zip(columns,values):p.text(x,110+i*52,str(value),16)
   p.text(25,420,'Output grids are H/8, H/16 and H/32. Raw outputs have 85 channels at every scale.',15)
  path=out/f'{size}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));evidence.setdefault('routes',{})[size]=routes
  views.append(dict(id=size,label='YOLOX regular family' if symbolic else 'YOLOX '+sz.upper(),task='detect',size='t/s/m/l/x' if symbolic else sz,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/yolox/{size}.svg',html=f'/diagrams/models/yolox/{size}.html',input='H from table' if symbolic else f'1 × 3 × {H} × {H}',verification='cpu'));print('YOLOX',size,'route findings',routes['total_findings'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family='yolox',slug='yolox',title='YOLOX',source_revision=rev,default_view='s',views=views),indent=2)+'\n');evidence.update(verification='CPU all six sizes, random weights; no downloads',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/yolox.py --source /path/to/libreyolo');ep.write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
