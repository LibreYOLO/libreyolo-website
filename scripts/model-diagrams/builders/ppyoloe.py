"""PP-YOLOE S/M/L/X, CSPResNet/CSP-PAN and the 17-bin gated head."""
from pathlib import Path
import argparse,os,sys,json,subprocess,math
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.ppyoloe.nn import LibrePPYOLOEModel,PPYOLOE_CONFIGS
 import torch
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/ppyoloe';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/ppyoloe.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];ev=json.loads(ep.read_text()) if args.diagram_only else dict(family='ppyoloe',source_revision=rev,source_files=['libreyolo/models/ppyoloe/nn.py'],shapes={});views=[]
 for size in [*PPYOLOE_CONFIGS,'family']:
  symbolic=size=='family';sz='s' if symbolic else size;cfg=PPYOLOE_CONFIGS[sz];C=round(64*cfg['width_mult']);counts=[max(round(n*cfg['depth_mult']),1) for n in [3,6,6,3]];r=max(round(3*cfg['depth_mult']),1);before=(r-1)//2+1;after=r-before;c=lambda k: (('C' if k==1 else str(k)+'C') if symbolic else str(round(k*C)));mid=[(round(C*2**i)+round(C*2**(i+1)))//2 for i in range(4)];halves=[x//2 for x in mid]
  if not symbolic and not args.diagram_only:
   model=LibrePPYOLOEModel(size,nb_classes=80).eval();obs={}
   def hook(name):
    def record(mod,ins,out):
     def shape(v):
      if isinstance(v,torch.Tensor):return list(v.shape)
      if isinstance(v,(list,tuple)):return [shape(x) for x in v]
      if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
      if isinstance(v,int):return v
      return None
     obs[name]={'input':shape(ins),'output':shape(out)}
    return record
   for name,mod in model.named_modules():
    if name and (name.count('.')<=3 or name.startswith('head.')):mod.register_forward_hook(hook(name))
   with torch.inference_mode():pred=model(torch.zeros(1,3,640,640))
   assert pred[0][0].shape==(1,8400,4);assert pred[0][1].shape==(1,8400,80);assert pred[1][1].shape==(1,8400,68);assert pred[1][4]==[400,1600,6400];ev['shapes'][size]=obs
  H=4550+(530 if symbolic else 0);d=Diagram('PP-YOLOE S/M/L/X family' if symbolic else 'PP-YOLOE-'+size.upper(),'Detection; 640 × 640 RGB; 80 classes; batch 1; unfused eval. Head order is stride 32,16,8.',width=2500,height=H,revision=rev,source_label='models/ppyoloe/nn.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/ppyoloe/nn.py',logo=WEBSITE/'public/icon-128.png')
  b=d.panel('backbone','CSPResNet backbone',40,230,610,1210);n=d.panel('neck','PP-YOLOE CSP-PAN',680,230,800,1210);h=d.panel('head','Efficient Task-aligned head',1510,230,950,1210)
  stages=[('input','Input','3 × 640 × 640','plain',''),('s1','ConvBNAct 3×3, s=2',c(.5)+' × 320 × 320','conv','ConvBNAct'),('s2','ConvBNAct 3×3',c(.5)+' × 320 × 320','conv','ConvBNAct'),('s3','ConvBNAct 3×3',c(1)+' × 320 × 320','conv','ConvBNAct')]
  for j,mul in enumerate([2,4,8,16],2):stages.append(('B'+str(j),'CSPResStage B'+str(j),c(mul)+f' ×{640//2**j} ×{640//2**j}; n='+('n'+str(j-2) if symbolic else str(counts[j-2]))+'; mid='+('M'+str(j) if symbolic else str(mid[j-2])),'aggregate','CSPResStage'))
  for i,(id,l,dt,k,bl) in enumerate(stages):box(b,id,40,70+i*128,530,l,dt,k,bl)
  seq(b,[x[0] for x in stages]);b.text(20,1140,'Every backbone stage downsamples by 2 and uses EffectiveSE.',15)
  td=[('N5','CSPStage with SPP',12,20,'aggregate','CSPStageSPP'),('red5','ConvBNAct 1×1',6,20,'conv','ConvBNAct'),('up4','Nearest upsample ×2',6,40,'norm',''),('cat4','Concat with B4',14,40,'concat',''),('N4','CSPStage',6,40,'aggregate','CSPStage'),('red4','ConvBNAct 1×1',3,40,'conv','ConvBNAct'),('up3','Nearest upsample ×2',3,80,'norm',''),('cat3','Concat with B3',7,80,'concat',''),('P3','CSPStage',3,80,'aggregate','CSPStage')];bu=[('down4','ConvBNAct 3×3, s=2',3,40,'conv','ConvBNAct'),('catn4','Concat with N4',9,40,'concat',''),('P4','CSPStage',6,40,'aggregate','CSPStage'),('down5','ConvBNAct 3×3, s=2',6,20,'conv','ConvBNAct'),('catn5','Concat with N5',18,20,'concat',''),('P5','CSPStage',12,20,'aggregate','CSPStage')]
  for x,ss in [(60,td),(485,bu)]:
   for i,(id,l,mul,g,k,bl) in enumerate(ss):box(n,id,x,95+i*103,280,l,c(mul)+f' ×{g} ×{g}'+('; blocks='+('r' if symbolic else str(r)) if bl.startswith('CSPStage') else ''),k,bl)
   seq(n,[x[0] for x in ss])
  sy=n.port('P3','right')[1];n.wire([(340,sy),(405,sy),(405,60),(625,60),(625,95)],start='P3',end='down4')
  for dest,source in [('N5','B5'),('cat4','B4'),('cat3','B3'),('catn4','N4'),('catn5','N5')]:
   px,py=n.port(dest,'left');n.text(px-50,py-10,source,12,weight=600);n.wire([(px-28,py),(px,py)],start=source,end=dest)
  n.text(25,1125,'Head receives P5, P4, P3, in that order.',16);n.text(25,1160,'Neck basic blocks do not add residuals.',15)
  # Shared spatial mean feeds separate learned class/reg gates; class has an
  # extra feature residual after its ESE stem, regression does not.
  box(h,'feature',340,85,380,'One scale feature','P5/P4/P3 channels: '+c(12)+'/'+c(6)+'/'+c(3));box(h,'avg',340,185,380,'AdaptiveAvgPool2d','1 ×1; shared mean for both stems','pool');h.connect('feature','avg')
  box(h,'classstem',25,370,350,'ESEAttn class stem','Same channel width as input','attention','ESEAttn');box(h,'regstem',580,370,350,'ESEAttn regression stem','Same channel width as input','attention','ESEAttn')
  # Common feature trunk drawn once, then split into class stem and residual.
  h.wire([(340,108),(200,108)],start='feature',arrow=False);h.dot(200,108);h.wire([(200,108),(200,370)],start='feature',end='classstem');h.wire([(720,108),(942,108),(942,305),(755,305),(755,370)],start='feature',end='regstem')
  h.wire([(530,231),(530,275)],start='avg',arrow=False);h.dot(530,275);h.wire([(530,275),(440,275),(440,393),(375,393)],start='avg',end='classstem');h.wire([(530,275),(530,393),(580,393)],start='avg',end='regstem')
  h.sum('classadd',200,510);h.connect('classstem','classadd');h.wire([(200,108),(10,108),(10,510),(187,510)],start='feature',end='classadd')
  box(h,'classpred',25,615,350,'Conv2d 3×3, p=1','80 logits; bias=True','conv2d');box(h,'regpred',580,615,350,'Conv2d 3×3, p=1','68 logits =4 × 17 bins; bias=True','conv2d');h.connect('classadd','classpred');h.connect('regstem','regpred');box(h,'sigmoid',25,750,350,'Sigmoid','1 ×8,400 ×80 scores','activation');box(h,'dfl',580,750,350,'DFL expectation over 17 bins','Bins 0...16 ; 4 distances/location','activation','Decode');h.connect('classpred','sigmoid');h.connect('regpred','dfl');box(h,'boxes',580,895,350,'Grid point minus/plus distances','Multiply per-level stride;1 ×8,400 ×4','plain');h.connect('dfl','boxes')
  h.text(20,1070,'Independent heads per level; no objectness term.',15);h.text(20,1110,'Flatten order: 400 locations, then 1,600, then 6,400.',15);h.text(20,1150,'NMS is external to this decoded eval graph.',15)
  defs=['ConvBNAct','RepVGGBlock','BasicBlock','CSPResStage','CSPStage','CSPStageSPP','EffectiveSE','ESEAttn','SPP']
  for i,name in enumerate(defs):
   p=d.panel('def'+name,name,40+(i%3)*820,1480+(i//3)*890,790,850,kind='aggregate' if name.startswith('CSP') else 'attention' if 'SE' in name else 'pool' if name=='SPP' else 'conv',dashed=True,block_type=name)
   def bb(id,y,l,dt='',k='plain',x=235,w=530,bl=''):return box(p,name+id,x,y,w,l,dt,k,bl,h=43)
   def sq(ids):seq(p,[name+id for id in ids])
   halfstr='Q' if symbolic else '/'.join(map(str,halves))
   if name=='ConvBNAct':
    for id,y,l,dt,k in [('c',110,'Conv2d','Kernel/stride/padding from occurrence; no bias','conv2d'),('bn',300,'BatchNorm2d','eps=.00001; momentum=.1','norm'),('act',490,'SiLU','x ×sigmoid(x)','activation')]:bb(id,y,l,dt,k)
    sq(['c','bn','act']);p.text(25,705,'Stride 1 unless marked. Conv1×1 p=0; Conv3×3 p=1.',15)
   elif name=='RepVGGBlock':
    bb('in',75,'Input','Width unchanged in all these blocks')
    p.wire([(500,118),(500,185)],start=name+'in',arrow=False);p.dot(500,185)
    for suffix,x,k,pad in [('a',25,3,1),('b',435,1,0)]:box(p,name+suffix,x,235,330,f'Conv2d{k}×{k}',f'p={pad}; no bias','conv2d');box(p,name+suffix+'bn',x,380,330,'BatchNorm2d','eps=.00001','norm');p.wire([(500,185),(x+165,185),(x+165,235)],start=name+'in',end=name+suffix);p.connect(name+suffix,name+suffix+'bn')
    p.sum(name+'sum',500,595);p.connect(name+'abn',name+'sum',via=[(190,595)],to_port='left');p.connect(name+'bbn',name+'sum',via=[(600,595)],to_port='right');bb('act',725,'SiLU');p.connect(name+'sum',name+'act');p.text(25,815,'No identity BN branch, SE or learnable alpha.',14)
   elif name=='BasicBlock':
    bb('in',80,'Input','Backbone widths '+halfstr+'; neck widths from table');bb('conv',250,'ConvBNAct 3×3','Same input/output width','conv',bl='ConvBNAct');bb('rep',420,'RepVGGBlock','Same width; no stride change','bottleneck',bl='RepVGGBlock');sq(['in','conv','rep']);p.sum(name+'add',500,665);p.connect(name+'rep',name+'add');p.connect(name+'in',name+'add',from_port='left',via=[(80,101.5),(80,665)],to_port='left');p.text(25,790,'Backbone adds identity. Neck returns RepVGG output directly.',15)
   elif name=='CSPResStage':
    bb('down',65,'ConvBNAct 3×3, s=2','Mid channels '+('M2/M3/M4/M5' if symbolic else '/'.join(map(str,mid))),'conv',bl='ConvBNAct');bb('left',185,'ConvBNAct 1×1',halfstr+' channels','conv',x=25,w=335,bl='ConvBNAct');bb('right',185,'ConvBNAct 1×1',halfstr+' channels','conv',x=430,w=335,bl='ConvBNAct');p.connect(name+'down',name+'left',via=[(500,145),(192.5,145)]);p.connect(name+'down',name+'right',via=[(500,145),(597.5,145)]);bb('blocks',340,'BasicBlock repeated stage n','Residual enabled','bottleneck',x=430,w=335,bl='BasicBlock');p.connect(name+'right',name+'blocks');bb('cat',500,'Concat two branches','Restore mid channels','concat');p.wire([(192.5,228),(192.5,450),(350,450),(350,500)],start=name+'left',end=name+'cat');p.wire([(597.5,383),(597.5,474),(650,474),(650,500)],start=name+'blocks',end=name+'cat');bb('attn',625,'EffectiveSE','Same mid channels','attention',bl='EffectiveSE');bb('out',745,'ConvBNAct 1×1','Output channels from backbone stage','conv',bl='ConvBNAct');sq(['cat','attn','out'])
   elif name in ['CSPStage','CSPStageSPP']:
    use_spp=name=='CSPStageSPP';half=c(6) if use_spp else ('Qneck' if symbolic else '/'.join(str(round(C*k)) for k in [1.5,3,6]));bb('in',55,'Input','Neck-stage channels from main graph');bb('left',165,'ConvBNAct 1×1',half+' channels','conv',x=25,w=335,bl='ConvBNAct');bb('right',165,'ConvBNAct 1×1',half+' channels','conv',x=430,w=335,bl='ConvBNAct');p.connect(name+'in',name+'left',via=[(500,135),(192.5,135)]);p.connect(name+'in',name+'right',via=[(500,135),(597.5,135)])
    bb('before',290,'BasicBlock ×'+('before' if symbolic and use_spp else 'r' if symbolic else str(before if use_spp else r)),'No residual; '+half+' channels','bottleneck',x=430,w=335,bl='BasicBlock');p.connect(name+'right',name+'before');last='before'
    if use_spp:
     bb('spp',420,'SPP (parallel pools)',half+' channels','spp',x=430,w=335,bl='SPP');p.connect(name+'before',name+'spp');last='spp'
     if after or symbolic:bb('after',535,'BasicBlock ×'+('after (may be 0)' if symbolic else str(after)),'No residual','bottleneck',x=430,w=335,bl='BasicBlock');p.connect(name+'spp',name+'after');last='after'
    bb('cat',675,'Concat two branches','Output channels of neck stage','concat');sy=p.port(name+last,'bottom')[1];p.wire([(192.5,208),(192.5,620),(350,620),(350,675)],start=name+'left',end=name+'cat');p.wire([(597.5,sy),(597.5,650),(650,650),(650,675)],start=name+last,end=name+'cat');bb('out',775,'ConvBNAct 1×1','Same output width','conv',bl='ConvBNAct');sq(['cat','out'])
   elif name=='EffectiveSE':
    for id,y,l,dt,k in [('in',80,'Input','Backbone mid channels','plain'),('mean',225,'Spatial mean','1 ×1 per channel','pool'),('fc',370,'Conv2d 1×1','Same channel width; bias=True','conv2d'),('act',515,'Hardsigmoid','clamp(x+3,0,6) /6','activation'),('mul',690,'Multiply original input by gate','Spatially broadcast channel weights','activation')]:bb(id,y,l,dt,k)
    sq(['in','mean','fc','act','mul']);p.connect(name+'in',name+'mul',from_port='left',to_port='left',via=[(75,101.5),(75,711.5)])
   elif name=='ESEAttn':
    bb('avg',75,'Shared spatial mean from head','1 ×1; feature channel width');bb('fc',230,'Conv2d 1×1','Same channel width; bias=True','conv2d');bb('sigmoid',385,'Sigmoid','Channel gate','activation');bb('mul',545,'Multiply original feature by gate','Original feature is the second input','activation');bb('conv',710,'ConvBNAct 1×1','Same feature channel width','conv',bl='ConvBNAct');sq(['avg','fc','sigmoid','mul','conv']);p.text(25,567,'Feature',15);p.wire([(95,566.5),(235,566.5)],start='feature',end=name+'mul')
   elif name=='SPP':
    bb('in',70,'Input',c(6)+' channels;20 ×20')
    for j,k in enumerate([5,9,13]):box(p,name+'p'+str(j),170+j*195,240,175,f'MaxPool{k}×{k}',f's=1; p={k//2}','pool',h=43);p.wire([(350+j*90,113),(350+j*90,160+j*15),(257.5+j*195,160+j*15),(257.5+j*195,240)],start=name+'in',end=name+'p'+str(j))
    bb('cat',535,'Concat input and three pooled tensors',c(24)+' channels','concat');p.wire([(235,91.5),(45,91.5),(45,510),(270,510),(270,535)],start=name+'in',end=name+'cat')
    for j in range(3):sx=257.5+j*195;px=405+j*130;yy=390+j*24;p.wire([(sx,283),(sx,yy),(px,yy),(px,535)],start=name+'p'+str(j),end=name+'cat')
    bb('out',710,'ConvBNAct 1×1',c(6)+' channels','conv',bl='ConvBNAct');sq(['cat','out'])
  p=d.panel('decode','DFL and output contract',40,4185,2420,225,kind='plain',block_type='Decode')
  for j,(id,l,dt,k) in enumerate([('reshape','Reshape box logits','4 sides × 17 bins ×locations','split'),('softmax','Softmax over 17 bins','Per side and location','activation'),('expect','Multiply by bins 0...16; sum','Four l/t/r/b distances','linear'),('grid','Center point ±distances; ×stride','Offset 0.5; strides32/16/8','plain')]):box(p,id,25+j*600,85,550,l,dt,k)
  for a,z in zip(['reshape','softmax','expect'],['softmax','expect','grid']):p.connect(a,z,from_port='right',to_port='left')
  p.text(25,185,'Raw diagnostics also include class logits[1,8400,80], distributions[1,8400,68], anchors[8400,4], points[8400,2], counts[400,1600,6400], strides[8400,1].',15)
  if symbolic:
   p=d.panel('familyvalues','Resolved family values',40,4450,2420,500);xs=[25,560,1050,1540,2030]
   for x,val in zip(xs,['Quantity','S','M','L','X']):p.text(x,55,val,16,weight=600)
   props=[('Stem C',lambda cv,cc,rr:str(cv)),('Backbone n0/n1/n2/n3',lambda cv,cc,rr:'/'.join(map(str,cc))),('Backbone mids M2/M3/M4/M5',lambda cv,cc,rr:'/'.join(str(round(cv*k)) for k in [1.5,3,6,12])),('Backbone block widths Q',lambda cv,cc,rr:'/'.join(str(round(cv*k)) for k in [.75,1.5,3,6])),('Neck output P5/P4/P3',lambda cv,cc,rr:'/'.join(str(round(cv*k)) for k in [12,6,3])),('Neck block widths Qneck',lambda cv,cc,rr:'/'.join(str(round(cv*k)) for k in [1.5,3,6])),('Neck repeats r; before/after SPP',lambda cv,cc,rr:f'{rr}; {(rr-1)//2+1}/{rr-((rr-1)//2+1)}')]
   for j,(label,fn) in enumerate(props):
    p.text(25,100+j*48,label,15)
    for x,ss in zip(xs[1:],'smlx'):cf=PPYOLOE_CONFIGS[ss];cv=round(64*cf['width_mult']);cc=[max(round(v*cf['depth_mult']),1) for v in [3,6,6,3]];rr=max(round(3*cf['depth_mult']),1);p.text(x,100+j*48,fn(cv,cc,rr),15)
  else:d.text(50,H-95,'Backbone block widths: '+halfstr+'. Neck block widths: '+', '.join(str(round(C*k)) for k in [1.5,3,6])+'. Raw anchor squares have sizes 160/80/40 pixels.',15)
  path=out/f'{size}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));ev.setdefault('routes',{})[size]=routes;views.append(dict(id=size,label='PP-YOLOE family' if symbolic else 'PP-YOLOE '+size.upper(),task='detect',size='s/m/l/x' if symbolic else size,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/ppyoloe/{size}.svg',html=f'/diagrams/models/ppyoloe/{size}.html',input='1 × 3 × 640 × 640',verification='cpu'));print('ppyoloe',size,routes['total_findings'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family='ppyoloe',slug='ppyoloe',title='PP-YOLOE',source_revision=rev,default_view='s',views=views),indent=2)+'\n');ev.update(verification='CPU all 4 sizes, no downloads; decoded and raw contracts asserted',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/ppyoloe.py --source /path/to/libreyolo');ep.write_text(json.dumps(ev,indent=2)+'\n')
if __name__=='__main__':main()
