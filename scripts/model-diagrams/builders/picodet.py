"""PicoDet s/m/l ESNet, CSP-PAN, shared class/reg head and exact SE gate."""
from pathlib import Path
import argparse,os,sys,json,subprocess,math
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]
def block_table(nn,size):
 cfg=nn.ESNet.ARCH[size];outchs=[24]+[nn._make_divisible(v*cfg['scale'],16) for v in [128,256,512]];rows=[];idx=0
 for stage,repeats in enumerate([3,7,3]):
  for j in range(repeats):
   ci=outchs[stage] if j==0 else outchs[stage+1];co=outchs[stage+1];mid=nn._make_divisible(int(co*cfg['ratios'][idx]),8);ds=j==0;gate=mid//2 if ds else mid;rows.append(dict(i=idx,ci=ci,co=co,mid=mid,ds=ds,A=ci if ds else ci//2,B=mid//2,G=gate,R=gate//4,O=co//2,div=8*2**stage));idx+=1
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.picodet import nn as nn
 import torch
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/picodet';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/picodet.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];ev=json.loads(ep.read_text()) if args.diagram_only else dict(family='picodet',source_revision=rev,source_files=['libreyolo/models/picodet/nn.py','libreyolo/postprocess/picodet.py'],shapes={});views=[];Hs={'s':320,'m':416,'l':640}
 for size in ['s','m','l','family']:
  symbolic=size=='family';sz='s' if symbolic else size;H=Hs[sz];F=nn.SIZE_SPEC[sz]['neck_ch'];T=nn.SIZE_SPEC[sz]['stacked_convs'];rows=block_table(nn,sz);chs=[rows[i]['co'] for i in [2,9,12]];grids=[H//8,H//16,H//32,math.ceil(H/64)];N=sum(g*g for g in grids);f='F' if symbolic else str(F)
  if not symbolic and not args.diagram_only:
   model=nn.LibrePICODETModel(size,nb_classes=80).eval();obs={}
   def hook(name):
    def record(mod,ins,out):
     def shape(v):
      if isinstance(v,torch.Tensor):return list(v.shape)
      if isinstance(v,(list,tuple)):return [shape(x) for x in v]
      if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
      return None
     obs[name]={'input':shape(ins),'output':shape(out)}
    return record
   for name,mod in model.named_modules():
    if name:mod.register_forward_hook(hook(name))
   with torch.inference_mode():cl,bo=model(torch.zeros(1,3,H,H))
   assert [list(x.shape) for x in cl]==[[1,80,g,g] for g in grids];assert [list(x.shape) for x in bo]==[[1,32,g,g] for g in grids];ev['shapes'][size]=obs
  d=Diagram('PicoDet S/M/L family' if symbolic else 'PicoDet-'+size.upper(),('H,F,T and per-block widths are resolved in the family tables.' if symbolic else f'{H} × {H} RGB; 80 classes; batch 1.')+' Unfused PyTorch eval; four feature scales.',width=2580,height=4600+(500 if symbolic else 0),revision=rev,source_label='models/picodet/nn.py; postprocess/picodet.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/picodet/nn.py',logo=WEBSITE/'public/icon-128.png')
  b=d.panel('backbone','ESNet backbone',40,230,610,1470);n=d.panel('neck','CSP-PAN',680,230,930,1470);h=d.panel('head','PicoHead and decoding',1640,230,900,1470)
  box(b,'input',40,65,530,'Input','3 × H × H' if symbolic else f'3 × {H} × {H}');box(b,'stem',40,150,530,'ConvBNAct 3×3, s=2','24 × '+('H/2 square' if symbolic else f'{H//2} × {H//2}'),'conv','ConvBNAct');box(b,'pool',40,235,530,'MaxPool 3×3, s=2, p=1','24 × '+('H/4 square' if symbolic else f'{H//4} × {H//4}'),'pool');seq(b,['input','stem','pool']);prev='pool'
  for row in rows:
   i=row['i'];label=f'Block {i}: '+('ESBlockDS' if row['ds'] else 'ESBlock')+('; mid=M'+str(i) if symbolic else f'; mid={row["mid"]}');detail=(f'Cout{i} × H/{row["div"]} square' if symbolic else f'{row["ci"]} input; {row["co"]} × {H//row["div"]} × {H//row["div"]}')+('; output C'+str([2,9,12].index(i)+3) if i in [2,9,12] else '')
   box(b,'B'+str(i),40,335+i*81,530,label,detail,'aggregate','ESBlockDS' if row['ds'] else 'ESBlock');b.connect(prev,'B'+str(i));prev='B'+str(i)
  b.text(20,1435,'Downsample blocks: 0,3,10. Outputs: 2,9,12.',15)
  for j,(cc,g) in enumerate(zip(chs,grids)):
   box(n,'T'+str(j+3),20+j*300,85,280,'C'+str(j+3)+': ConvBNAct 1×1',f'{cc if not symbolic else "Cin"} input; {f} × '+(f'H/{2**(j+3)} square' if symbolic else f'{g} × {g}'),'conv','ConvBNAct')
  td=[('up4','Nearest resize ×2',1,'norm',''),('cat4','Concat with T4',1,'concat',''),('N4','CSPLayer, n=1',1,'aggregate','CSPLayer'),('up3','Nearest resize ×2',0,'norm',''),('cat3','Concat with T3',0,'concat',''),('P3','CSPLayer, n=1',0,'aggregate','CSPLayer')];bu=[('down4','DepthwiseSeparable 5×5, s=2',1,'conv','DepthwiseSeparable'),('catn4','Concat with N4',1,'concat',''),('P4','CSPLayer, n=1',1,'aggregate','CSPLayer'),('down5','DepthwiseSeparable 5×5, s=2',2,'conv','DepthwiseSeparable'),('cat5','Concat with T5',2,'concat',''),('P5','CSPLayer, n=1',2,'aggregate','CSPLayer')]
  for x,stages in [(65,td),(510,bu)]:
   for i,(id,label,lev,k,bl) in enumerate(stages):width=('2F' if symbolic else str(2*F)) if k=='concat' else f;box(n,id,x,250+i*105,330,label,width+' × '+(f'H/{2**(lev+3)} square' if symbolic else f'{grids[lev]} × {grids[lev]}'),k,bl)
   seq(n,[x[0] for x in stages])
  sy=n.port('P3','right')[1];n.wire([(395,sy),(448,sy),(448,205),(675,205),(675,250)],start='P3',end='down4')
  for dest,source in [('up4','T5'),('cat4','T4'),('cat3','T3'),('catn4','N4'),('cat5','T5')]:
   px,py=n.port(dest,'left');n.text(px-52,py-10,source,12,weight=600);n.wire([(px-29,py),(px,py)],start=source,end=dest)
  box(n,'top1',65,995,330,'T5: DepthwiseSeparable 5×5, s=2',f+' × '+('ceil(H/64) square' if symbolic else f'{grids[3]} × {grids[3]}'),'conv','DepthwiseSeparable');box(n,'top2',510,995,330,'P5: DepthwiseSeparable 5×5, s=2',f+' × '+('ceil(H/64) square' if symbolic else f'{grids[3]} × {grids[3]}'),'conv','DepthwiseSeparable');n.connect('P5','top2');n.sum('topsum',452,1130);n.connect('top1','topsum',via=[(230,1130)],to_port='left');n.connect('top2','topsum',via=[(675,1130)],to_port='right');box(n,'P6',260,1240,385,'P6 output',f+' × '+('ceil(H/64) square' if symbolic else f'{grids[3]} × {grids[3]}'));n.connect('topsum','P6');n.text(25,1390,'All four CSP neck blocks disable residual addition.',15)
  box(h,'hin',165,80,570,'One scale feature',f+' channels; execute independently at P3/P4/P5/P6')
  for j in range(T if not symbolic else 1):box(h,'ht'+str(j),165,195+j*93,570,'DepthwiseSeparable 5×5',f+' channels; stack '+('T' if symbolic else str(T))+' total layers','conv','DepthwiseSeparable')
  seq(h,['hin']+['ht'+str(j) for j in range(T if not symbolic else 1)]);box(h,'pred',165,610,570,'Conv2d 1×1','112 output channels; bias=True','conv2d');h.connect('ht'+str((T if not symbolic else 1)-1),'pred');box(h,'split',165,720,570,'Split channels','80 class logits; 32 box-distribution logits','split');h.connect('pred','split')
  box(h,'classes',20,850,370,'Sigmoid class logits','80 probabilities/location','activation');box(h,'dfl',490,850,370,'Softmax over 8 bins per side','Weighted expectation of bins 0...7','activation');h.connect('split','classes',via=[(450,800),(205,800)]);h.connect('split','dfl',via=[(450,800),(675,800)]);box(h,'distance',490,980,370,'Multiply distances by stride','8,16,32,64 pixels','linear');h.connect('dfl','distance');box(h,'bbox',490,1100,370,'Grid center minus/plus l/t/r/b','Grid offset=0.5; xyxy boxes','plain');h.connect('distance','bbox');box(h,'nms',165,1280,570,'Threshold and class-aware NMS','Decode and NMS are outside native raw-head forward','pool');h.wire([(205,896),(205,1220),(275,1220),(275,1280)],start='classes',end='nms');h.wire([(675,1146),(675,1244),(625,1244),(625,1280)],start='bbox',end='nms');h.text(20,1410,'No separate regression tower or objectness branch.',15)
  # Six full recursive definitions. Letter widths are resolved per occurrence
  # in the concrete table, avoiding a fictitious uniform ESNet width ratio.
  names=['ConvBNAct','ESBlock','ESBlockDS','SELayer','CSPLayer','DarknetBottleneck']
  for i,name in enumerate(names):
   p=d.panel('def'+name,name,40+(i%3)*850,1740+(i//3)*1120,820,1080,kind='aggregate' if name.startswith('ES') or name=='CSPLayer' else 'attention' if name=='SELayer' else 'conv',dashed=True,block_type=name)
   def bb(id,y,label,detail='',kind='plain',x=250,w=530,bl=''):return box(p,name+id,x,y,w,label,detail,kind,bl,h=43)
   def sq(ids):seq(p,[name+id for id in ids])
   if name=='ConvBNAct':
    for id,y,l,dt,k in [('c',95,'Conv2d','Bias=False; p=k//2; k/stride/groups from occurrence','conv2d'),('bn',235,'BatchNorm2d','eps=.00001; momentum=.1','norm'),('act',375,'Hardswish or identity','Identity only on marked ES depthwise branches','activation')]:bb(id,y,l,dt,k)
    sq(['c','bn','act']);bb('dw',630,'ConvBNAct depthwise k×k','groups=input channels; k=5 in neck/head','conv',bl='ConvBNAct');bb('pw',800,'ConvBNAct pointwise 1×1','groups=1; output width from occurrence','conv',bl='ConvBNAct');sq(['dw','pw']);p.text(25,1005,'The lower pair defines DepthwiseSeparableConv.',16)
   elif name=='ESBlock':
    for id,y,l,dt,k,bl in [('input',55,'Input','Cin channels','plain',''),('split',125,'Split into two channel halves','A channels each; table resolves A','split',''),('pw',215,'ConvBNAct 1×1','A input; B output; Hardswish','conv','ConvBNAct'),('dw',315,'ConvBNAct depthwise 3×3','B channels; groups=B; no activation','conv','ConvBNAct'),('catmid',415,'Concat PW and DW outputs','G channels','concat',''),('se',510,'SELayer','G input; R reduced width','attention','SELayer'),('linear',605,'ConvBNAct 1×1','G input; O output; Hardswish','conv','ConvBNAct'),('catout',730,'Concat untouched half and new branch','Cout channels','concat',''),('reshape',825,'Reshape','1 ×2 ×O ×height ×width','split',''),('transpose',900,'Transpose channel-group axes 1 and 2','Make contiguous','split',''),('out',975,'Reshape','1 ×Cout ×height ×width','split','')]:bb(id,y,l,dt,k,bl=bl)
    sq(['input','split','pw','dw']);p.wire([(250,146.5),(40,146.5),(40,705),(345,705),(345,730)],start=name+'split',end=name+'catout');p.wire([(250,236.5),(125,236.5),(125,390),(350,390),(350,415)],start=name+'pw',end=name+'catmid');p.wire([(515,358),(515,377),(665,377),(665,415)],start=name+'dw',end=name+'catmid');sq(['catmid','se','linear']);p.wire([(515,648),(515,682),(665,682),(665,730)],start=name+'linear',end=name+'catout');sq(['catout','reshape','transpose','out'])
   elif name=='ESBlockDS':
    bb('input',55,'Input','Cin channels; no channel split before branches')
    for id,y,l,dt,k,x in [('dw1',205,'Depthwise 3×3, s=2','Cin channels; BN; no activation','conv',25),('pw1',440,'ConvBNAct 1×1','Cin input; O output; Hardswish','conv',25),('pw2',185,'ConvBNAct 1×1','Cin input; B output; Hardswish','conv',440),('dw2',285,'Depthwise 3×3, s=2','B channels; BN; no activation','conv',440),('se',385,'SELayer','G=B input; R reduced width','attention',440),('linear',490,'ConvBNAct 1×1','B input; O output; Hardswish','conv',440)]:bb(id,y,l,dt,k,x=x,w=335,bl='SELayer' if id=='se' else 'ConvBNAct')
    p.connect(name+'input',name+'dw1',via=[(515,135),(192.5,135)]);p.connect(name+'input',name+'pw2',via=[(515,135),(607.5,135)]);sq(['dw1','pw1']);sq(['pw2','dw2','se','linear']);bb('cat',630,'Concat two downsampled branches','Cout channels','concat');p.wire([(192.5,483),(192.5,580),(350,580),(350,630)],start=name+'pw1',end=name+'cat');p.wire([(607.5,533),(607.5,604),(670,604),(670,630)],start=name+'linear',end=name+'cat');bb('refinedw',755,'ConvBNAct depthwise 3×3','Cout channels; s=1; BN + Hardswish','conv',bl='ConvBNAct');bb('refinepw',875,'ConvBNAct pointwise 1×1','Cout channels; BN + Hardswish','conv',bl='ConvBNAct');sq(['cat','refinedw','refinepw']);p.text(25,1020,'This downsample block does not apply channel shuffle.',15)
   elif name=='SELayer':
    for id,y,l,dt,k in [('in',80,'Input','G channels','plain'),('pool',225,'AdaptiveAvgPool2d','1 ×1 spatial output','pool'),('reduce',360,'Conv2d 1×1','G input; R output; bias=True','conv2d'),('relu',470,'ReLU','','activation'),('expand',585,'Conv2d 1×1','R input; G output; bias=True','conv2d'),('gate',705,'HSigmoid (custom)','clamp((x+3)/6,0,6); range[0,6]','activation'),('mul',885,'Multiply original feature by gate','Broadcast across spatial positions','activation')]:bb(id,y,l,dt,k)
    sq(['in','pool','reduce','relu','expand','gate','mul']);p.connect(name+'in',name+'mul',from_port='left',to_port='left',via=[(80,101.5),(80,906.5)]);p.text(25,1020,'This source gate is not the standard [0,1] hardsigmoid.',15)
   elif name=='CSPLayer':
    bb('in',90,'Input',('2F' if symbolic else str(2*F))+' channels');bb('left',250,'ConvBNAct 1×1',('F/2' if symbolic else str(F//2))+' channels','conv',x=25,w=340,bl='ConvBNAct');bb('right',250,'ConvBNAct 1×1',('F/2' if symbolic else str(F//2))+' channels','conv',x=440,w=340,bl='ConvBNAct');p.wire([(515,133),(515,195)],start=name+'in',arrow=False);p.dot(515,195);p.wire([(515,195),(195,195),(195,250)],start=name+'in',end=name+'left');p.wire([(515,195),(610,195),(610,250)],start=name+'in',end=name+'right');bb('bottle',435,'DarknetBottleneck, n=1','No residual; hidden '+('F/2' if symbolic else str(F//2)),'bottleneck',x=25,w=340,bl='DarknetBottleneck');p.connect(name+'left',name+'bottle');bb('cat',650,'Concat',f+' channels','concat');p.wire([(195,478),(195,585),(350,585),(350,650)],start=name+'bottle',end=name+'cat');p.wire([(610,293),(610,609),(670,609),(670,650)],start=name+'right',end=name+'cat');bb('out',840,'ConvBNAct 1×1',f+' output channels','conv',bl='ConvBNAct');sq(['cat','out'])
   elif name=='DarknetBottleneck':
    bb('in',100,'Input',('F/2' if symbolic else str(F//2))+' channels');bb('point',295,'ConvBNAct 1×1','Width unchanged','conv',bl='ConvBNAct');bb('dw',490,'DepthwiseSeparable 5×5','Width unchanged; BN and Hardswish after both convs','conv',bl='DepthwiseSeparable');sq(['in','point','dw']);p.text(25,750,'No residual addition in the configured PicoDet CSP-PAN.',16)
  # Resolved per-block numbers form part of each concrete drawing.
  table=d.panel('widths','Per-block width and resolution values',40,4010,2500,465)
  if not symbolic:
   xs=[25,230,460,690,920,1150,1380,1610,1840,2070,2300];head=['Block','Cin','Cout','Mid M','A','B','G','R','O','Grid','Type']
   for x,label in zip(xs,head):table.text(x,50,label,15,weight=600)
   for j,row in enumerate(rows):
    vals=[row['i'],row['ci'],row['co'],row['mid'],row['A'],row['B'],row['G'],row['R'],row['O'],H//row['div'],'DS' if row['ds'] else 'ES']
    for x,value in zip(xs,vals):table.text(x,82+j*27,str(value),14)
  else:
   xs=[25,600,1220,1840]
   for x,label in zip(xs,['Block','S: Cin/Cout/M','M: Cin/Cout/M','L: Cin/Cout/M']):table.text(x,50,label,15,weight=600)
   tabs={ss:block_table(nn,ss) for ss in ['s','m','l']}
   for j in range(13):
    table.text(25,82+j*27,str(j),14)
    for x,ss in zip(xs[1:],['s','m','l']):rr=tabs[ss][j];table.text(x,82+j*27,f'{rr["ci"]}/{rr["co"]}/{rr["mid"]}',14)
   p=d.panel('familyvalues','Family input/head values and symbolic width definitions',40,4510,2500,450)
   for j,ss in enumerate(['s','m','l']):p.text(25,75+j*60,f'{ss.upper()}: H={Hs[ss]}; F={nn.SIZE_SPEC[ss]["neck_ch"]}; T={nn.SIZE_SPEC[ss]["stacked_convs"]}; P3/P4/P5/P6 grids='+str([Hs[ss]//8,Hs[ss]//16,Hs[ss]//32,math.ceil(Hs[ss]/64)]),17)
   p.text(25,290,'A=Cin for DS or Cin/2 for ES; B=M/2; G=B for DS or M for ES; R=G/4; O=Cout/2.',16);p.text(25,340,'Every convolution width is derived from its own block row, including the S variant-specific ratios.',16)
  path=out/f'{size}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));ev.setdefault('routes',{})[size]=routes;views.append(dict(id=size,label='PicoDet family' if symbolic else 'PicoDet '+size.upper(),task='detect',size='s/m/l' if symbolic else size,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/picodet/{size}.svg',html=f'/diagrams/models/picodet/{size}.html',input='H from family table' if symbolic else f'1 ×3 ×{H} ×{H}',verification='cpu'));print('picodet',size,routes['total_findings'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family='picodet',slug='picodet',title='PicoDet',source_revision=rev,default_view='s',views=views),indent=2)+'\n');ev.update(verification='CPU all 3 sizes, random weights, no downloads',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/picodet.py --source /path/to/libreyolo');ep.write_text(json.dumps(ev,indent=2)+'\n')
if __name__=='__main__':main()
