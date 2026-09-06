"""RTMDet detect/segment sizes with explicit CSPNeXt and dynamic-mask graphs."""
from pathlib import Path
import argparse,os,sys,json,subprocess
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.rtmdet.nn import LibreRTMDetModel,_SIZE_CONFIG
 from libreyolo.postprocess.rtmdet import _decode_masks
 import torch
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/rtmdet';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/rtmdet.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];ev=json.loads(ep.read_text()) if args.diagram_only else dict(family='rtmdet',source_revision=rev,source_files=['libreyolo/models/rtmdet/nn.py','libreyolo/postprocess/rtmdet.py'],shapes={});views=[]
 variants=[(sz,task,None) for task in ['detect','segment'] for sz in _SIZE_CONFIG]+[('t','detect',['t','s']),('m','detect',['m','l','x']),('t','segment',list(_SIZE_CONFIG))]
 for size,task,group in variants:
  symbolic=group is not None;segment=task=='segment';deep,widen,ch,F,r,exp=_SIZE_CONFIG[size];C=int(64*widen);counts=[max(round(n*deep),1) for n in [3,6,6,3]];vid=('family-'+''.join(group) if symbolic else size)+('-seg' if segment else '');title='RTMDet'+(' Ins' if segment else '')+(' '+''.join(group).upper()+' family' if symbolic else '-'+size.upper());c=lambda k:('C' if k==1 else str(k)+'C') if symbolic else str(k*C);f='F' if symbolic else str(F)
  if not symbolic and not args.diagram_only:
   model=LibreRTMDetModel(size,nc=80,enable_mask_head=segment).eval();obs={}
   def hook(name):
    def record(mod,ins,output):
     def shape(v):
      if isinstance(v,torch.Tensor):return list(v.shape)
      if isinstance(v,(list,tuple)):return [shape(x) for x in v]
      if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
      return None
     obs[name]={'input':shape(ins),'output':shape(output)}
    return record
   for name,mod in model.named_modules():
    if name and (name.count('.')<=3 or name.startswith('head.')):mod.register_forward_hook(hook(name))
   with torch.inference_mode():pred=model(torch.zeros(1,3,640,640))
   assert [list(x.shape) for x in pred[0]]==[[1,80,g,g] for g in [80,40,20]];assert [list(x.shape) for x in pred[1]]==[[1,4,g,g] for g in [80,40,20]]
   if segment:
    assert [list(x.shape) for x in pred[2]]==[[1,169,g,g] for g in [80,40,20]];assert pred[3].shape==(1,8,80,80)
    with torch.inference_mode():mask=_decode_masks(pred[3][0],pred[2][0][0,:,0,0].reshape(1,169),torch.tensor([[0.,0.,8.]]))
    assert mask.shape==(1,80,80);ev.setdefault('dynamic_mask_probe',{})[vid]=list(mask.shape)
   ev['shapes'][vid]=obs
  base_height=3350+(1050 if segment else 0);d=Diagram(title,'Detection'+(' and instance masks' if segment else '')+'; 640 × 640 RGB; 80 classes; batch 1; unfused native PyTorch eval.',width=2500,height=base_height+(550 if symbolic else 0),revision=rev,source_label='models/rtmdet/nn.py; postprocess/rtmdet.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/rtmdet/nn.py',logo=WEBSITE/'public/icon-128.png')
  b=d.panel('backbone','CSPNeXt backbone',40,230,610,1180);n=d.panel('neck','CSPNeXtPAFPN',680,230,780,1180);h=d.panel('head','Instance head' if segment else 'Detection head',1490,230,970,1180)
  stemhalf='C/2' if symbolic else str(C//2);stage=[('input','Input','3 × 640 × 640','plain',''),('st0','ConvBNAct 3×3, s=2',stemhalf+' × 320 × 320','conv','ConvBNAct'),('st1','ConvBNAct 3×3',stemhalf+' × 320 × 320','conv','ConvBNAct'),('st2','ConvBNAct 3×3',c(1)+' × 320 × 320','conv','ConvBNAct')]
  for j,mul in enumerate([2,4,8,16],2):
   grid=640//2**j;stage.append(('D'+str(j),'ConvBNAct 3×3, s=2',c(mul)+f' × {grid} × {grid}','conv','ConvBNAct'))
   if j==5:stage.append(('spp','SPPBottleneck',c(16)+' × 20 × 20','spp','SPP'))
   stage.append(('B'+str(j),f'CSPLayer B{j}'+(' (no residual)' if j==5 else ''),c(mul)+f' × {grid} × {grid}; n='+('n'+str(j-2) if symbolic else str(counts[j-2])),'aggregate','CSPLayer'))
  for i,(id,l,dt,k,bl) in enumerate(stage):box(b,id,40,60+i*82,530,l,dt,k,bl)
  seq(b,[x[0] for x in stage]);b.text(20,1140,'Backbone CSP layers include channel attention.',15)
  td=[('lat','ConvBNAct 1×1',8,20,'conv','ConvBNAct'),('u4','Nearest upsample ×2',8,40,'norm',''),('cat4','Concat with B4',16,40,'concat',''),('N4','CSPLayer',8,40,'aggregate','CSPLayer'),('red','ConvBNAct 1×1',4,40,'conv','ConvBNAct'),('u3','Nearest upsample ×2',4,80,'norm',''),('cat3','Concat with B3',8,80,'concat',''),('N3','CSPLayer',4,80,'aggregate','CSPLayer')]
  bu=[('down4','ConvBNAct 3×3, s=2',4,40,'conv','ConvBNAct'),('catr','Concat with red',8,40,'concat',''),('Nout4','CSPLayer',8,40,'aggregate','CSPLayer'),('down5','ConvBNAct 3×3, s=2',8,20,'conv','ConvBNAct'),('catl','Concat with lat',16,20,'concat',''),('Nout5','CSPLayer',16,20,'aggregate','CSPLayer')]
  for x,stages in [(65,td),(475,bu)]:
   for i,(id,l,mul,g,k,bl) in enumerate(stages):box(n,id,x,95+i*91,275,l,c(mul)+f' × {g} × {g}'+('; n='+('r' if symbolic else str(r)) if bl=='CSPLayer' else ''),k,bl)
   seq(n,[x[0] for x in stages])
  sy=n.port('N3','right')[1];n.wire([(340,sy),(397,sy),(397,62),(612.5,62),(612.5,95)],start='N3',end='down4')
  for dest,source in [('lat','B5'),('cat4','B4'),('cat3','B3'),('catr','red'),('catl','lat')]:
   px,py=n.port(dest,'left');n.text(px-48,py-10,source,12,weight=600);n.wire([(px-28,py),(px,py)],start=source,end=dest)
  n.text(25,900,'Neck CSP: no residuals and no channel attention.',15)
  for j,(id,mul,g) in enumerate([('N3',4,80),('Nout4',8,40),('Nout5',16,20)]):box(n,'P'+str(j+3),25+j*250,975,230,'ConvBNAct 3×3',f'P{j+3}: {f} × {g} × {g}','conv','ConvBNAct');n.text(25+j*250,958,id+' continuation',12);n.wire([(140+j*250,963),(140+j*250,975)],start=id,end='P'+str(j+3))
  n.text(25,1115,'These three output projections make every head input '+f+' channels.',14)
  h.text(20,62,'Run separately at P3/P4/P5: grids 80/40/20, strides 8/16/32.',14)
  branches=['class','reg','kernel'] if segment else ['class','reg'];width=285 if segment else 400;xs=[20,345,670] if segment else[50,520]
  for branch,x in zip(branches,xs):
   box(h,branch+'in',x,120,width,'Pyramid feature',f+' channels')
   for q in range(2):box(h,branch+str(q),x,240+q*110,width,'ConvBNAct 3×3',f+' channels','conv','ConvBNAct')
   seq(h,[branch+'in',branch+'0',branch+'1']);outchannels=80 if branch=='class' else 4 if branch=='reg' else 169;box(h,branch+'pred',x,490,width,'Conv2d 1×1',f'{outchannels} output channels; bias=True','conv2d');h.connect(branch+'1',branch+'pred')
   if branch=='reg':
    op='ReLU, then multiply stride' if segment else 'Exp, then multiply stride' if exp else 'Multiply by stride (linear)';box(h,'regscale',x,620,width,op,'4 l/t/r/b distances in canvas pixels','activation');h.connect('regpred','regscale')
   else:box(h,branch+'raw',x,620,width,'Raw '+branch+' output',str(outchannels)+' channels; no activation','plain');h.connect(branch+'pred',branch+'raw')
  h.text(20,790,'Class/reg 3×3 conv weights are shared across levels; BN is per level.',14)
  if segment:h.text(20,840,'Class and reg evaluate the same tower modules twice per level.',14);h.text(20,885,'Kernel towers are separate and are not shared across levels.',14);h.text(20,930,'Kernel output: 169 parameters for a three-layer dynamic mask head.',14);h.text(20,975,'Mask features are produced from all three neck outputs below.',14)
  else:h.text(20,840,'Class and regression towers have separate parameters.',15);h.text(20,890,'No per-level learned Scale modules. No DFL bins.',15)
  h.text(20,1070,'Total locations: 8,400. Classes remain logits in native outputs.',15)
  # Core definitions are shared only where source topology is actually identical.
  definitions=['ConvBNAct','DepthwiseSeparable','CSPNeXtBlock','CSPLayer','ChannelAttention','SPP']
  for i,name in enumerate(definitions):
   p=d.panel('def'+name,name,40+(i%3)*820,1450+(i//3)*810,790,770,kind='aggregate' if name=='CSPLayer' else 'attention' if name=='ChannelAttention' else 'pool' if name=='SPP' else 'conv',dashed=True,block_type=name)
   def bb(id,y,l,dt='',kind='plain',x=230,w=530,bl=''):return box(p,name+id,x,y,w,l,dt,kind,bl,h=43)
   def sq(ids):seq(p,[name+id for id in ids])
   hidden='Q' if symbolic else '/'.join(str(C*k) for k in [1,2,4,8])
   if name=='ConvBNAct':
    for id,y,l,dt,k in [('c',100,'Conv2d','k/stride/padding/channels from occurrence; no bias','conv2d'),('bn',280,'BatchNorm2d','eps=0.00001; momentum=0.1','norm'),('act',460,'SiLU','x × sigmoid(x)','activation')]:bb(id,y,l,dt,k)
    sq(['c','bn','act']);p.text(25,650,'Unmarked strides/groups are 1. Conv 3×3 uses p=1; Conv 1×1 p=0.',15)
   elif name=='DepthwiseSeparable':
    bb('dw',140,'ConvBNAct depthwise 5×5','p=2; groups=Q; '+hidden+' channels','conv',bl='ConvBNAct');bb('pw',355,'ConvBNAct pointwise 1×1',hidden+' channels','conv',bl='ConvBNAct');sq(['dw','pw']);p.text(25,575,'Both depthwise and pointwise convolutions have BN and SiLU.',15)
   elif name=='CSPNeXtBlock':
    bb('in',70,'Input',hidden+' channels');bb('c',190,'ConvBNAct 3×3',hidden+' channels','conv',bl='ConvBNAct');bb('dw',325,'DepthwiseSeparable 5×5',hidden+' channels','conv',bl='DepthwiseSeparable');sq(['in','c','dw']);p.sum(name+'add',495,555);p.connect(name+'dw',name+'add');p.connect(name+'in',name+'add',from_port='left',via=[(85,91.5),(85,555)],to_port='left');p.text(25,690,'Identity addition only in backbone B2/B3/B4.',15)
   elif name=='CSPLayer':
    bb('in',65,'Input','Output width from stage');bb('left',155,'ConvBNAct 1×1',hidden+' hidden channels','conv',x=25,w=340,bl='ConvBNAct');bb('right',155,'ConvBNAct 1×1',hidden+' hidden channels','conv',x=420,w=340,bl='ConvBNAct');p.connect(name+'in',name+'left',via=[(495,125),(195,125)]);p.connect(name+'in',name+'right',via=[(495,125),(590,125)]);bb('blocks',280,'CSPNeXtBlock repeated n times','n from backbone stage; r in neck','bottleneck',x=25,w=340,bl='CSPNeXtBlock');p.connect(name+'left',name+'blocks');bb('cat',420,'Concat two branches',('2Q' if symbolic else '/'.join(str(C*k) for k in [2,4,8,16]))+' channels','concat');p.wire([(195,323),(195,375),(345,375),(345,420)],start=name+'blocks',end=name+'cat');p.wire([(590,198),(590,399),(645,399),(645,420)],start=name+'right',end=name+'cat');bb('attn',530,'ChannelAttention (backbone only)','Neck bypasses this operation','attention',bl='ChannelAttention');bb('out',645,'ConvBNAct 1×1','Output channels from stage','conv',bl='ConvBNAct');sq(['cat','attn','out']);p.wire([(760,441.5),(777,441.5),(777,666.5),(760,666.5)],start=name+'cat',end=name+'out');p.text(650,500,'neck bypass',12)
   elif name=='ChannelAttention':
    for id,y,l,dt,k in [('in',70,'Input','Backbone stage output width','plain'),('mean',190,'AdaptiveAvgPool2d','1 ×1 spatial output','pool'),('fc',305,'Conv2d 1×1','Same channels; bias=True','conv2d'),('act',430,'Hardsigmoid','Clamp(x+3,0,6) /6; range[0,1]','activation'),('mul',590,'Multiply original input by gate','Broadcast channel weights','activation')]:bb(id,y,l,dt,k)
    sq(['in','mean','fc','act','mul']);p.connect(name+'in',name+'mul',from_port='left',to_port='left',via=[(80,91.5),(80,611.5)])
   elif name=='SPP':
    bb('in',70,'ConvBNAct 1×1',c(8)+' × 20 × 20','conv',bl='ConvBNAct')
    for j,k in enumerate([5,9,13]):box(p,name+'p'+str(j),170+j*195,245,175,f'MaxPool{k}×{k}',f's=1; p={k//2}','pool',h=43);p.wire([(345+j*90,113),(345+j*90,170+j*15),(257.5+j*195,170+j*15),(257.5+j*195,245)],start=name+'in',end=name+'p'+str(j))
    bb('cat',535,'Concat input and three parallel pools',c(32)+' channels','concat');p.wire([(230,91.5),(45,91.5),(45,510),(265,510),(265,535)],start=name+'in',end=name+'cat')
    for j in range(3):sx=257.5+j*195;px=400+j*135;yy=380+j*24;p.wire([(sx,288),(sx,yy),(px,yy),(px,535)],start=name+'p'+str(j),end=name+'cat')
    bb('out',655,'ConvBNAct 1×1',c(16)+' × 20 × 20','conv',bl='ConvBNAct');sq(['cat','out'])
  if segment:
   p=d.panel('maskfeature','Mask feature module',40,3110,900,1010,kind='plain',block_type='MaskFeatModule')
   for j,(id,g) in enumerate([('P3',80),('P4',40),('P5',20)]):box(p,'mf'+id,20+j*290,70,260,id,f+' ×'+str(g)+' ×'+str(g));box(p,'resize'+id,20+j*290,180,260,'Identity' if j==0 else 'Bilinear resize','80 × 80; align_corners=False','norm');p.connect('mf'+id,'resize'+id)
   box(p,'mfcat',185,355,530,'Concat all three features',('3F' if symbolic else str(3*F))+' × 80 × 80','concat')
   for j in range(3):sx=150+j*290;px=260+j*190;yy=270+j*24;p.wire([(sx,226),(sx,yy),(px,yy),(px,355)],start='resizeP'+str(j+3),end='mfcat')
   box(p,'mffuse',185,475,530,'Conv2d 1×1',f+' output channels; bias=True','conv2d');box(p,'mfconv',185,595,530,'ConvBNAct 3×3 repeated 4 times',f+' × 80 × 80','conv','ConvBNAct');box(p,'mfout',185,720,530,'Conv2d 1×1','8 × 80 × 80 prototypes; bias=True','conv2d');seq(p,['mfcat','mffuse','mfconv','mfout']);p.text(20,895,'Mask features are shared by all candidate instances.',16)
   q=d.panel('dynamicmask','Dynamic mask decoding after box NMS',970,3110,1490,1010,kind='plain')
   box(q,'rel',25,75,500,'Relative coordinates (2 channels)','(prior xy - grid xy) / (8 × prior stride)','plain');box(q,'proto',600,75,500,'8 prototype channels','Shared mask feature from left panel','plain');box(q,'di',305,210,530,'Concat coordinates and prototypes','N × 10 × 80 × 80; N ≤100','concat');q.wire([(275,121),(275,170),(395,170),(395,210)],start='rel',end='di');q.wire([(850,121),(850,194),(745,194),(745,210)],start='proto',end='di')
   box(q,'params',1050,215,410,'169 selected kernel parameters','Weights 80/64/8; biases 8/8/1','split')
   for j,(id,yy,inch,outch) in enumerate([('dc1',360,10,8),('dc2',535,8,8),('dc3',710,8,1)]):
    box(q,id,305,yy,530,'Dynamic Conv2d 1×1'+(' + ReLU' if j<2 else ''),f'{inch} inputs; {outch} outputs per instance; groups=N','conv2d');q.wire([(1100+j*125,261),(1100+j*125,300+j*24),(1465-j*24,300+j*24),(1465-j*24,yy+23),(835,yy+23)],start='params',end=id)
   seq(q,['di','dc1','dc2','dc3']);box(q,'maskout',305,865,1060,'Bilinear upsample ×8; invert resize; sigmoid >0.5','N ×640 ×640 masks on this canvas; coordinates use stride 8 grid','activation');q.connect('dc3','maskout',via=[(570,820),(835,820)])
  # Decode is outside the native head; exact reg transform is shown above.
  foot=base_height-155;d.text(50,foot,'Box decode: zero-offset grid points (0,8,16,... or stride 16/32) minus l/t and plus r/b. Sigmoid class logits; threshold; class NMS.',15)
  if symbolic:
   p=d.panel('familyvalues','Resolved values for this shared topology',40,base_height-95,2420,540);xs=[25,415,785,1155,1525,1895]
   for x,name in zip(xs,['Size','Stem C / half-stem','Head F','Backbone counts n0/n1/n2/n3','Neck repeats r','Regression transform']):p.text(x,55,name,14,weight=600)
   for j,sz in enumerate(group):
    dd,ww,cc,ff,rr,ee=_SIZE_CONFIG[sz];cv=int(64*ww);vals=[sz.upper(),f'{cv} / {cv//2}',str(ff),'/'.join(str(max(round(v*dd),1)) for v in [3,6,6,3]),str(rr),'ReLU × stride' if segment else 'exp × stride' if ee else 'linear × stride']
    for x,value in zip(xs,vals):p.text(x,105+j*59,value,15)
   p.text(25,455,'Backbone/neck widths are 2C,4C,8C,16C as labeled; Q is half the corresponding CSPLayer output width.',15)
  path=out/f'{vid}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{vid}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));ev.setdefault('routes',{})[vid]=routes;views.append(dict(id=vid,label=title,task=task,size='/'.join(group) if symbolic else size,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/rtmdet/{vid}.svg',html=f'/diagrams/models/rtmdet/{vid}.html',input='1 × 3 × 640 × 640',verification='cpu'));print('rtmdet',vid,routes['total_findings'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family='rtmdet',slug='rtmdet',title='RTMDet',source_revision=rev,default_view='s',views=views),indent=2)+'\n');ev.update(verification='CPU every detect/segment size; dynamic-mask probe for every segment size; no downloads',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/rtmdet.py --source /path/to/libreyolo');ep.write_text(json.dumps(ev,indent=2)+'\n')
if __name__=='__main__':main()
