"""MoGe-2 normal-only graph: summed DINOv2 taps and two UV-conditioned ConvStacks."""
from quicksrnet import *
from depth_anything import vit_definition
CFG={'s':(384,6,2,1),'b':(768,12,2,1),'l':(1024,16,4,2)}

def build(a,size,ev):
 sym=size=='family';D,heads,nt,nr=('D','A','T','R') if sym else CFG[size]
 dims=[D,256,128,64,32]
 d=diagram(a,'MoGe-2 normal family' if sym else 'MoGe-2 '+size.upper(),'Surface-normal prediction, input3 × 518 × 518 RGB,native eval. Shapes exclude batch.','moge2',1900,4790)
 p=d.panel('encoder','DINOv2 feature encoder',40,230,570,1960)
 rows=[('input','Resize and ImageNet normalization','518 × 518;bilinear,antialias=True','norm'),('patch','Conv patch14,stride14',f'3 to{D};37 × 37 patches','conv2d'),('cls','Prepend CLS + learned positions',f'1370 × {D}; no registers','aggregate'),('vit',f'Transformer blocks,n={"6T" if sym else 6*nt}',f'{heads} heads; taps every6 completed blocks','attention')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,65,65+i*190,440,label,detail,kind,block='vit' if id=='vit' else '')
 chain(p,[r[0] for r in rows])
 op(p,'taps',65,875,440,f'Take{nt} taps; LayerNorm; discard CLS',f'S/B:[5,11]; L:[5,11,17,23]; {D}×37²','norm');p.connect('vit','taps')
 op(p,'projections',65,1080,440,'Independent Conv1×1 per tap',f'{D} to{D}; each projection has own parameters','conv2d');p.connect('taps','projections')
 op(p,'stack',65,1270,440,'Stack projected maps along tap dimension',f'{nt} × {D} × 37 × 37','concat');p.connect('projections','stack')
 op(p,'sum',65,1460,440,'Sum over tap dimension',f'Feature F: {D} × 37 × 37','aggregate');p.connect('stack','sum')
 p.text(25,1750,'The normal-only graph omits points,mask andmetric-scale heads.',14)
 p.text(25,1800,'Last CLS token is returned by encoder, then ignored by this head.',13)
 for ishead,x in [(False,650),(True,1260)]:
  name='Normal head ConvStack' if ishead else 'UV-conditioned neck ConvStack';q=d.panel('head' if ishead else 'neck',name,x,230,600,1960)
  prefix='h' if ishead else 'n';previous=None
  for j,ch in enumerate(dims):
   hw=37*2**j;yy=65+j*300;cin=ch if ishead else (D+2 if isinstance(D,int) else 'D+2') if j==0 else 2
   inp=f'N{j}' if ishead else 'F concat UV0' if j==0 else f'UV{j}'
   op(q,prefix+'in'+str(j),25,yy,135,inp,f'{cin}channels',h=49)
   op(q,prefix+'proj'+str(j),215,yy,350,'Input Conv1×1',f'{cin} to{ch};{hw} × {hw}','conv2d');q.connect(prefix+'in'+str(j),prefix+'proj'+str(j),from_port='right',to_port='left')
   if j:
    q.sum(prefix+'add'+str(j),390,yy+108);q.connect(prefix+'proj'+str(j),prefix+'add'+str(j));q.connect(previous,prefix+'add'+str(j),from_port='right',to_port='right',via=[(580,yy-45.5),(580,yy+108)]);prev=prefix+'add'+str(j)
   else:prev=prefix+'proj0'
   count=0 if j in [0,4] else 1 if ishead else nr
   op(q,prefix+'res'+str(j),215,yy+155,350,f'ResidualConvBlock,n={count}',f'{ch} × {hw} × {hw}; '+('H'+str(j) if ishead else 'N'+str(j)),'conv' if count!=0 else 'plain',block='rcu'+str(ch) if count!=0 else '');q.connect(prev,prefix+'res'+str(j))
   if j<4:
    op(q,prefix+'up'+str(j),215,yy+230,350,'Resampler×2',f'{ch} to{dims[j+1]}; output{hw*2}²','aggregate',block='resample');q.connect(prefix+'res'+str(j),prefix+'up'+str(j));previous=prefix+'up'+str(j)
  if ishead:
   for j,(id,label,detail,kind) in enumerate([('pred','Output Conv1×1','32 to3;592 × 592','conv2d'),('resize','Bilinear resize to518 × 518','align_corners=False,antialias=False','pool'),('unit','Normalize3-channel vectors','3 × 518 × 518 unit normals','norm')]):op(q,id,95,1615+j*105,450,label,detail,kind)
   chain(q,['hres4','pred','resize','unit'])
  else:q.text(20,1850,'Every Ni is retained as an independent normal-head input.',13)
 vit_definition(d,D,heads,1370,y=2250)
 r=d.panel('resample','Resampler and UV geometry',1300,2250,560,1420,kind='aggregate',dashed=True,block_type='resample')
 op(r,'tr',60,65,440,'First3 transitions:ConvTranspose2×2','D to256,256to128,128to64;stride2','conv2d')
 op(r,'tc',60,225,440,'Conv3×3,replicate padding1','Output channels256,128,64 respectively','conv2d');r.connect('tr','tc')
 op(r,'bu',60,445,440,'Last transition:bilinear resize×2','64 channels;align_corners=False','pool')
 op(r,'bc',60,605,440,'Conv3×3,replicate padding1','64 to32','conv2d');r.connect('bu','bc')
 r.text(25,830,'Both ConvStacks use these exact four resamplers.',14)
 r.text(25,910,'UV levels:37²,74²,148²,296²,592²;2 coordinate channels.',14)
 r.text(25,970,'For square input, horizontal/vertical span is1/sqrt(2).',14)
 r.text(25,1030,'Endpoints are ±span×(resolution-1)/resolution.',14)
 r.text(25,1110,'UV0 concatenates with image features; later neck levels',14)
 r.text(25,1140,'receive coordinates alone before learned projection.',14)
 r.text(25,1260,'All ConvStack input blocks are Conv1×1, even when Cin=Co.',13)
 for j,ch in enumerate([256,128,64]):
  c=d.panel('rcu'+str(ch),f'ResidualConvBlock {ch} channels',40+j*620,3730,590,870,kind='conv',dashed=True,block_type='rcu'+str(ch))
  rows=[('in','Input',f'{ch} channels','plain'),('relu1','ReLU','','activation'),('c1','Conv3×3',f'{ch} to{ch};replicate padding1','conv2d'),('relu2','ReLU','','activation'),('c2','Conv3×3',f'{ch} to{ch};replicate padding1','conv2d')]
  for z,(id,label,detail,kind) in enumerate(rows):op(c,f'r{ch}'+id,95,65+z*120,430,label,detail,kind)
  chain(c,[f'r{ch}'+x[0] for x in rows]);c.sum(f'r{ch}sum',310,705);c.connect(f'r{ch}c2',f'r{ch}sum');c.connect(f'r{ch}in',f'r{ch}sum',from_port='left',to_port='left',via=[(30,89.5),(30,705)])
  c.text(25,820,'No normalization in the configured neck/head residual blocks.',13)
 if sym:d.text(50,4660,'s:D384,A6,T2,R1. b:D768,A12,T2,R1. l:D1024,A16,T4,R2. Normal-head middle levels always have one block.',15)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'moge-2','family' if sym else size+'-normal','Shared normal-only topology' if sym else size,'normal',size,'family' if sym else 'concrete','3×518×518',rec.get('device','source'))

def main():
 a=environment('Build all MoGe-2 normal-only sizes')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'moge2');records={}
  for size in CFG:
   names=['encoder','neck','normal_head','normal_head.output_blocks.4']
   if size in ['s','b']:records[size]=cpu_probe(nn.MoGe2NormalNet(size),(1,3,518,518),names)
   else:records[size]=meta_probe(construct_meta(lambda:nn.MoGe2NormalNet(size)),(1,3,518,518),names)
  write_evidence(a,'moge2',records,['libreyolo/models/moge2/nn.py:MoGe2NormalNet,DINOv2Encoder,ConvStack,Resampler,ResidualConvBlock','libreyolo/models/moge2/model.py:SUPPORTED_TASKS'],['s/bCPU,lmeta. This family predictsnormal maps,not depth.','Every ConvStack input block isConv1×1 evenwhenchannel counts match.','Encoder tap projections arestacked/summed,not concatenated. Normal field isresized592 to518 then L2normalized over3 channels.'])
 ev=read_evidence('moge2');views=[build(a,s,ev) for s in CFG]+[build(a,'family',ev)];manifest(a,'moge2','moge-2','MoGe-2',views)
if __name__=='__main__':main()
