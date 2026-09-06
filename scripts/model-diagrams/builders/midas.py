"""Two different MiDaS topologies: EfficientNet-Lite3 Small and ViT-L/16 DPT-Large."""
from quicksrnet import *
from depth_anything import vit_definition,dpt_topology,fusion_definitions
SMALL_GROUPS=[(32,32,24,3,1,1,32),(24,144,32,3,2,3,192),(32,192,48,5,2,3,288),(48,288,96,3,2,5,576),(96,576,136,5,1,5,816),(136,816,232,5,2,6,1392),(232,1392,384,3,1,1,1392)]

def small(a,ev):
 d=diagram(a,'MiDaS Small','Relative inverse depth,EfficientNet-Lite3,input3 × 256 × 256,native eval. Shapes exclude batch.','midas',1900,4770)
 p=d.panel('encoder','EfficientNet-Lite3 encoder',40,230,610,1560)
 rows=[('image','RGB + ImageNet normalization','3 × 256 × 256','norm'),('stem','Conv3×3,s2;BN;ReLU6','3 to32;128 × 128','conv')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,65,65+i*140,480,label,detail,kind)
 p.connect('image','stem');prev='stem';hw=128
 for j,(ci,e,co,k,s,n,er) in enumerate(SMALL_GROUPS):
  hw//=s
  op(p,f'group{j}',65,380+j*150,480,('Depthwise-separable' if j==0 else 'MBConv')+f' group,n={n}',f'{ci}/{e}/{co};k{k},firsts{s}; repeat expansion{er};{hw}²','conv',block='mbconv')
  p.connect(prev,f'group{j}');prev=f'group{j}'
 p.text(20,1480,'Taps after groups1,2,4,6:32×64²,48×32²,136×16²,384×8².',14)
 q=d.panel('decode','Projection and multiscale refinement',690,230,1170,1560)
 for j,(ci,fi,hw) in enumerate([(32,64,64),(48,128,32),(136,256,16),(384,512,8)]):
  xx=25+j*285;op(q,f'tap{j}',xx,65,250,f'Tap L{j+1}',f'{ci} × {hw} × {hw}')
  op(q,f'scratch{j}',xx,185,250,'Conv3×3,s1,p1',f'{ci} to{fi};biasFalse','conv2d');q.connect(f'tap{j}',f'scratch{j}')
  op(q,f'adapt{j}',xx,295,250,f'A{j+1}',f'{fi} × {hw} × {hw}');q.connect(f'scratch{j}',f'adapt{j}')
 for j,(fi,fo,hw) in enumerate([(512,256,16),(256,128,32),(128,64,64),(64,64,128)]):
  yy=490+j*220;op(q,f'f{j}',365,yy,540,f'FeatureFusion{4-j}',f'{fi} to{fo};bilinear×2; output{fo}×{hw}²','aggregate',block='fusion-small')
  if j:q.connect(f'f{j-1}',f'f{j}')
  op(q,f'fa{j}',100,yy+11,110,f'A{4-j}',f'{fi}ch',h=38);q.connect(f'fa{j}',f'f{j}',from_port='right',to_port='left')
 q.text(20,1450,'FeatureFusion4 has no lateral branch; other stages merge their matching adapted tap.',14)
 b=d.panel('mbdef','EfficientNet-Lite block primitives',40,1850,880,1130,kind='conv',dashed=True,block_type='mbconv')
 b.text(25,65,'Ci/expanded/Co,k,s are numeric tuples in the encoder. No squeeze-excitation in Lite3.',15)
 seq=[('exp','PointwiseConv1×1','Ci toexpanded; omitted for first depthwise-separable group','conv2d'),('bn1','BatchNorm + ReLU6','','norm'),('dw','DepthwiseConv k×k','Expanded groups; first stride s; repeated stride1','conv2d'),('bn2','BatchNorm + ReLU6','','norm'),('proj','PointwiseConv1×1','Expanded toCo','conv2d'),('bn3','BatchNorm','No projection activation','norm')]
 for j,(id,label,detail,kind) in enumerate(seq):op(b,id,170,145+j*120,620,label,detail,kind)
 chain(b,[s[0] for s in seq]);b.sum('mbsum',480,935);b.connect('bn3','mbsum');op(b,'skip',25,815,115,'Identity','Only s1,Ci=Co',h=55);b.connect('skip','mbsum',via=[(82.5,935)],to_port='left')
 b.text(25,1060,'TensorFlow SAME padding in timm preserves ceil(input/stride) dimensions.',14)
 h=d.panel('outputhead','Small output head',960,1850,900,1130)
 seq=[('h1','Conv3×3','64 to32;128 × 128','conv2d'),('hu','Bilinear resize×2','32 × 256 × 256;align_corners=False','pool'),('h2','Conv3×3','32 to32;s1,p1','conv2d'),('ha','ReLU','','activation'),('h3','Conv1×1','32 to1','conv2d'),('hr','ReLU','Nonnegative relative inverse depth','activation'),('ho','Output','1 × 256 × 256','plain')]
 for j,(id,label,detail,kind) in enumerate(seq):op(h,id,160,65+j*135,580,label,detail,kind)
 chain(h,[s[0] for s in seq])
 # Four concrete fusion contracts with common explicit RCU sequence.
 for j,(fi,fo) in enumerate([(512,256),(256,128),(128,64),(64,64)]):
  r=d.panel('fusion'+str(j),f'Fusion{4-j}: {fi} to{fo}',40+j*465,3040,425,850,kind='aggregate',dashed=True,block_type='fusion-small')
  op(r,f'f{j}top',25,65,175,'Top feature',f'{fi} channels')
  if j:
   op(r,f'f{j}lat',230,65,175,'Lateral A'+str(4-j),f'{fi} channels')
   op(r,f'f{j}rcu1',230,180,175,'ResidualConvUnit',f'{fi} channels','conv',block='midas-rcu'+str(fi));r.connect(f'f{j}lat',f'f{j}rcu1')
   r.sum(f'f{j}add',215,345);r.wire([(112.5,114),(112.5,345),(202,345)],start=f'f{j}top',end=f'f{j}add');r.wire([(317.5,229),(317.5,345),(228,345)],start=f'f{j}rcu1',end=f'f{j}add');prev=f'f{j}add'
  else:prev=f'f{j}top'
  for k,(id,label,detail,kind) in enumerate([('rcu2','ResidualConvUnit',f'{fi} channels','conv'),('up','Bilinear resize×2','align_corners=True','pool'),('out','Conv1×1',f'{fi} to{fo};biasTrue','conv2d')]):
   nid=f'f{j}'+id;op(r,nid,35,455+k*110,355,label,detail,kind,block='midas-rcu'+str(fi) if id=='rcu2' else '');r.connect(prev,nid);prev=nid
  rr=d.panel('rcu-small'+str(j),f'ResidualConvUnit {fi}',40+j*465,3950,425,660,kind='conv',dashed=True,block_type='midas-rcu'+str(fi))
  seq=[('x','Input',f'{fi} channels','plain'),('a1','ReLU','','activation'),('c1','Conv3×3',f'{fi} to{fi};s1,p1,biasTrue','conv2d'),('a2','ReLU','','activation'),('c2','Conv3×3',f'{fi} to{fi};s1,p1,biasTrue','conv2d')]
  for z,(nid,label,detail,kind) in enumerate(seq):op(rr,f'srcu{j}'+nid,65,60+z*90,325,label,detail,kind)
  chain(rr,[f'srcu{j}'+n[0] for n in seq]);rr.sum(f'srcu{j}sum',227.5,570);rr.connect(f'srcu{j}c2',f'srcu{j}sum');rr.connect(f'srcu{j}x',f'srcu{j}sum',from_port='left',to_port='left',via=[(25,84.5),(25,570)])
 rec=ev['records'].get('s',{});return finish_view(a,d,'midas','s-depth','Small EfficientNet-Lite3','depth','s','concrete','3×256×256',rec.get('device','source'))

def large(a,ev):
 d=diagram(a,'MiDaS DPT-Large','Relative inverse depth,ViT-L/16,input3 × 384 × 384,native eval. Shapes exclude batch.','midas',1900,4490)
 p=d.panel('encoder','ViT-L/16 backbone and CLS readout',40,230,570,1760)
 rows=[('image','Normalize(x-0.5)/0.5','3 × 384 × 384','norm'),('patch','Conv patch16,stride16','1024 × 24 × 24','conv2d'),('pos','CLS + learned positional embedding','577 × 1024','aggregate')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,65,65+i*135,440,label,detail,kind)
 chain(p,[r[0] for r in rows]);prev='pos'
 for j,end in enumerate([5,11,17,23]):
  yy=500+j*245;op(p,f'blk{j}',65,yy,440,'Transformer blocks,n=6',f'End atindex{end};16 heads;1024 width','attention',block='vit')
  if j:p.connect(prev,f'blk{j}',from_port='right',to_port='right',via=[(540,yy-220.5),(540,yy+24.5)])
  else:p.connect(prev,f'blk{j}')
  op(p,f'readout{j}',65,yy+115,440,'CLS-conditioned readout + reshape',f'T{j+1}:1024 × 24 × 24','linear',block='readout');p.connect(f'blk{j}',f'readout{j}');prev=f'blk{j}'
 p.text(20,1600,'Taps are captured before model.norm. That final norm executes',14);p.text(20,1630,'but its output is not used by MiDaS DPT-Large.',14)
 dpt_topology(d,1024,256,[256,512,1024,1024],24,384)
 vit_definition(d,1024,16,577,y=2070,eps='1e-6',layerscale=False)
 h=d.panel('head','Large output head',1300,2070,560,1420)
 rows=[('c1','Conv3×3','256 to128;192 × 192','conv2d'),('up','Bilinear resize×2','128 × 384 × 384;align_corners=True','pool'),('c2','Conv3×3','128 to32;s1,p1','conv2d'),('a','ReLU','','activation'),('c3','Conv1×1','32 to1','conv2d'),('a2','ReLU','','activation'),('out','Relative inverse depth','1 × 384 × 384','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(h,'lh'+id,65,65+j*175,430,label,detail,kind)
 chain(h,['lh'+s[0] for s in rows]);fusion_definitions(d,256,y=3600)
 r=d.panel('readoutdef','CLS readout',1450,3600,410,760,kind='linear',dashed=True,block_type='readout')
 rows=[('broadcast','Broadcast CLS to576 patch positions','Each token1024 channels','plain'),('cat','Concat patch and CLS','576 × 2048','concat'),('linear','Linear2048 to1024','','linear'),('gelu','GELU','','activation'),('map','Restore1024 × 24 × 24','Then DPT projection and resampling','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(r,'ro'+id,25,65+j*125,360,label,detail,kind)
 chain(r,['ro'+s[0] for s in rows]);rec=ev['records'].get('l',{});return finish_view(a,d,'midas','l-depth','Large ViT-L/16','depth','l','concrete','3×384×384',rec.get('device','source'))

def main():
 a=environment('Build both structurally different MiDaS variants')
 if a.verify:
  import torch,timm
  torch.set_num_threads(4);nn=nn_module(a,'midas')
  recs={'s':cpu_probe(nn.MiDaSSmall(),(1,3,256,256),['pretrained.layer1','pretrained.layer2','pretrained.layer3','pretrained.layer4','scratch.refinenet1','scratch.output_conv'])}
  m=construct_meta(nn.DPTLarge);recs['l']=meta_probe(m,(1,3,384,384),['pretrained.act_postprocess1','pretrained.act_postprocess2','pretrained.act_postprocess3','pretrained.act_postprocess4','scratch.output_conv']);recs['timm_version']=timm.__version__
  write_evidence(a,'midas',recs,['libreyolo/models/midas/nn.py:MiDaSSmall, DPTLarge, _make_efficientnet_lite3, ProjectReadout','timm==1.0.28:tf_efficientnet_lite3 and vit_large_patch16_384 (Apache-2.0)'],['Installedtimm1.0.28 with --no-deps; no existingpackagesupgraded.','Small checkedCPU256; Large checkedmeta384. No weights downloaded.','Small decoder progressively halves512 to256 to128 to64; Large stays256.','Large uses CLS-conditioned projectedreadout; source taps before finalnorm. Small is a distinctCNN topology, so no misleading shared family graph.'])
 ev=read_evidence('midas');manifest(a,'midas','midas','MiDaS',[small(a,ev),large(a,ev)])
if __name__=='__main__':main()
