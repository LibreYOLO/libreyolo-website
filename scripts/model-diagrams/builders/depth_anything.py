"""Depth Anything V2: DINOv2 plus numerically resolved DPT pyramid."""
from quicksrnet import *
from nafnet import product
CONFIG={'s':(384,12,6,64,[48,96,192,384],[2,5,8,11]),'b':(768,12,12,128,[96,192,384,768],[2,5,8,11]),'l':(1024,24,16,256,[256,512,1024,1024],[4,11,17,23]),'g':(1536,40,24,384,[1536]*4,[9,19,29,39])}

def vit_definition(d,D,heads,tokens,giant=False,y=2100,eps='1e-6',layerscale=True):
 p=d.panel('vitdef','Transformer block',40,y,580,1420,kind='attention',dashed=True,block_type='vit')
 rows=[('tx','Input tokens',f'{tokens} × {D}','plain'),('tn1','LayerNorm',f'{D} channels; epsilon{eps}','norm'),('ta','Multihead self-attention',f'{heads} heads; head width64','attention')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,100,65+i*105,380,label,detail,kind,block='mha' if id=='ta' else '')
 chain(p,[r[0] for r in rows]);prev='ta'
 if layerscale:op(p,'tls1',100,395,380,'LayerScale',f'{D} learned channel scalars','linear');p.connect(prev,'tls1');prev='tls1'
 p.sum('tadd1',290,520);p.connect(prev,'tadd1');p.connect('tx','tadd1',from_port='left',to_port='left',via=[(35,89.5),(35,520)])
 op(p,'tn2',100,615,380,'LayerNorm',f'{D} channels; epsilon{eps}','norm');p.connect('tadd1','tn2')
 if giant:
  op(p,'tffn',100,735,380,'SwiGLU FFN','1536 input; gated width4096','aggregate',block='swiglu');p.connect('tn2','tffn');prev='tffn'
 else:
  ff=4*D if isinstance(D,int) else '4D'
  for i,(id,label,detail,kind) in enumerate([('tf1','Linear',f'{D} to {ff}','linear'),('tg','GELU','','activation'),('tf2','Linear',f'{ff} to {D}','linear')]):op(p,id,100,735+i*95,380,label,detail,kind)
  chain(p,['tn2','tf1','tg','tf2']);prev='tf2'
 if layerscale:op(p,'tls2',100,1060,380,'LayerScale',f'{D} learned channel scalars','linear');p.connect(prev,'tls2');prev='tls2'
 p.sum('tadd2',290,1190);p.connect(prev,'tadd2');p.dot(290,575);p.wire([(290,575),(35,575),(35,1190),(277,1190)],start='tadd1',end='tadd2')
 op(p,'to',100,1280,380,'Output tokens',f'{tokens} × {D}');p.connect('tadd2','to')
 q=d.panel('attentiondef','Self-attention primitives',660,y,600,1420,kind='attention',dashed=True,block_type='mha')
 qkv=3*D if isinstance(D,int) else '3D'
 steps=[('aqkv','Fused QKV linear',f'{D} to {qkv}; biases enabled','linear'),('asplit','Split Q, K and V',f'{heads} heads; {tokens} tokens;64 channels/head','split')]
 for i,(id,label,detail,kind) in enumerate(steps):op(q,id,80,70+i*125,440,label,detail,kind)
 chain(q,['aqkv','asplit'])
 op(q,'aqk',35,365,310,'Q × transpose(K) / 8',f'{tokens} × {tokens} per head','attention')
 op(q,'av',415,365,145,'Values V','64/head','plain')
 q.wire([(220,244),(220,312),(190,312),(190,365)],start='asplit',end='aqk');q.wire([(465,244),(465,312),(487.5,312),(487.5,365)],start='asplit',end='av')
 op(q,'asoft',35,555,310,'Softmax over keys','','attention');q.connect('aqk','asoft')
 op(q,'amul',100,750,440,'Attention weights × V',f'{heads} heads;64 output channels/head','attention')
 q.wire([(190,604),(190,710),(220,710),(220,750)],start='asoft',end='amul');q.wire([(487.5,414),(487.5,710),(430,710),(430,750)],start='av',end='amul')
 op(q,'acat',100,955,440,'Concat heads',f'{tokens} × {D}','concat');q.connect('amul','acat')
 op(q,'aproj',100,1160,440,'Output linear',f'{D} to {D}','linear');q.connect('acat','aproj')
 q.text(25,1360,'No causal mask. Dropout and stochastic depth are inactive in eval.',14)
 if giant:
  s=d.panel('swiglu','Giant SwiGLU FFN',1300,y,560,1420,kind='aggregate',dashed=True,block_type='swiglu')
  for i,(id,label,detail,kind) in enumerate([('gi','Input','1536 channels','plain'),('gw','Linear fused w12','1536 to 8192','linear'),('gsplit','Split two equal halves','4096 channels each','split')]):op(s,id,90,70+i*140,380,label,detail,kind)
  chain(s,['gi','gw','gsplit']);op(s,'gact',25,560,230,'SiLU on first half','4096 channels','activation');op(s,'gidentity',310,560,230,'Second half','4096 channels','plain')
  s.wire([(200,399),(200,490),(140,490),(140,560)],start='gsplit',end='gact');s.wire([(385,399),(385,490),(425,490),(425,560)],start='gsplit',end='gidentity')
  product(s,'gmul',280,805);s.wire([(140,609),(140,805),(267,805)],start='gact',end='gmul');s.wire([(425,609),(425,805),(293,805)],start='gidentity',end='gmul')
  op(s,'gout',90,1010,380,'Linear w3','4096 to1536','linear');s.connect('gmul','gout')
  s.text(25,1260,'Hidden width rounds (4×1536)×2/3 to a multiple of8.',14)

def fusion_definitions(d,F,y=3600):
 r=d.panel('rcudef','ResidualConvUnit',40,y,570,760,kind='conv',dashed=True,block_type='rcu')
 rows=[('ri','Input',f'{F} channels','plain'),('ra1','ReLU','','activation'),('rc1','Conv2d 3×3',f'{F} to {F}; s1,p1,bias=True','conv2d'),('ra2','ReLU','','activation'),('rc2','Conv2d 3×3',f'{F} to {F}; s1,p1,bias=True','conv2d')]
 for i,(id,label,detail,kind) in enumerate(rows):op(r,id,100,65+i*95,370,label,detail,kind)
 chain(r,[n[0] for n in rows]);r.sum('radd',285,610);r.connect('rc2','radd');r.connect('ri','radd',from_port='left',to_port='left',via=[(35,89.5),(35,610)])
 r.text(25,705,'No BatchNorm in these DPT fusion units.',14)
 p=d.panel('fusiondef','FeatureFusionBlock with lateral input',650,y,760,760,kind='aggregate',dashed=True,block_type='fusion')
 op(p,'top',30,65,300,'Top-down feature',f'{F} channels');op(p,'lateral',430,65,300,'Lateral feature',f'{F} channels')
 op(p,'res1',430,170,300,'ResidualConvUnit',f'{F} channels','conv',block='rcu');p.connect('lateral','res1');p.sum('fadd',380,310)
 p.wire([(180,114),(180,310),(367,310)],start='top',end='fadd');p.wire([(580,219),(580,310),(393,310)],start='res1',end='fadd')
 op(p,'res2',180,395,400,'ResidualConvUnit',f'{F} channels','conv',block='rcu');p.connect('fadd','res2')
 op(p,'fup',180,495,400,'Bilinear resize','Target size shown at each occurrence; align_corners=True','pool');p.connect('res2','fup')
 op(p,'fconv',180,595,400,'Conv2d 1×1',f'{F} to {F}; bias=True','conv2d');p.connect('fup','fconv')
 p.text(25,712,'Deepest block has no lateral input and skips the first RCU/add.',14)


def dpt_topology(d,D,F,proj,grid,input_size,y=230,sky=False):
 sizes=[grid*4,grid*2,grid,(grid+1)//2]
 q=d.panel('dpt','DPT adapters and fusion',650,y,1210,1760)
 for i in range(4):
  x=25+i*298
  op(q,f'tap{i}',x,65,265,f'Tap T{i+1}',f'{D} × {grid} × {grid}')
  op(q,f'proj{i}',x,175,265,'Conv2d 1×1',f'{D} to {proj[i]}','conv2d');q.connect(f'tap{i}',f'proj{i}')
  lab=['ConvTranspose2d 4×4','ConvTranspose2d 2×2','Identity','Conv2d 3×3'][i]
  detail=[f's4,p0; {proj[i]} channels',f's2,p0; {proj[i]} channels',f'{proj[i]} channels',f's2,p1; {proj[i]} channels'][i]
  op(q,f'rs{i}',x,285,265,lab,detail,'plain' if i==2 else 'conv2d');q.connect(f'proj{i}',f'rs{i}')
  op(q,f'scratch{i}',x,395,265,'Conv2d 3×3',f'{proj[i]} to {F}; {sizes[i]} × {sizes[i]}; bias=False','conv2d');q.connect(f'rs{i}',f'scratch{i}')
  op(q,f'lat{i}',x+80,475,105,f'L{i+1}',f'{F} ch',h=40);q.connect(f'scratch{i}',f'lat{i}')
 for j,i in enumerate([3,2,1,0]):
  target=[grid,grid*2,grid*4,grid*8][j]
  yy=615+j*235
  op(q,f'fusion{j}',360,yy,490,f'FeatureFusionBlock {i+1}',f'{F} channels; output {target} × {target}','aggregate',block='fusion')
  if j==0:
   op(q,'start-L4',100,yy+11,120,'L4',f'{F} channels',h=38);q.connect('start-L4','fusion0',from_port='right',to_port='left')
  else:
   q.connect(f'fusion{j-1}',f'fusion{j}');op(q,f'use-L{i+1}',100,yy+11,120,f'L{i+1}',f'{F} channels',h=38);q.connect(f'use-L{i+1}',f'fusion{j}',from_port='right',to_port='left')
 q.text(25,1665,'L1...L4 are named continuations of the independent adapted feature maps.',14)
 q.text(25,1700,'First fusion resizes to the exact next grid, including odd-grid rounding.',14)
 return q

def build(a,size,ev):
 sym=size=='family'
 D,N,H,F,proj,taps=('D','N','A','F',['P1','P2','P3','P4'],['I1','I2','I3','I4']) if sym else CONFIG[size]
 d=diagram(a,'Depth Anything V2 family' if sym else 'Depth Anything V2 '+size.upper(),'Relative inverse depth, native eval, input 3 × 518 × 518. Shapes exclude batch.','depth_anything',1900,4610 if sym else 4490)
 p=d.panel('backbone','DINOv2 image encoder',40,230,570,1760)
 rows=[('image','RGB + ImageNet normalization','3 × 518 × 518','norm'),('patch','Conv2d patch embedding 14×14',f'3 to {D}; stride14; 37 × 37 grid','conv2d'),('token','Prepend CLS + add learned positions',f'1370 × {D}; no register tokens','aggregate')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,70,65+i*125,430,label,detail,kind)
 chain(p,[r[0] for r in rows]);prev='token';last=-1
 for j,idx in enumerate(taps):
  count=('I1+1' if j==0 else f'I{j+1}-I{j}') if sym else idx-last
  yy=485+j*225
  op(p,f'vit{j}',70,yy,430,f'Transformer blocks, n={count}',f'End at zero-based block {idx}; {H} heads','attention',block='vit')
  if j==0:p.connect(prev,f'vit{j}')
  else:p.connect(prev,f'vit{j}',from_port='right',to_port='right',via=[(540,yy-200.5),(540,yy+24.5)])
  op(p,f'tapout{j}',70,yy+95,430,'LayerNorm; remove CLS',f'T{j+1}: {D} × 37 × 37','norm');p.connect(f'vit{j}',f'tapout{j}')
  if j>0:
   # The next segment consumes pre-tap-normalization block output.
   pass
  prev=f'vit{j}';last=idx
 # Correct the repeated-segment connections to bypass tap-only normalizations.
 p.text(25,1485,'Tap normalization does not replace the ongoing token stream.',14)
 p.text(25,1525,'CLS is returned by the backbone but ignored by this DPT head.',14)
 p.text(25,1575,'s/b/l use GELU MLP; g uses SwiGLU width4096.',14)
 p.text(25,1625,'Learned position grid is already37×37 at this input size.',14)
 p.text(25,1675,'All LayerScale parameters initialize to1.0.',14)
 dpt_topology(d,D,F,proj,37,518)
 vit_definition(d,D,H,1370,giant=size=='g',y=2070)
 if size!='g':
  h=d.panel('depthhead','DPT output head',1300,2070,560,1420)
 else:h=d.panel('depthhead','DPT output head',1450,3600,410,760)
 half=F//2 if isinstance(F,int) else 'F/2'
 nodes=[('h1','Conv2d 3×3',f'{F} to {half}; 296 × 296','conv2d'),('hresize','Bilinear resize','518 × 518; align_corners=True','pool'),('h2','Conv2d 3×3',f'{half} to32; s1,p1','conv2d'),('hrelu','ReLU','','activation'),('h3','Conv2d 1×1','32 to1','conv2d'),('hrelu2','ReLU + final ReLU','Nonnegative inverse depth','activation'),('hout','Relative inverse depth','1 × 518 × 518','plain')]
 for i,(id,label,detail,kind) in enumerate(nodes):op(h,id,50,65+i*(92 if size=='g' else 160),310 if size=='g' else 460,label,detail,kind)
 chain(h,[n[0] for n in nodes]);fusion_definitions(d,F,y=3600)
 if sym:
  t=d.panel('variants','Family values',40,4380,1820,130)
  t.text(20,38,'s: D384,N12,A6,F64,P=[48,96,192,384], I=[2,5,8,11].   b: D768,N12,A12,F128,P=[96,192,384,768], same I.',14)
  t.text(20,72,'l: D1024,N24,A16,F256,P=[256,512,1024,1024], I=[4,11,17,23].',14)
  t.text(20,106,'g: D1536,N40,A24,F384,P=[1536,1536,1536,1536], I=[9,19,29,39]; SwiGLU replaces GELU MLP (see g view).',14)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'depth-anything-v2','family' if sym else size+'-depth','Family topology with explicit g FFN exception' if sym else size,'depth',size,'family' if sym else 'concrete','3×518×518',rec.get('device','source'))

def main():
 a=environment('Build all Depth Anything V2 native encoders')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'depth_anything');records={}
  for size,encoder in [('s','vits'),('b','vitb'),('l','vitl'),('g','vitg')]:
   names=['pretrained.patch_embed','depth_head.resize_layers.0','depth_head.resize_layers.1','depth_head.resize_layers.2','depth_head.resize_layers.3','depth_head.scratch.refinenet1','depth_head.scratch.output_conv2']
   if size in ['s','b']:records[size]=cpu_probe(nn.LibreDepthAnythingV2Net(encoder),(1,3,518,518),names)
   else:
    m=construct_meta(lambda:nn.LibreDepthAnythingV2Net(encoder))
    records[size]=meta_probe(m,(1,3,518,518),names)
  write_evidence(a,'depth_anything',records,['libreyolo/models/depth_anything/nn.py:DEPTH_ANYTHING_V2_CONFIGS','libreyolo/models/depth_anything/_vendor/dpt.py:DepthAnythingV2 and DPTHead','libreyolo/models/depth_anything/_vendor/dinov2.py:DINOv2 and get_intermediate_layers','libreyolo/models/depth_anything/_vendor/util/blocks.py:ResidualConvUnit and FeatureFusionBlock'],['s/b CPU forward; l/g meta shape propagation. Constructor-only torch.linspace stochastic-depth constants materialized on CPU to allow .item(), parameters remain meta. No weights.','37×37 patch grid produces adapters148,74,37,19; fusion uses exact resize19 to37 rather than assuming power-of-two18.','Giant has SwiGLU; other sizes GELU MLP. Source code Apache-2.0 according to repository THIRD_PARTY_NOTICES; weight licenses are separate.'])
 ev=read_evidence('depth_anything');views=[build(a,s,ev) for s in CONFIG]+[build(a,'family',ev)];manifest(a,'depth_anything','depth-anything-v2','Depth Anything V2',views)
if __name__=='__main__':main()
