"""LingBot-Vision semantic linear probe with axial rotary transformer internals."""
from quicksrnet import *
from nafnet import product
CONFIGS={'s':(384,12,6),'b':(768,12,12),'l':(1024,24,16),'g':(1536,40,24)}

def build(a,size,ev):
 sym=size=='family';D,N,H=('D','N','A') if sym else CONFIGS[size];ff='4D' if sym else 4*D;qkv='3D' if sym else 3*D
 d=diagram(a,'LingBot-Vision family' if sym else 'LingBot-Vision '+size.upper(),'Semantic segmentation, 150 classes, native eval, input 3 × 512 × 512. Shapes exclude batch.','lingbotvision',1800,2980)
 p=d.panel('network','Backbone and semantic probe',40,230,540,1860)
 rows=[('input','RGB + ImageNet normalization','3 × 512 × 512','norm'),('patch','Conv2d patch embedding 16×16',f'3 to {D}; stride16; 32 × 32 patch grid','conv2d'),('flatten','Flatten 1024 patches',f'1024 × {D}','plain'),('prefix','Prepend CLS + 4 storage tokens',f'1029 × {D}; learnable prefix','concat'),('blocks',f'Self-attention block, n={N}',f'{H} heads; head dimension64','attention'),('ln','Final LayerNorm',f'{D} channels; epsilon1e-5','norm'),('select','Select patch tokens',f'Discard CLS and 4 storage tokens; 1024 × {D}','split'),('grid','Restore patch grid',f'{D} × 32 × 32','plain'),('head','Conv2d 1×1 linear probe',f'{D} to 150; bias=True','conv2d'),('output','semantic_logits','150 × 32 × 32','plain')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,70,70+i*150,400,label,detail,kind,block='vit' if id=='blocks' else '')
 chain(p,[r[0] for r in rows])
 p.text(25,1615,'Network forward returns patch-grid logits in a dictionary.',14)
 p.text(25,1650,'Caller-level resizing is outside this network graph.',14)
 p.text(25,1710,'Mask token is stored for checkpoint compatibility.',14)
 p.text(25,1740,'Masked-token pretraining is not executed by this port.',14)
 p.text(25,1800,'g is implemented but unpublished in the public family.',14)
 b=d.panel('block','Self-attention block',620,230,550,1860,kind='attention',dashed=True,block_type='vit')
 rows=[('x','Input tokens',f'1029 × {D}','plain'),('n1','LayerNorm',f'{D} channels; epsilon1e-5','norm'),('att','Rotary self-attention',f'{H} heads, width64','attention'),('ls1','LayerScale',f'{D} learned scalars; init1e-5','linear')]
 for i,(id,label,detail,kind) in enumerate(rows):op(b,id,95,70+i*125,370,label,detail,kind,block='attention' if id=='att' else '')
 chain(b,[r[0] for r in rows]);b.sum('add1',280,620);b.connect('ls1','add1');b.connect('x','add1',from_port='left',to_port='left',via=[(30,94.5),(30,620)])
 op(b,'n2',95,720,370,'LayerNorm',f'{D} channels; epsilon1e-5','norm');b.connect('add1','n2')
 if size=='g':
  op(b,'ff',95,865,370,'SwiGLU FFN','1536 input; 4096 gated width','aggregate',block='swiglu');b.connect('n2','ff');prev='ff'
 else:
  for i,(id,label,detail,kind) in enumerate([('fc1','Linear',f'{D} to {ff}','linear'),('gelu','GELU','','activation'),('fc2','Linear',f'{ff} to {D}','linear')]):op(b,id,95,865+i*125,370,label,detail,kind)
  chain(b,['n2','fc1','gelu','fc2']);prev='fc2'
 op(b,'ls2',95,1300,370,'LayerScale',f'{D} learned scalars; init1e-5','linear');b.connect(prev,'ls2')
 b.sum('add2',280,1450);b.connect('ls2','add2');b.dot(280,678);b.wire([(280,678),(30,678),(30,1450),(267,1450)],start='add1',end='add2')
 op(b,'bo',95,1570,370,'Output tokens',f'1029 × {D}');b.connect('add2','bo')
 b.text(25,1720,'No stochastic depth or dropout in this inference port.',14)
 b.text(25,1760,'s/b/l use GELU MLP; g uses the separately drawn SwiGLU.',14)
 q=d.panel('attention','Rotary self-attention',1210,230,550,1860,kind='attention',dashed=True,block_type='attention')
 op(q,'qkv',80,70,390,'Fused QKV linear',f'{D} to {qkv}; '+('no bias' if size=='g' else 'K bias masked to zero'),'linear')
 op(q,'split',80,195,390,'Split Q, K, V and heads',f'{H} × 1029 × 64 for each tensor','split');q.connect('qkv','split')
 op(q,'qkrope',35,340,250,'Apply RoPE to Q and K','Only 1024 patch tokens; keep prefix','attention',block='rope')
 op(q,'v',355,340,150,'V unchanged','1029 tokens','plain')
 q.wire([(200,244),(200,292),(160,292),(160,340)],start='split',end='qkrope');q.wire([(400,244),(400,292),(430,292),(430,340)],start='split',end='v')
 op(q,'qk',35,505,250,'Q × transpose(K) / 8','Each head: 1029 × 1029','attention');q.connect('qkrope','qk')
 op(q,'soft',35,670,250,'Softmax over keys','All prefix and patch tokens attend','attention');q.connect('qk','soft')
 op(q,'weighted',110,850,340,'Attention weights × V','64 channels per head','attention')
 q.wire([(160,719),(160,804),(190,804),(190,850)],start='soft',end='weighted');q.wire([(430,389),(430,804),(370,804),(370,850)],start='v',end='weighted')
 op(q,'concat',110,1020,340,'Concat heads',f'1029 × {D}','concat');q.connect('weighted','concat')
 op(q,'project',110,1190,340,'Output linear',f'{D} to {D}; bias=True','linear');q.connect('concat','project')
 q.text(25,1395,'Q/K rotation uses fp32 sine/cosine tables.',14)
 q.text(25,1435,'Q/K convert back to input dtype before attention.',14)
 q.text(25,1520,'Head dimension is64 for every registered size.',14)
 q.text(25,1620,'Q,V biases exist for s/b/l; K bias is masked to zero.',14)
 q.text(25,1660,'The giant g configuration disables fused QKV bias.',14)
 r=d.panel('rope','Axial RoPE on patch tokens',40,2150,1080,680,kind='attention',dashed=True,block_type='rope')
 for i,(id,label,detail,kind) in enumerate([('coords','32×32 patch-center coordinates','Normalize each axis independently to [-1,1]','plain'),('freq','Periods and angles','16 frequencies per axis; base100; 2π × coordinate / period','linear'),('sin','Sine and cosine tables','1024 × 64; duplicate axial angle vector','activation')]):op(r,id,45,65+i*140,460,label,detail,kind)
 chain(r,['coords','freq','sin'])
 op(r,'rotate',600,65,430,'Rotate half of Q or K','Split [x1,x2], concatenate [-x2,x1]','split')
 op(r,'cosmul',600,235,430,'Multiply original x by cosine','Elementwise','linear')
 op(r,'sinmul',600,365,430,'Multiply rotated x by sine','Elementwise','linear');r.connect('rotate','sinmul',from_port='right',to_port='right',via=[(1050,89.5),(1050,389.5)])
 r.sum('ropeadd',815,520);r.connect('sinmul','ropeadd');r.connect('cosmul','ropeadd',from_port='left',to_port='left',via=[(560,259.5),(560,520)])
 r.text(25,590,'x_rotated = x*cos + rotate_half(x)*sin. Prefix tokens bypass this rotation.',15)
 r.text(25,630,'Coordinates and periods construct positions; there is no learned absolute position table.',15)
 if size=='g':
  s=d.panel('swiglu','Giant SwiGLU FFN',1160,2150,600,680,kind='aggregate',dashed=True,block_type='swiglu')
  op(s,'w1',25,80,250,'Linear w1','1536 to 4096','linear');op(s,'w2',325,80,250,'Linear w2','1536 to 4096','linear')
  op(s,'silu',25,220,250,'SiLU','','activation');s.connect('w1','silu');product(s,'gate',300,380)
  s.wire([(150,269),(150,380),(287,380)],start='silu',end='gate');s.wire([(450,129),(450,380),(313,380)],start='w2',end='gate')
  op(s,'w3',100,470,400,'Linear w3','4096 to 1536','linear');s.connect('gate','w3')
  s.text(25,610,'Hidden width: ceil_to_8((4×1536)×2/3)=4096.',14)
 else:
  t=d.panel('variants','MLP family configuration',1160,2150,600,680)
  t.text(25,85,'Size        D (width)       N (blocks)       A (heads)',16,weight=700)
  for i,size0 in enumerate(['s','b','l']):
   dd,nn,hh=CONFIGS[size0];t.text(25,145+i*60,f'{size0}             {dd}                    {nn}                    {hh}',16)
  t.text(25,390,'Every MLP expands width by4 with GELU.',15)
  t.text(25,438,'s:384/1536; b:768/3072; l:1024/4096.',15)
  t.text(25,540,'g:1536 width,40 blocks,24 heads, SwiGLU4096.',15)
  t.text(25,580,'g has a separate concrete view because its FFN differs.',14)
 record=ev['records'].get(size,{})
 return finish_view(a,d,'lingbot-vision','mlp-family' if sym else size+'-semantic','Shared s/b/l topology' if sym else size,'semantic',size,'family' if sym else 'concrete','3×512×512',record.get('device','source'))

def main():
 a=environment('Build LingBot-Vision native sizes including unpublished giant boundary')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'lingbotvision');records={}
  for size in CONFIGS:
   if size in ['s','b']:records[size]=cpu_probe(nn.LingBotVisionSemanticSegmenter(size,150),(1,3,512,512),['backbone.patch_embed','backbone.blocks','backbone.norm','predict'])
   else:
    with torch.device('meta'):m=nn.LingBotVisionSemanticSegmenter(size,150)
    records[size]=meta_probe(m,(1,3,512,512),['backbone.patch_embed','backbone.norm','predict'])
  write_evidence(a,'lingbotvision',records,['libreyolo/models/lingbotvision/nn.py:SIZE_CONFIGS, SelfAttention, RopePositionEmbedding, SwiGLUFFN, LingBotVisionSemanticSegmenter','libreyolo/models/lingbotvision/model.py:INPUT_SIZES and UNPUBLISHED_SIZES'],['s/b checked numerically on CPU; l/g use actual meta forward to avoid allocating300M+ parameters.','g is implemented but listed UNPUBLISHED_SIZES in public family.','Native forward returns150×32×32 logits, not full512×512.','No weights or network requests.'])
 ev=read_evidence('lingbotvision');views=[build(a,s,ev) for s in CONFIGS]+[build(a,'family',ev)];manifest(a,'lingbotvision','lingbot-vision','LingBot-Vision',views)
if __name__=='__main__':main()
