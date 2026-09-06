"""All six SegFormer sizes with numeric MiT attention and Mix-FFN internals."""
from quicksrnet import *
CONFIGS={'b0':([2,2,2,2],[32,64,160,256],256,512),'b1':([2,2,2,2],[64,128,320,512],256,512),'b2':([3,4,6,3],[64,128,320,512],768,512),'b3':([3,4,18,3],[64,128,320,512],768,512),'b4':([3,8,27,3],[64,128,320,512],768,512),'b5':([3,6,40,3],[64,128,320,512],768,640)}

def block(d,i,c,hw,heads,sr,symbolic):
 p=d.panel('block-'+str(i),f'Stage {i+1} transformer block',40+i*440,1770,420,1630,kind='attention',dashed=True,block_type='block-'+str(i))
 pre='s'+str(i);n=f'({hw})²' if symbolic else hw*hw;kv='(H/32)²' if symbolic else (hw//sr)**2
 hd=f'{c}/{heads}' if symbolic else c//heads;ff=f'4{c}' if symbolic else c*4
 def b(id,x,y,w,label,detail='',kind='plain'):return op(p,pre+id,x,y,w,label,detail,kind,h=45)
 b('in',95,65,300,'Input tokens',f'{n} × {c}')
 b('ln',95,145,300,'LayerNorm',f'{c} channels, epsilon 1e-5','norm');p.connect(pre+'in',pre+'ln')
 b('q',20,250,170,'Linear Q',f'{c} to {c}','linear')
 b('sr',230,250,170,'Conv2d reduction' if sr>1 else 'Identity K/V input',f'k=s={sr}; {c} channels' if sr>1 else f'{c} channels','conv2d' if sr>1 else 'plain')
 p.connect(pre+'ln',pre+'q',via=[(245,222),(105,222)])
 p.connect(pre+'ln',pre+'sr',via=[(245,222),(315,222)])
 b('sln',230,330,170,'Flatten + LayerNorm' if sr>1 else 'Identity',f'{kv} × {c}','norm' if sr>1 else 'plain');p.connect(pre+'sr',pre+'sln')
 b('k',230,415,170,'Linear K',f'{c} to {c}','linear');p.connect(pre+'sln',pre+'k')
 b('v',230,550,170,'Linear V',f'{c} to {c}','linear');p.connect(pre+'sln',pre+'v',from_port='right',to_port='right',via=[(412,352.5),(412,572.5)])
 b('qk',20,465,190,'Q × transpose(K)',f'Scale 1/sqrt({hd}); {heads} heads','attention')
 p.connect(pre+'q',pre+'qk');p.connect(pre+'k',pre+'qk',from_port='left',to_port='right',via=[(215,437.5),(215,487.5)])
 b('soft',20,550,190,'Softmax over keys',f'{n} queries, {kv} keys','attention');p.connect(pre+'qk',pre+'soft')
 b('av',20,650,190,'Weights × V',f'{heads} heads, width {hd}','attention');p.connect(pre+'soft',pre+'av')
 p.connect(pre+'v',pre+'av',from_port='left',to_port='right',via=[(215,572.5),(215,672.5)])
 b('cat',95,735,300,'Concat heads',f'{n} × {c}','concat');p.connect(pre+'av',pre+'cat',via=[(115,714),(245,714)])
 b('outproj',95,810,300,'Output linear',f'{c} to {c}','linear');p.connect(pre+'cat',pre+'outproj')
 p.sum(pre+'add1',245,915);p.connect(pre+'outproj',pre+'add1');p.connect(pre+'in',pre+'add1',from_port='left',to_port='left',via=[(5,87.5),(5,915)])
 rows=[('ln2',985,'LayerNorm',f'{c} channels','norm'),('fc1',1070,'Linear Mix-FFN',f'{c} to {ff}','linear'),('dw',1155,'Depthwise Conv2d 3×3',f'{ff} channels/groups; stride1, padding1','conv2d'),('gelu',1240,'GELU','','activation'),('fc2',1325,'Linear Mix-FFN',f'{ff} to {c}','linear')]
 for id,y,label,detail,kind in rows:b(id,95,y,300,label,detail,kind)
 chain(p,[pre+'add1']+[pre+r[0] for r in rows]);p.sum(pre+'add2',245,1430);p.connect(pre+'fc2',pre+'add2')
 p.dot(245,956);p.wire([(245,956),(5,956),(5,1430),(232,1430)],start=pre+'add1',end=pre+'add2')
 b('o',95,1495,300,'Output tokens',f'{n} × {c}');p.connect(pre+'add2',pre+'o')
 p.text(18,1590,'DropPath and all dropout are identities during eval.',12)

def build(a,size,ev):
 sym=size=='family'
 depths,cs,hd,H=([f'N{i}' for i in range(1,5)],[f'C{i}' for i in range(1,5)],'D','H') if sym else CONFIGS[size]
 d=diagram(a,'SegFormer family' if sym else 'SegFormer '+size.upper(),f'Semantic segmentation, 150 classes, {H} × {H} RGB, native eval. Shapes exclude batch.','segformer',1800,3850 if sym else 3550)
 p=d.panel('encoder','MiT encoder',40,230,570,1470)
 op(p,'image',60,60,440,'RGB input and ImageNet normalization',f'3 × {H} × {H}','norm')
 prev='image'
 for i in range(4):
  h=f'H/{4*2**i}' if sym else H//(4*2**i);ci=3 if i==0 else cs[i-1];y=155+i*310;k=7 if i==0 else 3;s=4 if i==0 else 2
  rows=[('patch','Conv2d overlap embedding',f'{ci} to {cs[i]}; k={k}, s={s}, p={k//2}','conv2d'),('norm','Flatten + LayerNorm',f'{h} × {h} spatial tokens, width {cs[i]}','norm'),('block',f'Transformer block, n={depths[i]}',f'{[1,2,5,8][i]} heads; reduction {[8,4,2,1][i]}','attention'),('output','LayerNorm + restore spatial map',f'{cs[i]} × {h} × {h}','norm')]
  for j,(id,label,detail,kind) in enumerate(rows):
   nid=f'stage{i}-{id}';op(p,nid,60,y+j*70,440,label,detail,kind,block='block-'+str(i) if id=='block' else '')
   p.connect(prev,nid);prev=nid
  op(p,f'F{i}',510,y+223,45,f'F{i+1}',h=26,block='signal-F'+str(i+1));p.connect(prev,f'F{i}',from_port='right',to_port='left')
 p.text(25,1430,'No learned position embeddings; Mix-FFN supplies spatial mixing.',14)
 q=d.panel('decoder','All-MLP decode head',650,230,1110,1470)
 target='H/4' if sym else H//4
 for col,i in enumerate([3,2,1,0]):
  x=25+270*col;h=f'H/{4*2**i}' if sym else H//(4*2**i)
  op(q,f'tap{i}',x,70,240,f'F{i+1}',f'{cs[i]} × {h} × {h}','plain',block='signal-F'+str(i+1))
  op(q,f'linear{i}',x,180,240,'Flatten + Linear',f'{cs[i]} to {hd}','linear');q.connect(f'tap{i}',f'linear{i}')
  op(q,f'reshape{i}',x,290,240,'Restore feature map',f'{hd} × {h} × {h}','plain');q.connect(f'linear{i}',f'reshape{i}')
  op(q,f'resize{i}',x,400,240,'Bilinear resize',f'{hd} × {target} × {target}','pool');q.connect(f'reshape{i}',f'resize{i}')
  sx,sy=q.port(f'resize{i}','bottom');q.wire([(sx,sy),(sx,530)],start=f'resize{i}',end='concat')
 op(q,'concat',25,530,1050,'Concat in reverse stage order: F4, F3, F2, F1',f'{"4D" if sym else 4*hd} × {target} × {target}','concat')
 rows=[('fuse','Conv2d 1×1',f'{"4D" if sym else 4*hd} to {hd}; bias=False','conv2d'),('bn','BatchNorm2d',f'{hd} channels','norm'),('relu','ReLU','','activation'),('drop','Dropout 0.1','Identity in eval','plain'),('classifier','Conv2d 1×1',f'{hd} to 150; bias=True','conv2d'),('up','Bilinear resize to input size','align_corners=False','pool'),('out','Semantic logits',f'150 × {H} × {H}','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(q,id,320,635+j*105,470,label,detail,kind)
 chain(q,['concat']+[n[0] for n in rows]);q.text(25,1430,'All linear and overlap/reduction convolutions have bias unless marked otherwise.',14)
 for i,c in enumerate(cs):block(d,i,c,f'H/{4*2**i}' if sym else H//(4*2**i),[1,2,5,8][i],[8,4,2,1][i],sym)
 if sym:
  t=d.panel('variants','Resolved family values',40,3460,1720,245)
  t.text(25,48,'Size       N1,N2,N3,N4          C1,C2,C3,C4                  D       H',16,weight=700)
  for i,(k,(ns,chs,dh,h)) in enumerate(CONFIGS.items()):t.text(25,80+i*24,f'{k}          {str(ns):23}       {str(chs):25}       {dh}      {h}',15)
 return finish_view(a,d,'segformer','family' if sym else size+'-semantic','Shared B0-B5 topology' if sym else size,'semantic',size,'family' if sym else 'concrete',f'3×{H}×{H}','cpu' if size in ev['records'] else 'source')

def main():
 a=environment('Build SegFormer B0 through B5 and symbolic family')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'segformer');records={}
  for size,(_,_,_,h) in CONFIGS.items():
   records[size]=cpu_probe(nn.LibreSegformerNet(size,num_classes=150),(1,3,h,h),['encoder.stages.0','encoder.stages.1','encoder.stages.2','encoder.stages.3','decode_head'])
  write_evidence(a,'segformer',records,['libreyolo/models/segformer/nn.py:SIZE_CONFIGS, OverlapPatchEmbeddings, EfficientSelfAttention, MixMLP, SegformerDecodeHead','libreyolo/models/segformer/model.py:INPUT_SIZES'],['B5 uses640 input; other sizes512. Attention reduced K/V tokens are20×20 for B5 and16×16 for other sizes.','Every LayerNorm epsilon is1e-5. Decoder concatenates projected stages in reverse order.'])
 ev=read_evidence('segformer');views=[build(a,s,ev) for s in CONFIGS]+[build(a,'family',ev)];manifest(a,'segformer','segformer','SegFormer',views)
if __name__=='__main__':main()
