from clip import *

def rope_definition(d,id,x,y,width,patches,prefix,offset):
 p=d.panel(id,'Rotary position on Q and K',x,y,1150,890,kind='attention',dashed=True)
 p.box(id+'input',30,65,320,'Q or K per head',detail=f'{patches} patch tokens, {width} channels')
 p.box(id+'prefix',750,65,340,'Prefix bypass',detail=f'{prefix} CLS token; no rotation')
 p.box(id+'rot',30,205,320,'Pair adjacent channels',detail='[a,b] becomes [-b,a]')
 p.box(id+'cos',410,205,270,'Cosine table',detail=f'2D grid offset {offset}')
 p.box(id+'sin',750,205,340,'Sine table',detail='Same spatial frequency grid')
 p.connect(id+'input',id+'rot')
 p.box(id+'xc',410,375,270,'Multiply input × cos',detail=f'{patches} × {width}')
 p.box(id+'rs',30,375,320,'Multiply rotation × sin',detail=f'{patches} × {width}')
 p.connect(id+'rot',id+'rs');p.connect(id+'cos',id+'xc');p.wire([(920,254),(920,310),(100,310),(100,375)],start=id+'sin',end=id+'rs')
 p.wire([(350,89.5),(375,89.5),(375,399.5),(410,399.5)],start=id+'input',end=id+'xc')
 p.sum(id+'add',370,560);p.connect(id+'rs',id+'add',via=[(190,560)],to_port='left');p.connect(id+'xc',id+'add',via=[(545,560)],to_port='right')
 p.box(id+'cat',210,690,570,'Concat preserved CLS and rotated patches',detail='Restore the original token order',kind='concat');p.connect(id+'add',id+'cat',via=[(370,645),(495,645)]);p.connect(id+'prefix',id+'cat',via=[(1120,89.5),(1120,714.5)],from_port='right',to_port='right')
 p.text(30,820,'No RoPE on V. Frequency base 10,000; XY grid. Family Np = Nv - 1.',16)
 return p

def draw(b,size,c,task,specs):
 sym=size=='family';E='Ev' if sym else c.embed_dim;T='Et' if sym else c.text_width;M='Mv' if sym else round(c.embed_dim*c.mlp_ratio);Mt='4Et' if sym else T*4;S='S' if sym else c.image_size;P='P' if sym else c.patch_size;N='Nv' if sym else c.num_patches+c.num_prefix_tokens;L='L' if sym else c.context_length;h='hv' if sym else c.num_heads;ht='ht' if sym else c.text_heads;D='D' if sym else c.projection_dim;hd='Ev/hv' if sym else c.head_dim;hp='Ev/8' if sym else E//8;prefix=1 if sym else c.num_prefix_tokens
 d=b.diagram('PE Core '+size+' '+task,f'Image input {S} × {S}, text length {L}. Classify example: 3 classes, one prompt each. Tensor sizes exclude batch.',2470,5080)
 p=d.panel('vision','Image tower',30,220,580,1740)
 items=[('Image',f'3 × {S} × {S}','plain'),(f'Conv2d {P}×{P} / {P}',f'{E} channels, bias=False','conv2d'),('Flatten patch grid',f'{"Nv-1" if sym else c.num_patches} × {E}','plain'),('LayerNorm before encoder',f'{N} × {E}','norm'),('Vision encoder block',f'{N} × {E}, repeats={"nv" if sym else c.depth}','attention'),('LayerNorm after encoder',f'{N} × {E}','norm'),('Latent attention pooling','1 query, 8 pooling heads','attention'),('Linear projection',f'{E} to {D}, bias=True','linear'),('L2 normalize',str(D),'plain')]
 vi=tower(p,'vision',items,f'{N} × {E}',f'1 × {E}' if prefix else None)
 p.text(30,1450,'RoPE applies to patch Q/K in every block.',16)
 p.text(30,1490,'g14 has no CLS token; other sizes have one.',16)
 p=d.panel('text','Text tower',640,220,580,1740)
 items=[('Text token IDs',f'{L} IDs, vocabulary 49,408','plain'),('Token embedding',f'{L} × {T}','linear'),('Embedded sequence',f'{L} × {T}','plain'),('Text transformer block',f'{L} × {T}, repeats={"nt" if sym else c.text_layers}','attention'),('Final LayerNorm',f'{L} × {T}','norm'),('Select EOT','Argmax token ID, width '+str(T),'plain'),('MatMul projection',f'{T} to {D}, no bias','linear'),('L2 normalize',str(D),'plain')]
 ti=tower(p,'text',items,f'{L} × {T}')
 p=d.panel('task','Task output',1250,220,1190,370,kind='pool')
 if task=='classify':
  p.box('if',20,65,270,'Image vector',detail=str(D));p.box('tf',820,65,340,'Class matrix',detail=f'3 × {D}');p.box('sim',390,160,370,'MatMul cosine similarity',detail='3 scores',kind='attention');p.wire([(155,114),(155,184.5),(390,184.5)],start='if',end='sim');p.wire([(990,114),(990,184.5),(760,184.5)],start='tf',end='sim');p.box('scale',390,245,370,'Multiply exp(logit_scale)',detail='3 logits');p.connect('sim','scale');p.text(20,337,'Prediction applies softmax over the class set.',16)
 else:
  p.box('if',30,75,420,'Image embedding',detail=str(D));p.box('tf',700,75,420,'Text embedding',detail=str(D));p.text(30,190,'L2-normalized outputs in a shared space.',17)
 for node,label in [(vi,'Image vector: task output'),(ti,'Text vector: task output')]:
  x,y=d.port(node);d.text(x,y+36,label,15,anchor='middle')
 p=d.panel('pool','Latent attention pooling',1250,640,510,1320,kind='attention',dashed=True)
 ids=chain(p,'pl',[('Learned latent','1 × '+str(E),'plain'),('Cross-attention',f'1 query over {N} image tokens','attention')],x=35,w=310,y=80,gap=110)
 p.dot(190,355);p.wire([p.port(ids[-1]),(190,355)],start=ids[-1],arrow=False)
 ids=chain(p,'pm',[('LayerNorm',f'1 × {E}','norm'),('Linear',f'{E} to '+('4Ev' if sym else str(E*4)),'linear'),('GELU','1 × '+('4Ev' if sym else str(E*4)),'activation'),('Linear',('4Ev' if sym else str(E*4))+f' to {E}','linear')],x=35,w=310,y=405,gap=110)
 p.wire([(190,355),(190,405)],end=ids[0]);p.sum('pooladd',190,895);p.connect(ids[-1],'pooladd');p.wire([(190,355),(455,355),(455,895),(203,895)],end='pooladd');p.box('select',35,980,310,'Select pooled token',detail=str(E));p.connect('pooladd','select')
 attention(d,'poolattn','Pooling cross-attention',1790,640,E,8,1,hp,K=N)
 residual(d,'vblock','Vision encoder block',30,2020,E,N,M,h,height=1370)
 attention(d,'vattn','Vision rotary self-attention',570,2020,E,h,N,hd,rope=f'{prefix} prefix; rotate patches',height=1370)
 residual(d,'tblock','Text transformer block',1250,2020,T,L,Mt,ht,height=1370)
 attention(d,'tattn','Text causal self-attention',1790,2020,T,ht,L,'Et/ht' if sym else T//ht,causal=True,height=1370)
 rope_definition(d,'rope',30,3450,hd,'Np' if sym else c.num_patches,prefix,'1.0' if sym else c.rope_grid_offset)
 p=d.panel('video','Video embedding',1220,3450,1220,400,kind='pool')
 ops=[('Two input frames',f'2 × 3 × {S} × {S}','plain'),('Image tower per frame',f'2 × {D}, before L2 normalization','attention'),('Mean frame embeddings',str(D),'pool'),('L2 normalize once',str(D),'plain')]
 for j,(label,detail,kind) in enumerate(ops):
  p.box('video'+str(j),25+j*300,90,275,label,detail=detail,kind=kind,font_size=14)
  if j:p.connect('video'+str(j-1),'video'+str(j),from_port='right',to_port='left')
 p.text(25,240,'Example uses two frames; encode_video accepts arbitrary frame count F.',17)
 p.text(25,280,'No temporal attention is added. Frame embeddings are averaged before normalization.',16)
 p=d.panel('variants','Variant values',30,4400,2410,480)
 cols=[('Size',25),('S / P',225),('Nv',440),('Ev / nv / hv',650),('Mv',1050),('Et / nt / ht',1250),('L',1660),('D',1840)]
 for label,x in cols:p.text(x,65,label,18,weight=700)
 for j,(key,cfg) in enumerate(specs.items()):
  for x,value in zip([25,225,440,650,1050,1250,1660,1840],[key,f'{cfg.image_size} / {cfg.patch_size}',cfg.num_patches+cfg.num_prefix_tokens,f'{cfg.embed_dim} / {cfg.depth} / {cfg.num_heads}',round(cfg.embed_dim*cfg.mlp_ratio),f'{cfg.text_width} / {cfg.text_layers} / {cfg.text_heads}',cfg.context_length,cfg.projection_dim]):p.text(x,115+j*45,str(value),17)
 p.text(25,370,'S/P: input/patch size. Nv: image tokens. Ev/Et: tower width. nv/nt: block counts. hv/ht: heads. Mv: vision MLP width.',16)
 p.text(25,410,'The symbolic graph covers t16, s16, b16, l14. g14 has a separate concrete graph because its CLS path is absent.',16)
 b.save(d,size+'-'+task,size,kind='family' if sym else 'concrete',verification='source' if sym else b.evidence[size]['device'],task=task,input=f'Image: 1 × 3 × {S} × {S}; text: 1 × {L}')

def main():
 a=setup();b=Book(a,'pe','PE Core');m=load(a.source,'pe');import torch
 torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
 for size,c in m.PE_CONFIGS.items():
  dev='cpu' if size=='t16' else 'meta'
  with torch.device(dev):model=m.LibrePEModel(c).eval()
  observed={};handles=[]
  for name,mm in model.named_modules():
   if name:handles.append(mm.register_forward_hook(lambda mm,inp,out,name=name:observed.__setitem__(name,{'input':list(inp[0].shape),'output':list(out.shape)}) if isinstance(out,torch.Tensor) else None))
  with torch.inference_mode():im=model.encode_image(torch.zeros(1,3,c.image_size,c.image_size,device=dev));tx=model.encode_text(torch.zeros(1,c.context_length,dtype=torch.long,device=dev))
  b.evidence[size]={'device':dev,'config':vars(c),'image_output':list(im.shape),'text_output':list(tx.shape),'modules':observed}
  for h in handles:h.remove()
  for task in ['classify','embed']:draw(b,size,c,task,m.PE_CONFIGS)
 for task in ['classify','embed']:draw(b,'family',m.PE_CONFIGS['t16'],task,m.PE_CONFIGS)
 b.finish()
if __name__=='__main__':main()
