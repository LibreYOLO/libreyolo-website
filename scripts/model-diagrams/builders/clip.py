from alexnet import *

def attention(d,id,title,x,y,E,h,N,head_dim,*,K=None,causal=False,relative=False,rope=False,height=1270,shift_mask=None,out_projection=True,inner_dim=None):
 """Primitive multi-head attention equation; Q/K/V occupy independent lanes."""
 K=N if K is None else K
 A=E if inner_dim is None else inner_dim
 p=d.panel(id,title,x,y,650,height,kind='attention',dashed=True,block_type=id)
 p.box(id+'qi',20,65,180,'Query input',detail=f'{N} × {E}',kind='plain')
 p.box(id+'kv',280,65,330,'Key/value input',detail=f'{K} × {E}',kind='plain')
 for key,xx in [('q',20),('k',240),('v',460)]:
  p.box(id+key,xx,200,170,'Linear '+key.upper(),detail=f'{E} to {A}',kind='linear')
  p.box(id+key+'h',xx,305,170,'Reshape heads',detail=f'{h} × {N if key=="q" else K} × {head_dim}',font_size=13)
  p.connect(id+key,id+key+'h')
 p.connect(id+'qi',id+'q');p.wire([(370,114),(370,165),(325,165),(325,200)],start=id+'kv',end=id+'k');p.wire([(520,114),(520,165),(545,165),(545,200)],start=id+'kv',end=id+'v')
 ro=100 if rope else 0
 if rope:
  for key,xx in [('q',20),('k',240)]:
   p.box(id+key+'rope',xx,380,170,'Rotary position',detail=str(rope),font_size=13);p.connect(id+key+'h',id+key+'rope')
 p.box(id+'qk',80,440+ro,320,'MatMul Q K-transpose',detail=f'{h} × {N} × {K}',kind='attention')
 p.wire([(105,429 if rope else 354),(105,403+ro),(170,403+ro),(170,440+ro)],start=id+('qrope' if rope else 'qh'),end=id+'qk');p.wire([(325,429 if rope else 354),(325,440+ro)],start=id+('krope' if rope else 'kh'),end=id+'qk')
 p.box(id+'scale',80,540+ro,320,'Scale',detail=f'Divide by sqrt({head_dim})',kind='attention');p.connect(id+'qk',id+'scale')
 prev=id+'scale'
 if causal or relative:
  p.box(id+'mask',440,625,185,'Causal mask' if causal else 'Position bias',detail=f'{N} × {K}' if causal else f'{h} × {N} × {K}',font_size=13,kind='plain');p.sum(id+'ma',240,700);p.connect(prev,id+'ma');p.connect(id+'mask',id+'ma',via=[(532.5,700)],to_port='right');prev=id+'ma'
 offset=ro+(100 if shift_mask is not None else 0)
 if shift_mask is not None:
  p.box(id+'smask',440,735,185,'Shift mask',detail=str(shift_mask),font_size=13);p.sum(id+'sa',240,800);p.connect(prev,id+'sa');p.connect(id+'smask',id+'sa',via=[(532.5,800)],to_port='right');prev=id+'sa'
 p.box(id+'soft',80,775+offset,320,'Softmax over keys',detail=f'{h} × {N} × {K}',kind='activation');p.connect(prev,id+'soft')
 p.box(id+'av',80,880+offset,320,'MatMul attention × V',detail=f'{h} × {N} × {head_dim}',kind='attention');p.connect(id+'soft',id+'av');p.connect(id+'vh',id+'av',via=[(545,380),(640,380),(640,904.5+offset)],to_port='right')
 output_ops=[('Merge heads',f'{N} × {A}','plain')]
 if out_projection:output_ops.append(('Linear output',f'{A} to {E}','linear'))
 ids=chain(p,id+'out',output_ops,x=80,w=320,y=985+offset,gap=100);p.connect(id+'av',ids[0])
 return p

def residual(d,id,title,x,y,E,N,M,heads,activation='GELU',height=1270):
 p=d.panel(id,title,x,y,510,height,kind='attention',dashed=True,block_type=id)
 p.text(25,65,f'Input {N} × {E}',16);p.dot(190,100)
 ids=chain(p,id+'a',[('LayerNorm',f'{N} × {E}','norm'),('Multi-head self-attention',f'{heads} heads','attention')],x=35,w=310,y=140,gap=100);p.wire([(190,100),(190,140)],end=ids[0]);p.sum(id+'s1',190,390);p.connect(ids[-1],id+'s1');p.wire([(190,100),(455,100),(455,390),(203,390)],end=id+'s1')
 p.dot(190,445);p.wire([(190,403),(190,445)],start=id+'s1',arrow=False)
 ids=chain(p,id+'m',[('LayerNorm',f'{N} × {E}','norm'),('Linear',f'{E} to {M}','linear'),(activation,f'{N} × {M}','activation'),('Linear',f'{M} to {E}','linear')],x=35,w=310,y=490,gap=110)
 p.wire([(190,445),(190,490)],end=ids[0]);p.sum(id+'s2',190,970);p.connect(ids[-1],id+'s2');p.wire([(190,445),(455,445),(455,970),(203,970)],end=id+'s2');p.text(25,1030,f'Output {N} × {E}',16)
 return p

def tower(p,id,items,position_shape,cls_shape=None):
 # Token concat and position addition are explicit branches before transformer entry.
 ids=chain(p,id+'prep',items[:3],x=30,w=310,y=65,gap=80)
 current=ids[-1];next_y=355
 if cls_shape:
  p.box(id+'cls',390,225,160,'Learned CLS',detail=cls_shape,font_size=14)
  p.box(id+'concat',30,355,310,'Concat CLS and patches',detail=position_shape,kind='concat');p.connect(current,id+'concat');p.connect(id+'cls',id+'concat',via=[(470,379.5)],to_port='right');current=id+'concat';next_y=510
 p.box(id+'pos',390,next_y-90,160,'Position embedding',detail=position_shape,font_size=13)
 p.sum(id+'posadd',185,next_y);p.connect(current,id+'posadd');p.connect(id+'pos',id+'posadd',via=[(470,next_y)],to_port='right')
 tail=chain(p,id+'tail',items[3:],x=30,w=310,y=next_y+80,gap=90);p.connect(id+'posadd',tail[0]);return tail[-1]

def draw_clip(b,size,cfg,task,specs):
 sym=size=='family';E='Ev' if sym else cfg.vision_width;T='Et' if sym else cfg.text_width;D='D' if sym else cfg.embed_dim;P='P' if sym else cfg.patch_size;grid='224/P' if sym else cfg.image_size//cfg.patch_size;Nv='Nv' if sym else grid*grid+1;hv='hv' if sym else cfg.vision_heads;ht='ht' if sym else cfg.text_heads;dv='Ev/hv' if sym else E//hv;dt='Et/ht' if sym else T//ht
 d=b.diagram('CLIP '+size+' '+task,'Image input 224 × 224; text length 77. Classify example uses 3 classes and one prompt per class.',2400,3450)
 p=d.panel('vision','Image tower',30,220,580,1480)
 items=[('Image','3 × 224 × 224','plain'),(f'Conv2d {P}×{P} / {P}',f'{E} × {grid} × {grid}, no bias','conv2d'),('Flatten and transpose',f'{str(Nv)+"-1" if sym else Nv-1} × {E}','plain'),('LayerNorm before blocks',f'{Nv} × {E}','norm'),('Vision transformer block',f'{Nv} × {E}, n={"nv" if sym else cfg.vision_layers}','attention'),('LayerNorm after blocks',f'{Nv} × {E}','norm'),('Select CLS',str(E),'plain'),('MatMul learned projection',f'{E} to {D}, no bias','linear'),('L2 normalize',str(D),'plain')]
 vi=tower(p,'vision',items,f'{Nv} × {E}',f'1 × {E}')
 p=d.panel('text','Text tower',640,220,580,1480)
 items=[('Text token IDs','77 IDs, vocabulary 49,408','plain'),('Token embedding',f'77 × {T}','linear'),('Embedded sequence',f'77 × {T}','plain'),('Text transformer block',f'77 × {T}, n={"nt" if sym else cfg.text_layers}','attention'),('LayerNorm',f'77 × {T}','norm'),('Select EOT position','Argmax token ID, width '+str(T),'plain'),('MatMul learned projection',f'{T} to {D}, no bias','linear'),('L2 normalize',str(D),'plain')]
 ti=tower(p,'text',items,f'77 × {T}')
 p=d.panel('output','Task output',1250,220,1120,420,kind='pool')
 if task=='classify':
  p.box('image-feature',30,65,280,'Normalized image',detail=str(D));p.box('text-feature',650,65,350,'Normalized class matrix',detail=f'3 × {D}');p.box('similarity',320,190,390,'MatMul cosine similarities',detail='3 class scores',kind='attention')
  p.wire([(170,114),(170,214.5),(320,214.5)],start='image-feature',end='similarity');p.wire([(825,114),(825,214.5),(710,214.5)],start='text-feature',end='similarity')
  p.box('scale',320,275,390,'Multiply exp(logit_scale)',detail='3 logits',kind='plain');p.connect('similarity','scale');p.text(30,382,'Prediction postprocess applies softmax across the configured class set.',16)
 else:
  p.box('image-feature',40,85,420,'Normalized image embedding',detail=str(D));p.box('text-feature',620,85,420,'Normalized text embedding',detail=str(D));p.text(40,210,'predict(task=embed) returns the image vector.',17);p.text(40,245,'embed_text returns text rows in the same space.',17)
 # Matching continuation labels avoid forcing unrelated tower outputs into one wire.
 p.text(30,35,'',14)
 for node,label in [(vi,'Normalized image: task output'),(ti,'Normalized text: task output')]:
  x,y=d.port(node);d.text(x,y+36,label,15,anchor='middle')
 residual(d,'vblock','Vision transformer block',30,1770,E,Nv,('4Ev' if sym else E*4),hv)
 attention(d,'vattn','Vision self-attention',570,1770,E,hv,Nv,dv)
 residual(d,'tblock','Text transformer block',1250,1770,T,77,('4Et' if sym else T*4),ht)
 attention(d,'tattn','Text causal self-attention',1790,1770,T,ht,77,dt,causal=True)
 # Last panel extends to x2440, so give this poster the actual required width.
 d.root.set('viewBox','0 0 2470 3450');d.root.set('width','2470');d.width=2470
 p=d.panel('variants','Variant values',1250,700,1120,430)
 cols=[('Size',25),('P',165),('Ev',295),('nv',450),('hv',585),('Et',720),('nt',850),('ht',965)]
 for name,x in cols:p.text(x,70,name,17,weight=700)
 for j,(key,c) in enumerate(specs.items()):
  for x,v in zip([25,165,295,450,585,720,850,965],[key,c.patch_size,c.vision_width,c.vision_layers,c.vision_heads,c.text_width,c.text_layers,c.text_heads]):p.text(x,120+j*47,str(v),17)
 p.text(25,295,'D: 512 for b32/b16, 768 for l14. Nv: 50, 197, 257 respectively.',17)
 p.text(25,340,'Ev/Et: vision/text width. nv/nt: repeats. hv/ht: heads. P: patch size.',16)
 p.text(25,379,'Concrete views contain all resolved widths, head counts and token counts.',16)
 b.save(d,size+'-'+task,size,kind='family' if sym else 'concrete',verification='source' if sym else b.evidence[size]['device'],task=task,input='Image: 1 × 3 × 224 × 224; text: 1 × 77')

def main():
 a=setup();b=Book(a,'clip','CLIP');m=load(a.source,'clip');import torch
 torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
 for size,c in m.CLIP_CONFIGS.items():
  dev='cpu' if size=='b32' else 'meta'
  with torch.device(dev):model=m.CLIPModel(c).eval()
  observed={};handles=[]
  for name,mm in model.named_modules():
   if name:handles.append(mm.register_forward_hook(lambda mm,inp,out,name=name:observed.__setitem__(name,{'input':list(inp[0].shape),'output':list(out.shape)}) if isinstance(out,torch.Tensor) else None))
  with torch.inference_mode():image=model.encode_image(torch.zeros(1,3,224,224,device=dev));text=model.encode_text(torch.zeros(1,77,dtype=torch.long,device=dev))
  b.evidence[size]={'device':dev,'config':vars(c),'image_output':list(image.shape),'text_output':list(text.shape),'modules':observed}
  for h in handles:h.remove()
  for task in ['classify','embed']:draw_clip(b,size,c,task,m.CLIP_CONFIGS)
 for task in ['classify','embed']:draw_clip(b,'family',m.CLIP_CONFIGS['b32'],task,m.CLIP_CONFIGS)
 b.finish()
if __name__=='__main__':main()
