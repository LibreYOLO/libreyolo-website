from clip import *
def draw(b,size,c,task,specs):
 sym=size=='family';E='E' if sym else c.vision_width;D='D' if sym else c.projection_size;N='N' if sym else c.num_patches;P='P' if sym else c.patch_size;S='S' if sym else c.image_size;h='h' if sym else c.vision_heads;n='n' if sym else c.vision_layers;M='M' if sym else c.vision_intermediate;hd='E/h' if sym else E//h;grid='floor(S/P)' if sym else c.image_size//c.patch_size
 d=b.diagram('SigLIP2 '+size+' '+task,f'Image input {S} × {S}; text length 64. Classify example: 3 classes, one prompt each. Fixed-resolution SigLIP graph.',2470,3850)
 p=d.panel('vision','Image tower',30,220,580,1730)
 items=[('Image',f'3 × {S} × {S}','plain'),(f'Conv2d {P}×{P} / {P}',f'{E} × {grid} × {grid}, bias=True','conv2d'),('Flatten and transpose',f'{N} × {E}','plain'),('Vision encoder block',f'{N} × {E}, repeats={n}','attention'),('LayerNorm after encoder',f'{N} × {E}','norm'),('Attention pooling head',f'{N} tokens to 1 token','attention'),('Select pooled token',str(E),'plain'),('L2 normalize',str(E),'plain')]
 vi=tower(p,'vision',items,f'{N} × {E}')
 p.text(30,1200,'No CLS token. Convolution uses valid padding.',16)
 p.text(30,1240,'so400m uses 27 × 27 patches at 384 px.',16)
 p=d.panel('text','Text tower',640,220,580,1730)
 items=[('Text token IDs','64 IDs, vocabulary 256,000','plain'),('Token embedding',f'64 × {E}','linear'),('Embedded sequence',f'64 × {E}','plain'),('Text encoder block',f'64 × {E}, repeats={n}','attention'),('Final LayerNorm',f'64 × {E}','norm'),('Select last token',str(E),'plain'),('Linear projection',f'{E} to {D}, bias=True','linear'),('L2 normalize',str(D),'plain')]
 ti=tower(p,'text',items,f'64 × {E}')
 p.text(30,1200,'Bidirectional attention, no padding mask.',16)
 p=d.panel('output','Task output',1250,220,1190,390,kind='pool')
 if task=='classify':
  p.box('image-feat',20,65,270,'Image vector',detail=str(E));p.box('text-feat',820,65,340,'Class matrix',detail=f'3 × {D}')
  p.box('cosine',390,160,370,'MatMul cosine similarity',detail='3 scores',kind='attention');p.wire([(155,114),(155,184.5),(390,184.5)],start='image-feat',end='cosine');p.wire([(990,114),(990,184.5),(760,184.5)],start='text-feat',end='cosine')
  p.box('scale',390,245,370,'Scale and add learned bias',detail='exp(logit_scale) × cosine + logit_bias');p.connect('cosine','scale');p.text(20,345,'Postprocess: softmax by default; optional multi_label=True applies sigmoid.',16)
 else:
  p.box('image-feat',30,70,410,'Image embedding',detail=str(E));p.box('text-feat',710,70,410,'Text embedding',detail=str(D));p.text(30,185,'Both outputs are L2-normalized in the same feature space.',17)
 for node,label in [(vi,'Image vector: task output'),(ti,'Text vector: task output')]:
  x,y=d.port(node);d.text(x,y+36,label,15,anchor='middle')
 # MAP output is attention(probe, tokens), followed by residual MLP; no residual to the probe.
 p=d.panel('pool','Attention pooling head',1250,660,510,1290,kind='attention',dashed=True)
 ids=chain(p,'poolprep',[('Learned probe',f'1 × {E}','plain'),('Cross-attention',f'Q: probe; K/V: {N} image tokens','attention')],x=35,w=310,y=80,gap=110)
 p.dot(190,355);p.wire([p.port(ids[-1]),(190,355)],start=ids[-1],arrow=False)
 ids=chain(p,'poolmlp',[('LayerNorm',f'1 × {E}','norm'),('Linear',f'{E} to {M}','linear'),('GELU (tanh)',f'1 × {M}','activation'),('Linear',f'{M} to {E}','linear')],x=35,w=310,y=405,gap=110)
 p.wire([(190,355),(190,405)],end=ids[0]);p.sum('pooladd',190,895);p.connect(ids[-1],'pooladd');p.wire([(190,355),(455,355),(455,895),(203,895)],end='pooladd');p.box('select',35,980,310,'Select token 0',detail=str(E));p.connect('pooladd','select')
 attention(d,'mapattn','Pooling cross-attention',1790,660,E,h,1,hd,K=N)
 residual(d,'vblock','Vision encoder block',30,2020,E,N,M,h,activation='GELU (tanh)')
 attention(d,'vattn','Vision self-attention',570,2020,E,h,N,hd)
 residual(d,'tblock','Text encoder block',1250,2020,E,64,M,h,activation='GELU (tanh)')
 attention(d,'tattn','Text self-attention',1790,2020,E,h,64,hd)
 p=d.panel('variants','Variant values',30,3370,2410,290)
 columns=[('Size',25),('S',220),('P',400),('N',580),('E = D',790),('n',1050),('h',1290),('M',1540),('Head width',1840)]
 for label,x in columns:p.text(x,70,label,18,weight=700)
 for j,(key,cfg) in enumerate(specs.items()):
  for x,value in zip([25,220,400,580,790,1050,1290,1540,1840],[key,cfg.image_size,cfg.patch_size,cfg.num_patches,cfg.vision_width,cfg.vision_layers,cfg.vision_heads,cfg.vision_intermediate,cfg.vision_width//cfg.vision_heads]):p.text(x,120+j*45,str(value),18)
 p.text(25,225,'S: image size. P: patch size. N: image tokens. E/D: tower/output width. n: repeats in each tower. h: heads. M: MLP width.',17)
 b.save(d,size+'-'+task,size,kind='family' if sym else 'concrete',verification='source' if sym else 'meta',task=task,input=f'Image: 1 × 3 × {S} × {S}; text: 1 × 64')

def main():
 a=setup();b=Book(a,'siglip2','SigLIP2');m=load(a.source,'siglip2');import torch
 torch.backends.mha.set_fastpath_enabled(False)
 for size,c in m.SIGLIP2_CONFIGS.items():
  with torch.device('meta'):model=m.SigLIP2Model(c).eval()
  observed={};handles=[]
  for name,mm in model.named_modules():
   if name:handles.append(mm.register_forward_hook(lambda mm,inp,out,name=name:observed.__setitem__(name,{'input':list(inp[0].shape),'output':list(out.shape)}) if isinstance(out,torch.Tensor) else None))
  with torch.inference_mode():image=model.encode_image(torch.zeros(1,3,c.image_size,c.image_size,device='meta'));text=model.encode_text(torch.zeros(1,64,dtype=torch.long,device='meta'))
  b.evidence[size]={'device':'meta','config':vars(c),'image_output':list(image.shape),'text_output':list(text.shape),'modules':observed}
  for h in handles:h.remove()
  for task in ['classify','embed']:draw(b,size,c,task,m.SIGLIP2_CONFIGS)
 for task in ['classify','embed']:draw(b,'family',m.SIGLIP2_CONFIGS['b16'],task,m.SIGLIP2_CONFIGS)
 b.finish()
if __name__=='__main__':main()
