from clip import *
def draw(b,size,c,specs):
 sym=size=='family';E='Ev' if sym else c.vision_hidden;T='Et' if sym else c.text_hidden;D='D' if sym else c.projection_dim;S='S' if sym else c.image_size;P='P' if sym else c.patch_size;N='Nv' if sym else c.num_patches_side**2+1;patches='Np' if sym else N-1;hv='hv' if sym else c.vision_heads;ht='ht' if sym else c.text_heads
 d=b.diagram('OWLv2 '+size,f'Open-vocabulary detection, {S} × {S} image, 3 class prompts of 16 tokens. Tensor sizes exclude batch.',2660,5270)
 p=d.panel('image','Vision tower',25,220,610,1740)
 items=[('Image',f'3 × {S} × {S}','plain'),(f'Conv2d {P}×{P} / {P}',f'{E} channels, bias=False','conv2d'),('Flatten patches',f'{patches} × {E}','plain'),('LayerNorm before encoder',f'{N} × {E}','norm'),('Vision encoder block',f'{N} × {E}, n={"nv" if sym else c.vision_layers}','attention'),('Post LayerNorm',f'{N} × {E}','norm')]
 tail=tower(p,'vision',items,f'{N} × {E}',f'1 × {E}')
 p.box('clsfeat',365,1000,215,'Select CLS and broadcast',detail=f'{patches} × {E}',font_size=13);p.box('patchfeat',30,1100,310,'Select patch tokens',detail=f'{patches} × {E}');p.connect(tail,'patchfeat');p.connect(tail,'clsfeat',from_port='right',to_port='top',via=[(472.5,794.5)])
 p.box('multiply',30,1240,310,'Multiply patch × CLS',detail=f'{patches} × {E}');p.connect('patchfeat','multiply');p.connect('clsfeat','multiply',via=[(472.5,1264.5)],to_port='right');p.box('feature',30,1380,310,'LayerNorm',detail=f'{patches} × {E}',kind='norm');p.connect('multiply','feature')
 p=d.panel('text','Text tower',670,220,610,1740)
 items=[('Tokenized class prompts','3 × 16 IDs, vocabulary 49,408','plain'),('Token embedding',f'3 × 16 × {T}','linear'),('Embedded sequences',f'3 × 16 × {T}','plain'),('Text encoder block',f'16 × {T} per class, n={"nt" if sym else c.text_layers}','attention'),('Final LayerNorm',f'3 × 16 × {T}','norm'),('Select EOS position','Argmax token ID per class','plain'),('Linear text projection',f'{T} to {D}, no bias','linear'),('L2 normalize',f'3 × {D}','plain')];tower(p,'text',items,f'16 × {T}')
 p.text(30,1380,'Causal attention plus padding mask.',17)
 p.text(30,1430,'Query validity is first token ID > 0.',17)
 p=d.panel('class','Per-patch class scoring',1315,220,1315,1740,kind='pool')
 p.box('image-in',25,65,560,'Image patch features',detail=f'{patches} × {E}');p.box('text-in',785,65,500,'Text embeddings',detail=f'3 × {D}')
 a=chain(p,'imageclass',[('Linear class embedding',f'{E} to {T}','linear'),('L2 normalize with epsilon',f'{patches} × {T}, eps=1e-6','plain')],x=25,w=560,y=200,gap=110);p.connect('image-in',a[0])
 p.box('textnorm',785,200,500,'L2 normalize with epsilon',detail=f'3 × {D}, eps=1e-6');p.connect('text-in','textnorm')
 p.box('cosine',25,515,560,'Einsum cosine similarity',detail=f'{patches} × 3',kind='attention');p.connect(a[-1],'cosine');p.connect('textnorm','cosine',via=[(1035,465),(650,465),(650,539.5)],to_port='right')
 p.box('shift',785,650,500,'Linear logit shift',detail=f'{E} to 1 per patch',kind='linear');p.box('scale',785,840,500,'Linear logit scale',detail=f'{E} to 1 per patch',kind='linear');p.wire([(585,89.5),(700,89.5),(700,864.5)],start='image-in',arrow=False);p.dot(700,674.5);p.wire([(700,674.5),(785,674.5)],start='image-in',end='shift');p.wire([(700,864.5),(785,864.5)],start='image-in',end='scale')
 p.sum('shiftadd',305,720);p.connect('cosine','shiftadd');p.connect('shift','shiftadd',via=[(1035,720)],to_port='right')
 ids=chain(p,'scaleact',[('ELU',f'{patches} × 1','activation'),('Add 1',f'{patches} × 1','plain')],x=785,w=500,y=985,gap=100);p.connect('scale',ids[0])
 p.box('logits',25,1210,560,'Multiply shifted logits by scale',detail=f'{patches} × 3');p.connect('shiftadd','logits');p.connect(ids[-1],'logits',via=[(1035,1234.5)],to_port='right')
 p.box('maskinvalid',25,1360,560,'Mask invalid class prompts',detail='Invalid scores become dtype minimum');p.connect('logits','maskinvalid')
 p.text(25,1550,'Returned logits are per patch and per requested class.',17)
 residual(d,'visionblock','Vision encoder block',25,2020,E,N,('Mv' if sym else c.vision_intermediate),hv,activation='QuickGELU')
 attention(d,'visionattention','Vision self-attention',570,2020,E,hv,N,'Ev/hv' if sym else E//hv)
 for j,(id,outdim,title) in enumerate([('box',4,'Box prediction'),('obj',1,'Objectness prediction')]):
  p=d.panel(id,title,1250+j*700,2020,680,1270,kind='bottleneck',dashed=True)
  items=[('Image patch features',f'{patches} × {E}','plain'),('Linear',f'{E} to {E}','linear'),('GELU',f'{patches} × {E}','activation'),('Linear',f'{E} to {E}','linear'),('GELU',f'{patches} × {E}','activation'),('Linear',f'{E} to {outdim}','linear')]
  if outdim==4:items +=[('Add grid box bias','Logit normalized corner and patch-size priors','plain'),('Sigmoid',f'{patches} × 4 normalized cxcywh','activation')]
  else:items +=[('Squeeze last axis',f'{patches} objectness logits','plain')]
  chain(p,id+'op',items,w=560,gap=120)
  p.text(25,1140,'All prediction-head Linear layers include bias.',16)
 residual(d,'textblock','Text encoder block',25,3360,T,16,('Mt' if sym else c.text_intermediate),ht,activation='QuickGELU')
 attention(d,'textattention','Text causal self-attention',570,3360,T,ht,16,'Et/ht' if sym else T//ht,causal=True)
 p=d.panel('quick','QuickGELU',1250,3360,680,600,kind='attention',dashed=True)
 ids=chain(p,'qg',[('Input x','Elementwise activation','plain'),('Multiply by 1.702','Same shape','plain'),('Sigmoid','Same shape','activation'),('Multiply by original x','Same shape','plain')],w=500,gap=105);p.connect(ids[0],ids[-1],from_port='right',to_port='right',via=[(620,89.5),(620,404.5)])
 p=d.panel('post','Detection outputs',1950,3360,680,1270,kind='pool')
 chain(p,'postop',[('Class logits','Sigmoid, max class per patch','activation'),('Score threshold','Keep matching patch predictions','plain'),('Convert normalized cxcywh to XYXY','Scale to the original image dimensions','plain'),('Results','Boxes, confidence and requested class IDs','plain')],w=560,gap=130)
 p.text(25,720,'Objectness logits are a separate returned branch.',16)
 p.text(25,780,'No DETR query decoder or FPN exists in this graph.',16)
 p=d.panel('table','Variant values',25,4700,2605,360)
 for x,label in [(25,'Size'),(220,'S / P'),(460,'Np / Nv'),(830,'Ev / nv / hv'),(1300,'Et / nt / ht'),(1770,'Mv / Mt'),(2190,'D')]:p.text(x,65,label,18,weight=700)
 for j,(key,sp) in enumerate(specs.items()):
  for x,value in zip([25,220,460,830,1300,1770,2190],[key,f'{sp.image_size} / {sp.patch_size}',f'{sp.num_patches_side**2} / {sp.num_patches_side**2+1}',f'{sp.vision_hidden} / {sp.vision_layers} / {sp.vision_heads}',f'{sp.text_hidden} / {sp.text_layers} / {sp.text_heads}',f'{sp.vision_intermediate} / {sp.text_intermediate}',sp.projection_dim]):p.text(x,120+j*60,str(value),17)
 p.text(25,270,'Nv includes CLS; Np is patch count. Ev/Et: tower widths; nv/nt: repeats; hv/ht: heads; Mv/Mt: MLP widths; D: text projection.',17)
 b.save(d,size+'-detect',size,kind='family' if sym else 'concrete',verification='source' if sym else 'meta',task='detect',input=f'Image: 1 × 3 × {S} × {S}; text: 3 × 16')

def main():
 a=setup();b=Book(a,'owlv2','OWLv2');m=load(a.source,'owlv2');import torch
 for size,c in m.OWLV2_DIMS.items():
  with torch.device('meta'):model=m.Owlv2DetectionModel(c).eval()
  with torch.inference_mode():out=model(torch.ones(3,16,dtype=torch.long,device='meta'),torch.zeros(1,3,c.image_size,c.image_size,device='meta'),torch.ones(3,16,device='meta'))
  b.evidence[size]={'device':'meta','config':vars(c),'outputs':{k:list(v.shape) for k,v in out.items()}};draw(b,size,c,m.OWLV2_DIMS)
 draw(b,'family',m.OWLV2_DIMS['b16'],m.OWLV2_DIMS);b.finish()
if __name__=='__main__':main()
