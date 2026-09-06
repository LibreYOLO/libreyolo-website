from alexnet import *
def draw(b,size,specs):
 family=size=='family';s=next(iter(specs.values())) if family else specs[size];E='E' if family else s['embed_dim'];heads='h' if family else s['num_heads'];depth='n' if family else s['depth'];M='4E' if family else E*4;Q='3E' if family else E*3;D='E/h' if family else E//heads
 d=b.diagram(b.title+(' family' if family else ' '+size),'Classification, 224 × 224 input, patch size 16, 1,000 classes. Token shapes exclude batch.',1800,1980)
 p=d.panel('network','Network',30,220,470,1360)
 items=[('Input','3 × 224 × 224','plain'),('Conv2d 16×16 / 16',f'{E} × 14 × 14, bias=True','conv2d'),('Flatten and transpose',f'196 × {E}','plain')];ids=chain(p,'net',items,x=35,w=290,y=65)
 p.box('cls',340,209,105,'CLS token',detail=f'1 × {E}',font_size=13)
 p.box('cat',35,330,290,'Concat tokens',detail=f'197 × {E}',kind='concat')
 p.connect(ids[-1],'cat');p.connect('cls','cat',from_port='bottom',to_port='right',via=[(392.5,354.5)])
 p.box('pos',340,413,105,'Position',detail=f'197 × {E}',font_size=13)
 p.sum('posadd',180,490);p.connect('cat','posadd');p.connect('pos','posadd',via=[(392.5,490)],to_port='right')
 items=[('Transformer block',f'197 × {E}, n={depth}','attention'),('LayerNorm',f'197 × {E}, eps=1e-6','norm'),('Select CLS token',str(E),'plain'),('Linear classifier',f'{E} to 1,000 logits','linear')]
 ids=chain(p,'tail',items,x=35,w=290,y=555,gap=110);p.connect('posadd',ids[0]);p.text(35,1070,'Learned CLS and absolute position.',16);p.text(35,1104,'Dropout probabilities are zero.',16);p.text(35,1138,'All Linear projections include bias.',16)
 if b.family=='deit':p.text(35,1192,'Plain DeiT: no distillation token.',16)
 p=d.panel('block','Transformer block',530,220,510,1360,kind='attention',dashed=True,block_type='block')
 p.text(30,65,f'Input: 197 × {E}',16);p.dot(200,95)
 ids=chain(p,'b',[('LayerNorm',f'197 × {E}','norm'),('Self-attention',f'{heads} heads, width {D} each','attention')],x=45,w=310,y=125,gap=100)
 p.wire([(200,95),(200,125)],end=ids[0]);p.sum('a1',200,365);p.connect(ids[-1],'a1');p.wire([(200,95),(445,95),(445,365),(213,365)],end='a1')
 p.dot(200,420);p.wire([(200,378),(200,420)],start='a1',arrow=False)
 ids=chain(p,'m',[('LayerNorm',f'197 × {E}','norm'),('Linear',f'{E} to {M}','linear'),('GELU',f'197 × {M}','activation'),('Linear',f'{M} to {E}','linear')],x=45,w=310,y=465,gap=95)
 p.wire([(200,420),(200,465)],end=ids[0]);p.sum('a2',200,920);p.connect(ids[-1],'a2');p.wire([(200,420),(445,420),(445,920),(213,920)],end='a2');p.text(45,981,f'Output: 197 × {E}',16)
 p=d.panel('attention','Self-attention',1070,220,700,1360,kind='attention',dashed=True,block_type='attention')
 p.box('qkv',185,65,330,'Linear QKV',detail=f'{E} to {Q}',kind='linear');p.box('split',185,165,330,'Reshape and split Q, K, V',detail=f'Each: {heads} × 197 × {D}',kind='split');p.connect('qkv','split')
 for id,x in [('q',30),('k',265),('v',500)]:p.box(id,x,280,170,id.upper(),detail=f'{heads} × 197 × {D}',kind='plain')
 p.wire([(250,214),(250,247),(115,247),(115,280)],start='split',end='q');p.wire([(350,214),(350,280)],start='split',end='k');p.wire([(450,214),(450,247),(585,247),(585,280)],start='split',end='v')
 p.box('scores',95,410,355,'MatMul Q K-transpose',detail=f'{heads} × 197 × 197',kind='attention')
 p.wire([(115,329),(115,372),(185,372),(185,410)],start='q',end='scores');p.wire([(350,329),(350,410)],start='k',end='scores')
 p.box('scale',95,515,355,'Scale',detail=f'Divide by sqrt({D})',kind='attention');p.connect('scores','scale')
 p.box('soft',95,620,355,'Softmax over keys',detail=f'{heads} × 197 × 197',kind='activation');p.connect('scale','soft')
 p.box('av',95,745,355,'MatMul attention × V',detail=f'{heads} × 197 × {D}',kind='attention');p.connect('soft','av');p.connect('v','av',via=[(585,769.5)],to_port='right')
 ids=chain(p,'out',[('Transpose and merge heads',f'197 × {E}','plain'),('Linear output projection',f'{E} to {E}','linear')],x=95,w=355,y=870,gap=110);p.connect('av',ids[0]);p.text(35,1190,'SDPA is drawn as its primitive equation.',16);p.text(35,1225,'No causal mask or relative position bias.',16)
 p=d.panel('table','Variant values',30,1620,1740,245)
 columns=[('Size',30),('E: token width',230),('n: block repeats',520),('h: heads',860),('MLP width',1120),('Head width',1420)]
 for label,x in columns:p.text(x,70,label,17,weight=700)
 for j,(key,s) in enumerate(specs.items()):
  for x,v in zip([30,230,520,860,1120,1420],[key,s['embed_dim'],s['depth'],s['num_heads'],s['embed_dim']*4,s['embed_dim']//s['num_heads']]):p.text(x,107+j*32,str(v),17)
 b.save(d,size+'-classify',size,kind='family' if family else 'concrete',verification='source' if family else b.evidence[size]['device'].replace('meta','meta'))

def run(family,title,cls):
 a=setup();b=Book(a,family,title);m=load(a.source,family);import torch
 specs={k:(vars(v) if not isinstance(v,dict) else v) for k,v in m.ARCH_DEFS.items()}
 torch.set_num_threads(2)
 for size in specs:
  device='cpu' if size==next(iter(specs)) else 'meta'
  with torch.device(device):model=getattr(m,cls)(size=size)
  b.evidence[size]=shapes(model,[1,3,224,224],device);draw(b,size,specs)
 draw(b,'family',specs);b.finish()
if __name__=='__main__':run('vit','ViT','VisionTransformer')
