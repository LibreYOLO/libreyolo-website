from clip import *
def draw(b,id,size,c,task,frames,classes,specs):
 sym=size=='family';E='E' if sym else c.hidden_size;S='S' if sym else c.crop_size;F='F' if sym else frames;G='S/16' if sym else c.grid_size;Z='F/2' if sym else frames//2;N='N' if sym else frames//2*c.grid_size*c.grid_size;h='h' if sym else c.num_attention_heads;hd='E/h' if sym else c.hidden_size//c.num_attention_heads;M='M' if sym else int(c.hidden_size*c.mlp_ratio);Mp='4E' if sym else c.hidden_size*4;depth='n' if sym else c.num_hidden_layers;chunk='r' if sym else 2*((hd//3)//2);remain='u' if sym else hd-3*chunk
 d=b.diagram('V-JEPA 2 '+id,f'{task.capitalize()}, {F} frames at {S} × {S}. Video layout F × C × H × W, excluding batch.',2470,4920 if task=='classify' else 3290)
 p=d.panel('network','Video encoder',30,220,620,1690)
 items=[('Video clip',f'{F} × 3 × {S} × {S}','plain'),('Permute to C, F, H, W',f'3 × {F} × {S} × {S}','plain'),('Conv3d tubelets','Kernel/stride 2×16×16, bias=True','conv2d'),('Flatten spatiotemporal grid',f'{N} × {E}; grid {Z} × {G} × {G}','plain'),('Encoder block',f'{N} × {E}, repeats={depth}','attention'),('Final LayerNorm',f'{N} × {E}, eps=1e-6','norm')]
 ids=chain(p,'enc',items,w=500,gap=110)
 p.text(30,900,'No CLS token or absolute position embedding.',17);p.text(30,940,'3D rotary position applies to Q and K.',17)
 if task=='embed':
  ids=chain(p,'embedding',[('Mean over all tokens',str(E),'pool'),('L2 normalize',str(E),'plain')],w=500,y=1080,gap=110);p.connect('enc5',ids[0],via=[(280,1000)])
  p.text(30,1410,f'embed_tokens also exposes {Z} × {G} × {G} × {E}.',16)
 else:
  ids=chain(p,'poolflow',[('Pooler self-attention block',f'{N} × {E}, repeats=3','attention'),('Pooler cross-attention block',f'1 × {E}, one learned query','attention'),('Squeeze query axis',str(E),'plain'),('Linear classifier',f'{E} to {classes} logits','linear')],w=500,y=1030,gap=120);p.connect('enc5',ids[0],via=[(280,980)])
 residual(d,'encoder-block','Encoder block',700,220,E,N,M,h,height=1690)
 attention(d,'encoder-attn','Encoder rotary self-attention',1245,220,E,h,N,hd,rope='Rotate time, height, width',height=1450)
 p=d.panel('rotary','Factorized 3D rotary',1920,220,520,1690,kind='attention',dashed=True)
 p.box('qk-in',25,65,330,'Q or K head',detail=f'{N} × {hd}')
 p.box('split-rope',25,175,330,'Split channel slices',detail=f'Time {chunk}, H {chunk}, W {chunk}, rest {remain}',kind='split');p.connect('qk-in','split-rope')
 for i,axis in enumerate(['Time','Height','Width']):
  p.box('rotate'+str(i),200,300+i*160,280,axis+' rotary slice',detail=f'{N} × {chunk}',kind='attention')
  p.wire([(75+i*50,224),(75+i*50,260+i*160),(340,260+i*160),(340,300+i*160)],start='split-rope',end='rotate'+str(i))
 p.box('rot-cat',25,900,460,'Concat rotated slices and remainder',detail=f'{N} × {hd}',kind='concat')
 for i in range(3):
  p.wire([(340,349+i*160),(340,375+i*160),(45+i*50,375+i*160),(45+i*50,900)],start='rotate'+str(i),end='rot-cat')
 p.wire([(355,199.5),(495,199.5),(495,930),(485,930)],start='split-rope',end='rot-cat')
 p.text(25,1050,'Remainder channels are unchanged.',16);p.text(25,1090,'V receives no rotary transformation.',16)
 p.text(25,1190,'Primitive slice definition is below.',16)
 # Elementary rotation shared by the three axes, with exact upstream sine/cosine convention.
 p=d.panel('rotation-math','Rotary slice primitive',30,1980,1190,970,kind='attention',dashed=True)
 p.box('slice',25,65,330,'Input channel slice',detail=f'{N} × {chunk}');p.box('position',780,65,370,'Axis position IDs',detail='Time-major, then row, then column')
 p.box('frequency',780,205,370,'Multiply position × frequency',detail=f'omega[j] = 10000^(-2j/{chunk})');p.connect('position','frequency')
 p.box('sin',780,335,370,'Sin and duplicate half table',detail='Concatenate half-width table twice');p.box('cos',390,335,330,'Cos and duplicate half table',detail='Concatenate half-width table twice');p.connect('frequency','sin');p.connect('frequency','cos',from_port='left',via=[(555,229.5)])
 p.box('rotate-pair',25,205,330,'Pair adjacent channels',detail='[a,b] becomes [-b,a]');p.connect('slice','rotate-pair')
 p.box('times-sin',25,520,330,'Multiply rotation × sin',detail=f'{N} × {chunk}');p.connect('rotate-pair','times-sin');p.wire([(965,384),(965,440),(100,440),(100,520)],start='sin',end='times-sin')
 p.box('times-cos',390,520,330,'Multiply input × cos',detail=f'{N} × {chunk}');p.connect('cos','times-cos');p.connect('slice','times-cos',from_port='right',via=[(375,89.5),(375,544.5)],to_port='left')
 p.sum('rotation-add',370,730);p.connect('times-sin','rotation-add',via=[(190,730)],to_port='left');p.connect('times-cos','rotation-add',via=[(555,730)],to_port='right');p.text(25,835,'Preserves the upstream adjacent-pair rotation with duplicated half-width sine/cosine tables.',16)
 p=d.panel('variants','Variant values',1250,1980,1190,970)
 cols=[('Size',25),('S',210),('E',365),('n',545),('h',705),('M',865)]
 for label,x in cols:p.text(x,70,label,17,weight=700)
 for j,(key,sp) in enumerate(specs.items()):
  for x,v in zip([25,210,365,545,705,865],[key,sp['crop_size'],sp['hidden_size'],sp['num_hidden_layers'],sp['num_attention_heads'],int(sp['hidden_size']*sp['mlp_ratio'])]):p.text(x,120+j*47,str(v),17)
 p.text(25,365,'Encoder artifacts: F=64 for every size.',17)
 p.text(25,410,'Released probes:',17,weight=700)
 for j,line in enumerate(['l256 ssv2: F=16, 174 classes','l256 diving48: F=32, 48 classes','g384 ssv2: F=64, 174 classes','g384 diving48: F=32, 48 classes']):p.text(25,455+j*42,line,17)
 p.text(25,675,'N=(F/2) × (S/16)². r=2 floor(floor((E/h)/3)/2).',16)
 p.text(25,715,'u=(E/h)-3r. For l/g: r=20,u=4; h256: r=26,u=2.',16)
 p.text(25,755,'E: width; n: depth; h: heads; M: encoder MLP width.',16)
 p.text(25,805,'Pooler MLP is 4E, including g variants.',16)
 if task=='classify':
  residual(d,'pool-self','Pooler self-attention block',30,3020,E,N,Mp,h,height=1400)
  attention(d,'pool-self-attn','Pooler self-attention',570,3020,E,h,N,hd,height=1400)
  p=d.panel('crossblock','Pooler cross-attention block',1250,3020,510,1400,kind='attention',dashed=True)
  p.box('query',25,65,285,'Learned query',detail=f'1 × {E}');p.box('kvnorm',25,220,285,'LayerNorm on K/V tokens',detail=f'{N} × {E}',kind='norm');p.box('cross',25,375,285,'Cross-attention',detail='No output projection',kind='attention');p.connect('kvnorm','cross');p.connect('query','cross',via=[(365,89.5),(365,399.5)],from_port='right',to_port='right')
  p.sum('crossadd',167.5,560);p.connect('cross','crossadd');p.connect('query','crossadd',via=[(470,89.5),(470,560)],from_port='right',to_port='right')
  p.dot(167.5,615);p.wire([(167.5,573),(167.5,615)],start='crossadd',arrow=False)
  ids=chain(p,'crossmlp',[('LayerNorm',f'1 × {E}','norm'),('Linear',f'{E} to {Mp}','linear'),('GELU',f'1 × {Mp}','activation'),('Linear',f'{Mp} to {E}','linear')],x=25,w=285,y=650,gap=110);p.wire([(167.5,615),(167.5,650)],end=ids[0]);p.sum('mlpadd',167.5,1160);p.connect(ids[-1],'mlpadd');p.wire([(167.5,615),(470,615),(470,1160),(180.5,1160)],end='mlpadd')
  attention(d,'crossattn','Pooler cross-attention',1790,3020,E,h,1,hd,K=N,height=1400,out_projection=False)
 b.save(d,id,size,kind='family' if sym else 'concrete',verification='source' if sym else 'meta',task=task,input=f'1 × {F} × 3 × {S} × {S}')

def main():
 a=setup();b=Book(a,'vjepa2','V-JEPA 2');m=load(a.source,'vjepa2');import torch
 probes={('l256','ssv2'):(16,174),('l256','diving48'):(32,48),('g384','ssv2'):(64,174),('g384','diving48'):(32,48)}
 variants=[(size,'embed',64,None,size+'-embed') for size in m.VJEPA2_CONFIGS]+[(size,'classify',f,k,size+'-classify-'+dataset) for (size,dataset),(f,k) in probes.items()]
 for size,task,frames,classes,id in variants:
  c=m.VJEPA2Config.for_size(size,frames_per_clip=frames)
  with torch.device('meta'):model=m.LibreVJEPA2Encoder(c) if task=='embed' else m.LibreVJEPA2Classifier(c,classes)
  b.evidence[id]=shapes(model,[1,frames,3,c.crop_size,c.crop_size],'meta');b.evidence[id]['config']=vars(c);draw(b,id,size,c,task,frames,classes,m.VJEPA2_CONFIGS)
 for task in ['embed','classify']:
  c=m.VJEPA2Config.for_size('l256');draw(b,'family-'+task,'family',c,task,64,'K',m.VJEPA2_CONFIGS)
 b.finish()
if __name__=='__main__':main()
