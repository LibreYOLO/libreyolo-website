from clip import *
def draw(b,size,task):
 sym=size=='family';S=518 if task=='semantic' else 224;G=S//14;N=G*G+1
 d=b.diagram('DINOv2 '+size+' '+task,f'{task.capitalize()}, {S} × {S} input. DINOv2-S: 384 channels, 12 blocks, 6 heads. Tensor sizes exclude batch.',2260,4150 if task!='embed' else 2040)
 p=d.panel('encoder','DINOv2-S encoder',25,220,660,1600)
 items=[('Input',f'3 × {S} × {S}','plain'),('Normalize RGB','ImageNet mean/std' if task=='semantic' else 'Classification transform outside network','plain'),('Conv2d 14×14 / 14',f'384 × {G} × {G}, bias=True','conv2d'),('Flatten patch grid',f'{G*G} × 384','plain')]
 ids=chain(p,'prep',items,x=30,w=330,y=65,gap=80)
 p.box('cls',440,305,175,'Learned CLS',detail='1 × 384');p.box('tokens',30,455,330,'Concat CLS and patches',detail=f'{N} × 384',kind='concat');p.connect(ids[-1],'tokens');p.connect('cls','tokens',via=[(527.5,479.5)],to_port='right')
 p.box('pos',440,555,175,'Position embedding',detail='1,370 × 384',font_size=13);p.box('interp',410,660,215,'Bicubic interpolation',detail='37×37 to 16×16 patches' if S==224 else '37×37 unchanged',font_size=13);p.connect('pos','interp')
 p.sum('addpos',195,755);p.connect('tokens','addpos');p.connect('interp','addpos',via=[(517.5,755)],to_port='right')
 for i in range(4):
  p.box('layers'+str(i),30,830+i*150,330,f'Encoder blocks {i*3+1} to {i*3+3}',detail=f'{N} × 384, repeats=3',kind='attention')
  if i:p.connect('layers'+str(i-1),'layers'+str(i))
  else:p.connect('addpos','layers0')
  if task!='embed':
   p.box('tap'+str(i),430,830+i*150,205,f'Tap stage {3*(i+1)}',detail=f'384 × {G} × {G}',font_size=14)
   p.connect('layers'+str(i),'tap'+str(i),from_port='right',to_port='left')
 p.text(30,1480,'One attention window; zero register tokens.',16)
 p.text(30,1520,'Taps apply LayerNorm, remove CLS, reshape.',16)
 # Exact DINOv2 residual unit includes learned layer scales.
 p=d.panel('block','Encoder block',715,220,620,1600,kind='attention',dashed=True)
 p.text(30,65,f'Input {N} × 384',16);p.dot(195,105)
 ids=chain(p,'enc1',[('LayerNorm',f'{N} × 384, eps=1e-6','norm'),('Self-attention','6 heads, 64 channels per head','attention'),('Multiply layer scale','384 learned channel weights','plain')],w=330,y=150,gap=105);p.wire([(195,105),(195,150)],end=ids[0]);p.sum('r1',195,520);p.connect(ids[-1],'r1');p.wire([(195,105),(555,105),(555,520),(208,520)],end='r1')
 p.dot(195,580);p.wire([(195,533),(195,580)],start='r1',arrow=False)
 ids=chain(p,'enc2',[('LayerNorm',f'{N} × 384','norm'),('Linear','384 to 1,536','linear'),('GELU',f'{N} × 1,536','activation'),('Linear','1,536 to 384','linear'),('Multiply layer scale','384 learned channel weights','plain')],w=330,y=625,gap=105);p.wire([(195,580),(195,625)],end=ids[0]);p.sum('r2',195,1190);p.connect(ids[-1],'r2');p.wire([(195,580),(555,580),(555,1190),(208,1190)],end='r2');p.text(30,1260,f'Output {N} × 384',16)
 attention(d,'selfattn','Encoder self-attention',1370,220,384,6,N,64)
 if task=='embed':
  p=d.panel('head','Embedding output',1370,1530,860,290,kind='pool');chain(p,'emb',[('Final LayerNorm',f'{N} × 384','norm'),('Select CLS token','384-dimensional whole-image embedding','plain')],w=590,y=60,gap=90)
  d.text(30,1895,'n, s, m and l use the same encoder graph. The embedding wrapper removes projector and task-head parameters.',17)
 else:
  p=d.panel('projector','Projector P4',25,1880,960,1510,kind='aggregate',dashed=True)
  for i in range(4):p.box('feature'+str(i),20+i*235,65,215,f'Tap stage {3*(i+1)}',detail=f'384 × {G} × {G}',font_size=14)
  p.box('fuse',60,215,830,'Concat four selected features',detail=f'1,536 × {G} × {G}',kind='concat')
  for i in range(4):p.wire([(127.5+i*235,114),(127.5+i*235,155+i*12),(140+i*220,155+i*12),(140+i*220,215)],start='feature'+str(i),end='fuse')
  ids=chain(p,'cv1',[('Conv2d 1×1','1,536 to 256, bias=False','conv2d'),('Channel LayerNorm',f'256 × {G} × {G}','norm'),('SiLU',f'256 × {G} × {G}','activation'),('Split channels','a: 128 channels; b: 128 channels','split')],x=545,w=345,y=330,gap=85);p.connect('fuse',ids[0],via=[(475,295),(717.5,295)])
  for i in range(3):
   p.box('bneck'+str(i),545,720+i*160,345,'Bottleneck (no shortcut)',detail=f'128 × {G} × {G}',kind='bottleneck')
   p.connect(ids[-1] if i==0 else 'bneck'+str(i-1),'bneck'+str(i))
  p.box('c2cat',60,1230,830,'Concat a, b, bottleneck 1, 2, 3',detail=f'640 × {G} × {G}',kind='concat')
  # Each of the five concat inputs receives a distinct, spacious lane.
  p.wire([(545,609.5),(80,609.5),(80,1230)],start=ids[-1],end='c2cat')
  for i,(source,yy) in enumerate([(ids[-1],685),('bneck0',825),('bneck1',985),('bneck2',1145)]):
   sx,sy=p.port(source);lane=180+i*100;p.dot(sx,yy);p.wire([(sx,sy),(sx,yy)],start=source,arrow=False);p.wire([(sx,yy),(lane,yy),(lane,1230)],start=source,end='c2cat')
  p.box('outproj',60,1340,830,'Conv2d 1×1, channel LayerNorm, SiLU',detail=f'640 to 256; output 256 × {G} × {G}',kind='conv')
  p.connect('c2cat','outproj');p.text(60,1440,'The projection ConvX is expanded in the definition to the right.',16)
  p=d.panel('bottleneck','Projector Bottleneck',1020,1880,540,880,kind='bottleneck',dashed=True)
  chain(p,'bn',[('Conv2d 3×3 / 1','128 to 128, p=1, bias=False','conv2d'),('Channel LayerNorm',f'128 × {G} × {G}','norm'),('SiLU',f'128 × {G} × {G}','activation'),('Conv2d 3×3 / 1','128 to 128, p=1, bias=False','conv2d'),('Channel LayerNorm',f'128 × {G} × {G}','norm'),('SiLU',f'128 × {G} × {G}','activation')],w=450,gap=105)
  p.text(30,780,'No residual addition in this C2f configuration.',16)
  p=d.panel('projectorout','Projector output ConvX',1020,2810,540,580,kind='conv',dashed=True)
  chain(p,'proj',[('Conv2d 1×1','640 to 256, p=0, bias=False','conv2d'),('Channel LayerNorm',f'256 × {G} × {G}','norm'),('SiLU',f'256 × {G} × {G}','activation'),('Final channel LayerNorm',f'256 × {G} × {G}','norm')],w=450,gap=105)
  p=d.panel('head','Task head',1600,1880,630,1510,kind='pool')
  if task=='classify':items=[('Projector P4 output',f'256 × 16 × 16','plain'),('AdaptiveAvgPool2d','256 × 1 × 1','pool'),('Flatten','256','plain'),('Dropout (eval identity)','p=0.2','plain'),('Linear classifier','256 to 1,000 logits','linear')]
  else:items=[('Projector P4 output','256 × 37 × 37','plain'),('Lateral Conv2d 1×1','256 to 256, bias=True','conv2d'),('Smoothing Conv2d 3×3','256 to 256, p=1, bias=True','conv2d'),('GroupNorm','32 groups, 256 channels','norm'),('GELU','256 × 37 × 37','activation'),('Dropout2d (eval identity)','p=0.1','plain'),('Predict Conv2d 1×1','256 to 19 class logits','conv2d'),('Bilinear resize','19 × 518 × 518, align_corners=False','plain')]
  chain(p,'task',items,w=520,gap=120)
  d.text(35,3470,'C2f repeats three Bottlenecks. Each receives the preceding output; every intermediate feature is concatenated.',17)
  d.text(35,3510,'Only P4 is configured in all four sizes, so semantic fusion contains one lateral and no multi-level additions.',17)
  p=d.panel('familytable','Size equivalence',25,3580,2205,340)
  for x,label in [(30,'Size'),(320,'Encoder width / depth / heads'),(1050,'Projector width'),(1550,'Selected layers')]:p.text(x,70,label,18,weight=700)
  for j,key in enumerate(['n','s','m','l']):
   for x,value in [(30,key),(320,'384 / 12 / 6'),(1050,'256'),(1550,'3, 6, 9, 12')]:p.text(x,120+j*45,value,18)
  p.text(30,305,'All fields used by these heads are identical across sizes at the pinned revision; detector decoder settings do not participate.',17)
 b.save(d,size+'-'+task,size,kind='family' if sym else 'concrete',verification='cpu' if size=='n' else 'source',task=task,input=f'1 × 3 × {S} × {S}')

def main():
 a=setup();b=Book(a,'dinov2','DINOv2',sourcefile='model.py');m=load(a.source,'dinov2','model.py');r=load(a.source,'rfdetr');import torch
 torch.set_num_threads(2)
 fields=['encoder','hidden_dim','out_feature_indexes','projector_scale']
 assert all(all(getattr(r.RFDETR_CONFIGS[key],f)==getattr(r.RFDETR_CONFIGS['n'],f) for f in fields) for key in ['n','s','m','l'])
 for task in ['semantic','classify','embed']:
  cls={'semantic':m._DINOv2ModelWrapper,'classify':m._DINOv2ClassifierWrapper,'embed':m._DINOv2EmbedderWrapper}[task];kwargs={'config':'n','device':'cpu','load_dinov2_weights':False}
  if task!='embed':kwargs['nb_classes']=19 if task=='semantic' else 1000
  model=cls(**kwargs).eval();S=518 if task=='semantic' else 224
  ev=shapes(model,[1,3,S,S],'cpu');b.evidence[task]=ev
  for size in ['n','s','m','l','family']:draw(b,size,task)
 b.evidence['size_equivalence']={'compared_fields':fields,'sizes':['n','s','m','l'],'all_equal':True,'runtime':'Only n executed; other views use source-equivalent factory settings.'}
 b.finish()
if __name__=='__main__':main()
