from alexnet import *
def main():
 a=setup();b=Book(a,'vgg','VGG');m=load(a.source,'vgg');import torch
 for size in m.ARCH_DEFS:
  with torch.device('meta'):model=m.VGG(size,init_weights=False)
  ev=shapes(model,[1,3,224,224]);b.evidence[size]=ev
  d=b.diagram('VGG-'+size,'Classification, 224 × 224 input, 1,000 classes. Tensor sizes exclude batch.',2360,1500)
  stages=[];cur=[]
  for name,layer in model.features.named_children():
   cur.append(op(layer,ev['modules']['features.'+name]['output'][1:]))
   if isinstance(layer,torch.nn.MaxPool2d):stages.append(cur);cur=[]
  ids=[]
  for i,items in enumerate(stages):
   p=d.panel('s'+str(i),'Stage '+str(i+1),30+i*390,220,365,1080)
   items=[('Input',dim([3 if i==0 else [64,128,256,512][i-1],224//2**i,224//2**i]),'plain')]+items
   ids.append(chain(p,'s'+str(i),items,x=22,w=320,gap=72))
  p=d.panel('head','Classifier',1980,220,350,1080,kind='pool')
  items=[op(model.avgpool,[512,7,7]),('Flatten','25,088','plain')]+[op(l,ev['modules']['classifier.'+n]['output'][1:]) for n,l in model.classifier.named_children()];ids.append(chain(p,'head',items,x=20,w=310))
  # Named continuations keep independent column flows separate without long return routes.
  for i in range(5):
   sx,sy=d.port(ids[i][-1]);d.text(sx,sy+28,f'Continue at '+('classifier' if i==4 else f'stage {i+2}'),14,anchor='middle')
  d.text(50,1350,'Conv2d: bias=True, 3×3, stride 1, padding 1. MaxPool2d: 2×2, stride 2, padding 0.',17)
  d.text(50,1382,'Stage inputs are the preceding stage outputs. Dropout is identity in eval. Output: 1,000 logits.',17)
  b.save(d,size+'-classify',size)
 for bn in [False,True]:
  suffix='bn' if bn else '';d=b.diagram('VGG family'+(' with BatchNorm' if bn else ''),'Shared stage structure. n1..n5 are convolution counts; stage widths and input geometry are fixed.',2000,1770)
  for i,c in enumerate([64,128,256,512,512]):
   p=d.panel('p'+str(i),'Stage '+str(i+1),30+i*395,220,370,1040,kind='conv',dashed=True)
   prev=3 if i==0 else [64,128,256,512,512][i-1];hw=224//2**i
   items=[('Input',dim([prev,hw,hw]),'plain'),('Conv2d 3×3 / 1',f'{prev} to {c}, p=1','conv2d')]
   if bn:items.append(('BatchNorm2d',str(c)+' channels','norm'))
   items.append(('ReLU',dim([c,hw,hw]),'activation'))
   first=chain(p,'base'+str(i),items,w=310)
   p.text(25,415,f'Repeated unit, n{i+1}-1 copies',16,weight=700)
   items=[('Conv2d 3×3 / 1',f'{c} to {c}, p=1','conv2d')]
   if bn:items.append(('BatchNorm2d',str(c)+' channels','norm'))
   items.append(('ReLU',dim([c,hw,hw]),'activation'))
   repeated=chain(p,'repeat'+str(i),items,y=450,w=310)
   sx,sy=p.port(first[-1],'right');p.connect(first[-1],repeated[0],from_port='right',via=[(350,sy),(350,435),(185,435)])
   p.box('pool'+str(i),30,800,310,'MaxPool2d 2×2 / 2',detail=dim([c,hw//2,hw//2]),kind='pool')
   p.connect(repeated[-1],'pool'+str(i))
   p.text(25,920,'Continue at '+('classifier' if i==4 else f'stage {i+2}'),16)
  d.text(50,1310,'Variant',18,weight=700)
  for i in range(5):d.text(340+i*150,1310,f'n{i+1}',18,weight=700)
  for y,size,vals in [(1350,'16',[2,2,3,3,3]),(1390,'19',[2,2,4,4,4])]:
   d.text(50,y,size+suffix,18)
   for i,v in enumerate(vals):d.text(340+i*150,y,str(v),18)
  p=d.panel('classifier','Classifier',30,1440,1940,190,kind='pool')
  items=[('AvgPool 7×7','512 × 7 × 7','pool'),('Flatten','25,088','plain'),('Linear','25,088 to 4,096','linear'),('ReLU','4,096','activation'),('Dropout','eval identity','plain'),('Linear','4,096 to 4,096','linear'),('ReLU','4,096','activation'),('Dropout','eval identity','plain'),('Linear','4,096 to 1,000','linear')]
  for j,(label,detail,kind) in enumerate(items):
   p.box('h'+str(j),20+j*213,75,192,label,detail=detail,kind=kind)
   if j:p.connect('h'+str(j-1),'h'+str(j),from_port='right',to_port='left')
  b.save(d,'family-'+('bn' if bn else 'plain'),'16/19'+suffix,kind='family',verification='source')
 b.finish()
if __name__=='__main__':main()
