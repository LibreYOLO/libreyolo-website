"""FCN with dilated ResNet-50/101. Also provides exact shared ResNet block drawings."""
from quicksrnet import *

def resnet_backbone(d,size,input_size=520,x=40,y=230,width=560,height=1130,normalize=True):
 p=d.panel('backbone','Dilated ResNet backbone',x,y,width,height)
 n3='N3' if size=='family' else 6 if size=='r50' else 23
 rows=[('input','Input RGB',f'3 × {input_size} × {input_size}','plain'),('norm','ImageNet normalization' if normalize else 'Normalized input', '(x-mean)/std, per RGB channel' if normalize else 'Normalization occurs in preprocessing','norm'),('stemconv','Conv2d 7×7, stride 2','3 to 64; padding 3','conv2d'),('stembn','BatchNorm2d + ReLU','64 × 260 × 260','norm'),('pool','MaxPool2d 3×3, stride 2','64 × 130 × 130; padding 1','pool'),('layer1','Bottleneck stage 1, n=3','256 × 130 × 130; width 64','bottleneck'),('layer2','Bottleneck stage 2, n=4','512 × 65 × 65; width 128','bottleneck'),('layer3',f'Bottleneck stage 3, n={n3}','1024 × 65 × 65; width 256','bottleneck'),('layer4','Bottleneck stage 4, n=3','2048 × 65 × 65; width 512','bottleneck')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,75,60+i*105,410,label,detail,kind,block=id if id.startswith('layer') else '')
 chain(p,[r[0] for r in rows])
 p.text(25,1050,'Output stride 8. Stage 3/4 replace spatial strides with dilation.',14)
 return p

def resnet_block_definitions(d,size,y=1440,counts_override=None,dilated=True):
 counts=counts_override or [3,4,'N3' if size=='family' else 6 if size=='r50' else 23,3]
 for stage in range(4):
  xx=40+(stage%2)*880;yy=y+(stage//2)*1140
  p=d.panel('stage-def'+str(stage),f'Stage {stage+1} bottlenecks (n={counts[stage]})',xx,yy,840,1080,kind='bottleneck',dashed=True,block_type='layer'+str(stage+1))
  base=[64,128,256,512][stage];cout=base*4;cin=[64,256,512,1024][stage]
  stride=2 if (stage==1 or (not dilated and stage>1)) else 1
  for first in [True,False]:
   off=0 if first else 420;prefix=f's{stage}-'+('first-' if first else 'rest-')
   ci=cin if first else cout;ss=stride if first else 1;di=([1,1,1,2] if first else [1,1,2,4])[stage] if dilated else 1
   p.text(off+25,70,'First block' if first else f'Repeated block, n={counts[stage]-1 if isinstance(counts[stage],int) else str(counts[stage])+"-1"}',16,weight=700)
   def b(id,yy,label,detail='',kind='plain'):
    return op(p,prefix+id,off+145,yy,250,label,detail,kind,h=42)
   rows=[('in',105,'Input',f'{ci} channels','plain'),('c1',180,'Conv2d 1×1',f'{ci} to {base}, stride 1','conv2d'),('bn1',245,'BatchNorm2d',f'{base} channels','norm'),('a1',310,'ReLU','', 'activation'),('c2',375,'Conv2d 3×3',f'{base} to {base}; s={ss}, d=p={di}','conv2d'),('bn2',440,'BatchNorm2d',f'{base} channels','norm'),('a2',505,'ReLU','','activation'),('c3',570,'Conv2d 1×1',f'{base} to {cout}, stride 1','conv2d'),('bn3',635,'BatchNorm2d',f'{cout} channels','norm')]
   for id,by,label,detail,kind in rows:b(id,by,label,detail,kind)
   chain(p,[prefix+r[0] for r in rows]);p.sum(prefix+'add',off+270,800);p.connect(prefix+'bn3',prefix+'add')
   if first:
    op(p,prefix+'sc',off+15,295,110,'Conv 1×1',f'{ci} to {cout}; s={ss}','conv2d',h=58)
    op(p,prefix+'sbn',off+15,435,110,'BatchNorm',f'{cout} channels','norm',h=58)
    p.connect(prefix+'in',prefix+'sc',from_port='left',via=[(off+70,126),(off+70,270)])
    p.connect(prefix+'sc',prefix+'sbn');p.connect(prefix+'sbn',prefix+'add',via=[(off+70,800)],to_port='left')
   else:p.connect(prefix+'in',prefix+'add',from_port='left',to_port='left',via=[(off+40,126),(off+40,800)])
   b('act',870,'ReLU');p.connect(prefix+'add',prefix+'act')
   p.text(off+25,965,'Conv bias=False; addition precedes final ReLU.',13)
  p.text(25,1035,'s: stride; d: dilation; p: padding. First block projects the residual branch.',14)

def head(p,prefix,x,y,ci,hidden,tag):
 nodes=[('in',tag,f'{ci} × 65 × 65','plain'),('conv','Conv2d 3×3',f'{ci} to {hidden}; s=1, p=1, bias=False','conv2d'),('bn','BatchNorm2d',f'{hidden} channels','norm'),('relu','ReLU','','activation'),('drop','Dropout 0.1','Identity in eval','plain'),('class','Conv2d 1×1',f'{hidden} to 21; bias=True','conv2d'),('resize','Bilinear resize to 520 × 520','align_corners=False','pool'),('out',prefix+' logits','21 × 520 × 520','plain')]
 for i,(id,label,detail,kind) in enumerate(nodes):op(p,prefix+id,x,y+100*i,440,label,detail,kind)
 chain(p,[prefix+n[0] for n in nodes])

def build(a,size,verification):
 d=diagram(a,'FCN family' if size=='family' else 'FCN '+size.upper(),'Semantic segmentation, 21 classes, 520 × 520 RGB, native eval with auxiliary output. Shapes exclude batch.','fcn',1800,3920)
 p=resnet_backbone(d,size)
 q=d.panel('heads','FCN heads',640,230,1120,1130)
 head(q,'main',55,75,2048,512,'From stage 4 (S4)');head(q,'aux',625,75,1024,256,'From stage 3 (S3)')
 for idx,layer in [(3,'layer3'),(4,'layer4')]:
  y=p.port(layer,'right')[1];op(p,'source'+str(idx),490,y-13,50,'S'+str(idx),h=26,block='signal-S'+str(idx));p.connect(layer,'source'+str(idx),from_port='right',to_port='left')
 q.text(25,955,'Native output is an OrderedDict with out and aux; both paths execute during eval.',15)
 q.text(25,995,'The public semantic result can use argmax over the 21 out channels.',15)
 q.text(25,1035,'This is the dilated-ResNet adaptation, not the original VGG FCN-8s topology.',15)
 resnet_block_definitions(d,size)
 d.text(50,3775,'Shared family variable N3: 6 for r50, 23 for r101. All other stage counts and dimensions are identical.',16)
 d.text(50,3810,'ResNet backbone comes from torchvision (BSD-3-Clause); native FCN head and orchestration are in LibreYOLO.',15)
 return finish_view(a,d,'fcn','family' if size=='family' else size+'-semantic','Shared r50/r101 topology' if size=='family' else size,'semantic',size,'family' if size=='family' else 'concrete','3×520×520',verification)

def main():
 a=environment('Build FCN r50/r101 and shared family diagrams')
 if a.verify:
  nn=nn_module(a,'fcn');records={}
  for size in ['r50','r101']:records[size]=cpu_probe(nn.LibreFCNModel(size,num_classes=21,aux_loss=True),(1,3,520,520),['backbone.conv1','backbone.layer1','backbone.layer2','backbone.layer3','backbone.layer4','classifier','aux_classifier'])
  write_evidence(a,'fcn',records,['libreyolo/models/fcn/nn.py:LibreFCNModel and FCNHead','libreyolo/models/fcn/model.py:INPUT_SIZES and class defaults','torchvision.models.resnet:Bottleneck'],['ResNet50/101 use dilation replacement [False,True,True]. First stage3/4 blocks retain previous dilation1/2; remaining blocks use2/4.','Auxiliary FCN head executes in native eval.','Reproduction requires installed BSD-3-Clause torchvision; no pretrained weights.'])
 ev=read_evidence('fcn');views=[build(a,s,'cpu' if s in ev['records'] else 'source') for s in ['r50','r101']]+[build(a,'family','source')];manifest(a,'fcn','fcn','FCN',views)
if __name__=='__main__':main()
