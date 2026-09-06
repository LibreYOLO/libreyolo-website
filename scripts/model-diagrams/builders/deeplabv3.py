"""DeepLabv3 ASPP with ResNet-50/101 or dilated MobileNetV3-Large."""
from quicksrnet import *
from fcn import resnet_backbone,resnet_block_definitions
from nafnet import product

def mb_specs(model):
 specs=[]
 for name,module in model.backbone.named_children():
  if not hasattr(module,'block'):continue
  convs=[];se=None
  for sub in module.block:
   if type(sub).__name__=='Conv2dNormActivation':
    c=sub[0];convs.append({'cin':c.in_channels,'cout':c.out_channels,'k':c.kernel_size[0],'s':c.stride[0],'d':c.dilation[0],'g':c.groups,'act':type(sub[-1]).__name__ if len(sub)>2 else 'none'})
   elif type(sub).__name__=='SqueezeExcitation':se={'cin':sub.fc1.in_channels,'reduced':sub.fc1.out_channels}
  depth=next(c for c in convs if c['g']>1)
  specs.append({'index':int(name),'cin':convs[0]['cin'],'expanded':depth['cin'],'cout':convs[-1]['cout'],'k':depth['k'],'s':depth['s'],'d':depth['d'],'act':depth['act'],'expand':len(convs)==3,'se':se,'residual':module.use_res_connect})
 return specs

def aspp_head(d,ci,hw):
 p=d.panel('aspp','ASPP and dense prediction head',640,230,1220,1800,kind='plain')
 p.text(25,67,f'B denotes the backbone output: {ci} × {hw} × {hw}. All five branches consume B.',16)
 for i in range(5):
  x=25+i*238
  op(p,f'b{i}',x,100,210,'B',f'{ci} channels','plain',h=40,block='signal-B')
  if i<4:
   label='Conv2d 1×1' if i==0 else 'Atrous Conv2d 3×3'
   detail=f'{ci} to 256; d={1 if i==0 else [12,24,36][i-1]}'
   op(p,f'c{i}',x,210,210,label,detail,'conv2d');p.connect(f'b{i}',f'c{i}')
   op(p,f'n{i}',x,310,210,'BatchNorm2d','256 channels','norm');p.connect(f'c{i}',f'n{i}')
   op(p,f'a{i}',x,410,210,'ReLU',f'256 × {hw} × {hw}','activation');p.connect(f'n{i}',f'a{i}')
  else:
   steps=[('pool','AdaptiveAvgPool2d(1)',f'{ci} × 1 × 1','pool'),('c','Conv2d 1×1',f'{ci} to 256','conv2d'),('n','BatchNorm2d','256 channels','norm'),('a','ReLU','','activation'),('resize','Bilinear resize',f'256 × {hw} × {hw}','pool')]
   for j,(id,label,detail,kind) in enumerate(steps):op(p,id+str(i),x,180+j*75,210,label,detail,kind,h=45)
   chain(p,['b4','pool4','c4','n4','a4','resize4'])
  bottom='a'+str(i) if i<4 else 'resize4'
  sx,sy=p.port(bottom,'bottom');p.wire([(sx,sy),(sx,640)],start=bottom,end='concat')
 op(p,'concat',25,640,1162,'Concat five 256-channel branches',f'1280 × {hw} × {hw}','concat')
 steps=[('projection','Conv2d 1×1','1280 to 256; bias=False','conv2d'),('pn','BatchNorm2d','256 channels','norm'),('pa','ReLU','','activation'),('drop','Dropout 0.5','Identity in eval','plain'),('hc','Conv2d 3×3','256 to 256; s=1, p=1, bias=False','conv2d'),('hn','BatchNorm2d','256 channels','norm'),('ha','ReLU','','activation'),('logits','Conv2d 1×1','256 to 21; bias=True','conv2d'),('resize-out','Bilinear resize to 520 × 520','align_corners=False','pool'),('output','Semantic logits','21 × 520 × 520','plain')]
 for j,(id,label,detail,kind) in enumerate(steps):op(p,id,385,730+j*95,450,label,detail,kind)
 chain(p,['concat']+[s[0] for s in steps]);p.text(25,1738,'No auxiliary FCN head in this native inference graph. ASPP conv padding equals dilation.',15)

def cba_def(d,x,y,w=560):
 p=d.panel('cbadef','ConvNormActivation definition',x,y,w,500,kind='conv',dashed=True,block_type='cba')
 for j,(id,label,detail,kind) in enumerate([('cc','Conv2d','Numeric channels, kernel, stride and dilation at occurrence','conv2d'),('cn','BatchNorm2d','One normalization parameter set per output channel','norm'),('ca','ReLU or Hardswish','Actual activation named at occurrence','activation')]):op(p,id,45,65+j*105,w-90,label,detail,kind)
 chain(p,['cc','cn','ca']);p.text(25,420,'A projection marked no activation stops after BatchNorm2d.',14)

def mb_backbone(d,specs,records):
 p=d.panel('backbone','Dilated MobileNetV3-Large',40,230,560,1800)
 op(p,'input',55,60,450,'Normalized RGB input','3 × 520 × 520')
 op(p,'stem',55,140,450,'ConvNormActivation 3×3','3 to 16; s=2, p=1; Hardswish','conv',block='cba');p.connect('input','stem')
 prev='stem'
 for j,s in enumerate(specs):
  shape=records.get('hooks',{}).get('backbone.'+str(s['index']),{}).get('output')
  if shape:hw=shape[-1]
  else:hw=[260,130,130,65,65,65,33,33,33,33,33,33,33,33,33][j]
  op(p,'mb'+str(s['index']),55,230+j*90,450,f'Inverted residual block {s["index"]}',f'{s["cin"]} / {s["expanded"]} / {s["cout"]} channels; output {s["cout"]} × {hw} × {hw}','bottleneck',block='mb-'+str(s['index']))
  p.connect(prev,'mb'+str(s['index']));prev='mb'+str(s['index'])
 op(p,'final',55,1590,450,'ConvNormActivation 1×1','160 to 960; stride 1; Hardswish','conv',block='cba');p.connect(prev,'final')
 op(p,'bout',55,1670,450,'Backbone output B','960 × 33 × 33','plain',block='signal-B');p.connect('final','bout')

def mb_definitions(d,specs):
 # Group only exact repeated numeric configurations; every occurrence stays named.
 groups={}
 for s in specs:
  key=tuple((k,str(v)) for k,v in s.items() if k!='index')
  groups.setdefault(key,[]).append(s)
 for j,ss in enumerate(groups.values()):
  s=ss[0];ids=', '.join(str(t['index']) for t in ss)
  p=d.panel('mbdef-'+str(j),'Inverted residual '+ids,40+(j%4)*465,2140+(j//4)*1130,425,1080,kind='bottleneck',dashed=True,block_type='mb-'+str(s['index']))
  pre='m'+str(j)
  def b(id,y,label,detail='',kind='plain'):return op(p,pre+id,120,y,275,label,detail,kind,h=46,block='cba' if id in ['exp','dw','proj'] else '')
  b('in',65,'Input',f'{s["cin"]} channels')
  b('exp',150,'ConvNormActivation 1×1' if s['expand'] else 'Identity expansion',f'{s["cin"]} to {s["expanded"]}; '+s['act'],'conv' if s['expand'] else 'plain')
  b('dw',235,f'Depthwise ConvNormAct {s["k"]}×{s["k"]}',f'{s["expanded"]} groups; s={s["s"]}, d={s["d"]}; '+s['act'],'conv')
  chain(p,[pre+'in',pre+'exp',pre+'dw']);prev=pre+'dw'
  if s['se']:
   for id,y,label,detail,kind in [('pool',330,'AdaptiveAvgPool2d(1)',f'{s["expanded"]} × 1 × 1','pool'),('reduce',410,'Conv2d 1×1',f'{s["expanded"]} to {s["se"]["reduced"]}','conv2d'),('relu',490,'ReLU','','activation'),('expand-se',570,'Conv2d 1×1',f'{s["se"]["reduced"]} to {s["expanded"]}','conv2d'),('hsig',650,'Hardsigmoid','Channel attention weights','activation')]:b(id,y,label,detail,kind)
   chain(p,[pre+'dw',pre+'pool',pre+'reduce',pre+'relu',pre+'expand-se',pre+'hsig']);product(p,pre+'mul',257.5,750);p.connect(pre+'hsig',pre+'mul')
   p.dot(257.5,306);p.wire([(257.5,306),(80,306),(80,750),(244.5,750)],start=pre+'dw',end=pre+'mul');prev=pre+'mul';py=805
  else:py=410
  b('proj',py,'Conv2d 1×1 + BatchNorm',f'{s["expanded"]} to {s["cout"]}; no activation','conv');p.connect(prev,pre+'proj')
  if s['residual']:
   p.sum(pre+'add',257.5,py+115);p.connect(pre+'proj',pre+'add');p.connect(pre+'in',pre+'add',from_port='left',to_port='left',via=[(30,88),(30,py+115)])
  else:p.text(30,py+119,'No residual addition: input/output channels differ',13)
  p.text(25,1010,'SE shown only when present; stride is the executed stride.',13)


def build(a,size,ev,specs):
 mb=size=='mv3';family=size=='family'
 height=6240 if mb else 4560
 d=diagram(a,'DeepLabv3 '+('ResNet family' if family else size.upper()),'Semantic segmentation, 21 classes, 520 × 520 RGB, native eval. Shapes exclude batch.','deeplabv3',1900,height)
 if mb:mb_backbone(d,specs,ev['records'].get('mv3',{}));mb_definitions(d,specs)
 else:
  p=resnet_backbone(d,size,normalize=False,height=1130)
  op(p,'b-tag',490,p.port('layer4','right')[1]-13,45,'B',h=26,block='signal-B');p.connect('layer4','b-tag',from_port='right',to_port='left')
  resnet_block_definitions(d,size,y=2140)
 cba_def(d,40,5540 if mb else 1410)
 if mb:
  table=d.panel('mbtable','Concrete MobileNet block settings',640,5540,1220,500)
  table.text(25,68,'Block      Input / expanded / output channels      Kernel / stride / dilation      SE reduction      Residual',15,weight=700)
  for j,s in enumerate(specs):
   se=s['se']['reduced'] if s['se'] else 'none'
   table.text(25,103+j*23,f"{s['index']:>2}               {s['cin']} / {s['expanded']} / {s['cout']}                          {s['k']} / {s['s']} / {s['d']}                           {se}                    {s['residual']}",14)
 aspp_head(d,960 if mb else 2048,33 if mb else 65)
 if family:d.text(50,height-140,'Shared ResNet family: N3=6 for r50, N3=23 for r101. MobileNetV3 has a separate topology and view.',16)
 if mb:d.text(50,height-140,'Blocks 13-15 execute dilation 2 with stride 1. Output is 33×33 at a 520×520 input, giving nominal stride 16.',16)
 verification='cpu' if size in ev['records'] else 'source'
 return finish_view(a,d,'deeplabv3','resnet-family' if family else size+'-semantic','ResNet shared topology' if family else size,'semantic',size,'family' if family else 'concrete','3×520×520',verification)

def main():
 a=environment('Build every DeepLabv3 registered backbone')
 if a.verify:
  nn=nn_module(a,'deeplabv3');records={};specs=[]
  for size in ['r50','r101','mv3']:
   m=nn.LibreDeepLabv3Net(size,num_classes=21)
   names=['backbone.layer1','backbone.layer2','backbone.layer3','backbone.layer4','classifier.0','classifier'] if size!='mv3' else [f'backbone.{i}' for i in range(17)]+['classifier.0','classifier']
   records[size]=cpu_probe(m,(1,3,520,520),names)
   if size=='mv3':specs=mb_specs(m)
  data=write_evidence(a,'deeplabv3',records,['libreyolo/models/deeplabv3/nn.py:LibreDeepLabv3Net, DeepLabHead, ASPP, ASPPConv, ASPPPooling','libreyolo/models/deeplabv3/model.py:INPUT_SIZES','torchvision.models.mobilenetv3:InvertedResidual (BSD-3-Clause)'],['ResNet output stride8; MobileNet output stride16 and actual33×33 map for520 input.','ASPP rates12,24,36 are unchanged across all backbones. No auxiliary FCN head.'])
  data['mobilenet_specs']=specs;evidence_path('deeplabv3').write_text(json.dumps(data,indent=2)+'\n')
 ev=read_evidence('deeplabv3');specs=ev.get('mobilenet_specs')
 if not specs:specs=mb_specs(nn_module(a,'deeplabv3').LibreDeepLabv3Net('mv3'))
 views=[build(a,s,ev,specs) for s in ['r50','r101','mv3','family']];manifest(a,'deeplabv3','deeplabv3','DeepLabv3',views)
if __name__=='__main__':main()
