"""Exact operation definitions for EfficientNetV2 and MobileNetV4 conv variants."""
from alexnet import *
import math

def describe(model,base,ev,activation):
 import torch.nn as nn
 entries=[];se_span=None
 def append_module(module,name):
  nonlocal se_span
  if isinstance(module,nn.Identity):return
  shp=ev['modules'][name]['output'][1:]
  if isinstance(module,nn.Conv2d):
   if type(module).__name__=='Conv2dSame':
    inp=ev['modules'][name]['input'];ph=max((math.ceil(inp[-2]/module.stride[0])-1)*module.stride[0]+module.kernel_size[0]-inp[-2],0);pw=max((math.ceil(inp[-1]/module.stride[1])-1)*module.stride[1]+module.kernel_size[1]-inp[-1],0)
    entries.append(('Pad SAME',f'L/R={pw//2}/{pw-pw//2}, T/B={ph//2}/{ph-ph//2}','plain'))
   entries.append(op(module,shp));return
  if type(module).__name__=='ConvNormAct':
   append_module(module.conv,name+'.conv');append_module(module.bn,name+'.bn')
   if module.apply_act:entries.append(('ReLU',dim(shp),'activation'))
  elif type(module).__name__=='SqueezeExcite':
   start=len(entries);entries.append(('Spatial mean',f'{shp[0]} × 1 × 1','pool'))
   for nm,child in module.named_children():append_module(child,name+'.'+nm)
   entries.append(('Multiply gate with feature',dim(shp),'plain'));se_span=(start,len(entries)-1)
  else:entries.append(op(module,shp))
 name=type(model).__name__
 if name=='ConvBnAct':
  append_module(model.conv,base+'.conv');append_module(model.bn1,base+'.bn1');entries.append((activation,dim(ev['modules'][base]['output'][1:]),'activation'))
 elif name=='EdgeResidual':
  append_module(model.conv_exp,base+'.conv_exp');append_module(model.bn1,base+'.bn1');entries.append((activation,dim(ev['modules'][base+'.bn1']['output'][1:]),'activation'))
  if hasattr(model,'se'):append_module(model.se,base+'.se')
  append_module(model.conv_pwl,base+'.conv_pwl');append_module(model.bn2,base+'.bn2')
 elif name=='InvertedResidual':
  for conv,bn in [('conv_pw','bn1'),('conv_dw','bn2')]:
   append_module(getattr(model,conv),base+'.'+conv);append_module(getattr(model,bn),base+'.'+bn);entries.append((activation,dim(ev['modules'][base+'.'+bn]['output'][1:]),'activation'))
  append_module(model.se,base+'.se');append_module(model.conv_pwl,base+'.conv_pwl');append_module(model.bn3,base+'.bn3')
 elif name=='UniversalInvertedResidual':
  for nm in ['dw_start','pw_exp','dw_mid','se','pw_proj','dw_end','layer_scale']:append_module(getattr(model,nm),base+'.'+nm)
 else:raise ValueError(name)
 return {'name':name,'ops':entries,'se':se_span,'skip':model.has_skip,'input':ev['modules'][base]['input'][1:],'output':ev['modules'][base]['output'][1:]}

def render_definition(d,key,group,x,y,w=425,h=1420,symbolic=False):
 p=d.panel(key,group.get('title',key),x,y,w,h,kind='bottleneck',dashed=True,block_type=key)
 p.text(22,59,group['name'],16);p.text(22,85,'Input '+dim(group['input']),14);p.dot(160,112)
 entries=group['ops'];ids=chain(p,key+'op',entries,x=20,y=145,w=280,gap=68)
 p.wire([(160,112),(160,145)],end=ids[0]);end_y=145+(len(entries)-1)*68+49
 if group['se']:
  a,z=group['se'];sy=145+a*68-10;ey=145+z*68+24.5
  # The SE feature is intentionally forked; its gate joins only at Multiply.
  p.dot(160,sy);p.wire([(160,sy),(335,sy),(335,ey),(300,ey)],end=ids[z])
 if group['skip']:
  p.sum(key+'sum',160,end_y+65);p.connect(ids[-1],key+'sum');p.wire([(160,112),(390,112),(390,end_y+65),(173,end_y+65)],end=key+'sum');end_y+=78
 p.text(22,end_y+36,'Output '+dim(group['output']),14)
 p.text(22,end_y+65,'Repeat '+str(group['count'])+' times',15,weight=700)
 return p

def run(family,title,cls,sizes):
 a=setup();b=Book(a,family,title);m=load(a.source,family);import torch
 torch.set_num_threads(2);allgroups={};activation='SiLU' if family=='efficientnetv2' else 'ReLU'
 for size,img in sizes.items():
  dev='cpu' if size==next(iter(sizes)) else 'meta'
  with torch.device(dev):model=getattr(m,cls)(size)
  ev=shapes(model,[1,3,img,img],dev);b.evidence[size]=ev
  groups=[];stage_counts=[]
  for si,stage in enumerate(model.blocks):
   stage_counts.append(len(stage))
   for bi,block in enumerate(stage):
    g=describe(block,f'blocks.{si}.{bi}',ev,activation);g['stage']=si+1
    if groups and all(groups[-1][key]==g[key] for key in ['name','ops','se','skip','input','output','stage']):groups[-1]['count']+=1
    else:g['count']=1;g['title']=f'Layer {si+1}, unit {bi+1}';groups.append(g)
  allgroups[size]=groups
  cols=4;rows=math.ceil(len(groups)/cols);rh=1450 if family=='efficientnetv2' else 1080;height=max(2050,rows*rh+400)
  d=b.diagram(title+' '+size,f'Classification, {img} × {img} input, 1,000 classes. Tensor sizes exclude batch.',2240,height)
  p=d.panel('network','Network',25,220,415,height-340)
  stem=[('Input',f'3 × {img} × {img}','plain')]
  if type(model.conv_stem).__name__=='Conv2dSame':stem.append(('Pad SAME','L/R=0/1, T/B=0/1','plain'))
  stem += [op(model.conv_stem,ev['modules']['conv_stem']['output'][1:]),op(model.bn1,ev['modules']['bn1']['output'][1:]),(activation,dim(ev['modules']['bn1']['output'][1:]),'activation')]
  ops=stem+[(f'Layer {g["stage"]}: {g["name"]}',f'{dim(g["output"])}, repeat {g["count"]}','aggregate') for g in groups]
  if family=='efficientnetv2':
   ops +=[op(model.conv_head,ev['modules']['conv_head']['output'][1:]),op(model.bn2,ev['modules']['bn2']['output'][1:]),('SiLU',dim(ev['modules']['bn2']['output'][1:]),'activation'),('AdaptiveAvgPool2d',str(model.num_features)+' × 1 × 1','pool'),('Flatten',str(model.num_features),'plain'),op(model.classifier,[1000])]
  else:
   ops +=[op(model.global_pool,ev['modules']['global_pool']['output'][1:]),op(model.conv_head,ev['modules']['conv_head']['output'][1:]),op(model.norm_head,ev['modules']['norm_head']['output'][1:]),('ReLU','1,280 × 1 × 1','activation'),('Flatten','1,280','plain'),op(model.classifier,[1000])]
  chain(p,'net',ops,w=355,gap=76)
  for j,g in enumerate(groups):render_definition(d,'unit'+str(j),g,465+(j%cols)*440,220+(j//cols)*rh,h=rh-30)
  d.text(50,height-95,'Repeated units are grouped only when operation parameters, tensor shapes and shortcut behavior match.',17)
  b.save(d,size+'-classify',size,verification=dev,input=f'1 × 3 × {img} × {img}')
 # Distinct MobileNetV4 stage topology requires separate concrete graphs; a shared variable-only graph would be false.
 if family=='efficientnetv2':family_view(b,allgroups,sizes)
 else:
  b.evidence['family_scope']={'reason':'s/m/l use different ordered depthwise-start and depthwise-mid kernels, omissions and repeats. Each concrete graph is its own topology; no shared multiplier graph is claimed.'}
 b.finish()

def family_view(b,allgroups,sizes):
 import copy,re
 groups=allgroups['b3'];d=b.diagram('EfficientNetV2 family','Six-stage family. Per-unit symbols I, C, M, R are input, output, expanded and squeeze channels; Hi/Ho are spatial sizes.',2240,7500)
 p=d.panel('net','Network and variant values',25,220,415,7100)
 items=[('Input','3 × S × S','plain'),('Pad SAME','Stride-2 stem pads right/bottom by 1','plain'),('Conv2d 3×3 / 2','Cstem × (S/2) × (S/2)','conv2d'),('BatchNorm2d','Cstem channels','norm'),('SiLU','Cstem × (S/2) × (S/2)','activation')]
 for j,name in enumerate(['ConvBnAct','EdgeResidual','EdgeResidual','InvertedResidual','InvertedResidual','InvertedResidual']):items.append((f'Layer {j+1}: {name}',f'C{j+1} channels, N{j+1} blocks','aggregate'))
 items +=[('Conv2d 1×1','C6 to Chead','conv2d'),('BatchNorm2d','Chead channels','norm'),('SiLU','Chead channels','activation'),('AdaptiveAvgPool2d','Chead × 1 × 1','pool'),('Flatten','Chead','plain'),('Linear','Chead to 1,000 logits','linear')];chain(p,'familyflow',items,w=350,gap=80)
 y=1660
 for size,gs in allgroups.items():
  p.text(22,y,size+', S='+str(sizes[size]),18,weight=700);y+=36
  for i in range(1,7):
   st=[g for g in gs if g['stage']==i];p.text(22,y,f'C{i}={st[-1]["output"][0]}, N{i}={sum(g["count"] for g in st)}',16);y+=31
  y+=45
 p.text(22,y,'Cstem: 32, 32, 32, 40 (b0..b3)',15);p.text(22,y+35,'Chead: 1280, 1280, 1408, 1536',15)
 p.text(22,y+90,'r=0 skips an optional repeated unit.',15)
 p.text(22,y+125,'Entries run once; repeated units run r times.',15)
 p.text(22,y+180,'Bias: SE convolutions and classifier only.',15)
 positions={}
 for j,original in enumerate(groups):
  g=copy.deepcopy(original);stage=g['stage'];position=positions.get(stage,0);positions[stage]=position+1
  g['title']=f'Layer {stage}, '+('entry' if position==0 else 'repeated unit');g['input']=['I','Hi','Hi'];g['output']=['C','Ho','Ho'];g['count']='r'
  conv_indices=[k for k,item in enumerate(g['ops']) if item[2]=='conv2d'];conv_role={}
  if g['name']=='ConvBnAct':roles=['C']
  elif g['name']=='EdgeResidual':roles=['M','C']
  else:roles=['M','M','R','M','C']
  for k,role in zip(conv_indices,roles):conv_role[k]=role
  spatial='Hi';width='I';in_se=False
  for k,(label,detail,kind) in enumerate(g['ops']):
   if label=='Spatial mean':in_se=True;detail='M × 1 × 1'
   elif kind=='conv2d':
    role=conv_role[k]
    if '/ 2' in label:spatial='Ho'
    elif g['name']=='ConvBnAct':spatial='Ho'
    kernel=int(re.search(r'Conv2d (\d+)',label).group(1));depthwise=(g['name']=='InvertedResidual' and k==conv_indices[1]);padding='0' if kernel==1 or '/ 2' in label else str(kernel//2)
    detail=f'{width} to {role}, p={padding}, g='+('M' if depthwise else '1');width=role
   elif label=='Pad SAME':detail='Pad to ceil(Hi / 2); asymmetry allowed'
   elif label=='Multiply gate with feature':in_se=False;spatial='Ho';detail='M × Ho × Ho'
   else:detail=f'{width} × '+('1 × 1' if in_se else f'{spatial} × {spatial}')
   g['ops'][k]=(label,detail,kind)
  pp=render_definition(d,'pattern'+str(j),g,465+j%4*440,220+j//4*1750,h=1720,symbolic=True)
  yy=1425
  pp.text(20,yy,'Variant   I / C / M / R      Hi / Ho    r',13,weight=700)
  for size in sizes:
   matches=[gg for gg in allgroups[size] if gg['stage']==stage]
   if position>=len(matches):text=size+': omitted, r=0'
   else:
    real=matches[position];convs=[item for item in real['ops'] if item[2]=='conv2d'];channels=[int(item[1].split(' × ')[0]) for item in convs];mid=channels[0] if len(channels)>1 else '-';rd=channels[2] if real['se'] else '-';text=f'{size}: {real["input"][0]}/{real["output"][0]}/{mid}/{rd}  {real["input"][-1]}/{real["output"][-1]}  r={real["count"]}'
   yy+=35;pp.text(20,yy,text,13)
 b.save(d,'family-classify','b0/b1/b2/b3',kind='family',verification='source',input='S in {224,240,260,300}')
if __name__=='__main__':run('efficientnetv2','EfficientNetV2','EfficientNetV2',{'b0':224,'b1':240,'b2':260,'b3':300})
