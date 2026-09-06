"""L2CS gaze: five ResNet backbones, honest yaw/pitch tuple ordering."""
from quicksrnet import *
from fcn import resnet_block_definitions
COUNTS={'r18':[2,2,2,2],'r34':[3,4,6,3],'r50':[3,4,6,3],'r101':[3,4,23,3],'r152':[3,8,36,3]}

def basic_defs(d,counts,y=1610):
 for stage in range(4):
  p=d.panel('basic'+str(stage),f'BasicBlock stage {stage+1}',40+(stage%2)*880,y+(stage//2)*1010,840,950,kind='bottleneck',dashed=True,block_type='stage'+str(stage))
  co=64*2**stage;ci=64 if stage==0 else co//2
  for first in [True,False]:
   off=0 if first else 420;pre=f'bb{stage}{first}-';cin=ci if first else co;stride=2 if stage>0 and first else 1
   p.text(off+25,62,'First block' if first else f'Repeat n={counts[stage]-1 if isinstance(counts[stage],int) else str(counts[stage])+"-1"}',16,weight=700)
   rows=[('in','Input',f'{cin} channels','plain'),('c1','Conv2d 3×3',f'{cin} to{co}; stride{stride},padding1','conv2d'),('n1','BatchNorm2d',f'{co} channels','norm'),('a','ReLU','','activation'),('c2','Conv2d 3×3',f'{co} to{co}; stride1,padding1','conv2d'),('n2','BatchNorm2d',f'{co} channels','norm')]
   for i,(id,label,detail,kind) in enumerate(rows):op(p,pre+id,off+145,105+i*95,250,label,detail,kind,h=42)
   chain(p,[pre+r[0] for r in rows]);p.sum(pre+'sum',off+270,740);p.connect(pre+'n2',pre+'sum')
   if first and stage>0:
    op(p,pre+'skipc',off+15,265,110,'Conv1×1',f'{cin} to{co};s2','conv2d',h=58);op(p,pre+'skipbn',off+15,430,110,'BatchNorm',f'{co} channels','norm',h=58)
    p.connect(pre+'in',pre+'skipc',from_port='left',via=[(off+70,126),(off+70,230)]);p.connect(pre+'skipc',pre+'skipbn');p.connect(pre+'skipbn',pre+'sum',via=[(off+70,740)],to_port='left')
   else:p.connect(pre+'in',pre+'sum',from_port='left',to_port='left',via=[(off+40,126),(off+40,740)])
   op(p,pre+'out',off+145,825,250,'ReLU',f'{co} output channels','activation');p.connect(pre+'sum',pre+'out')

def build(a,size,ev):
 family=size.endswith('family');basic=size.startswith('basic') or size in ['r18','r34']
 counts=[f'N{i+1}' for i in range(4)] if family else COUNTS[size]
 channels=[64,128,256,512] if basic else [256,512,1024,2048];feature=channels[-1]
 d=diagram(a,'L2CS '+size.upper(),'Face-crop gaze, 90 bins per angle, 3 × 448 × 448 normalized RGB input, native eval. Shapes exclude batch.','l2cs',1800,4100 if family and not basic else 3920)
 p=d.panel('backbone','ResNet face encoder',40,230,620,1280)
 rows=[('input','Normalized face crop','3 × 448 × 448','plain'),('conv','Conv2d 7×7, stride2','3 to64; padding3;64×224×224','conv2d'),('bn','BatchNorm2d + ReLU','64 × 224 × 224','norm'),('pool','MaxPool2d 3×3, stride2','64 × 112 × 112; padding1','pool')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,95,65+i*105,430,label,detail,kind)
 chain(p,[r[0] for r in rows]);prev='pool'
 for i in range(4):
  op(p,f'stage{i}',95,535+i*145,430,('BasicBlock' if basic else 'Bottleneck')+f' stage{i+1}, n={counts[i]}',f'{channels[i]} × {112//2**i} × {112//2**i}','bottleneck',block='stage'+str(i) if basic else 'layer'+str(i+1));p.connect(prev,f'stage{i}');prev=f'stage{i}'
 p.text(25,1180,'Stage1 preserves stride4; stages2/3/4 downsample to stride32.',14)
 q=d.panel('heads','Parallel angle heads and decode',700,230,1060,1280)
 op(q,'gap',285,65,490,'AdaptiveAvgPool2d(1) + flatten',f'{feature}-element feature vector','pool')
 for name,x in [('yaw',35),('pitch',575)]:
  op(q,name+'fc',x,240,450,'Linear '+name,f'{feature} to90 logits','linear')
  q.wire([(440 if name=='yaw' else 610,114),(440 if name=='yaw' else 610,190),(x+225,190),(x+225,240)],start='gap',end=name+'fc')
  op(q,name+'softmax',x,425,450,'Softmax in float32','90 probabilities','attention');q.connect(name+'fc',name+'softmax')
  op(q,name+'expect',x,610,450,'Angular-bin expectation','4 × sum(probability[i] × i) -180 degrees','linear');q.connect(name+'softmax',name+'expect')
  op(q,name+'rad',x,795,450,'Multiply π/180',name+' in radians','linear');q.connect(name+'expect',name+'rad')
 q.text(25,1040,'Native model tuple order: (yaw_logits, pitch_logits).',16,weight=700)
 q.text(25,1090,'External decode returns columns [pitch, yaw] in radians.',16)
 q.text(25,1140,'Face detection/cropping is a caller pipeline, not part of this network.',15)
 q.text(25,1190,'Unused upstream fc_finetune layer is absent.',15)
 if basic:basic_defs(d,counts)
 else:resnet_block_definitions(d,size,y=1610,counts_override=counts,dilated=False)
 if family:
  keys=['r18','r34'] if basic else ['r50','r101','r152']
  d.text(50,d.height-115,'Stage repeat variables: '+ '    '.join(k+'='+str(COUNTS[k]) for k in keys),16)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'l2cs',size if family else size+'-gaze',size,'gaze',size,'family' if family else 'concrete','3×448×448',rec.get('device','source'))

def main():
 a=environment('Build all L2CS registered ResNet backbones')
 if a.verify:
  nn=nn_module(a,'l2cs');records={}
  for size in COUNTS:records[size]=cpu_probe(nn.build_l2cs(size,num_bins=90),(1,3,448,448),['layer1','layer2','layer3','layer4','avgpool','fc_yaw_gaze','fc_pitch_gaze'])
  write_evidence(a,'l2cs',records,['libreyolo/models/l2cs/nn.py:_RESNET_LAYERS, L2CS.forward','libreyolo/models/l2cs/utils.py:logits_to_angles','libreyolo/models/l2cs/model.py:INPUT_SIZES'],['All five native sizes checked on CPU with448 face crop.','Network tuple is yaw then pitch; external decoded tensor is pitch then yaw. No detector weights downloaded.'])
 ev=read_evidence('l2cs');views=[build(a,s,ev) for s in COUNTS]+[build(a,s,ev) for s in ['basic-family','bottleneck-family']];manifest(a,'l2cs','l2cs','L2CS-Net',views)
if __name__=='__main__':main()
