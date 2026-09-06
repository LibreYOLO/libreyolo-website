"""Tiny Efficient Edge Detector, native wrapper plus exact side/fusion heads."""
from quicksrnet import *
from nafnet import product

def edge_wrapper(a,core):
 spec=importlib.util.spec_from_file_location('edge_common_diagram',a.source/'libreyolo/models/edge_common.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.EdgeInferenceNet(core)

def smish_def(d,x,y,w=560,h=780):
 p=d.panel('smishdef','Smish',x,y,w,h,kind='activation' if False else 'conv',dashed=True,block_type='smish')
 rows=[('si','Input x','','plain'),('sig','Sigmoid','','activation'),('plus','Add scalar1','','linear'),('log','Natural logarithm','','activation'),('tanh','Tanh','','activation')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,95,65+i*105,w-175,label,detail,kind)
 chain(p,[r[0] for r in rows]);product(p,'smul',95+(w-175)/2,650);p.connect('tanh','smul');p.connect('si','smul',from_port='left',to_port='left',via=[(30,89.5),(30,650)])
 p.text(25,735,'Smish(x) = x × tanh(log(1 + sigmoid(x))).',14)

def dense_def(d,family,cin,cout,n,x,y,w=570,bn=False,activation='Smish',id='dense'):
 p=d.panel(id+'-def',f'Dense layer: first {cin} to{cout}, repeat {cout} to{cout}',x,y,w,1070,kind='bottleneck',dashed=True,block_type=id)
 pre=id+'-';op(p,pre+'p',100,65,w-150,'Primary feature P',f'First layer {cin} channels; later {cout}')
 seq=[('a0',activation,'Before first convolution','activation'),('c1','Conv2d 3×3',f'{cin} to{cout} first; {cout} to{cout} later; padding2','conv2d')]
 if bn:seq.append(('bn1','BatchNorm2d',f'{cout} channels','norm'))
 seq.append(('a1',activation,'','activation'));seq.append(('c2','Conv2d 3×3',f'{cout} to{cout}; padding0','conv2d'))
 if bn:seq.append(('bn2','BatchNorm2d',f'{cout} channels','norm'))
 for i,(key,label,detail,kind) in enumerate(seq):op(p,pre+key,100,155+i*90,w-150,label,detail,kind,block='smish' if activation=='Smish' and key.startswith('a') else '')
 chain(p,[pre+'p']+[pre+s[0] for s in seq]);p.sum(pre+'sum',100+(w-150)/2,820);p.connect(pre+seq[-1][0],pre+'sum')
 op(p,pre+'r',15,725,65,'R',f'{cout} ch',h=35);p.connect(pre+'r',pre+'sum',via=[(47.5,820)],to_port='left')
 op(p,pre+'half',100,910,w-150,'Multiply0.5',f'Return (new primary, unchanged residual R); n={n}','linear');p.connect(pre+'sum',pre+'half')
 p.text(20,1025,'Padding2 expands by2; padding0 restores the original grid.',13)

def up_definition(d,family,channels,scales,activation,y):
 # Four scalar scales are distinct repeated-upsample topologies.
 for j,(ci,scale) in enumerate(zip(channels,scales)):
  p=d.panel('updef'+str(j),f'Side head {j+1}: input {ci}, steps {scale}',40+j*580,y,540,850 if scale==1 else 1170,kind='conv',dashed=True,block_type='up'+str(j))
  pre=f'up{j}-';prev=None
  for i in range(scale):
   cin=ci if i==0 else 16;cout=1 if i==scale-1 else 16;yy=65+i*270;k=2**scale;pad=[0,0,1,3,7][scale]
   for z,(id,label,detail,kind) in enumerate([('c','Conv2d 1×1',f'{cin} to{cout}; bias=True','conv2d'),('a',activation,'','activation'),('t',f'ConvTranspose2d {k}×{k}',f'{cout} channels; stride2,padding{pad}','conv2d')]):
    nid=pre+str(i)+id;op(p,nid,70,yy+z*80,400,label,detail,kind,block='smish' if id=='a' and activation=='Smish' else '')
    if prev:p.connect(prev,nid)
    prev=nid
  p.text(25,(780 if scale==1 else 1100),'Every step doubles height/width; final output is1 channel.',13)

def main():
 a=environment('Build TEED tiny with complete edge and Smish internals')
 if a.verify:
  m=edge_wrapper(a,nn_module(a,'teed').TEEDCore());record=cpu_probe(m,(1,3,352,352),['core.block_1','core.block_2','core.dblock_3','core.up_block_1','core.up_block_2','core.up_block_3','core.block_cat','core'])
  write_evidence(a,'teed',{'t':record},['libreyolo/models/teed/nn.py:TEEDCore, _DenseLayer, DoubleFusion, Smish','libreyolo/models/edge_common.py:EdgeInferenceNet'],['Native wrapper converts RGB[0,1] toBGR255 and subtracts[103.939,116.779,123.68], then sigmoid of fused logits.','No BatchNorm in TEED. Core returns3 side logits plusfusion; wrapper returnsonlyfused probability.'])
 ev=read_evidence('teed');d=diagram(a,'TEED Tiny','Edge probability, input3 × 352 × 352 RGB, native eval. Shapes exclude batch.','teed',1780,4820)
 p=d.panel('main','Feature path',40,230,550,1560)
 rows=[('rgb','RGB toBGR, multiply255','Subtract BGR means103.939,116.779,123.68','norm'),('c1','Conv2d 3×3, stride2','3 to16; padding1;176 × 176','conv2d'),('a1','Smish','','activation'),('c2','Conv2d 3×3','16 to16; s1,p1','conv2d'),('a2','Smish: B1','16 × 176 × 176','activation'),('c3','Conv2d 3×3','16 to32; s1,p1','conv2d'),('a3','Smish','','activation'),('c4','Conv2d 3×3: B2','32 × 176 × 176; no final activation','conv2d'),('pool','MaxPool3×3,stride2: D2','32 × 88 × 88; padding1','pool')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,70,65+i*145,410,label,detail,kind,block='smish' if id.startswith('a') else '')
 chain(p,[r[0] for r in rows]);p.text(25,1440,'B1/B2/D2 labels are named continuations into branches.',13)
 q=d.panel('densepath','Residual branch and dense block',630,230,550,1560)
 op(q,'side',75,65,400,'Conv2d1×1 fromB1','16 to32; stride2; S1:32×88×88','conv2d')
 op(q,'d2',75,210,400,'D2','32 × 88 × 88');q.sum('add',275,365)
 q.connect('d2','add');q.connect('side','add',from_port='left',to_port='left',via=[(30,89.5),(30,365)])
 op(q,'pre',75,520,400,'Conv2d1×1 fromD2','32 to48; stride1; residual R','conv2d')
 op(q,'dense',75,740,400,'DenseBlock, n=1','Primary:add(32ch), residual:R(48ch)','bottleneck',block='dense')
 q.connect('add','dense',from_port='right',to_port='right',via=[(515,365),(515,764.5)]);q.connect('pre','dense')
 op(q,'b3',75,930,400,'B3 feature','48 × 88 × 88');q.connect('dense','b3')
 q.text(25,1135,'Dense residual R is projected from D2 before summation.',14)
 q.text(25,1190,'Unlike DexiNed, TEED uses Smish and omits BatchNorm.',14)
 r=d.panel('sides','Three side heads and fusion',1220,230,520,1560)
 for j,(b,ci,up) in enumerate([('B1',16,1),('B2',32,1),('B3',48,2)]):
  op(r,f'side{j}',65,65+j*225,390,f'UpConvBlock from{b}',f'{ci} input channels; {up} upsample steps','conv',block='up'+str(j))
  op(r,f'L{j}',65,165+j*225,390,f'Side logit L{j+1}','1 × 352 × 352');r.connect(f'side{j}',f'L{j}')
 for j in range(3):op(r,f'lin{j}',55+150*j,780,110,f'L{j+1}',h=26);r.wire([(110+150*j,806),(110+150*j,855)],start=f'lin{j}',end='cat')
 op(r,'cat',45,855,430,'Concat three side logits','3 × 352 × 352','concat')
 op(r,'fusion',45,1010,430,'DoubleFusion','1 × 352 × 352 fused logits','aggregate',block='doublefusion');r.connect('cat','fusion')
 op(r,'sigmoid',45,1200,430,'Sigmoid','1 × 352 × 352 edge probability','activation');r.connect('fusion','sigmoid')
 r.text(20,1400,'Core side logits remain available before wrapper selection.',13)
 dense_def(d,'teed',32,48,1,40,1850,w=550,id='dense')
 f=d.panel('fusiondef','DoubleFusion',630,1850,550,1560,kind='aggregate',dashed=True,block_type='doublefusion')
 rows=[('fi','Concat side logits','3 × 352 × 352','plain'),('fa','Smish','','activation'),('fdw1','GroupedConv2d3×3','3 to24;groups3;s1,p1; multiplier8','conv2d'),('fb','Smish','','activation'),('fdw2','DepthwiseConv2d3×3','24 channels/groups;s1,p1','conv2d')]
 for i,(id,label,detail,kind) in enumerate(rows):op(f,id,75,65+i*155,400,label,detail,kind,block='smish' if id in ['fa','fb'] else '')
 chain(f,[r[0] for r in rows]);f.sum('fadd',275,940);f.connect('fdw2','fadd');f.connect('fdw1','fadd',from_port='left',to_port='left',via=[(25,399.5),(25,940)])
 op(f,'sumch',75,1060,400,'Sum along24 channels','1 × 352 × 352','aggregate');f.connect('fadd','sumch')
 op(f,'fsmish',75,1240,400,'Smish','Fused edge logits','activation',block='smish');f.connect('sumch','fsmish')
 f.text(25,1430,'PixelShuffle(1) is identity and has no spatial effect.',14)
 smish_def(d,1220,1850,w=520,h=780)
 up_definition(d,'teed',[16,32,48],[1,1,2],'Smish',3500)
 rec=ev['records'].get('t',{});view=finish_view(a,d,'teed','t-edge','Tiny','edge','t','concrete','3×352×352',rec.get('device','source'));manifest(a,'teed','teed','TEED',[view])
if __name__=='__main__':main()
