"""NAFNet default s/l U-Nets and their shared symbolic topology."""
from quicksrnet import *

def product(p,id,x,y):
 p.sum(id,x,y,description='Elementwise multiplication')
 for g in p.ops:
  if g.get('id')==id:
   g.set('data-label','Multiply')
   for e in g:
    if e.tag.endswith('text'):e.text='×'

def block(d,index,c,spatial,y=1730):
 x=40+index*352
 p=d.panel('block-'+str(index),f'NAFBlock, channels {c}',x,y,336,1490,kind='conv',dashed=True,block_type='naf-'+str(index))
 double=2*c if isinstance(c,int) else '2'+c
 def b(id,yy,label,detail='',kind='plain'):
  return op(p,str(index)+id,57,yy,230,label,detail,kind,h=45)
 nodes=[('i',60,'Input',f'{c} × {spatial} × {spatial}','plain'),('n1',130,'Channel LayerNorm',f'{c} channels; epsilon 1e-6','norm'),('c1',205,'Conv2d 1×1',f'{c} to {double}; bias=True','conv2d'),('dw',280,'Depthwise Conv2d 3×3',f'{double} channels/groups; s=1, p=1','conv2d'),('g1',355,'SimpleGate',f'{double} to {c} channels','aggregate'),('pool',435,'Average pool',f'{c} × 1 × 1 at this input','pool'),('sca',510,'Conv2d 1×1',f'{c} to {c}; attention weights','conv2d')]
 for id,yy,label,detail,kind in nodes:b(id,yy,label,detail,kind)
 ids=[str(index)+v[0] for v in nodes];chain(p,ids)
 p.dot(172,416)
 product(p,str(index)+'mul',172,605)
 p.connect(str(index)+'sca',str(index)+'mul')
 p.wire([(172,416),(33,416),(33,605),(159,605)],start=str(index)+'g1',end=str(index)+'mul')
 for id,yy,label,detail,kind in [('c3',655,'Conv2d 1×1',f'{c} to {c}','conv2d'),('beta',730,'Multiply learned beta',f'{c} channel scalars; initialized zero','linear')]:b(id,yy,label,detail,kind)
 p.sum(str(index)+'add1',172,820)
 chain(p,[str(index)+'mul',str(index)+'c3',str(index)+'beta',str(index)+'add1'])
 p.connect(str(index)+'i',str(index)+'add1',from_port='left',to_port='left',via=[(9,82.5),(9,820)])
 tail=[('n2',870,'Channel LayerNorm',f'{c} channels; epsilon 1e-6','norm'),('c4',945,'Conv2d 1×1',f'{c} to {double}','conv2d'),('g2',1020,'SimpleGate',f'{double} to {c} channels','aggregate'),('c5',1095,'Conv2d 1×1',f'{c} to {c}','conv2d'),('gamma',1170,'Multiply learned gamma',f'{c} channel scalars; initialized zero','linear')]
 for id,yy,label,detail,kind in tail:b(id,yy,label,detail,kind)
 p.sum(str(index)+'add2',172,1270)
 chain(p,[str(index)+'add1']+[str(index)+v[0] for v in tail]+[str(index)+'add2'])
 p.dot(172,850);p.wire([(172,850),(9,850),(9,1270),(159,1270)],start=str(index)+'add1',end=str(index)+'add2')
 b('o',1330,'Output',f'{c} × {spatial} × {spatial}');p.connect(str(index)+'add2',str(index)+'o')
 p.text(18,1415,'No nonlinear activation or dropout.',13)

def build(a,size,width,verification):
 symbolic=size=='family'
 cs=[f'C{i}' for i in range(5)] if symbolic else [width*2**i for i in range(5)]
 title='NAFNet family' if symbolic else f'NAFNet {size.upper()}'
 d=diagram(a,title,'RGB restoration, 256 × 256 input, native NAFNetLocal eval. Shapes exclude batch.','nafnet',1800,3540)
 p=d.panel('net','Encoder, bottleneck and decoder',40,230,1090,1430)
 def b(id,x,y,label,detail='',kind='plain',block=''):
  return op(p,id,x,y,355,label,detail,kind,block=block)
 b('image',55,70,'Input and pad to multiple of 16','3 × 256 × 256; no padding at this canvas')
 b('intro',55,160,'Conv2d 3×3, stride 1',f'3 to {cs[0]}; padding 1','conv2d');p.connect('image','intro')
 for i,n in enumerate([1,1,1,28]):
  yy=255+i*180;h=256//2**i
  b(f'e{i}',55,yy,f'NAFBlock, n={n}',f'{cs[i]} × {h} × {h}','conv',f'naf-{i}')
  b(f'down{i}',55,yy+80,'Conv2d 2×2, stride 2',f'{cs[i]} to {cs[i+1]}; output {h//2} × {h//2}','conv2d')
  p.connect('intro' if i==0 else f'down{i-1}',f'e{i}');p.connect(f'e{i}',f'down{i}')
 b('middle',55,1015,'NAFBlock, n=1',f'{cs[4]} × 16 × 16','conv','naf-4');p.connect('down3','middle')
 for j,i in enumerate([3,2,1,0]):
  yy=1015-j*235;h=256//2**i;ci=cs[i+1];cout=2*ci if isinstance(ci,int) else '2'+ci
  b(f'up{j}',640,yy,'Conv2d 1×1 + PixelShuffle ×2',f'{ci} to {cout}; shuffle gives {cs[i]} × {h} × {h}','aggregate',f'up-{j}')
  p.sum(f'add{j}',817.5,yy-42)
  b(f'd{j}',640,yy-145,'NAFBlock, n=1',f'{cs[i]} × {h} × {h}','conv',f'naf-{i}')
  p.connect(f'up{j}',f'add{j}',from_port='top',to_port='bottom');p.connect(f'add{j}',f'd{j}',from_port='top',to_port='bottom')
  if j==0:p.connect('middle','up0',from_port='right',to_port='left')
  else:p.connect(f'd{j-1}',f'up{j}',from_port='top',to_port='bottom')
  # Named skips avoid long crossed collections between unlike stage orders.
  ex,ey=p.port(f'e{i}','right');p.box(f'src{i}',445,ey-13,50,f'E{i}',h=26,center=True,font_size=14,block_type=f'skip-{i}');p.connect(f'e{i}',f'src{i}',from_port='right',to_port='left')
  p.box(f'dst{i}',545,yy-55,50,f'E{i}',h=26,center=True,font_size=14,block_type=f'skip-{i}');p.wire([(595,yy-42),(804.5,yy-42)],start=f'dst{i}',end=f'add{j}')
 b('ending',640,80,'Conv2d 3×3, stride 1',f'{cs[0]} to 3; padding 1','conv2d');p.connect('d3','ending',from_port='top',to_port='bottom')
 p.text(35,1158,'Matching E0, E1, E2, E3 labels carry identical encoder tensors.',14)
 p.text(35,1187,'Final image residual and crop are shown in the output panel.',14)
 if symbolic:
  p.text(35,1250,'C0 is base width; C1=2C0, C2=4C0, C3=8C0, C4=16C0.',15)
  p.text(35,1282,'s: C0=32, C1=64, C2=128, C3=256, C4=512.',15)
  p.text(35,1314,'l: C0=64, C1=128, C2=256, C3=512, C4=1024.',15)
 p.text(35,1376,'Default preset depths: encoder [1,1,1,28], middle 1, decoder [1,1,1,1].',14)
 q=d.panel('output','Image residual and output',1170,230,590,400)
 op(q,'ending-copy',30,65,240,'Ending convolution','3 × 256 × 256')
 op(q,'input-copy',320,65,240,'Padded input image','3 × 256 × 256')
 q.sum('image-add',295,205)
 q.wire([(150,114),(150,205),(282,205)],start='ending-copy',end='image-add')
 q.wire([(440,114),(440,205),(308,205)],start='input-copy',end='image-add')
 op(q,'crop',90,265,410,'Crop to original image size','3 × 256 × 256 restored RGB','plain');q.connect('image-add','crop')
 g=d.panel('gate','SimpleGate',1170,680,590,450,kind='aggregate',dashed=True)
 op(g,'gin',95,60,400,'Split channels into equal halves','Each half retains the same spatial grid','split')
 op(g,'ga',30,170,240,'First channel half',kind='plain')
 op(g,'gb',320,170,240,'Second channel half',kind='plain')
 g.wire([(210,109),(210,143),(150,143),(150,170)],start='gin',end='ga')
 g.wire([(380,109),(380,143),(440,143),(440,170)],start='gin',end='gb')
 product(g,'gate-product',295,305)
 g.wire([(150,219),(150,305),(282,305)],start='ga',end='gate-product');g.wire([(440,219),(440,305),(308,305)],start='gb',end='gate-product')
 g.text(90,397,'Elementwise product; channel count is halved.',15)
 u=d.panel('updef','Upsample primitive',1170,1180,590,480,kind='aggregate',dashed=True)
 op(u,'u1',90,65,410,'Conv2d 1×1, bias=False','Channels double; spatial dimensions fixed','conv2d')
 op(u,'u2',90,155,410,'PixelShuffle ×2','Channels divide by 4; height/width double','aggregate');u.connect('u1','u2')
 u.text(25,285,'Every concrete upsample lists its numeric channel counts.',14)
 u.text(25,326,'TLC calibration uses a 256×256 training canvas.',14)
 u.text(25,356,'At this input, each pool spans the entire feature map.',14)
 u.text(25,386,'Larger native images use local average pooling.',14)
 u.text(25,437,'Convolutions use bias unless explicitly marked otherwise.',14)
 for i,c in enumerate(cs):block(d,i,c,256//2**i)
 d.text(50,3290,'Fresh s/l preset topology. Loaded checkpoints can infer different depths, for example SIDD encoder [2,2,4,8] and middle 12.',16)
 d.text(50,3323,'Those checkpoint-derived custom layouts are not mislabeled as the constructor presets. No pretrained checkpoint was loaded.',16)
 return finish_view(a,d,'nafnet',('family' if symbolic else size+'-restore'),('Shared s/l topology' if symbolic else size.upper()),'restore',size,'family' if symbolic else 'concrete','3×256×256',verification)

def main():
 a=environment('Build NAFNet s/l and shared family diagrams')
 if a.verify:
  import torch
  torch.set_num_threads(4);torch.manual_seed(0)
  nn=nn_module(a,'nafnet');records={}
  for size,width in [('s',32),('l',64)]:
   m=nn.NAFNetLocal(width=width,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1])
   records[size]=cpu_probe(m,(1,3,256,256),['intro','encoders.0','encoders.1','encoders.2','encoders.3','middle_blks','ups.0','ups.3','ending'])
  write_evidence(a,'nafnet',records,['libreyolo/models/nafnet/model.py:NAFNET_SIZE_CONFIGS and infer_nafnet_config','libreyolo/models/nafnet/nn.py:NAFNetLocal, NAFNet, NAFBlock, SimpleGate, LayerNorm2d, AvgPool2d'],['Default constructors s/l have encoder [1,1,1,28], middle1, decoder [1,1,1,1]. SIDD checkpoint can infer [2,2,4,8]/middle12; not loaded or asserted as default.','At a 256×256 canvas, TLC kernels calibrated from base384 cover each feature map and use global adaptive mean.','No weights or network requests.'])
 ev=read_evidence('nafnet');views=[build(a,size,w,'cpu' if size in ev['records'] else 'source') for size,w in [('s',32),('l',64)]]
 views.append(build(a,'family',32,'source'));manifest(a,'nafnet','nafnet','NAFNet',views)
if __name__=='__main__':main()
