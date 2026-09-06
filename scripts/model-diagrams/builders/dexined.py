"""DexiNed six side heads with all cross-stage residual/projection paths."""
from quicksrnet import *
from teed import edge_wrapper,dense_def

def main():
 a=environment('Build DexiNed base edge network')
 if a.verify:
  m=edge_wrapper(a,nn_module(a,'dexined').DexiNedCore());rec=cpu_probe(m,(1,3,352,352),['core.block_1','core.block_2','core.dblock_3','core.dblock_4','core.dblock_5','core.dblock_6']+[f'core.up_block_{i}' for i in range(1,7)]+['core'])
  write_evidence(a,'dexined',{'b':rec},['libreyolo/models/dexined/nn.py:DexiNedCore.forward, _DenseLayer, UpConvBlock, DoubleConvBlock, SingleConvBlock','libreyolo/models/edge_common.py:EdgeInferenceNet'],['All six side outputs have352×352 shape. Native wrapper selectsfused logits and sigmoid.','side_5 is defined in core for parameter compatibility but unused by forward. It is excluded from executed graph.','Dense residual is fixed through repeated layers; each updatedprimary is0.5*(convpath+residual).'])
 ev=read_evidence('dexined');d=diagram(a,'DexiNed Base','Edge probability, input3 × 352 × 352 RGB, native eval. Shapes exclude batch.','dexined',1840,7130)
 p=d.panel('feature','Feature path and additions',40,230,580,2070)
 rows=[('rgb','RGB toBGR, multiply255','Subtract103.939,116.779,123.68 BGR means','norm'),('B1','B1: DoubleConvBlock','3/32/64 channels; first stride2;64×176×176','conv'),('B2','B2: DoubleConvBlock','64/128/128; stride1;no finalReLU;128×176×176','conv'),('D2','D2: MaxPool3×3,stride2','128 × 88 × 88; padding1','pool'),('A2','sum','S1','128×88×88'),('B3','B3: DenseBlock n=2','Primary A2:128ch; residual PD3:256ch;88×88','dense3'),('D3','D3: MaxPool3×3,stride2','256 × 44 × 44; padding1','pool'),('A3','sum','S2','256×44×44'),('B4','B4: DenseBlock n=3','Primary A3:256ch; residual PD4:512ch;44×44','dense4'),('D4','D4: MaxPool3×3,stride2','512 × 22 × 22; padding1','pool'),('A4','sum','S3','512×22×22'),('B5','B5: DenseBlock n=3','Primary A4:512ch; residual PD5:512ch;22×22','dense5'),('A5','sum','S4','512×22×22'),('B6','B6: DenseBlock n=3','Primary A5:512ch; residual PD6:256ch;22×22','dense6')]
 prev=None
 for i,(id,label,detail,kind) in enumerate(rows):
  y=65+i*137
  if label=='sum':
   p.sum(id,295,y+24.5);op(p,id+'skip',20,y+11.5,100,detail,h=26);p.connect(id+'skip',id,from_port='right',to_port='left');p.text(320,y+31,id+': '+kind,13)
  else:op(p,id,90,y,430,label,detail,'bottleneck' if kind.startswith('dense') else kind,block=kind if kind.startswith('dense') else id if id in ['B1','B2'] else '')
  if prev:p.connect(prev,id)
  prev=id
 p.text(20,2020,'Named S/PD inputs are computed in the adjacent projection panel.',13)
 q=d.panel('projections','Cross-stage projections',660,230,550,2070)
 specs=[('S1','B1',64,128,2,88),('S2','A2',128,256,2,44),('S3','A3',256,512,2,22),('S4','A4',512,512,1,22),('PD3','D2',128,256,1,88),('R2','D2',128,256,2,44),('PD4','D3+R2',256,512,1,44),('PD5','D4',512,512,1,22),('PD6','B5',512,256,1,22)]
 for j,(name,src,ci,co,s,hw) in enumerate(specs):
  y=65+j*190
  if name=='PD4':
   q.sum('in'+name,110,y+24.5)
   op(q,'PD4-D3',20,y-30,65,'D3',h=26);op(q,'PD4-R2',135,y-30,65,'R2',h=26)
   q.wire([(52.5,y-4),(52.5,y+24.5),(97,y+24.5)],start='PD4-D3',end='inPD4');q.wire([(167.5,y-4),(167.5,y+24.5),(123,y+24.5)],start='PD4-R2',end='inPD4')
  else:op(q,'in'+name,30,y,165,src,f'{ci} input channels',h=48)
  op(q,'op'+name,230,y,285,'Conv1×1 + BatchNorm',f'{ci} to{co}; stride{s}','conv',block='single')
  q.connect('in'+name,'op'+name,from_port='right',to_port='left');op(q,name,230,y+85,285,name,f'{co} × {hw} × {hw}');q.connect('op'+name,name)
 q.text(20,1840,'D3+R2 is an elementwise sum before the PD4 projection.',13)
 q.text(20,1900,'All projection convolutions have bias; BatchNorm follows.',13)
 q.text(20,1960,'No activation inside SingleConvBlock.',13)
 r=d.panel('heads','Six side logits and fused probability',1250,230,550,2070)
 channels=[64,128,256,512,512,256];scales=[1,1,2,3,4,4]
 for i,(ci,scale) in enumerate(zip(channels,scales)):
  y=65+i*220
  op(r,f'up{i}',75,y,400,f'UpConvBlock from B{i+1}',f'{ci} channels; {scale} upsample steps','conv',block='up'+str(i))
  op(r,f'out{i}',75,y+95,400,f'L{i+1}: side logits','1 × 352 × 352');r.connect(f'up{i}',f'out{i}')
 for i in range(6):
  x=25+i*85;op(r,f'concat-in{i}',x,1460,65,f'L{i+1}',h=26);r.wire([(x+32.5,1486),(x+32.5,1535)],start=f'concat-in{i}',end='concat')
 op(r,'concat',25,1535,500,'Concat six side logits','6 × 352 × 352','concat')
 op(r,'final',25,1660,500,'Conv2d1×1, stride1','6 to1; bias=True; noBatchNorm','conv2d');r.connect('concat','final')
 op(r,'sigmoid',25,1800,500,'Sigmoid','1 × 352 × 352 edge probability','activation');r.connect('final','sigmoid')
 r.text(20,1990,'Core returns six side logits + fusion; wrapper selects fusion.',13)
 # Concrete dense units grouped by their actual stage widths/repeats.
 for j,(ci,co,n) in enumerate([(128,256,2),(256,512,3),(512,512,3),(512,256,3)]):dense_def(d,'dexined',ci,co,n,40+(j%2)*900,2380+(j//2)*1130,w=850,bn=True,activation='ReLU',id='dense'+str(j+3))
 # Primitive stem and projection definitions.
 prim=d.panel('primitive','Stem and projection primitives',40,4700,1760,570,kind='conv',dashed=True)
 for j,(ci,mid,co,ss,finalact) in enumerate([(3,32,64,2,True),(64,128,128,1,False)]):
  xx=25+j*610
  prim.text(xx,58,f'B{j+1} DoubleConvBlock',18,weight=700)
  steps=[('c1','Conv2d3×3',f'{ci} to{mid};s{ss},p1','conv2d'),('bn1','BatchNorm',str(mid),'norm'),('relu','ReLU','','activation'),('c2','Conv2d3×3',f'{mid} to{co};s1,p1','conv2d'),('bn2','BatchNorm',str(co),'norm')]
  if finalact:steps.append(('outrelu','ReLU','','activation'))
  for i,(id,label,detail,kind) in enumerate(steps):op(prim,f'stem{j}'+id,xx,100+i*66,540,label,detail,kind,h=42)
  chain(prim,[f'stem{j}'+s[0] for s in steps])
 op(prim,'sconv',1270,120,430,'SingleConvBlock: Conv2d1×1','Numeric Cin/Cout/stride in projection panel','conv2d');op(prim,'sbn',1270,280,430,'BatchNorm2d','No activation','norm');prim.connect('sconv','sbn')
 # All side heads expand their exact kernels/channels.
 for j,(ci,scale) in enumerate(zip(channels,scales)):
  u=d.panel('up-def'+str(j),f'UpConvBlock side{j+1}',40+(j%3)*600,5340+(j//3)*830,560,760,kind='conv',dashed=True,block_type='up'+str(j))
  prev=None
  for i in range(scale):
   cin=ci if i==0 else 16;co=1 if i==scale-1 else 16;k=2**scale;pad=[0,0,1,3,7][scale]
   # Each step is a short horizontal primitive chain.
   for z,(id,label,detail,kind) in enumerate([('c','Conv1×1',f'{cin} to{co}','conv2d'),('a','ReLU','','activation'),('t',f'Transpose{k}×{k}',f'{co}ch;s2,p{pad}','conv2d')]):op(u,f'u{j}{i}{id}',25+180*z,65+i*145,155,label,detail,kind,h=49)
   u.connect(f'u{j}{i}c',f'u{j}{i}a',from_port='right',to_port='left');u.connect(f'u{j}{i}a',f'u{j}{i}t',from_port='right',to_port='left')
   if prev:u.connect(prev,f'u{j}{i}c',via=[(462.5,65+i*145-32),(102.5,65+i*145-32)])
   prev=f'u{j}{i}t'
  u.text(20,705,'Every step doubles the grid; output1 × 352 × 352.',13)
 rec=ev['records'].get('b',{});view=finish_view(a,d,'dexined','b-edge','Base','edge','b','concrete','3×352×352',rec.get('device','source'));manifest(a,'dexined','dexined','DexiNed', [view])
if __name__=='__main__':main()
