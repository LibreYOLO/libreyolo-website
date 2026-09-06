"""PIDNet three streams, spatial gates, distinct PAPPM/DAPPM and native head."""
from quicksrnet import *
from nafnet import product
CFG={'s':(2,3,32,96,128),'m':(2,3,64,96,128),'l':(3,4,64,112,256)}

def build(a,size,ev):
 sym=size=='family';m,n,P,F,HP=(2,3,'P',96,128) if sym else CFG[size];big=size=='l'
 mult=lambda k:f'{k}P' if sym else k*P
 d=diagram(a,'PIDNet S/M family' if sym else 'PIDNet '+size.upper(),'Cityscapes semantic segmentation,19 classes,input3 × 1024 × 1024, native eval. Shapes exclude batch.','pidnet',1900,5280)
 p=d.panel('stem','Shared stem and integral stream I',40,230,570,1510)
 rows=[('input','RGB + ImageNet normalization','3 × 1024 × 1024','norm'),('stem','Two Conv3×3/BN/ReLU operations',f'3 to{P} to{P}; bothstride2; {P} × 256 × 256','conv'),('l1',f'BasicBlock stage1,n={m}',f'{P} to{P}; stride1;256 × 256','bottleneck'),('l2',f'ReLU; BasicBlock stage2,n={m}; ReLU',f'{P} to{mult(2)}; firststride2;128 × 128','bottleneck'),('l3',f'BasicBlock stage3,n={n}; ReLU',f'{mult(2)} to{mult(4)}; firststride2;64 × 64','bottleneck'),('l4',f'BasicBlock stage4,n={n}; ReLU',f'{mult(4)} to{mult(8)}; firststride2;32 × 32','bottleneck'),('l5','Bottleneck stage5,n=2',f'{mult(8)} to{mult(16)}; inner{mult(8)}; firststride2;16²','bottleneck'),('ppm','DAPPM' if big else 'PAPPM',f'{mult(16)} to{mult(4)}; branchwidth{F};16 × 16','pool'),('iout','Bilinear resize to128 × 128',f'I: {mult(4)} × 128 × 128','pool')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,65,65+i*150,440,label,detail,kind,block='ppm' if id=='ppm' else 'residual' if id.startswith('l') else 'convstack' if id=='stem' else '')
 chain(p,[r[0] for r in rows]);p.text(25,1450,'Stage2 output fans into P, I and D streams.',14)
 q=d.panel('pstream','Proportional stream P',650,230,570,1510)
 rows=[('p3',f'BasicBlock layer3_,n={m}',f'From sharedstage2; {mult(2)} to{mult(2)};128²','bottleneck'),('pag3','PagFM3 with compressed I3',f'I3 Conv1×1+BN: {mult(4)} to{mult(2)}','aggregate'),('p4',f'ReLU; BasicBlock layer4_,n={m}',f'{mult(2)} to{mult(2)};128²','bottleneck'),('pag4','PagFM4 with compressed I4',f'I4 Conv1×1+BN: {mult(8)} to{mult(2)}','aggregate'),('p5','ReLU; Bottleneck layer5_,n=1',f'{mult(2)} to{mult(4)}; inner{mult(2)};128²','bottleneck')]
 for i,(id,label,detail,kind) in enumerate(rows):op(q,id,65,65+i*255,440,label,detail,kind,block='pag' if id.startswith('pag') else 'residual')
 chain(q,[r[0] for r in rows]);q.text(25,1405,'P: '+str(mult(4))+' × 128 × 128.',15)
 r=d.panel('dstream','Derivative stream D',1260,230,600,1510)
 c3=mult(2) if big else P
 rows=[('d3','BasicBlock layer3_d,n=1',f'From sharedstage2; {mult(2)} to{c3}; nofinalReLU','bottleneck'),('diff3','Add resized diff3(I3)',f'Conv3×3+BN: {mult(4)} to{c3}; thenresize128²','aggregate'),('d4','ReLU; '+('BasicBlock' if big else 'Bottleneck')+' layer4_d,n=1',f'{c3} to{mult(2)}; inner{P if not big else mult(2)};128²','bottleneck'),('diff4','Add resized diff4(I4)',f'Conv3×3+BN: {mult(8)} to{mult(2)}; thenresize128²','aggregate'),('d5','ReLU; Bottleneck layer5_d,n=1',f'{mult(2)} to{mult(4)}; inner{mult(2)};128²','bottleneck')]
 for i,(id,label,detail,kind) in enumerate(rows):op(r,id,65,65+i*255,470,label,detail,kind,block='residual' if kind=='bottleneck' else '')
 chain(r,[r[0] for r in rows]);r.text(25,1405,'D: '+str(mult(4))+' × 128 × 128.',15)
 # Decoder gate equations each have drawn input paths to actual arithmetic.
 g=d.panel('pagdef','PagFM (used twice)',40,1800,590,1150,kind='aggregate',dashed=True,block_type='pag')
 for name,x in [('P',20),('Y',325)]:
  op(g,name+'in',x,65,245,name+' input',f'{mult(2)} channels')
  op(g,name+'proj',x,180,245,'Conv1×1 + BatchNorm',f'{mult(2)} to{P}','conv');g.connect(name+'in',name+'proj')
 op(g,'yr',325,300,245,'Bilinear resize projectedY','Match P grid128×128','pool');g.connect('Yproj','yr');product(g,'simprod',295,445)
 g.wire([(142.5,229),(142.5,445),(282,445)],start='Pproj',end='simprod');g.wire([(447.5,349),(447.5,445),(308,445)],start='yr',end='simprod')
 op(g,'reduce',90,550,410,'Sum channels then sigmoid','Similarity S:1 × 128 × 128','aggregate');g.connect('simprod','reduce')
 op(g,'pa',20,720,245,'Multiply (1-S) × P',f'{mult(2)} channels','linear');op(g,'ya',325,720,245,'Multiply S × resize(Y)',f'{mult(2)} channels','linear')
 g.wire([(295,599),(295,665)],start='reduce',arrow=False);g.wire([(295,665),(142.5,665),(142.5,720)],start='reduce',end='pa');g.wire([(295,665),(447.5,665),(447.5,720)],start='reduce',end='ya');g.dot(295,665);g.sum('pagadd',295,955)
 g.wire([(142.5,769),(142.5,955),(282,955)],start='pa',end='pagadd');g.wire([(447.5,769),(447.5,955),(308,955)],start='ya',end='pagadd')
 g.text(25,1065,'P/Y in the weighted mix are unprojected feature tensors.',14)
 b=d.panel('bag','Bag' if big else 'LightBag',670,1800,590,1150,kind='aggregate',dashed=True)
 op(b,'datt',75,65,440,'Sigmoid(D)',f'Edge gate E: {mult(4)} channels','activation')
 if big:
  op(b,'bp',20,255,250,'Multiply E × P',f'{mult(4)} channels','linear');op(b,'bi',320,255,250,'Multiply (1-E) × I',f'{mult(4)} channels','linear')
  b.sum('bagadd',295,490);b.wire([(145,304),(145,490),(282,490)],start='bp',end='bagadd');b.wire([(445,304),(445,490),(308,490)],start='bi',end='bagadd')
  op(b,'bagconv',75,650,440,'BatchNorm; ReLU; Conv3×3',f'{mult(4)} to{mult(4)};s1,p1','conv');b.connect('bagadd','bagconv')
 else:
  op(b,'bp',20,255,250,'Multiply (1-E) × I',f'{mult(4)} channels','linear');op(b,'bi',320,255,250,'Multiply E × P',f'{mult(4)} channels','linear')
  b.sum('bpadd',145,375);b.sum('biadd',445,375);b.connect('bp','bpadd');b.connect('bi','biadd')
  op(b,'bag-P',20,362,55,'P',h=26);op(b,'bag-I',515,362,55,'I',h=26);b.connect('bag-P','bpadd',from_port='right',to_port='left');b.connect('bag-I','biadd',from_port='left',to_port='right')
  op(b,'bpc',20,460,250,'Conv1×1 + BatchNorm',f'{mult(4)} to{mult(4)}','conv');op(b,'bic',320,460,250,'Conv1×1 + BatchNorm',f'{mult(4)} to{mult(4)}','conv');b.connect('bpadd','bpc');b.connect('biadd','bic')
  b.sum('bagadd',295,710);b.wire([(145,509),(145,710),(282,710)],start='bpc',end='bagadd');b.wire([(445,509),(445,710),(308,710)],start='bic',end='bagadd')
 b.wire([(295,114),(295,205)],start='datt',arrow=False);b.wire([(295,205),(145,205),(145,255)],start='datt',end='bp');b.wire([(295,205),(445,205),(445,255)],start='datt',end='bi');b.dot(295,205)
 b.text(25,890,'P, I and D are the final128×128 stream outputs.',14)
 b.text(25,950,'Large uses Bag; S/M use LightBag with two learned paths.',14)
 h=d.panel('head','SegmentHead and native output',1300,1800,560,1150)
 rows=[('hin','Fused feature',f'{mult(4)} × 128 × 128','plain'),('hbn1','BatchNorm2d',str(mult(4))+' channels','norm'),('hrelu1','ReLU','','activation'),('hc1','Conv2d3×3',f'{mult(4)} to{HP};s1,p1,bias=False','conv2d'),('hbn2','BatchNorm2d',str(HP)+' channels','norm'),('hrelu2','ReLU','','activation'),('hc2','Conv2d1×1',f'{HP} to19;bias=True','conv2d'),('hout','Semantic logits','19 × 128 × 128','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(h,id,75,65+j*115,410,label,detail,kind)
 chain(h,[r[0] for r in rows]);h.text(20,1070,'Full-canvas resizing belongs to public postprocessing.',13)
 # PPM branch details and exact parallel-vs-cascaded topology.
 s=d.panel('ppmdef','DAPPM' if big else 'PAPPM',40,3010,1820,1340,kind='pool',dashed=True,block_type='ppm')
 for j in range(5):
  xx=20+j*355
  op(s,f'scale{j}',xx,70,320,'Identity' if j==0 else ['AvgPool5,s2,p2','AvgPool9,s4,p4','AvgPool17,s8,p8','AdaptiveAvgPool1'][j-1],f'Input {mult(16)} × 16 × 16','plain' if j==0 else 'pool')
  op(s,f'scaleproj{j}',xx,210,320,'BatchNorm; ReLU; Conv1×1',f'{mult(16)} to{F}; noConv bias','conv');s.connect(f'scale{j}',f'scaleproj{j}')
  if j:
   op(s,f'scaleup{j}',xx,355,320,'Bilinear resize16×16',f'{F} channels','pool');s.connect(f'scaleproj{j}',f'scaleup{j}')
   s.sum(f'psum{j}',xx+160,510);s.connect(f'scaleup{j}',f'psum{j}');op(s,f'prev{j}',xx+15,497,85,f'Z{j-1}' if big else 'Z0',h=26);s.connect(f'prev{j}',f'psum{j}',from_port='right',to_port='left')
   if big:op(s,f'process{j}',xx,620,320,'BN; ReLU; Conv3×3',f'{F} to{F};s1,p1; output Z{j}','conv');s.connect(f'psum{j}',f'process{j}')
  else:op(s,'Z0',xx+110,355,100,'Z0',f'{F}ch',h=40);s.connect('scaleproj0','Z0')
 if not big:
  for j in range(1,5):
   xx=20+j*355;op(s,f'U{j}',xx+110,620,100,f'U{j}',f'{F}ch',h=40);s.connect(f'psum{j}',f'U{j}')
 names=['Z0','Z1','Z2','Z3','Z4'] if big else ['U1','U2','U3','U4']
 for j,name in enumerate(names):
  xx=65+j*230;op(s,'collect-'+name,xx,750,100,name,h=26);s.wire([(xx+50,776),(xx+50,830)],start='collect-'+name,end='pconcat')
 op(s,'pconcat',25,830,1220,'Concat '+','.join(names),f'{F*len(names)} channels','concat')
 if not big:
  op(s,'pgroup',200,940,1045,'BN; ReLU; groupedConv3×3 (groups4)',f'{F*4} to{F*4};s1,p1','conv');s.connect('pconcat','pgroup',via=[(635,915),(722.5,915)])
  op(s,'Z0-again',25,1060,100,'Z0',h=26);op(s,'allscales',200,1050,1045,'Concat Z0 and grouped scale output',f'{F*5} channels','concat')
  s.connect('pgroup','allscales');s.connect('Z0-again','allscales',from_port='right',to_port='left');collector='allscales'
 else:collector='pconcat'
 op(s,'shortcut',1290,750,490,'Input shortcut: BN; ReLU; Conv1×1',f'{mult(16)} to{mult(4)}','conv')
 op(s,'compress',1290,1050,490,'BN; ReLU; Conv1×1 compression',f'{F*5} to{mult(4)}','conv')
 if big:s.connect(collector,'compress',from_port='right',to_port='left',via=[(1265,854.5),(1265,1074.5)])
 else:s.connect(collector,'compress',from_port='right',to_port='left')
 s.sum('ppm-final-add',1535,1210);s.connect('compress','ppm-final-add');s.connect('shortcut','ppm-final-add',from_port='right',to_port='right',via=[(1800,774.5),(1800,1210)])
 s.text(25,1290,'All pool branches consume the same input. '+('Z1...Z4 are sequentially dependent processed sums.' if big else 'Each scale independently adds the same Z0 before grouped processing.'),14)
 # Residual/normalization definitions use fully specified per-occurrence channel tuples above.
 z=d.panel('primitives','Residual block and composite-convolution definitions',40,4410,1820,630,kind='bottleneck',dashed=True,block_type='residual')
 z.text(25,62,'Ci/Co are the explicit input/output channel numbers printed at each occurrence; s is its printed stride.',15)
 for j,(name,steps) in enumerate([('BasicBlock',['Conv3×3 Ci toCo,s,p1','BN','ReLU','Conv3×3 Co toCo,s1,p1','BN']),('Bottleneck',['Conv1×1 Ci toinner','BN','ReLU','Conv3×3 inner toinner,s,p1','BN','ReLU','Conv1×1 inner to2inner','BN'])]):
  xx=25+j*900;z.text(xx,120,name,18,weight=700)
  for k,label in enumerate(steps):op(z,f'prim{j}{k}',xx+150,165+k*42,620,label,kind='conv2d' if label.startswith('Conv') else 'norm',h=32)
  chain(z,[f'prim{j}{k}' for k in range(len(steps))]);z.sum(f'primadd{j}',xx+460,525);z.connect(f'prim{j}{len(steps)-1}',f'primadd{j}')
  op(z,f'skipprim{j}',xx,350,130,'Identity or1×1+BN','Project when Ci!=Co or s!=1',h=62);z.connect(f'skipprim{j}',f'primadd{j}',via=[(xx+65,525)],to_port='left')
 z.text(25,600,'Last BasicBlock in a multi-block stage omits final ReLU. Bottlenecks and single D blocks omit it; main graph marks later ReLUs.',13)
 if sym:d.text(50,5150,'S/M family: P=32 forS,64 forM; m=2,n=3,PPM branch96,head128. L is separate: m3,n4,P64,branch112,head256.',14)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'pidnet','sm-family' if sym else size+'-semantic','Shared S/M topology' if sym else size,'semantic',size,'family' if sym else 'concrete','3×1024×1024',rec.get('device','source'))

def main():
 a=environment('Build PIDNet S/M/L and shared S/M topology')
 if a.verify:
  nn=nn_module(a,'pidnet');records={}
  for size in CFG:records[size]=cpu_probe(nn.LibrePIDNetNet(size,19),(1,3,1024,1024),['layer1','layer2','layer3','layer4','layer5','layer5_','layer5_d','spp','dfm','final_layer'])
  write_evidence(a,'pidnet',records,['libreyolo/models/pidnet/nn.py:SIZE_CONFIGS, LibrePIDNetNet.forward, PagFM, PAPPM, DAPPM, LightBag, Bag','libreyolo/models/pidnet/model.py:INPUT_SIZES'],['All three1024 presets checked onCPU. Native output is19×128×128, not19×1024×1024.','S/M have PAPPM andLightBag; L hasDAPPM andBag plusa differentD stream.','BasicBlock expansion1, Bottleneck expansion2. Last block activation flags preserved in main graph.'])
 ev=read_evidence('pidnet');views=[build(a,s,ev) for s in CFG]+[build(a,'family',ev)];manifest(a,'pidnet','pidnet','PIDNet',views)
if __name__=='__main__':main()
