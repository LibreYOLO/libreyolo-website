"""PP-LiteSeg: rectangular t50/b50/t75/b75 recipes, STDC, SPPM and UAFM."""
from quicksrnet import *
from nafnet import product
CFG={'t50':([2,2,2],[64,128,128],[128,64,32],512,1024),'b50':([4,5,3],[96,128,128],[128,96,64],512,1024),'t75':([2,2,2],[64,128,128],[128,64,32],768,1536),'b75':([4,5,3],[96,128,128],[128,96,64],768,1536)}

def stdc(d,i,down,y):
 co=[256,512,1024][i];ci=[64,256,512][i] if down else co
 p=d.panel(f'stdc-{i}-{down}',f'STDC {co}, '+('first block, stride2' if down else 'repeated block, stride1'),40+i*600,y,560,1110,kind='bottleneck',dashed=True,block_type=f'stdc-{i}')
 op(p,'i'+str(i)+str(down),100,60,360,'Input',f'{ci} channels')
 pre=f'stdc{i}{down}-';iid='i'+str(i)+str(down)
 op(p,pre+'c1',100,150,360,'ConvBNReLU 1×1',f'{ci} to {co//2}; stride1','conv',block='cbr');p.connect(iid,pre+'c1')
 op(p,pre+'pool',15,260,160,'AvgPool 3×3' if down else 'Identity a', 'stride2, padding1' if down else f'{co//2} channels','pool' if down else 'plain')
 p.connect(pre+'c1',pre+'pool',from_port='left',via=[(95,174.5),(95,230)])
 op(p,pre+'a',50,365,85,'a',f'{co//2} ch','plain');p.connect(pre+'pool',pre+'a')
 op(p,pre+'dw',235,260,305,'Depthwise Conv + BN' if down else 'Identity trunk',f'k3,s2,p1; {co//2} groups; no ReLU' if down else f'{co//2} channels','conv' if down else 'plain',block='cbr')
 p.connect(pre+'c1',pre+'dw',via=[(280,230),(387.5,230)])
 prev=pre+'dw'
 for j,(name,cin,cout) in enumerate([('b',co//2,co//4),('c',co//4,co//8),('d',co//8,co//8)]):
  yy=390+j*135;op(p,pre+name+'conv',235,yy,305,'ConvBNReLU 3×3',f'{cin} to {cout}; s1,p1','conv',block='cbr');p.connect(prev,pre+name+'conv');prev=pre+name+'conv'
  op(p,pre+name,145,yy+12,65,name,f'{cout} ch',h=35);p.connect(prev,pre+name,from_port='left',to_port='right')
 for j,name in enumerate(['a','b','c','d']):
  xx=55+j*130;op(p,pre+'in-'+name,xx,810,70,name,h=26)
  p.wire([(xx+35,836),(xx+35,890)],start=pre+'in-'+name,end=pre+'concat')
 op(p,pre+'concat',40,890,480,'Concat a,b,c,d',f'{co//2}+{co//4}+{co//8}+{co//8} = {co} channels','concat')
 p.text(20,1000,'Named a,b,c,d connections keep all four inputs distinct.',13)
 p.text(20,1032,'Each later convolution consumes only the previous result.',13)

def uafm(d,j,cin,cout,up,hw,y):
 p=d.panel(f'uafm{j}',f'UAFM {j+1}: {cin} to {cout} channels',40+j*600,y,560,1390,kind='aggregate',dashed=True,block_type=f'uafm{j}')
 pre=f'u{j}-'
 for id,x,label in [('x',20,'Decoder input X'),('s',310,'Projected skip S')]:op(p,pre+id,x,65,230,label,f'{cin} channels')
 op(p,pre+'up',20,165,230,'Bilinear resize ×'+str(up) if up!=1 else 'Identity resize',f'{cin} × {hw}','pool' if up!=1 else 'plain');p.connect(pre+'x',pre+'up')
 op(p,pre+'si',310,165,230,'Identity skip projection',f'{cin} × {hw}','plain');p.connect(pre+'s',pre+'si')
 for id,x in [('x',20),('s',310)]:
  op(p,pre+id+'reduce',x,275,230,'Channel mean and max','Two 1-channel spatial maps','pool');p.connect(pre+'up' if id=='x' else pre+'si',pre+id+'reduce')
 op(p,pre+'concat',90,395,380,'Concat mean(X),max(X),mean(S),max(S)','4 spatial channels','concat')
 p.wire([(135,324),(135,367),(180,367),(180,395)],start=pre+'xreduce',end=pre+'concat');p.wire([(425,324),(425,367),(380,367),(380,395)],start=pre+'sreduce',end=pre+'concat')
 steps=[('c1','ConvBNReLU 3×3','4 to 2; s1,p1','conv'),('c2','Conv2d 3×3 + BatchNorm','2 to 1; s1,p1; no activation','conv'),('sig','Sigmoid','Spatial attention A: 1 channel','activation')]
 for i,(id,label,detail,kind) in enumerate(steps):op(p,pre+id,90,500+i*90,380,label,detail,kind,block='cbr' if id.startswith('c') else '')
 chain(p,[pre+'concat']+[pre+s[0] for s in steps])
 op(p,pre+'xa',20,835,230,'Multiply X × A',f'{cin} channels','linear')
 op(p,pre+'sa',310,835,230,'Multiply S × (1-A)',f'{cin} channels','linear')
 p.wire([(280,729),(280,785),(135,785),(135,835)],start=pre+'sig',end=pre+'xa');p.wire([(280,729),(280,785),(425,785),(425,835)],start=pre+'sig',end=pre+'sa');p.dot(280,785)
 p.text(28,927,'X and S denote the resized decoder and skip tensors above.',12)
 p.sum(pre+'sum',280,1010)
 p.wire([(135,884),(135,1010),(267,1010)],start=pre+'xa',end=pre+'sum');p.wire([(425,884),(425,1010),(293,1010)],start=pre+'sa',end=pre+'sum')
 op(p,pre+'out',90,1115,380,'ConvBNReLU 3×3',f'{cin} to {cout}; s1,p1; output {cout} × {hw}','conv',block='cbr');p.connect(pre+'sum',pre+'out')
 p.text(20,1240,'Projection skips are identities for these registered recipes.',13)
 p.text(20,1280,'Attention uses spatial channel statistics, not token attention.',13)

def build(a,size,ev):
 sym=size=='family'
 ns,proj,dec,H,W=([f'N{i}' for i in range(3)],[ 'P8',128,128],[128,'D16','D8'],'H','W') if sym else CFG[size]
 dims=lambda stride:f'H/{stride} × W/{stride}' if sym else f'{H//stride} × {W//stride}'
 d=diagram(a,'PP-LiteSeg family' if sym else 'PP-LiteSeg '+size.upper(),f'Cityscapes semantic segmentation, 19 classes, {H} × {W} RGB, native eval. Shapes exclude batch.','ppliteseg',1840,6570 if sym else 6290)
 p=d.panel('backbone','STDC backbone',40,230,560,1490)
 rows=[('input','RGB + ImageNet normalization',f'3 × {H} × {W}','norm'),('stem1','ConvBNReLU 3×3','3 to 32; s2,p1; '+dims(2),'conv'),('stem2','ConvBNReLU 3×3','32 to 64; s2,p1; '+dims(4),'conv')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,70,65+i*135,420,label,detail,kind,block='cbr')
 chain(p,[r[0] for r in rows]);prev='stem2'
 for i,stride in enumerate([8,16,32]):
  op(p,f'b{i}',70,505+i*200,420,f'STDC stage s{stride}, n={ns[i]}',f'{[256,512,1024][i]} × {dims(stride)}; first stride2','bottleneck',block=f'stdc-{i}');p.connect(prev,f'b{i}');prev=f'b{i}'
  op(p,f'F{stride}',495,518+i*200,55,f'F{stride}',h=26,block=f'signal-F{stride}');p.connect(prev,f'F{stride}',from_port='right',to_port='left')
 p.text(25,1170,'Every STDC stage begins with one downsampling block.',14)
 p.text(25,1210,'Remaining blocks keep spatial resolution and channel width.',14)
 p.text(25,1310,'50 / 75 indicate source validation scale factors.',15)
 p.text(25,1350,'They are not channel-width multipliers.',15)
 q=d.panel('project','Projection and context paths',640,230,560,1490)
 for j,stride in enumerate([8,16,32]):
  y=65+j*220
  op(q,f'pi{stride}',75,y,410,f'F{stride}',f'{[256,512,1024][j]} × {dims(stride)}',block=f'signal-F{stride}')
  op(q,f'po{stride}',75,y+100,410,'ConvBNReLU 3×3',f'{[256,512,1024][j]} to {proj[j]}; s1,p1; P{stride}','conv',block='cbr');q.connect(f'pi{stride}',f'po{stride}')
 op(q,'contextin',75,815,410,'F32 context input','1024 × '+dims(32),block='signal-F32')
 op(q,'context',75,930,410,'SPPM','128 × '+dims(32),'pool',block='sppm');q.connect('contextin','context')
 q.text(25,1055,'P8/P16/P32 are independent skips; context reads raw F32.',13)
 cbr=d.panel('cbrdef','ConvBNReLU',665,1330,510,360,kind='conv',dashed=True,block_type='cbr')
 for j,(nid,lab,kind) in enumerate([('primitive-c','Conv2d','conv2d'),('primitive-n','BatchNorm2d','norm'),('primitive-r','ReLU','activation')]):
  op(cbr,nid,75,55+j*92,360,lab,'Parameters specified at each occurrence' if j==0 else '',kind)
 chain(cbr,['primitive-c','primitive-n','primitive-r'])
 r=d.panel('decode','UAFM decoder and logits',1240,230,560,1490)
 prev=None
 for j,stride in enumerate([32,16,8]):
  cin=128 if j<2 else dec[1];yy=65+j*250
  op(r,f'dec{j}',90,yy,410,f'UAFM {j+1}',f'{cin} to {dec[j]}; skip P{stride}; resize ×{[1,2,2][j]}','aggregate',block=f'uafm{j}')
  if prev:r.connect(prev,f'dec{j}')
  op(r,f'dshape{j}',90,yy+110,410,'Fused feature',f'{dec[j]} × {dims(stride)}');r.connect(f'dec{j}',f'dshape{j}');prev=f'dshape{j}'
 op(r,'head',90,830,410,'ConvBNReLU 3×3',f'{dec[2]} to {dec[2]}; s1,p1','conv',block='cbr');r.connect(prev,'head')
 op(r,'drop',90,930,410,'Dropout 0.0','Identity');op(r,'class',90,1030,410,'Conv2d 1×1',f'{dec[2]} to 19; bias=False','conv2d');op(r,'up',90,1130,410,'Bilinear resize ×8',f'19 × {H} × {W}','pool');chain(r,['head','drop','class','up'])
 r.text(25,1320,'Native eval returns only main logits.',14)
 r.text(25,1360,'Three auxiliary heads are training-only and do not execute.',14)
 # Pooling paths are explicit and sum before the context convolution.
 s=d.panel('sppm','SPPM: simple pyramid pooling',40,1780,1760,600,kind='pool',dashed=True,block_type='sppm')
 for i,scale in enumerate([1,2,4]):
  x=35+i*320
  for j,(id,label,detail,kind) in enumerate([('pool','Adaptive average pool',f'1024 × {scale} × {scale}','pool'),('conv','ConvBNReLU 1×1','1024 to 128','conv'),('up','Bilinear resize','128 × '+dims(32),'pool')]):op(s,f'sp{i}{id}',x,65+j*120,265,label,detail,kind,block='cbr' if id=='conv' else '')
  chain(s,[f'sp{i}pool',f'sp{i}conv',f'sp{i}up'])
 s.sum('spadd1',1120,185);s.sum('spadd2',1120,365)
 s.wire([(167.5,354),(167.5,410),(1085,410),(1085,185),(1107,185)],start='sp0up',end='spadd1')
 s.wire([(487.5,354),(487.5,434),(1155,434),(1155,185),(1133,185)],start='sp1up',end='spadd1')
 s.connect('spadd1','spadd2');s.wire([(807.5,354),(807.5,365),(1107,365)],start='sp2up',end='spadd2')
 op(s,'spout',1260,340,455,'ConvBNReLU 3×3','128 to 128; s1,p1; context output','conv',block='cbr');s.connect('spadd2','spout',from_port='right',to_port='left')
 s.text(35,522,'Each pooling branch independently reads F32. Three 128-channel results are added, not concatenated.',15)
 for i in range(3):stdc(d,i,True,2440);stdc(d,i,False,3610)
 for j,stride in enumerate([32,16,8]):uafm(d,j,128 if j<2 else dec[1],dec[j],[1,2,2][j],dims(stride),4780)
 d.text(50,6212,'ConvBNReLU = Conv2d, BatchNorm2d, ReLU in sequence. Numeric channels/kernels at each occurrence; bias=False.',15)
 if sym:
  t=d.panel('table','Family variables and concrete recipes',40,6250,1760,190)
  t.text(25,50,'t50/t75: N=[2,2,2], P8=64, D16=64, D8=32.   b50/b75: N=[4,5,3], P8=96, D16=96, D8=64.',15)
  t.text(25,95,'50: H=512, W=1024.   75: H=768, W=1536. P16=P32=128 and first UAFM output=128 for every recipe.',15)
  t.text(25,140,'Block definitions resolve every backbone channel width; only repeats, decoder widths and input canvas vary.',15)
 record=ev['records'].get(size,{})
 return finish_view(a,d,'ppliteseg','family' if sym else size+'-semantic','Shared topology' if sym else size,'semantic',size,'family' if sym else 'concrete',f'3×{H}×{W}',record.get('device','source'))

def main():
 a=environment('Build every rectangular PP-LiteSeg recipe')
 if a.verify:
  nn=nn_module(a,'ppliteseg');records={}
  for size,(_,_,_,h,w) in CFG.items():records[size]=cpu_probe(nn.LibrePPLiteSegNet(size,19),(1,3,h,w),['encoder.backbone','encoder.context_module','encoder.proj_convs.0','decoder.up_stages.0','decoder.up_stages.1','decoder.up_stages.2','seg_head'])
  write_evidence(a,'ppliteseg',records,['libreyolo/models/ppliteseg/nn.py:SIZE_CONFIGS, STDCBlock, STDCBackbone, SPPM, UAFM, LibrePPLiteSegNet'],['All four actual rectangular recipe canvases checked on CPU.','50/75 are validation scaling tokens, not width multipliers. SPPM adds3 branches, UAFM uses sigmoid spatial attention and complementary weights.','Projection skip inside UAFM is identity for all registered configurations. Training auxiliary heads excluded from eval graph.'])
 ev=read_evidence('ppliteseg');views=[build(a,s,ev) for s in CFG]+[build(a,'family',ev)];manifest(a,'ppliteseg','ppliteseg','PP-LiteSeg',views)
if __name__=='__main__':main()
