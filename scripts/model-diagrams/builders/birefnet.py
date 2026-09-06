"""BiRefNet dual-scale Swin, bilateral image references and deformable ASPP.

FeyNobg imports render() with its own explicit stage3 depth24.
"""
from quicksrnet import *
from nafnet import product


def render(a,family,size,E,depths,window,ev,symbolic=False):
 def mul(n):return f'{n}E' if symbolic else int(E*n)
 dims=[mul(1),mul(2),mul(4),mul(8)];heads=[3,6,12,24] if E==96 else [6,12,24,48]
 if symbolic:dims=['E','2E','4E','8E'];heads=['A','2A','4A','8A']
 channels=[mul(16),mul(8),mul(4),mul(2)];ip=[mul(2),mul(2),mul(1),mul(.5),mul(.25)];out=[mul(8),mul(4),mul(2),mul(1)]
 slug='birefnet' if family=='birefnet' else 'feynobg';title='BiRefNet' if family=='birefnet' else 'FeyNobg'
 d=diagram(a,title+(' family' if symbolic else ' '+size.upper()),'Alpha-matte logits,normalized RGB input3 × 1024 × 1024,native eval. Shapes exclude batch.','birefnet' if family=='birefnet' else 'feynobg',1960,6240)
 p=d.panel('dual','Dual-scale shared encoder',40,230,600,2100)
 for name,x,side in [('full',20,1024),('half',320,512)]:
  op(p,name+'image',x,65,260,name+' image',f'3 × {side} × {side}')
  op(p,name+'bb',x,220,260,'Shared Swin-v1 encoder',f'E={"E" if symbolic else E};window{window}','attention',block='swin-backbone');p.connect(name+'image',name+'bb')
 p.text(25,385,'Half image uses bilinear resize,align_corners=True.',14)
 for j,c in enumerate(dims):
  y=500+j*285;grid=256//2**j
  op(p,f'f{j}',20,y,260,f'Full stage{j+1}',f'{c} × {grid}²')
  op(p,f'h{j}',320,y,260,f'Resize half stage{j+1}',f'{c} × {grid//2}² to{grid}²','pool')
  op(p,f'cat{j}',65,y+130,470,f'Concat two scales: X{j+1}',f'{mul(2**(j+1))} × {grid}²','concat')
  p.wire([(150,y+49),(150,y+100),(205,y+100),(205,y+130)],start=f'f{j}',end=f'cat{j}');p.wire([(450,y+49),(450,y+100),(395,y+100),(395,y+130)],start=f'h{j}',end=f'cat{j}')
 op(p,'context',65,1780,470,'Resize X1,X2,X3 to32²; concat withX4',f'{mul(30)} × 32 × 32','concat')
 op(p,'squeeze',65,1910,470,'BasicDecBlk squeeze',f'{mul(30)} to{mul(16)};32²','conv',block='basicdec');p.connect('context','squeeze')
 q=d.panel('decode','Bilateral-reference decoder',680,230,600,2100)
 for j in range(4):
  yy=65+j*420;ch=channels[j];ci=(mul(18) if j==0 else mul(10) if j==1 else mul(5) if j==2 else mul(2.5));hw=32*2**j
  op(q,f'dcat{j}',75,yy,450,f'Concat current feature and IPT{5-j}',f'{ch} + {ip[j]} = {ci} channels at{hw}²','concat')
  op(q,f'dblock{j}',75,yy+105,450,'BasicDecBlk',f'{ci} to{out[j]};{hw}²','conv',block='basicdec');q.connect(f'dcat{j}',f'dblock{j}')
  if j<3:
   op(q,f'gate{j}',75,yy+205,450,'Gradient-reference attention',f'{out[j]} channels multiplied by spatial gate','attention',block='gdt');q.connect(f'dblock{j}',f'gate{j}')
   op(q,f'up{j}',75,yy+295,450,'Bilinear resize×2',f'{out[j]} × {hw*2}²','pool');q.connect(f'gate{j}',f'up{j}');q.sum(f'dadd{j}',300,yy+385);q.connect(f'up{j}',f'dadd{j}')
   op(q,f'lat{j}',15,yy+365,155,f'Conv1×1(X{3-j})',f'{out[j]} to{out[j]}',h=40);q.connect(f'lat{j}',f'dadd{j}',from_port='right',to_port='left')
   # Connect to next concat after it exists, below.
  else:
   op(q,'upfull',75,yy+205,450,'Bilinear resize to1024²',f'{out[j]} channels','pool');q.connect(f'dblock{j}','upfull')
 for j in range(3):q.connect(f'dadd{j}',f'dcat{j+1}')
 op(q,'finalcat',75,1740,450,'Concat full-resolution feature and IPT1',f'{mul(1)} + {ip[4]} = {mul(1.25)} channels','concat');q.connect('upfull','finalcat')
 op(q,'logits',75,1880,450,'Conv1×1 to one logit',f'{mul(1.25)} to1;1 × 1024 × 1024','conv2d');q.connect('finalcat','logits')
 d.connect('squeeze','dcat0',from_port='right',to_port='left',via=[(660,2164.5),(660,319.5)])
 q.text(25,2040,'Sigmoid to alpha belongs to shared matte postprocessing.',14)
 i=d.panel('image-ref','Image-patch reference paths',1320,230,600,2100)
 i.text(25,65,'Each IPT path consumes the original normalized1024 RGB image.',14)
 for j,(name,ci,co,hw) in enumerate(zip([5,4,3,2,1],[3072,768,192,48,3],ip,[32,64,128,256,1024])):
  yy=150+j*350
  op(i,f'pack{j}',65,yy,470,f'Tile packing for IPT{name}',f'{1024//hw}×{1024//hw} contiguous tiles;{ci}×{hw}²','split')
  op(i,f'iptc{j}',65,yy+100,470,'Conv3×3 thenConv3×3',f'{ci} to64 to{co};boths1,p1','conv',block='simpleconvs');i.connect(f'pack{j}',f'iptc{j}')
  op(i,f'ipt{j}',65,yy+210,470,f'IPT{name}',f'{co} × {hw}²');i.connect(f'iptc{j}',f'ipt{j}')
 i.text(25,2010,'Tile packing concatenates whole spatial tiles as channels;',14);i.text(25,2040,'it is not the pixel-interleaving PixelUnshuffle operation.',14)
 # Swin primitive backbone and window block, with explicit sizes for both calls.
 b=d.panel('swin-backbone','Shared Swin-v1 backbone',40,2390,600,1690,kind='attention',dashed=True,block_type='swin-backbone')
 op(b,'patch','65' if False else 65,65,470,'Conv patch4,stride4 + LayerNorm',f'3 to{mul(1)}; fullgrid256² / halfgrid128²','conv')
 prev='patch'
 for j,ch in enumerate(dims):
  yy=235+j*330
  op(b,f'stage{j}',65,yy,470,f'Swin block stage{j+1},n={depths[j]}',f'{ch} channels;heads{heads[j]};MLPwidth{mul(4*2**j)}','attention',block='swinblock');b.connect(prev,f'stage{j}')
  op(b,f'norm{j}',65,yy+100,470,'Output-stage LayerNorm','Full/half grids '+str(256//2**j)+'² / '+str(128//2**j)+'²','norm');b.connect(f'stage{j}',f'norm{j}')
  if j<3:
   op(b,f'merge{j}',65,yy+205,470,'PatchMerging',f'Concat2×2 gives{mul(4*2**j)};LN;Linear to{mul(2*2**j)}','aggregate',block='patchmerge');b.connect(f'stage{j}',f'merge{j}',from_port='right',to_port='right',via=[(555,yy+24.5),(555,yy+229.5)]);prev=f'merge{j}'
 b.text(25,1605,'Stage normalization is an output tap; merging uses pre-tap tokens.',13)
 s=d.panel('swinblock','Swin block and attention',680,2390,600,1690,kind='attention',dashed=True,block_type='swinblock')
 rows=[('x','Input tokens','Stage widths '+str(dims),'plain'),('ln','LayerNorm','epsilon1e-5','norm'),('pad','Pad spatial grid to window multiple',f'Window{window}; alternating shift0/{window//2 if isinstance(window,int) else "W/2"}','plain'),('split','Partition windows',f'{window*window if isinstance(window,int) else "W²"} tokens perwindow','split'),('qkv','Linear QKV; split heads','QKV widths '+str([mul(3*2**j) for j in range(4)]),'linear'),('qk','QK transpose /sqrt32 +relative bias','Learned relative bias; shifted mask0/-100','attention'),('soft','Softmax thenweights × V','Each head has32 channels','attention'),('proj','Concat heads; output Linear','Restore stage channel width','linear'),('reverse','Reverse windows,undo shift,crop','Restore unpadded stage grid','aggregate')]
 for j,(id,label,detail,kind) in enumerate(rows):op(s,id,65,65+j*120,470,label,detail,kind)
 chain(s,[r[0] for r in rows]);s.sum('sadd1',300,1220);s.connect('reverse','sadd1');s.connect('x','sadd1',from_port='left',to_port='left',via=[(25,89.5),(25,1220)])
 op(s,'ffn',65,1320,470,'LN; Linear to4C; GELU; Linear toC','Numeric widths '+str([mul(4*2**j) for j in range(4)]),'linear',block='ffn')
 s.connect('sadd1','ffn');s.sum('sadd2',300,1510);s.connect('ffn','sadd2');s.dot(300,1280);s.wire([(300,1280),(25,1280),(25,1510),(287,1510)],start='sadd1',end='sadd2')
 s.text(25,1620,'PatchMerging reads2×2 spatial neighbors, thenLN and4C-to2C Linear.',13)
 bd=d.panel('basicdec','BasicDecBlk',1320,2390,600,1060,kind='conv',dashed=True,block_type='basicdec')
 rows=[('cin','Conv3×3','Occurrence Cin to64;s1,p1,biasTrue','conv2d'),('bn1','BatchNorm2d + ReLU','64 channels','norm'),('aspp','ASPPDeformable','64 to64','attention'),('cout','Conv3×3','64 tooccurrence Cout;s1,p1,biasTrue','conv2d'),('bn2','BatchNorm2d','Cout channels;no finalactivation','norm')]
 for j,(id,label,detail,kind) in enumerate(rows):op(bd,'bd'+id,65,65+j*165,470,label,detail,kind,block='aspp' if id=='aspp' else '')
 chain(bd,['bd'+r[0] for r in rows])
 g=d.panel('gdt','Gradient-reference attention',1320,3500,600,580,kind='attention',dashed=True,block_type='gdt')
 for j,(id,label,detail,kind) in enumerate([('c','Conv3×3;BN;ReLU',f'Inputchannels{out[:3]} to16;s1,p1','conv'),('a','Conv1×1 +Sigmoid','16to1;spatial gate','attention')]):op(g,'gdt'+id,65,65+j*155,470,label,detail,kind)
 g.connect('gdtc','gdta');product(g,'gdtmul',300,420);g.connect('gdta','gdtmul');op(g,'gdtX',15,350,80,'Feature',h=35);g.connect('gdtX','gdtmul',via=[(55,420)],to_port='left');g.text(25,525,'Attention executes in eval; gradient-prediction supervision does not.',13)
 # Exact deformable ASPP including duplicated1×1 branch.
 aP=d.panel('aspp','ASPPDeformable: five parallel branches',40,4140,1880,760,kind='attention',dashed=True,block_type='aspp')
 for j,k in enumerate([1,1,3,7]):
  xx=25+j*375;op(aP,'deform'+str(j),xx,90,340,f'DeformableConv{k}×{k}',f'64to256;padding{k//2}','conv',block='deform')
  op(aP,'dbn'+str(j),xx,235,340,'BatchNorm2d thenReLU','256 channels','norm');aP.connect('deform'+str(j),'dbn'+str(j))
 op(aP,'gp',1525,90,330,'Global avgpool;Conv1×1;BN;ReLU','64to256;resize back toinputgrid','pool')
 for j in range(5):
  source='dbn'+str(j) if j<4 else'gp';sx,sy=aP.port(source,'bottom');aP.wire([(sx,sy),(sx,420)],start=source,end='aspconcat')
 op(aP,'aspconcat',25,420,1830,'Concat five256-channel results','1280 channels','concat')
 op(aP,'aspout',350,580,1180,'Conv1×1;BatchNorm;ReLU;Dropout0.5','1280to64;dropout isidentity in eval','conv');aP.connect('aspconcat','aspout')
 aP.text(25,710,'Two distinct1×1 deformable branches exist: aspp1 plus the first entry of parallel_block_sizes=(1,3,7).',14)
 dc=d.panel('deform','Modulated deformable convolution',40,4960,1200,1100,kind='conv',dashed=True,block_type='deform')
 op(dc,'offset',25,65,500,'Offset Conv k×k','64to2k²:2,18,98 offsets for k1,3,7; biasTrue','conv2d')
 op(dc,'mod',650,65,500,'Mask Conv k×k','64tok²:1,9,49 values;biasTrue','conv2d')
 op(dc,'modsig',650,230,500,'Multiply2 × Sigmoid','Modulation range(0,2)','activation');dc.connect('mod','modsig')
 op(dc,'sample',25,420,500,'Base kernel grid + learned offsets','Bilinear sample input features at displaced locations','aggregate');dc.connect('offset','sample')
 op(dc,'weighted',220,655,760,'Multiply sampled values by modulation and kernel weights','Learned kernel[256,64,k,k]; sum over channels/kernelpositions','linear')
 dc.wire([(275,469),(275,605),(390,605),(390,655)],start='sample',end='weighted');dc.wire([(900,279),(900,605),(800,605),(800,655)],start='modsig',end='weighted')
 op(dc,'dcout',220,845,760,'Output256 channels','Stride1; spatial size preserved','plain');dc.connect('weighted','dcout')
 dc.text(25,1030,'regular_conv supplies its weights to torchvision deform_conv2d; it is not an extra executed convolution.',14)
 ipt=d.panel('simpleconvs','Image-reference primitives and values',1280,4960,640,1100,kind='conv',dashed=True,block_type='simpleconvs')
 op(ipt,'ipin',65,65,510,'Packed image tiles','Cin3072,768,192,48,3 fromcoarse tofull')
 op(ipt,'ipc1',65,245,510,'Conv3×3','Cin to64;stride1,padding1','conv2d');ipt.connect('ipin','ipc1')
 op(ipt,'ipc2',65,425,510,'Conv3×3','64 toIPT outputchannels '+str(ip),'conv2d');ipt.connect('ipc1','ipc2')
 ipt.text(25,650,'No normalization or activation between these two convolutions.',14)
 if symbolic:
  ipt.text(25,750,'Tiny:E96,A3,depths[2,2,6,2],window7.',14)
  ipt.text(25,795,'Large:E192,A6,depths[2,2,18,2],window12.',14)
 elif family=='feynobg':ipt.text(25,750,'FeyNobg changes only Swin stage3 depth to24.',14)
 ipt.text(25,885,'Training-only multiscale and gradient-label heads are stored',14);ipt.text(25,918,'for checkpoint loading, but excluded from this eval graph.',14)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,slug,'family' if symbolic else size+'-matte','Shared dual-scale topology' if symbolic else size,'matte',size,'family' if symbolic else'concrete','3×1024×1024 normalized RGB',rec.get('device','source'))

def main():
 a=environment('Build BiRefNet Tiny/Large and shared symbolic view')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'birefnet');records={}
  for size in ['t','l']:
   m=construct_meta(lambda:nn.LibreBiRefNetModel(size));records[size]=meta_probe(m,(1,3,1024,1024),['bb','squeeze_module','decoder.decoder_block4','decoder.decoder_block1','decoder.conv_out1'])
  records['t']['cpu_smoke64']=cpu_probe(nn.LibreBiRefNetModel('t'),(1,3,64,64),['squeeze_module','decoder.conv_out1'])
  write_evidence(a,'birefnet',records,['libreyolo/models/birefnet/nn.py:BIREFNET_DIMS,LibreBiRefNetModel,Decoder,BasicDecBlk,ASPPDeformable,DeformableConv2d','libreyolo/models/birefnet/model.py:INPUT_SIZES'],['Default1024graphsshape-checkedonmeta; Tiny additionallyranfullCPU64smoke.','Dual-scaleencoderweightsare shared; eachstagechannelsdouble afterhalf-featureupsample/concat.','ASPP haskernel1,1,3,7 deformablebranches plusglobalpool,not ordinary atrousdilations. Modulatoris2*sigmoid.','Gradientattentionexecutesduringeval; supervisionheads donot.'])
 ev=read_evidence('birefnet');views=[render(a,'birefnet','t',96,[2,2,6,2],7,ev),render(a,'birefnet','l',192,[2,2,18,2],12,ev),render(a,'birefnet','family',96,[2,2,'N3',2],'W',ev,True)];manifest(a,'birefnet','birefnet','BiRefNet',views)
if __name__=='__main__':main()
