"""ZipDepth base and separately trained NPU head, balanced global mode."""
from quicksrnet import *
from nafnet import product

def seq(p,pre,rows,x=75,y=65,w=430,step=105):
 for j,(id,label,detail,kind) in enumerate(rows):op(p,pre+id,x,y+j*step,w,label,detail,kind)
 chain(p,[pre+r[0] for r in rows]);return pre+rows[-1][0]

def build(a,size,ev):
 family=size=='family'
 d=diagram(a,'ZipDepth family' if family else 'ZipDepth '+size.upper(),'Relative inverse depth,input3 × 384 × 384,native unfused eval,global_mode=balanced. Shapes exclude batch.','zipdepth',1900,7030)
 p=d.panel('encoder','Encoder',40,230,590,1700)
 rows=[('norm','RGB + ImageNet normalization','3 × 384 × 384','norm'),('half','ConvBN 3×3,s2','3 to24;half skip24×192²','conv'),('quarter','ConvBN3×3,s2','24 to48;48×96²','conv'),('s1','QARepBlock,n=2','48 to48;S1:48×96²','bottleneck'),('d2','QARepBlock stride2','48 to96;48²','bottleneck'),('s2','QARepBlock,n=2','96 to96;48²','bottleneck'),('multi','MinimalMultiScale thenStripPooling','96×48²;S2','aggregate'),('d3','QARepBlock stride2','96 to192;24²','bottleneck'),('s3','QARepBlock,n=6','192 to192;24²','bottleneck'),('ca','ChannelAttention thenGlobalContext','192×24²;S3','aggregate'),('d4','QARepBlock stride2','192 to384;12²','bottleneck'),('s4','QARepBlock,n=2','384×12²','bottleneck'),('spp','LightweightSPPF','384×12²;S4','pool')]
 for j,(id,label,detail,kind) in enumerate(rows):op(p,id,70,65+j*120,450,label,detail,kind,block='qarep' if kind=='bottleneck' else 'sppf' if id=='spp' else id if id in ['multi','ca'] else 'convbn' if kind=='conv' else '')
 chain(p,[r[0] for r in rows])
 q=d.panel('neck','Cross-scale exchange and FPN',670,230,570,1700)
 rows=[('cross','MinimalCrossScale(S3,S4)','S3:192×24², S4:384×12²','aggregate'),('proj','ConvBN1×1 on exchangedS4','384 to288;12²','conv'),('f3','UltraLightFusion3','high192/low288 to192;24²','aggregate'),('f2','UltraLightFusion2','high96/low192 to144;48²','aggregate'),('f1','UltraLightFusion1','high48/low144 to96;96²','aggregate'),('fh','UltraLightFusion half','high24/low96 to32;192²','aggregate')]
 for j,(id,label,detail,kind) in enumerate(rows):op(q,id,65,65+j*245,440,label,detail,kind,block='crossdef' if id=='cross' else 'uf' if id.startswith('f') else 'convbn')
 chain(q,[r[0] for r in rows]);q.text(25,1580,'High-resolution inputs come fromS3,S2,S1 andhalf-stem skip.',13)
 h=d.panel('prediction','Half-resolution depth and final2× reconstruction',1280,230,580,1700)
 seq(h,'head',[('feat','Half feature','32×192×192','plain'),('depth','Conv2d3×3','32 to1;s1,p1;halfdepth192²','conv2d'),('up','Convex upsample' if size=='b' else 'Nearest/bilinear learned blend' if size=='bnpu' else 'Selected trained2× head','b:9-neighbor convex. bnpu:learnedinterpolation alpha.','aggregate'),('relu','ReLU','Nonnegative relative inverse depth','activation'),('out','Output','1×384×384','plain')],y=65,step=250)
 h.text(25,1450,'Both trained heads also read the32-channel half feature.',14)
 h.text(25,1510,'b andbnpu have different head parameters; no weight-free rewrite.',13)
 h.text(25,1570,'Public loaded models fuse QARep branches; this view is unfused.',13)
 # Reparameterizable block, explicit branches.
 r=d.panel('qarep','QARepBlock',40,1990,590,1100,kind='bottleneck',dashed=True,block_type='qarep')
 r.text(20,65,'Stage widths48,96,192,384. Transitions48to96,96to192,192to384.',13)
 for label,x,k in [('3×3',20,3),('1×1',320,1)]:
  op(r,'qconv'+str(k),x,165,250,'Conv2d'+label,'Ci toCo;declaredstride;biasFalse','conv2d');op(r,'qbn'+str(k),x,335,250,'BatchNorm2d','Co channels','norm');r.connect('qconv'+str(k),'qbn'+str(k))
 r.sum('qadd',295,555);r.wire([(145,384),(145,555),(282,555)],start='qbn3',end='qadd');r.wire([(445,384),(445,555),(308,555)],start='qbn1',end='qadd')
 r.sum('qidadd',295,725);r.connect('qadd','qidadd');op(r,'qid',20,650,120,'Input identity','Only Ci=Co,s1',h=55);r.connect('qid','qidadd',via=[(80,725)],to_port='left')
 op(r,'qa',75,855,440,'ReLU','One fusedConv3×3 can replacebranches afterload','activation');r.connect('qidadd','qa');r.text(20,1025,'Identity branch has no BatchNorm; downsample blocks omit it.',13)
 ca=d.panel('cadef','ChannelAttention',670,1990,570,1100,kind='attention',dashed=True,block_type='ca')
 seq(ca,'ca',[('x','Input','192×24²','plain'),('pool','Mean overH,W','192×1×1','pool'),('fc1','Conv1×1','192to24;biasFalse','conv2d'),('a','ReLU','','activation'),('fc2','Conv1×1','24to192;biasFalse','conv2d'),('sig','Sigmoid','','activation')],step=135,w=420)
 product(ca,'caout',285,970);ca.connect('casig','caout');ca.connect('cax','caout',from_port='left',to_port='left',via=[(25,89.5),(25,970)])
 sp=d.panel('strip','StripPoolingAttention',1280,1990,580,1100,kind='attention',dashed=True,block_type='multi')
 op(sp,'sh',20,65,245,'Mean overwidth','96×48×1','pool');op(sp,'sw',315,65,245,'Mean overheight','96×1×48','pool');sp.sum('stripadd',290,280)
 sp.wire([(142.5,114),(142.5,280),(277,280)],start='sh',end='stripadd');sp.wire([(437.5,114),(437.5,280),(303,280)],start='sw',end='stripadd')
 last=seq(sp,'sp',[('conv','DepthwiseConv1×1','96channels/groups;biasFalse','conv2d'),('bn','BatchNorm2d','96channels','norm'),('sig','Sigmoid','','activation')],y=410,step=145);sp.connect('stripadd','spconv');product(sp,'stripmul',290,945);sp.connect(last,'stripmul');op(sp,'stripX',20,900,80,'Input X',h=35);sp.connect('stripX','stripmul',from_port='right',to_port='left')
 ms=d.panel('multiscale','MinimalMultiScale',40,3150,590,1100,kind='aggregate',dashed=True,block_type='multi')
 for j,x in enumerate([20,320]):op(ms,'ms'+str(j),x,65,250,'DepthwiseConv3×3',f'96groups;dilation{j+1},padding{j+1}','conv2d')
 ms.sum('madd',295,310);ms.wire([(145,114),(145,310),(282,310)],start='ms0',end='madd');ms.wire([(445,114),(445,310),(308,310)],start='ms1',end='madd')
 op(ms,'mbn',75,455,440,'BatchNorm2d','96 channels','norm');ms.connect('madd','mbn');ms.sum('mres',295,685);ms.connect('mbn','mres');op(ms,'mx',20,620,110,'Input X','96×48²',h=49);ms.connect('mx','mres',via=[(75,685)],to_port='left');ms.text(20,870,'Residual addition has no finalactivation.',14)
 gc=d.panel('gc','GlobalContextBlock',670,3150,570,1100,kind='attention',dashed=True,block_type='ca')
 rows=[('weight','Conv1×1;flatten','192to1;576spatial logits','conv2d'),('soft','Softmax over576positions','Context weights','attention'),('bmm','Batch matmul X × weights','192×576 times576×1','attention'),('reduce','Conv1×1','192to48','conv2d'),('bn','BatchNorm2d + ReLU','48channels','norm'),('expand','Conv1×1','48to192','conv2d')]
 seq(gc,'gc',rows,step=135,w=420);gc.sum('gcadd',285,970);gc.connect('gcexpand','gcadd');op(gc,'gcX',15,885,90,'Input X',h=35);gc.connect('gcX','gcadd',via=[(60,970)],to_port='left')
 sf=d.panel('sppf','LightweightSPPF',1280,3150,580,1100,kind='pool',dashed=True,block_type='sppf')
 op(sf,'sfconv',95,65,420,'ConvBN1×1','384to96;12²','conv',block='convbn')
 for j in range(3):op(sf,'pool'+str(j),95,220+j*170,420,'MaxPool5×5,stride1,padding2','96×12²','pool');sf.connect('sfconv' if j==0 else 'pool'+str(j-1),'pool'+str(j))
 for j,name in enumerate(['x','p1','p2','p3']):op(sf,'sftap'+str(j),25+j*140,775,100,name,h=26);sf.wire([(75+j*140,801),(75+j*140,855)],start='sftap'+str(j),end='sfcat')
 op(sf,'sfcat',25,855,530,'Concat x,p1,p2,p3','384×12²','concat');op(sf,'sfout',25,980,530,'ConvBN1×1','384to384','conv',block='convbn');sf.connect('sfcat','sfout')
 cross=d.panel('crossdef','Bidirectional MinimalCrossScale',40,4310,880,920,kind='aggregate',dashed=True,block_type='crossdef')
 for name,x,ci,co,mode in [('low',20,384,192,'Nearest resize12² to24²'),('high',470,192,384,'AvgPool2×2,stride2')]:
  seq(cross,name,[('c','GroupedConv1×1',f'{ci}to{co};groups4;biasFalse','conv2d'),('r',mode,'Match targetgrid','pool'),('s','Multiply0.3','','linear')],x=x,y=65,w=390,step=170)
  cross.sum(name+'add',x+195,690);cross.connect(name+'s',name+'add');op(cross,name+'id',x,600,90,'Original'+('S3' if name=='low' else'S4'),h=38);cross.connect(name+'id',name+'add',via=[(x+45,690)],to_port='left')
 cross.text(25,840,'Both exchange paths read originalS3/S4, before either result is updated.',14)
 uf=d.panel('uf','UltraLightFusion and ConvBN',960,4310,900,920,kind='aggregate',dashed=True,block_type='uf')
 op(uf,'high',25,65,400,'High-resolution source','Channels192,96,48,24 across four fusions')
 op(uf,'low',475,65,400,'Bilinear resize low-resolution source','Channels288,192,144,96;align_cornersFalse','pool')
 for name,x in [('high',25),('low',475)]:op(uf,name+'p',x,260,400,'GroupedConv1×1,groups4','Both project to192,144,96,32 respectively','conv2d');uf.connect(name,name+'p')
 uf.sum('ufadd',450,480);uf.wire([(225,309),(225,480),(437,480)],start='highp',end='ufadd');uf.wire([(675,309),(675,480),(463,480)],start='lowp',end='ufadd')
 op(uf,'ufbn',235,615,430,'BatchNorm2d thenReLU','192,144,96,32channels respectively','norm');uf.connect('ufadd','ufbn');uf.text(25,820,'ConvBN elsewhere means Conv2d (biasFalse),BatchNorm2d,ReLU.',14)
 # Both head graphs are always visible, selection is explicitly tied to checkpoint size.
 for unfold,x in [(True,40),(False,970)]:
  v=d.panel('convex' if unfold else'npu','b:9-neighbor convex head' if unfold else'bnpu:unfold-free trained head',x,5290,890,1480,kind='aggregate',dashed=True)
  if unfold:
   rows=[('conv','Conv3×3 +BN +ReLU','32to8;192²;biasFalse','conv'),('mask','Conv1×1','8to36 =9neighbors×4subpixels','conv2d'),('reshape','Reshape andsoftmax over9neighbors','9×4×192×192;temperature1','attention'),('neighbors','Replicate-pad halfdepth;Unfold3×3','9×1×192×192 neighborhood values','split'),('weight','Multiply masks × neighbors,sum9','4×192×192','aggregate'),('shuffle','PixelShuffle2','1×384×384','aggregate')]
  else:
   rows=[('conv','Conv1×1 +BN +ReLU','32to16;192²;biasFalse','conv'),('dw','DepthwiseConv5×5 +BN +ReLU','16groups;padding2;192²','conv'),('alpha','Conv1×1;bilinear×2;sigmoid','16to1;alpha1×384×384','activation'),('near','Nearest×2 andbilinear×2 halfdepth','Two1×384×384 candidate maps','pool'),('blend','alpha×nearest + (1-alpha)×bilinear','Learned convex interpolation','aggregate')]
  seq(v,'cv' if unfold else'np',rows,x=125,y=65,w=640,step=190)
  prefix='cv' if unfold else 'np';before=prefix+('reshape' if unfold else 'alpha');independent=prefix+('neighbors' if unfold else 'near');combine=prefix+('weight' if unfold else 'blend')
  for edge in list(v.wires):
   if edge.get('data-from')==before and edge.get('data-to')==independent:v.wires.remove(edge)
  op(v,prefix+'depthinput',20,647,80,'Depth',h=26);v.connect(prefix+'depthinput',independent,from_port='right',to_port='left')
  v.connect(before,combine,from_port='right',to_port='right',via=[(830,469.5),(830,849.5)])
  v.text(25,1330,'Both heads finish withReLU. They are separately trained checkpoint variants.',14)
 d.text(50,6870,'Balanced mode includes strip pooling andGC context; it does not execute the optional full-mode global-token attention.',15)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'zipdepth','family' if family else size+'-depth','Shared base withbothhead alternatives' if family else size,'depth',size,'family' if family else'concrete','3×384×384',rec.get('device','source'))

def main():
 a=environment('Build ZipDepth b/bnpu balanced native architectures')
 if a.verify:
  nn=nn_module(a,'zipdepth');records={}
  for size in ['b','bnpu']:records[size]=cpu_probe(nn.LibreZipDepthNet(size),(1,3,384,384),['encoder.stem_half','encoder.stage1','encoder.stage2','encoder.stage3','encoder.stage4','encoder.cross_scale','decoder.fuse_half','decoder.head_half','decoder.convex_up'])
  write_evidence(a,'zipdepth',records,['libreyolo/models/zipdepth/nn.py:SIZE_CONFIGS,ZipDepthEncoder,ZipDepthDecoder,FastConvexUpsample','libreyolo/models/zipdepth/model.py:post-loadfuse_for_inference'],['Bothbase/NPUnativeunfused384graphscheckedCPU. Defaultbalancedmodeonly.','NPUdecoderisaseparatelytrainedhead; b usesreplicated3×3unfoldand9-waysoftmax.','Onlyb,bnpuareregisteredsizes;small/large/giantupstreamconfigsarenotpublicsizecodes.'])
 ev=read_evidence('zipdepth');views=[build(a,s,ev) for s in ['b','bnpu','family']];manifest(a,'zipdepth','zipdepth','ZipDepth',views)
if __name__=='__main__':main()
