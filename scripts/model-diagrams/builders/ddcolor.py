"""DDColor Tiny/Large: ConvNeXt, wide UNet, cyclic color-query transformer, Lab chroma."""
from quicksrnet import *


def build(a,size,ev):
 family=size=='family';base=96 if size=='t' else 192;dims=[base*2**i for i in range(4)];depths=[3,3,9 if size=='t' else 27,3]
 if family:dims=['C1','C2','C3','C4'];depths=[3,3,'N3',3]
 d=diagram(a,'DDColor family' if family else 'DDColor '+size.upper(),'Image colorization,input3 × 512 × 512 grayscale RGB,native eval. Output is2-channel Lab chroma.','ddcolor',1900,4370)
 p=d.panel('encoder','ConvNeXt encoder',40,230,570,1570)
 op(p,'input',65,65,440,'ImageNet normalization','3 × 512 × 512')
 prev='input'
 for j in range(4):
  yy=215+j*300
  op(p,f'down{j}',65,yy,440,'Conv4×4,s4 thenLayerNorm' if j==0 else 'LayerNorm thenConv2×2,s2',f'{3 if j==0 else dims[j-1]} to{dims[j]};grid{128//2**j}²','conv');p.connect(prev,f'down{j}')
  op(p,f'block{j}',65,yy+95,440,f'ConvNeXt block,n={depths[j]}',f'{dims[j]} channels','bottleneck',block='convnext');p.connect(f'down{j}',f'block{j}')
  op(p,f'tap{j}',65,yy+190,440,f'Output LayerNorm: E{j+1}',f'{dims[j]} × {128//2**j}²','norm');p.connect(f'block{j}',f'tap{j}');prev=f'block{j}'
  if j<3:
   # Next stage receives pre-output-normalization state, via right corridor.
   pass
 # Fix ongoing state routes around output-only norms.
 for e in list(p.wires):
  if e.get('data-from','').startswith('block') and e.get('data-to','').startswith('down'):
   p.wires.remove(e)
 for j in range(3):p.connect(f'block{j}',f'down{j+1}',from_port='right',to_port='right',via=[(540,334.5+j*300),(540,539.5+j*300)])
 p.text(20,1510,'A pooled classification feature is computed but ignored by DDColor.',13)
 q=d.panel('unet','Wide UNet and pixel features',650,230,580,1570)
 for j,(ci,skip,co,hw) in enumerate([(dims[3],dims[2],512,32),(512,dims[1],512,64),(512,dims[0],256,128)]):
  yy=65+j*355;op(q,f'unet{j}',65,yy,450,'UnetBlockWide',f'Input{ci};skip{skip};output{co} × {hw}²','conv',block='wide')
  if j:q.connect(f'unet{j-1}',f'unet{j}')
  q.text(85,yy+104,'From '+('E4 + E3' if j==0 else f'previous + E{3-j}'),14)
 op(q,'lastshuffle',65,1180,450,'PixelShuffleICNR ×4','256to4096conv;output256×512²;blur','aggregate',block='shuffle');q.connect('unet2','lastshuffle')
 q.text(20,1460,'Color transformer memories:512×32²,512×64²,256×128².',13)
 r=d.panel('queries','Color queries and chroma output',1270,230,590,1570)
 rows=[('init','Learned query content andposition','100 × 256 each;3 learned level embeddings','linear'),('layers','Color decoder layer,n=9','Memory levels cycle0,1,2 three times','attention'),('ln','Final LayerNorm','100 × 256','norm'),('mlp','Color embedding MLP','256to256to256to256;ReLU afterfirsttwo','linear'),('einsum','Dot queries with pixel features','[100,256] × [256,512,512] =100 color maps','attention'),('concat','Concat color maps and normalized input','100+3=103 channels at512²','concat'),('refine','Spectral-normalized Conv1×1','103to2;biasTrue;noactivation','conv2d'),('out','Lab a,b prediction','2 × 512 × 512;no sigmoid oroutputdenormalization','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(r,id,65,65+j*175,460,label,detail,kind,block='colorlayer' if id=='layers' else 'colormlp' if id=='mlp' else '')
 chain(r,[s[0] for s in rows]);r.text(20,1510,'Postprocessing combines resized a,b with the original LabL.',13)
 c=d.panel('convnext','ConvNeXt block',40,1860,570,1540,kind='bottleneck',dashed=True,block_type='convnext')
 ffs=[4*x if isinstance(x,int) else '4'+x for x in dims]
 rows=[('x','Input feature','Stage widths '+str(dims),'plain'),('dw','DepthwiseConv7×7,padding3','Groups equalstagewidth;biasTrue','conv2d'),('ln','Channels-last LayerNorm','epsilon1e-6','norm'),('fc1','Linear expansion','Outputwidths '+str(ffs),'linear'),('gelu','GELU','','activation'),('fc2','Linear projection','Restore stagewidth','linear'),('scale','LayerScale','Learned channelvector,init1e-6','linear')]
 for j,(id,label,detail,kind) in enumerate(rows):op(c,'c'+id,75,65+j*175,420,label,detail,kind)
 chain(c,['c'+s[0] for s in rows]);c.sum('cadd',285,1400);c.connect('cscale','cadd');c.connect('cx','cadd',from_port='left',to_port='left',via=[(25,89.5),(25,1400)])
 w=d.panel('wide','UnetBlockWide and shuffle',650,1860,580,1540,kind='conv',dashed=True,block_type='wide')
 rows=[('proj','Spectral-normalized Conv1×1 +BN','Ci to4Co;Co512,512,256','conv'),('relu','ReLU thenPixelShuffle2','Co channels atdoubledgrid','aggregate'),('blur','ReplicationPad(left1,top1);AvgPool2,s1','Blur preserves expandedsize','pool'),('concat','Concat with BatchNorm(skip),thenReLU',f'Concat widths512+{dims[2]},512+{dims[1]},256+{dims[0]}','concat'),('conv','Spectral-normalized Conv3×3','Concat toCo;s1,p1;biasTrue','conv2d'),('act','ReLU thenBatchNorm','Output512,512,256channels','norm')]
 for j,(id,label,detail,kind) in enumerate(rows):op(w,'w'+id,75,65+j*210,430,label,detail,kind)
 chain(w,['w'+s[0] for s in rows]);w.text(20,1445,'Final×4shuffle uses256to4096Conv1×1,ReLU,PixelShuffle4,blur;',13);w.text(20,1480,'it has spectral normalization and no extraBatchNorm.',13)
 z=d.panel('colorlayer','One color decoder layer',1270,1860,590,1540,kind='attention',dashed=True,block_type='colorlayer')
 rows=[('x','Input query state','100 × 256','plain'),('ca','Cross-attention,8 heads','Q=query+querypos;K=memory+sinepos;V=memory','attention'),('add1','Residual add thenLayerNorm','100 × 256','norm'),('sa','Self-attention,8 heads','Q,K=query+querypos;V=query','attention'),('add2','Residual add thenLayerNorm','100 × 256','norm'),('f1','Linear FFN','256to2048','linear'),('relu','ReLU','','activation'),('f2','Linear FFN','2048to256','linear'),('add3','Residual add thenLayerNorm','100 × 256','norm')]
 previous=None
 for j,(id,label,detail,kind) in enumerate(rows):
  yy=65+j*145
  if id.startswith('add'):
   z.sum('sum'+id,295,yy);op(z,'d'+id,75,yy+45,440,'LayerNorm','256 channels','norm')
   if previous:z.connect(previous,'sum'+id)
   z.connect('sum'+id,'d'+id)
   source={'add1':'dx','add2':'dadd1','add3':'dadd2'}[id];sy=z.port(source,'left')[1]
   z.connect(source,'sum'+id,from_port='left',to_port='left',via=[(25,sy),(25,yy)])
  else:
   op(z,'d'+id,75,yy,440,label,detail,kind)
   if previous:z.connect(previous,'d'+id)
  previous='d'+id
 z.text(20,1470,'Cross-attention executes before self-attention; dropout0,post-norm.',13)
 aP=d.panel('primitives','Attention, position and color-MLP primitives',40,3460,1820,730,kind='attention',dashed=True)
 for j,(id,label,detail,kind) in enumerate([('qkv','Separate Q/K/V Linear256to256','8 heads,32channels/head','linear'),('qk','QK transpose /sqrt32','Cross-memory lengths1024,4096,16384;selflength100','attention'),('soft','Softmax overkeys','','attention'),('av','Weights × V;concatheads;Linear256to256','','attention')]):op(aP,id,25,65+j*140,850,label,detail,kind)
 chain(aP,['qkv','qk','soft','av'])
 aP.text(960,95,'Memory Conv1×1 projections:512to256,512to256,256to256.',14)
 aP.text(960,145,'Each projected memory adds its learned256-channel levelvector.',14)
 aP.text(960,195,'2D normalized sine/cosine positions provide256 channels.',14)
 for j,(id,label,kind) in enumerate([('cml1','Linear256to256','linear'),('cmla1','ReLU','activation'),('cml2','Linear256to256','linear'),('cmla2','ReLU','activation'),('cml3','Linear256to256','linear')]):op(aP,id,990,290+j*68,780,label,kind=kind,h=42)
 chain(aP,['cml1','cmla1','cml2','cmla2','cml3'])
 if family:d.text(50,4260,'Tiny:C=[96,192,384,768],N3=9. Large:C=[192,384,768,1536],N3=27. All decoder dimensions and100-query settings are shared.',14)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'ddcolor','family' if family else size+'-restore','Shared T/L topology' if family else size,'restore',size,'family' if family else'concrete','3×512×512 grayscaleRGB',rec.get('device','source'))

def main():
 a=environment('Build both DDColor registered colorization sizes')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'ddcolor');records={}
  for size,enc in [('t','convnext-t'),('l','convnext-l')]:
   factory=lambda:nn.DDColor(encoder_name=enc,input_size=(512,512),num_output_channels=2,last_norm='Spectral',do_normalize=False,num_queries=100,num_scales=3,dec_layers=9)
   names=['encoder','decoder.layers.0','decoder.layers.1','decoder.layers.2','decoder.last_shuf','decoder.color_decoder','refine_net']
   if size=='t':records[size]=cpu_probe(factory(),(1,3,512,512),names,input_range=(0,1))
   else:records[size]=meta_probe(construct_meta(factory),(1,3,512,512),names)
  write_evidence(a,'ddcolor',records,['libreyolo/models/ddcolor/model.py:DDCOLOR_SIZE_CONFIGS,_init_model','libreyolo/models/ddcolor/nn.py:DDColor,DuelDecoder,MultiScaleColorDecoder','libreyolo/models/ddcolor/unet.py:UnetBlockWide,CustomPixelShuffleICNR','libreyolo/models/ddcolor/convnext.py:ConvNeXt','libreyolo/models/ddcolor/transformer.py:post-normlayers'],['Tiny CPU512,Large meta512.','Actual registeredmodeluses512input,100queries,2Labchromaoutputs,Spectralnorm,not constructor defaults256input/256queries/3outputs.','Memoryscale order0,1,2 repeated3 times; perlayercross-attention precedes self-attention.'])
 ev=read_evidence('ddcolor');views=[build(a,s,ev) for s in ['t','l','family']];manifest(a,'ddcolor','ddcolor','DDColor',views)
if __name__=='__main__':main()
