from sam import *
SPECS={'tiny':dict(widths=[96,192,384,768],depths=[1,2,7,2],heads=[1,2,4,8],windows=[8,4,14,7],global_blocks=[5,7,9],background=7),'small':dict(widths=[96,192,384,768],depths=[1,2,11,2],heads=[1,2,4,8],windows=[8,4,14,7],global_blocks=[7,10,13],background=7),'base-plus':dict(widths=[112,224,448,896],depths=[2,3,16,3],heads=[2,4,8,16],windows=[8,4,14,7],global_blocks=[12,16,20],background=14),'large':dict(widths=[144,288,576,1152],depths=[2,6,36,4],heads=[2,4,8,16],windows=[8,4,16,8],global_blocks=[23,33,43],background=7)}
CONFIG_SOURCES={'tiny':('LibreYOLO/LibreSAM2tiny','c30b06ef250f14fe1ac8712bdddc9242e9bcc4b4'),'small':('LibreYOLO/LibreSAM2small','6c26f791263446531a8cc89f1d73d4d57f9ff51b'),'base-plus':('LibreYOLO/LibreSAM2base-plus','44a0f4a23a9e099b0d8a24b47c9599f94c87e226'),'large':('facebook/sam2.1-hiera-large','665f8e2ad61cf5f53d65644ff27c8ee525124610')}

def v2_outputs(d,x,y):
 p=decoder_outputs(d,x,y,extra_object=True,high_res=True)
 # Widen the panel for the two independently projected high-resolution paths.
 p.root.find('{http://www.w3.org/2000/svg}rect').set('width','1050')
 for start,end,cy,label,detail,yy in [('up1','up2',355,'High-res feature s1','64 × 128 × 128',280),('up4','up5',655,'High-res feature s0','32 × 256 × 256',580)]:
  for wire in list(p.wires):
   if wire.get('data-from')==start and wire.get('data-to')==end:p.wires.remove(wire)
  p.sum('hiadd'+start,570,cy);p.connect(start,'hiadd'+start);p.connect('hiadd'+start,end)
  p.box('hi'+start,765,yy,250,label,detail=detail,font_size=14);p.connect('hi'+start,'hiadd'+start,via=[(890,cy)],to_port='right')
 for group in list(p.ops):
  if group.get('id','').startswith('quality'):p.ops.remove(group)
 for wire in list(p.wires):
  if wire.get('data-from','').startswith('quality') or wire.get('data-to','').startswith('quality'):p.wires.remove(wire)
 # Delete registry entries so the replacement nodes remain unique.
 for id in list(d.nodes):
  if id.startswith('quality'):del d.nodes[id]
 chain(p,'quality',[('Select IoU token 1','256','plain'),('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 4','linear'),('Sigmoid','4 mask-quality scores','activation'),('Select scores 1 to 3','3 quality scores','plain')],x=415,w=310,y=970,gap=85)
 chain(p,'object',[('Select object token 0','256','plain'),('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 1 object-score logit','linear')],x=765,w=250,y=970,gap=105)
 return p

def heads(d):
 prompt_panel(d,660,220);decoder_setup(d,1300,220,tokens=8,extra_object=True);v2_outputs(d,1960,220)
 two_way(d,'block1',25,2010,1,8);two_way(d,'block2',1360,2010,2,8)
 attention(d,'self','Token self-attention',25,3840,256,8,8,32,K=8)
 attention(d,'toimage','Token-to-image attention',700,3840,256,8,8,16,K=4096,inner_dim=128)
 attention(d,'totoken','Image-to-token attention',1375,3840,256,8,4096,16,K=8,inner_dim=128)
 p=d.panel('decoder-notes','Image-mode decoder',2050,3840,960,1270)
 p.text(25,90,'Q0: original eight output/prompt tokens; P: Fourier image position.',17)
 p.text(25,140,'Final cross-attention uses the token-to-image equation, with residual and LayerNorm.',17)
 p.text(25,210,'Features s0 and s1 are computed once, then added at the two upscaling steps.',17)
 p.text(25,280,'The no-memory embedding is added to the 64×64 image feature.',17)
 p.text(25,350,'This is image inference. Video memory attention and memory encoder do not run.',17)
 p.text(25,420,'multimask_output=True selects mask/IoU slots 1, 2 and 3.',17)
 p.text(25,490,'Single-mask eval can fall back to the best multimask using stability thresholds.',17)

def fpn(d,x,y,widths):
 p=d.panel('fpn','Image feature pyramid',x,y,1500,1010,kind='aggregate',dashed=True)
 for i,(c,h) in enumerate(zip(widths,[256,128,64,32])):
  xx=25+i*370
  ids=chain(p,'level'+str(i),[(f'Stage {i+1} map',f'{c} × {h} × {h}','plain'),('Lateral Conv2d 1×1',f'{c} to 256, bias=True','conv2d')],x=xx,w=320,y=65,gap=100)
  if i==2:
   p.sum('fpnsum',xx+160,410);p.connect(ids[-1],'fpnsum');prev='fpnsum'
  else:prev=ids[-1]
  if i<2:
   p.box('project'+str(i),xx,560,320,'High-res Conv2d 1×1',detail=f'256 to {32 if i==0 else 64}',kind='conv2d');p.connect(prev,'project'+str(i));p.text(xx,660,f's{i}: {32 if i==0 else 64} × {h} × {h}',16)
  if i==2:
   p.box('nomem',xx,560,320,'Add no-memory embedding',detail='256 × 64 × 64');p.connect(prev,'nomem');p.text(xx,660,'Main image embedding: 256 × 64 × 64',15)
 p.box('upsample',1135,340,320,'Nearest upsample ×2',detail='256 × 64 × 64',kind='pool');p.connect('level31','upsample');p.connect('upsample','fpnsum',from_port='left',to_port='right',via=[(1100,364.5),(1100,410)])
 p.text(25,800,'Only the 32×32 to 64×64 top-down addition runs for fpn_top_down_levels=[2,3].',17)
 p.text(25,845,'The 128×128 and 256×256 lateral outputs remain independent. The 32×32 output is not passed to the mask decoder.',17)
 return p

def draw(b,size,spec,title='SAM 2',encoder_title='Hiera encoder'):
 sym=size=='family';Cs=[f'C{i+1}' for i in range(4)] if sym else spec['widths'];ns=[f'n{i+1}' for i in range(4)] if sym else spec['depths'];hs=[f'h{i+1}' for i in range(4)] if sym else spec['heads'];wins=[f'w{i+1}' for i in range(4)] if sym else spec['windows']
 d=b.diagram(title+' '+size,'Image segmentation, 1024 × 1024 input, one box, three mask outputs. Eight decoder tokens include an object-score token.',3040,8650)
 p=d.panel('image',encoder_title,25,220,605,1720)
 items=[('RGB input after resize/normalize','3 × 1,024 × 1,024','plain'),('Conv2d 7×7 / 4',f'256 × 256 × {Cs[0]}, p=3','conv2d'),('Add image position','Interpolated background + tiled window term','plain')]
 for i in range(4):items.append((f'Stage {i+1} Hiera blocks',f'{256//2**i} × {256//2**i} × {Cs[i]}, n={ns[i]}','attention'))
 items +=[('Feature pyramid','Lateral 1×1 projections to 256 channels','aggregate'),('Main image embedding','256 × 64 × 64, plus no-memory embedding','plain')]
 chain(p,'im',items,w=490,gap=120)
 p.text(25,1270,'Stages 2, 3, 4 pool queries with stride 2.',16);p.text(25,1320,'Global blocks (0-based): '+('g from table' if sym else ', '.join(map(str,spec['global_blocks']))),16)
 p.text(25,1370,'Other blocks use stage-specific local windows.',16)
 heads(d)
 # Stage-specific primitive definitions with resolved rows alongside the shared entry-block rule.
 for i in range(4):
  c=Cs[i];ci=Cs[i-1] if i else Cs[0];hi=256//2**i;mid=f'4{c}' if sym else c*4
  p=d.panel('hiera'+str(i),f'Hiera stage {i+1} block',25+i*760,5200,730,2000,kind='attention',dashed=True)
  p.text(25,65,f'Output spatial size {hi} × {hi}, channels {c}',17)
  p.text(25,105,'Entry block pools 2×2 Q and shortcut.' if i else 'Stage 1 has no query pooling.',16)
  p.dot(220,150)
  ops=[('LayerNorm',f'Entry width {ci}; remaining blocks {c}','norm'),('Partition local windows','Entry window '+str(wins[i-1] if i else wins[i])+'; others '+str(wins[i]),'plain'),('Linear QKV',f'Entry {ci} to '+(f'3{c}' if sym else str(c*3)),'linear'),('Reshape Q, K and V',f'{hs[i]} heads; '+('C/h' if sym else str(c//hs[i]))+' channels per head','split'),('MaxPool2d on Q only','2×2 / 2 in entry block' if i else 'Identity in this stage','pool'),('MatMul Q K-transpose','Q pooled at entry; K/V keep input tokens','attention'),('Scale attention scores','Divide by sqrt('+('C/h' if sym else str(c//hs[i]))+')','attention'),('Softmax over keys','Local windows or full spatial token grid','activation'),('MatMul probabilities × V',f'{c} total output channels','attention'),('Linear output',f'{c} to {c}','linear'),('Reverse windows and crop',f'{hi} × {hi} × {c}','plain')]
  ids=chain(p,'h'+str(i),ops,x=25,w=390,y=200,gap=85);p.wire([(220,150),(220,200)],end=ids[0])
  if i:
   p.box('shortcutproj'+str(i),470,250,235,'Linear shortcut',detail=f'{ci} to {c}',kind='linear',font_size=14)
   p.box('shortcutpool'+str(i),470,430,235,'MaxPool2d 2×2 / 2',detail=f'{hi} × {hi} × {c}',kind='pool',font_size=13)
   p.connect(ids[0],'shortcutproj'+str(i),from_port='right',to_port='top',via=[(440,224.5),(440,225),(587.5,225)])
   p.connect('shortcutproj'+str(i),'shortcutpool'+str(i))
  p.sum('ha'+str(i),220,1160);p.connect(ids[-1],'ha'+str(i))
  if i:
   p.box('shortcutchoice'+str(i),470,1040,235,'Choose shortcut',detail='Projected entry / identity rest',font_size=13);p.connect('shortcutpool'+str(i),'shortcutchoice'+str(i));p.wire([(220,150),(720,150),(720,1064.5),(705,1064.5)],end='shortcutchoice'+str(i));p.connect('shortcutchoice'+str(i),'ha'+str(i),via=[(587.5,1160)],to_port='right')
  else:p.wire([(220,150),(680,150),(680,1160),(233,1160)],end='ha'+str(i))
  p.dot(220,1230);p.wire([(220,1173),(220,1230)],start='ha'+str(i),arrow=False)
  ids=chain(p,'hm'+str(i),[('LayerNorm',f'{hi} × {hi} × {c}','norm'),('Linear',f'{c} to {mid}','linear'),('GELU',f'{hi} × {hi} × {mid}','activation'),('Linear',f'{mid} to {c}','linear')],x=25,w=390,y=1270,gap=110);p.wire([(220,1230),(220,1270)],end=ids[0]);p.sum('hmadd'+str(i),220,1810);p.connect(ids[-1],'hmadd'+str(i));p.wire([(220,1230),(680,1230),(680,1810),(233,1810)],end='hmadd'+str(i))
  p.text(25,1880,'Non-entry blocks use the raw-input identity shortcut.',16)
  p.text(25,1920,'Global blocks bypass window partition/unpartition.',16)
 fpn(d,25,7260,Cs)
 p=d.panel('variants','Variant values',1560,7260,1450,1010)
 for j,(key,sp) in enumerate(SPECS.items()):
  yy=75+j*205;p.text(25,yy,key,19,weight=700);p.text(25,yy+42,'C1..C4: '+', '.join(map(str,sp['widths']))+'; n1..n4: '+', '.join(map(str,sp['depths'])),17);p.text(25,yy+82,'h1..h4: '+', '.join(map(str,sp['heads']))+'; w1..w4: '+', '.join(map(str,sp['windows'])),17);p.text(25,yy+122,'Global block indices (0-based): '+', '.join(map(str,sp['global_blocks'])),17)
 p.text(25,930,'The decoder and three output feature shapes are fixed across all four sizes.',17)
 b.save(d,size+'-box-segment',size,kind='family' if sym else 'concrete',verification='source' if sym else 'meta',task='segment',input='Image: 1 × 3 × 1024 × 1024; box: 1 × 1 × 4')

def main():
 a=setup();b=Book(a,'sam2','SAM 2',sourcefile='libreyolo/models/sam/sam2.py');import torch,transformers
 from transformers import Sam2Config,Sam2Model
 from transformers.models.sam2.configuration_sam2 import Sam2HieraDetConfig,Sam2VisionConfig
 b.evidence['configuration_sources']=CONFIG_SOURCES;b.evidence['backend']={'package':'transformers','version':transformers.__version__,'license':'Apache-2.0','source':'transformers/models/sam2/modeling_sam2.py'}
 for size,sp in SPECS.items():
  bc=Sam2HieraDetConfig(hidden_size=sp['widths'][0],embed_dim_per_stage=sp['widths'],blocks_per_stage=sp['depths'],num_attention_heads_per_stage=sp['heads'],global_attention_blocks=sp['global_blocks'],window_size_per_stage=sp['windows'],window_positional_embedding_background_size=[sp['background']]*2)
  vc=Sam2VisionConfig(backbone_config=bc,backbone_channel_list=sp['widths'][::-1])
  with torch.device('meta'):model=Sam2Model(Sam2Config(vision_config=vc)).eval()
  with torch.inference_mode():out=model(pixel_values=torch.zeros(1,3,1024,1024,device='meta'),input_boxes=torch.zeros(1,1,4,device='meta'),multimask_output=True)
  b.evidence[size]={'device':'meta','backbone_config':bc.to_dict(),'image_embeddings':[list(t.shape) for t in out.image_embeddings],'pred_masks':list(out.pred_masks.shape),'iou_scores':list(out.iou_scores.shape),'object_score_logits':list(out.object_score_logits.shape)};draw(b,size,sp)
 draw(b,'family',SPECS['tiny']);b.finish()
if __name__=='__main__':main()
