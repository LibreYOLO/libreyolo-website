from sam2 import *
from pe import rope_definition

def vision(d,y):
 p=d.panel('vit','SAM 3 vision encoder',25,y,640,1590)
 items=[('Image','3 × 1,008 × 1,008','plain'),('Conv2d 14×14 / 14','5,184 × 1,024 patch tokens','conv2d'),('Tile learned position embedding','24×24 pretrain grid to 72×72 grid','plain'),('Add patch position','5,184 × 1,024; no CLS token','plain'),('ViT blocks, repeats=32','1,024 width, 16 heads, MLP 4,736','attention'),('Reshape final features','1,024 × 72 × 72','plain')];chain(p,'visionflow',items,w=540,gap=140)
 p.text(25,1090,'Global blocks (1-based): 8, 16, 24, 32.',17);p.text(25,1140,'All other blocks use nine 24×24 windows.',17);p.text(25,1190,'Head width 64; axial RoPE on Q and K.',17)
 p=d.panel('vitblock','Vision ViT block',705,y,620,1590,kind='attention',dashed=True)
 p.dot(230,90);items=[('LayerNorm','72 × 72 × 1,024','norm'),('Window partition when local','Nine 24×24 windows, no padding','plain'),('Rotary self-attention','576 local / 5,184 global tokens','attention'),('Reverse windows','72 × 72 × 1,024','plain')];ids=chain(p,'visionb1',items,w=400,y=135,gap=105);p.wire([(230,90),(230,135)],end=ids[0]);p.sum('visions1',230,635);p.connect(ids[-1],'visions1');p.wire([(230,90),(570,90),(570,635),(243,635)],end='visions1')
 p.dot(230,695);p.wire([(230,648),(230,695)],start='visions1',arrow=False)
 ids=chain(p,'visionb2',[('LayerNorm','1,024 channels','norm'),('Linear','1,024 to 4,736','linear'),('GELU','4,736 channels','activation'),('Linear','4,736 to 1,024','linear')],w=400,y=740,gap=115);p.wire([(230,695),(230,740)],end=ids[0]);p.sum('visions2',230,1260);p.connect(ids[-1],'visions2');p.wire([(230,695),(570,695),(570,1260),(243,1260)],end='visions2')
 attention(d,'vlocal','Local vision self-attention',1360,y,1024,16,576,64,rope='2D axial; scale 1',height=1590)
 attention(d,'vglobal','Global vision self-attention',2040,y,1024,16,5184,64,rope='2D axial; scale 1/3',height=1590)

def vision_fpn(d,y):
 p=d.panel('fpn','Vision feature pyramid',25,y,2985,940,kind='aggregate',dashed=True)
 specs=[(4,288,[('ConvTranspose2d 2×2 / 2','1,024 to 512, 144×144','conv2d'),('GELU','512 × 144 × 144','activation'),('ConvTranspose2d 2×2 / 2','512 to 256, 288×288','conv2d')],256),(2,144,[('ConvTranspose2d 2×2 / 2','1,024 to 512, 144×144','conv2d')],512),(1,72,[],1024),(.5,36,[('MaxPool2d 2×2 / 2','1,024 × 36 × 36','pool')],1024)]
 for i,(scale,h,ops,cin) in enumerate(specs):
  items=[('Shared final ViT feature','1,024 × 72 × 72','plain')]+ops+[('Conv2d 1×1',f'{cin} to 256','conv2d'),('Conv2d 3×3 / 1',f'256 to 256, p=1','conv2d')]
  ids=chain(p,'fpn'+str(i),items,x=25+i*745,w=670,y=65,gap=100)
  xx,yy=p.port(ids[-1]);p.text(xx,yy+45,f'256 × {h} × {h}'+(' (not used by image heads)' if i==3 else ''),17,anchor='middle')
 p.text(25,875,'These branches are independent resamplings, not top-down additions. Three finest levels enter the image heads.',17)

def draw_tracker(b):
 d=b.diagram('SAM 3 large visual prompts (default config)','Transformers 5.16.1 default architecture, 1008 × 1008 input, one box. Gated checkpoint configuration was not accessible.',3040,8350)
 p=d.panel('image','Image-mode visual path',25,220,605,1720)
 chain(p,'main',[('Image encoder','32-block ViT, 1,024 channels','attention'),('Independent FPN resamplings','256-channel maps at 288,144,72,36','aggregate'),('Select three finest feature levels','288×288, 144×144, 72×72','plain'),('Project high-resolution features','32 × 288 × 288 and 64 × 144 × 144','conv2d'),('Add no-memory embedding','256 × 72 × 72 main image feature','plain')],w=490,gap=180)
 p.text(25,1230,'No video memory runs in this adapter.',17)
 heads(d)
 # The verified SAM3 tracker decoder shares the SAM2 topology at a 72x72 image grid.
 for e in d.root.iter():
  if e.tag.endswith('text') and e.text:
   e.text=e.text.replace('4,096','5,184').replace('65,536','82,944').replace('64 × 64','72 × 72').replace('128 × 128','144 × 144').replace('256 × 256','288 × 288').replace('64×64','72×72').replace('1024','1008')
 vision(d,5200);vision_fpn(d,6850)
 d.text(35,7960,'Default-config scope: architecture verified from Apache-2.0 Transformers classes; no gated checkpoint or license acceptance was used.',17)
 b.save(d,'large-visual-default','large',task='segment',verification='meta',input='Image: 1 × 3 × 1008 × 1008; box: 1 × 1 × 4')

def fusion_block(d,id,title,x,y,decoder=False):
 p=d.panel(id,title,x,y,650,1840,kind='attention',dashed=True)
 N=201 if decoder else 5184;prev=None;yy=75
 for j,(name,detail) in enumerate([('Self-attention',f'Q/K/V={N}, with position on Q/K'),('Prompt cross-attention',f'Q={N}, K/V=32 text tokens')]+([('Vision cross-attention','Q=201, K/V=5,184, box-relative bias')] if decoder else [])):
  p.dot(230,yy);begin=yy
  if not decoder:
   p.box(id+'norm'+str(j),30,yy+35,400,'LayerNorm',detail=f'{N} × 256',kind='norm');start=id+'norm'+str(j)
   if prev:p.connect(prev,start)
   else:p.wire([(230,yy),(230,yy+35)],end=start)
   ay=yy+120
  else:ay=yy+45;start=None
  p.box(id+'attn'+str(j),30,ay,400,name,detail=detail,kind='attention',font_size=15)
  if start:p.connect(start,id+'attn'+str(j))
  elif prev:p.connect(prev,id+'attn'+str(j))
  else:p.wire([(230,yy),(230,ay)],end=id+'attn'+str(j))
  p.sum(id+'sum'+str(j),230,ay+120);p.connect(id+'attn'+str(j),id+'sum'+str(j));p.wire([(230,begin),(585,begin),(585,ay+120),(243,ay+120)],end=id+'sum'+str(j));prev=id+'sum'+str(j)
  if decoder:p.box(id+'norm'+str(j),30,ay+180,400,'LayerNorm',detail=f'{N} × 256',kind='norm');p.connect(prev,id+'norm'+str(j));prev=id+'norm'+str(j)
  yy=ay+285
 p.dot(230,yy);p.wire([p.port(prev),(230,yy)],start=prev,arrow=False)
 ops=[] if decoder else [('LayerNorm',f'{N} × 256','norm')]
 ops +=[('Linear','256 to 2,048','linear'),('ReLU',f'{N} × 2,048','activation'),('Linear','2,048 to 256','linear')]
 ids=chain(p,id+'ff',ops,w=400,y=yy+45,gap=105);p.wire([(230,yy),(230,yy+45)],end=ids[0]);sy=yy+45+(len(ops)-1)*105+120;p.sum(id+'ffsum',230,sy);p.connect(ids[-1],id+'ffsum');p.wire([(230,yy),(585,yy),(585,sy),(243,sy)],end=id+'ffsum')
 if decoder:p.box(id+'ffnorm',30,sy+75,400,'LayerNorm',detail='201 × 256',kind='norm');p.connect(id+'ffsum',id+'ffnorm')
 return p

def draw_text(b):
 d=b.diagram('SAM 3 large concept text (default config)','Transformers 5.16.1 default architecture, 1008 × 1008 input, one 32-token concept prompt, no geometry prompts.',3040,10580)
 p=d.panel('flow','Concept segmentation',25,220,650,1720)
 ops=[('Vision encoder and FPN','Finest maps: 288×288,144×144,72×72','attention'),('Flatten coarsest retained map','5,184 × 256; sine image position','plain'),('Multimodal encoder, repeats=6','Image state attends to 32 prompt tokens','attention'),('DETR decoder, repeats=6','200 query tokens + 1 presence token','attention'),('Final query states','200 × 256; presence state 1 × 256','plain'),('Box, score and mask heads','200 candidates','aggregate')];chain(p,'flow',ops,w=550,gap=190)
 p=d.panel('text','Concept text tower',705,220,650,1720)
 items=[('Input token IDs','32 IDs, vocabulary 49,408','plain'),('Token embedding','32 × 1,024','linear'),('Embedded sequence','32 × 1,024','plain'),('Causal transformer blocks','24 blocks, 16 heads, MLP 4,096','attention'),('Final LayerNorm','32 × 1,024','norm'),('Linear token projection','1,024 to 256, preserve 32 tokens','linear')];tower(p,'text',items,'32 × 1,024')
 p.text(30,1240,'No EOT pooling is used by concept detection.',17)
 p.text(30,1290,'Geometry encoder is not called for text-only input.',17)
 p=d.panel('queries','Queries, reference boxes and outputs',1390,220,1620,1720)
 p.box('learnedqueries',25,65,550,'Learned query tokens',detail='200 × 256')
 p.box('presence',650,65,300,'Learned presence token',detail='1 × 256')
 p.box('queryconcat',25,250,550,'Prepend presence token',detail='201 × 256',kind='concat');p.connect('learnedqueries','queryconcat');p.connect('presence','queryconcat',via=[(800,274.5)],to_port='right')
 chain(p,'refs',[('Learned reference box logits','200 × 4','plain'),('Sigmoid','200 normalized cxcywh boxes','activation'),('Sine encode box coordinates','200 × 512','plain'),('Reference-position MLP','512 to 256 to 256','linear')],x=1010,w=570,y=65,gap=140)
 p.text(25,460,'The query and reference-box streams are independent.',17)
 p.text(25,510,'Every decoder layer receives both streams.',17)
 p.box('currentq',25,650,550,'Normalized object query states',detail='200 × 256')
 p.box('boxmlp',25,810,550,'Box MLP (3 layers)',detail='256 to 256 to 256 to 4',kind='linear');p.connect('currentq','boxmlp')
 p.box('currentrefs',1010,650,570,'Current reference boxes',detail='200 × 4')
 p.box('inverse',1010,810,570,'Inverse sigmoid',detail='200 × 4');p.connect('currentrefs','inverse')
 p.sum('boxadd',800,995);p.connect('boxmlp','boxadd',via=[(300,995)],to_port='left');p.connect('inverse','boxadd',via=[(1295,995)],to_port='right')
 p.box('refined',515,1100,570,'Sigmoid refined boxes',detail='200 × 4; next layer reference');p.connect('boxadd','refined')
 p.box('presence-norm',25,1250,550,'LayerNorm presence state',detail='1 × 256',kind='norm')
 p.box('presence-mlp',25,1380,550,'Presence MLP (3 layers)',detail='256 to 256 to 256 to 1',kind='linear');p.connect('presence-norm','presence-mlp')
 p.box('presence-clamp',25,1510,550,'Clamp presence logit',detail='[-10,10]');p.connect('presence-mlp','presence-clamp')
 p.text(1010,1310,'Final outputs:',18,weight=700);p.text(1010,1370,'200 boxes and matching logits;',17);p.text(1010,1420,'200 masks at 288×288;',17);p.text(1010,1470,'one presence logit;',17);p.text(1010,1520,'one semantic map at 288×288.',17)
 fusion_block(d,'encoder-block','Multimodal encoder block',25,2010)
 fusion_block(d,'decoder-block','DETR decoder block',705,2010,decoder=True)
 attention(d,'encself','Encoder image self-attention',1390,2010,256,8,5184,32,height=1840)
 attention(d,'encprompt','Image-to-prompt attention',2070,2010,256,8,5184,32,K=32,height=1840)
 attention(d,'decself','Decoder query self-attention',25,3920,256,8,201,32,K=201)
 attention(d,'dectext','Query-to-prompt attention',705,3920,256,8,201,32,K=32)
 attention(d,'decvision','Query-to-image attention',1390,3920,256,8,201,32,K=5184,relative=True)
 p=d.panel('rpb','Box relative position bias',2070,3920,940,1270,kind='attention',dashed=True)
 chain(p,'rpb',[('Reference boxes cxcywh','Convert to XYXY, 200 × 4','plain'),('Subtract box boundaries from grid','72 x-coordinates and 72 y-coordinates','plain'),('Signed logarithmic encoding','sign(8d) log2(abs(8d)+1) / log2(8)','plain'),('Separate X and Y MLPs','Each Linear 2 to256, ReLU, Linear to8','linear'),('Broadcast and add axis biases','8 × 200 × 72 × 72','plain'),('Flatten and prepend presence row','8 × 201 × 5,184; presence row zero','plain')],w=820,gap=130)
 p.text(25,995,'The resulting tensor is the vision-attention bias.',17)
 vision(d,5260);vision_fpn(d,6910)
 p=d.panel('pixel','Mask pixel decoder',25,7910,960,1960,kind='pool')
 p.dot(310,60)
 ids=chain(p,'pixelpre',[('Encoder image state','5,184 × 256','plain'),('LayerNorm','5,184 × 256','norm'),('Prompt cross-attention','Q=5,184; K/V=32, 8 heads ×32','attention')],w=560,y=90,gap=95)
 p.sum('pixel-residual',310,435);p.connect(ids[-1],'pixel-residual');p.connect(ids[0],'pixel-residual',from_port='left',to_port='left',via=[(10,114.5),(10,435)])
 p.box('pixel-reshape',30,515,560,'Reshape image state',detail='256 × 72 × 72');p.connect('pixel-residual','pixel-reshape');prev='pixel-reshape'
 for j,h in enumerate([144,288]):
  yy=655+j*580
  p.box('pixel-resize'+str(j),30,yy,560,'Nearest resize ×2',detail=f'256 × {h} × {h}',kind='pool');p.connect(prev,'pixel-resize'+str(j))
  p.box('pixel-skip'+str(j),660,yy,265,'FPN skip',detail=f'256 × {h} × {h}')
  p.sum('pixel-sum'+str(j),310,yy+110);p.connect('pixel-resize'+str(j),'pixel-sum'+str(j));p.connect('pixel-skip'+str(j),'pixel-sum'+str(j),via=[(792.5,yy+110)],to_port='right')
  ids=chain(p,'pixel-smooth'+str(j),[('Conv2d 3×3 / 1','256 to 256, p=1','conv2d'),('GroupNorm','8 groups, 256 channels','norm'),('ReLU',f'256 × {h} × {h}','activation')],w=560,y=yy+195,gap=95);p.connect('pixel-sum'+str(j),ids[0]);prev=ids[-1]
 p=d.panel('maskhead','Mask and semantic heads',1020,7910,960,1960,kind='pool')
 ids=chain(p,'mask',[('200 decoder queries','200 × 256','plain'),('Linear','256 to 256','linear'),('ReLU','200 × 256','activation'),('Linear','256 to 256','linear'),('ReLU','200 × 256','activation'),('Linear','256 to 256 mask coefficients','linear')],w=530,y=65,gap=120)
 p.box('pixel-feature',635,65,295,'Pixel feature',detail='256 × 288 × 288')
 p.box('instance-proj',635,220,295,'Conv2d 1×1',detail='256 to 256',kind='conv2d');p.connect('pixel-feature','instance-proj')
 p.box('mask-dot',30,940,880,'Einsum mask coefficients × pixel feature',detail='200 × 288 × 288 masks',kind='attention');p.connect(ids[-1],'mask-dot',via=[(295,855),(300,855)]);p.connect('instance-proj','mask-dot',via=[(782.5,850),(800,850)])
 p.box('semantic-proj',635,1180,295,'Conv2d 1×1',detail='256 to 1 semantic map',kind='conv2d');p.connect('pixel-feature','semantic-proj',from_port='right',to_port='right',via=[(945,89.5),(945,1204.5)])
 p.text(30,1330,'Instance and semantic projections have independent weights.',17)
 p=d.panel('scoring','Query-to-concept scoring',2015,7910,995,1960,kind='pool')
 ids=chain(p,'scores-pre',[('Prompt tokens','32 × 256','plain'),('Text scoring MLP','256 to 2,048 to 256','linear')],w=590,y=65,gap=150)
 p.sum('score-residual',325,460);p.connect(ids[-1],'score-residual');p.connect(ids[0],'score-residual',from_port='left',to_port='left',via=[(10,89.5),(10,460)])
 ids=chain(p,'scores',[('LayerNorm','32 × 256','norm'),('Masked mean over prompt tokens','256','pool'),('Linear text projection','256 to 256','linear')],w=590,y=560,gap=140);p.connect('score-residual',ids[0])
 p.box('score-query',675,560,285,'Decoder queries',detail='200 × 256')
 p.box('query-linear',675,720,285,'Linear query projection',detail='256 to 256',kind='linear',font_size=14);p.connect('score-query','query-linear')
 p.box('score-dot',30,1070,930,'Dot projected queries with projected pooled text',detail='200 match logits',kind='attention');p.connect(ids[-1],'score-dot',via=[(325,980),(350,980)]);p.connect('query-linear','score-dot',via=[(817.5,960),(800,960)])
 p.box('score-scale',30,1220,930,'Multiply by 1/16',detail='200 logits');p.connect('score-dot','score-scale')
 p.box('score-clamp',30,1370,930,'Clamp match logits',detail='[-12,12]');p.connect('score-scale','score-clamp')
 p.text(25,1530,'Mask and box selection is performed by the processor.',17)
 # Text block definition and elementary rotary math remain visible in this drawing.
 p=residual(d,'textblock','Text transformer block',25,10020,1024,32,4096,16,height=1270)
 attention(d,'textattn','Text causal self-attention',570,10020,1024,16,32,64,causal=True)
 rope_definition(d,'rope',1250,10020,64,'576 local / 5,184 global',0,0)
 p=d.panel('mlp3defs','Box and presence MLPs',25,11430,1160,1030,kind='bottleneck',dashed=True)
 chain(p,'three-mlp',[('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 4 (box) or 1 (presence)','linear')],w=990,gap=145)
 p.text(25,900,'The two heads use independent weights. ReLU follows only the first two Linear layers.',17)
 p=d.panel('mlp2defs','Two-layer MLP definitions',1220,11430,1790,1030,kind='bottleneck',dashed=True)
 cols=[('Box position',512,256,256),('X/Y relative bias',2,256,8),('Text scoring',256,2048,256)]
 for j,(name,ci,cm,co) in enumerate(cols):
  xx=25+j*585;p.text(xx,65,name,20,weight=700);chain(p,'two-mlp'+str(j),[('Linear',f'{ci} to {cm}','linear'),('ReLU',str(cm),'activation'),('Linear',f'{cm} to {co}','linear')],x=xx,w=540,y=150,gap=190)
 d.height=12700;d.root.set('height','12700');d.root.set('viewBox','0 0 3040 12700')
 d.text(35,12555,'Default-config scope is explicit: no gated checkpoint was downloaded or assumed. Independent tracker and concept models are instantiated by LibreSAM3.',17)
 b.save(d,'large-text-default','large',task='segment',verification='meta',input='Image: 1 × 3 × 1008 × 1008; text: 1 × 32; no geometry prompts')

def main():
 a=setup();b=Book(a,'sam3','SAM 3',sourcefile='libreyolo/models/sam/sam3.py');import torch,transformers
 from transformers import Sam3Config,Sam3Model,Sam3TrackerConfig,Sam3TrackerModel
 b.evidence['scope']='Transformers 5.16.1 default configurations. facebook/sam3 config request returned401; no gated weights or acceptance flow used.'
 b.evidence['backend']={'version':transformers.__version__,'license':'Apache-2.0','sources':['transformers/models/sam3/modeling_sam3.py','transformers/models/sam3_tracker/modeling_sam3_tracker.py']}
 tc=Sam3TrackerConfig()
 with torch.device('meta'):tracker=Sam3TrackerModel(tc).eval()
 with torch.inference_mode():out=tracker(pixel_values=torch.zeros(1,3,1008,1008,device='meta'),input_boxes=torch.zeros(1,1,4,device='meta'),multimask_output=True)
 b.evidence['tracker']={'config':tc.to_dict(),'pred_masks':list(out.pred_masks.shape),'iou_scores':list(out.iou_scores.shape),'image_embeddings':[list(t.shape) for t in out.image_embeddings]};draw_tracker(b)
 c=Sam3Config()
 with torch.device('meta'):model=Sam3Model(c).eval()
 with torch.inference_mode():
  visionout=model.vision_encoder(torch.zeros(1,3,1008,1008,device='meta'))
  text=torch.zeros(1,32,256,device='meta');positions=visionout.fpn_position_encoding[-2]
  encoder=model.detr_encoder(vision_features=[visionout.fpn_hidden_states[-2]],text_features=text,vision_pos_embeds=[positions])
  decoder=model.detr_decoder(vision_features=encoder.last_hidden_state,text_features=encoder.text_features,vision_pos_encoding=encoder.pos_embeds_flattened,spatial_shapes=torch.tensor([[72,72]],device='cpu'))
  masks=model.mask_decoder(decoder_queries=decoder.intermediate_hidden_states[-1],backbone_features=list(visionout.fpn_hidden_states[:-1]),encoder_hidden_states=encoder.last_hidden_state,prompt_features=text)
 b.evidence['text']={'config':c.to_dict(),'verification':'meta components with static CPU spatial-shape metadata; token embeddings supplied at the text-projection boundary','vision_levels':[list(t.shape) for t in visionout.fpn_hidden_states],'encoder':list(encoder.last_hidden_state.shape),'decoder':list(decoder.intermediate_hidden_states.shape),'masks':list(masks.pred_masks.shape),'semantic':list(masks.semantic_seg.shape)}
 draw_text(b);b.finish()
if __name__=='__main__':main()
