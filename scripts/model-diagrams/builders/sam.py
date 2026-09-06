"""SAM-1 architecture plus explicit shared prompt and mask-decoder schematics."""
from clip import *
SAM_SPECS={'base':{'width':768,'depth':12,'heads':12,'mlp':3072,'global':[2,5,8,11]},'large':{'width':1024,'depth':24,'heads':16,'mlp':4096,'global':[5,11,17,23]},'huge':{'width':1280,'depth':32,'heads':16,'mlp':5120,'global':[7,15,23,31]}}
SAM_CONFIG_SOURCES={'base':{'repo':'facebook/sam-vit-base','revision':'70c1a07f894ebb5b307fd9eaaee97b9dfc16068f'},'large':{'repo':'LibreYOLO/LibreSAMlarge','revision':'968f146bd38c19687620ad44405de79350ef476e'},'huge':{'repo':'LibreYOLO/LibreSAMhuge','revision':'e039bce0f0565ffbb58f99e446a43080c60b9695'}}

def prompt_panel(d,x,y,image_size=1024):
 p=d.panel('prompt','Box prompt encoder',x,y,610,1720,kind='pool')
 ids=chain(p,'box',[('One box','2 corner coordinates × 2','plain'),('Shift pixel centers','Add 0.5 to x and y','plain'),('Normalize coordinates',f'Divide by {image_size}, map to [-1,1]','plain'),('MatMul random Fourier matrix','2 coords to 128 frequencies','linear'),('Multiply by 2 pi','2 × 128','plain')],w=365,y=65,gap=95)
 p.box('sine',30,605,220,'Sin',detail='2 × 128',kind='activation');p.box('cosine',340,605,240,'Cos',detail='2 × 128',kind='activation');p.dot(212.5,550);p.wire([p.port(ids[-1]),(212.5,550)],start=ids[-1],arrow=False);p.wire([(212.5,550),(140,550),(140,605)],start=ids[-1],end='sine');p.wire([(212.5,550),(460,550),(460,605)],start=ids[-1],end='cosine')
 p.box('fouriercat',30,750,365,'Concat sin and cos',detail='2 × 256',kind='concat');p.wire([(140,654),(140,710),(140,750)],start='sine',end='fouriercat');p.wire([(460,654),(460,720),(330,720),(330,750)],start='cosine',end='fouriercat')
 p.box('cornerid',420,840,160,'Corner type',detail='2 learned × 256',font_size=13);p.sum('corneradd',212.5,950);p.connect('fouriercat','corneradd');p.connect('cornerid','corneradd',via=[(500,950)],to_port='right')
 p.box('sparse',30,1050,365,'Sparse prompt embeddings',detail='2 × 256');p.connect('corneradd','sparse')
 p.box('dense',30,1210,365,'Learned no-mask embedding',detail='Broadcast to 256 × 64 × 64')
 p.text(30,1350,'Selected input: one box, no mask prompt.',16)
 p.text(30,1390,'Points use the same Fourier encoding, plus',16);p.text(30,1420,'positive/negative or not-a-point embeddings.',16)
 p.text(30,1470,'Dense image position: this encoding evaluated',16);p.text(30,1500,'on the centers of the 64 × 64 feature grid.',16)
 return p

def decoder_setup(d,x,y,tokens=7,extra_object=False):
 p=d.panel('decode','Mask decoder inputs',x,y,630,1720)
 p.box('imagefeatures',25,65,370,'Image embedding',detail='256 × 64 × 64')
 p.box('denseprompt',435,185,170,'Dense prompt',detail='256 × 64 × 64',font_size=13);p.sum('denseadd',210,290);p.connect('imagefeatures','denseadd');p.connect('denseprompt','denseadd',via=[(520,290)],to_port='right')
 p.box('flattenimage',25,385,370,'Flatten image features',detail='4,096 × 256');p.connect('denseadd','flattenimage')
 p.box('learnedout',25,550,370,'Learned output tokens',detail=('Object + IoU + 4 mask = 6 × 256' if extra_object else 'IoU + 4 masks = 5 × 256'))
 p.box('promptinput',435,650,170,'Sparse prompts',detail='2 × 256',font_size=13);p.box('tokencat',25,780,370,'Concat output and prompt tokens',detail=f'{tokens} × 256',kind='concat');p.connect('learnedout','tokencat');p.connect('promptinput','tokencat',via=[(520,804.5)],to_port='right')
 p.box('tw1',25,900,370,'Two-way block 1',detail=f'{tokens} queries, 4,096 image tokens',kind='attention');p.connect('tokencat','tw1')
 p.box('tw2',25,1040,370,'Two-way block 2',detail=f'{tokens} queries, 4,096 image tokens',kind='attention');p.connect('tw1','tw2')
 p.box('final',25,1180,370,'Final token-to-image attention',detail='Q+Q0, K+P, V=updated image',kind='attention');p.connect('tw2','final')
 p.sum('finaladd',210,1320);p.connect('final','finaladd');p.connect('tw2','finaladd',from_port='right',to_port='right',via=[(420,1064.5),(420,1320)])
 p.box('finalnorm',25,1410,370,'Final LayerNorm on queries',detail=f'{tokens} × 256');p.connect('finaladd','finalnorm')
 p.text(25,1540,'Q0 is the original token sequence.',16);p.text(25,1580,'P is fixed Fourier image position.',16)
 # Image input to the first block is a separate, named continuation.
 p.text(25,470,'4,096 image tokens feed block 1 keys/values.',15)
 return p

def decoder_outputs(d,x,y,extra_object=False,high_res=False):
 p=d.panel('output','Masks and quality outputs',x,y,750,1720,kind='pool')
 p.box('query-out',25,65,310,'Updated query tokens',detail='1 IoU + 4 mask tokens'+(' + 1 object token' if extra_object else ''))
 p.box('image-out',415,65,310,'Updated image tokens',detail='4,096 × 256')
 ids=chain(p,'up',[('Reshape to image','256 × 64 × 64','plain'),('ConvTranspose2d 2×2 / 2','256 to 64; 64 × 128 × 128','conv2d'),('Channel LayerNorm','64 × 128 × 128','norm'),('GELU','64 × 128 × 128','activation'),('ConvTranspose2d 2×2 / 2','64 to 32; 32 × 256 × 256','conv2d'),('GELU','32 × 256 × 256','activation')],x=415,w=310,y=180,gap=100);p.connect('image-out',ids[0])
 hp=chain(p,'hyper',[('Select four mask tokens','4 × 256','plain'),('Four independent Linear layers','Each 256 to 256','linear'),('ReLU','4 × 256','activation'),('Four independent Linear layers','Each 256 to 256','linear'),('ReLU','4 × 256','activation'),('Four independent Linear layers','Each 256 to 32','linear')],x=25,w=310,y=180,gap=100);p.connect('query-out',hp[0])
 p.box('matmul',25,875,700,'MatMul mask coefficients × upscaled image',detail='4 × 32 times 32 × 65,536; reshape to 4 × 256 × 256',kind='attention');p.connect(hp[-1],'matmul',via=[(180,820),(200,820)]);p.connect(ids[-1],'matmul',via=[(570,840),(570,875)])
 p.box('selectmask',25,1020,310,'Select masks 1 to 3',detail='3 × 256 × 256');p.connect('matmul','selectmask',via=[(375,970),(180,970)])
 q=chain(p,'quality',[('IoU token Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 256','linear'),('ReLU','256','activation'),('Linear','256 to 4 scores','linear'),('Select scores 1 to 3','3 quality scores','plain')],x=415,w=310,y=1070,gap=90)
 p.text(25,1260,'Mask logits are resized to the input canvas,',15);p.text(25,1295,'then unpadded/resized to the original image.',15)
 p.text(25,1380,'multimask_output=True is drawn.',15);p.text(25,1415,'False selects mask/score 0 instead.',15)
 if extra_object:p.text(25,1510,'Object token: Linear 256, ReLU, Linear 256,',14);p.text(25,1540,'ReLU, Linear 1 produces object-score logit.',14)
 return p

def two_way(d,id,x,y,index,tokens=7):
 p=d.panel(id,f'Two-way block {index}',x,y,1300,1770,kind='attention',dashed=True)
 p.box(id+'q',100,70,350,'Query state',detail=f'{tokens} × 256');p.box(id+'k',800,70,350,'Image state',detail='4,096 × 256')
 p.box(id+'self',100,200,350,'Token self-attention',detail='8 heads, 32 channels per head',kind='attention');
 if index==1:p.connect(id+'q',id+'self')
 else:
  p.box(id+'selfpe',520,70,200,'Original tokens Q0',detail=f'{tokens} × 256',font_size=14);p.sum(id+'selfpos',275,160);p.connect(id+'q',id+'selfpos');p.connect(id+'selfpe',id+'selfpos',via=[(620,160)],to_port='right');p.connect(id+'selfpos',id+'self')
 if index==2:
  p.text(480,190,'Self-attention Q/K include Q0.',15)
  p.sum(id+'selfadd',275,295);p.connect(id+'self',id+'selfadd');p.connect(id+'q',id+'selfadd',from_port='left',to_port='left',via=[(70,94.5),(70,295)]);prev=id+'selfadd'
 else:prev=id+'self';p.text(480,190,'First self-attention has no PE or residual.',15)
 p.box(id+'n1',100,350,350,'LayerNorm',detail=f'{tokens} × 256',kind='norm');p.connect(prev,id+'n1')
 p.box(id+'qpe',520,350,200,'Original tokens Q0',detail=f'{tokens} × 256',font_size=14);p.sum(id+'qadd',275,460);p.connect(id+'n1',id+'qadd');p.connect(id+'qpe',id+'qadd',via=[(620,460)],to_port='right')
 p.box(id+'ipe',800,225,350,'Image position P',detail='4,096 × 256');p.sum(id+'kadd',975,460);p.connect(id+'ipe',id+'kadd');p.connect(id+'k',id+'kadd',from_port='right',to_port='right',via=[(1200,94.5),(1200,460)])
 p.box(id+'cross',100,565,350,'Token-to-image attention',detail=f'Q={tokens}, K/V=4,096; 8 heads × 16',kind='attention',h=65);p.connect(id+'qadd',id+'cross');p.wire([(975,473),(975,515),(735,515),(735,585),(450,585)],start=id+'kadd',end=id+'cross');p.dot(1280,610);p.wire([(1280,610),(450,610)],start=id+'k',end=id+'cross')
 p.sum(id+'crossadd',275,690);p.connect(id+'cross',id+'crossadd');p.connect(id+'n1',id+'crossadd',from_port='left',to_port='left',via=[(70,374.5),(70,690)])
 ids=chain(p,id+'ff',[('LayerNorm',f'{tokens} × 256','norm'),('Linear','256 to 2,048','linear'),('ReLU' if True else 'GELU',f'{tokens} × 2,048','activation'),('Linear','2,048 to 256','linear')],x=100,w=350,y=760,gap=95);p.connect(id+'crossadd',ids[0]);p.sum(id+'ffadd',275,1140);p.connect(ids[-1],id+'ffadd');p.connect(ids[0],id+'ffadd',from_port='left',to_port='left',via=[(70,784.5),(70,1140)])
 p.box(id+'n3',100,1210,350,'LayerNorm',detail=f'{tokens} × 256',kind='norm');p.connect(id+'ffadd',id+'n3')
 p.box(id+'qpe2',520,1220,200,'Original tokens Q0',detail=f'{tokens} × 256',font_size=14);p.sum(id+'qadd2',275,1330);p.connect(id+'n3',id+'qadd2');p.connect(id+'qpe2',id+'qadd2',via=[(620,1330)],to_port='right')
 p.box(id+'reverse',800,1430,350,'Image-to-token attention',detail=f'Q=4,096, K/V={tokens}; 8 heads × 16',kind='attention',h=65)
 p.wire([(975,473),(975,515),(1180,515),(1180,1370),(975,1370),(975,1430)],start=id+'kadd',end=id+'reverse');p.wire([(275,1343),(275,1380),(700,1380),(700,1450),(800,1450)],start=id+'qadd2',end=id+'reverse');p.wire([(450,1234.5),(480,1234.5),(480,1475),(800,1475)],start=id+'n3',end=id+'reverse')
 p.sum(id+'imageadd',975,1560);p.connect(id+'reverse',id+'imageadd');p.wire([(1150,94.5),(1280,94.5),(1280,1560)],start=id+'k',arrow=False);p.wire([(1280,1560),(988,1560)],start=id+'k',end=id+'imageadd')
 p.box(id+'n4',800,1640,350,'LayerNorm on image state',detail='4,096 × 256',kind='norm');p.connect(id+'imageadd',id+'n4');p.text(100,1690,f'Updated queries: {tokens} × 256',16)
 return p

def common_heads(d,tokens=7):
 prompt_panel(d,660,220);decoder_setup(d,1300,220,tokens);decoder_outputs(d,1960,220)
 two_way(d,'block1',25,2010,1,tokens);two_way(d,'block2',1360,2010,2,tokens)
 attention(d,'self','Token self-attention',25,3840,256,8,tokens,32,K=tokens)
 attention(d,'toimage','Token-to-image attention',700,3840,256,8,tokens,16,K=4096,inner_dim=128)
 attention(d,'totoken','Image-to-token attention',1375,3840,256,8,4096,16,K=tokens,inner_dim=128)
 p=d.panel('decoder-notes','Decoder conventions',2050,3840,660,1270)
 p.text(25,80,'Q/K add positional terms; V is the raw state.',17);p.text(25,125,'Final attention repeats token-to-image attention.',17);p.text(25,170,'Its output is added to queries, then normalized.',17)
 p.text(25,240,'Four mask tokens have independent MLP weights.',17);p.text(25,285,'IoU token uses a separate three-layer MLP.',17)
 p.text(25,355,'All decoder Linear projections include bias.',17)
 p.text(25,425,'Prompt input for this view is one box.',17);p.text(25,470,'Dense prompt is the learned no-mask embedding.',17)
 p.text(25,540,'Point prompts use Fourier position with a',17);p.text(25,575,'positive, negative or padding type embedding.',17)
 p.text(25,645,'Encode once and cache the image features;',17);p.text(25,680,'new prompts rerun prompt encoder and decoder.',17)

def draw_sam(b,size,spec):
 sym=size=='family';E='E' if sym else spec['width'];h='h' if sym else spec['heads'];depth='n' if sym else spec['depth'];M='M' if sym else spec['mlp'];D='E/h' if sym else E//h
 d=b.diagram('SAM '+size,'Promptable segmentation, 1024 × 1024 input, one box, multimask output. Image memory and video tracking are outside this graph.',2740,7030)
 p=d.panel('image','Image encoder',25,220,605,1720)
 ids=chain(p,'image',[('Input after resize/pad','3 × 1,024 × 1,024','plain'),('Conv2d 16×16 / 16',f'64 × 64 × {E}, NHWC','conv2d'),('Add learned absolute position',f'64 × 64 × {E}','plain'),('Window/global ViT blocks',f'64 × 64 × {E}, repeats={depth}','attention'),('Permute to NCHW',f'{E} × 64 × 64','plain'),('Conv2d 1×1',f'{E} to 256, bias=False','conv2d'),('Channel LayerNorm','256 × 64 × 64','norm'),('Conv2d 3×3 / 1','256 to 256, p=1, bias=False','conv2d'),('Channel LayerNorm','256 × 64 × 64','norm')],w=490,y=65,gap=120)
 p.text(25,1270,'Global blocks (1-based): '+('g from table' if sym else ', '.join(str(v+1) for v in spec['global'])),16)
 p.text(25,1310,'Other blocks use 14 × 14 windows.',16)
 p.text(25,1350,'No CLS token in the image encoder.',16)
 common_heads(d)
 p=d.panel('visionblock','Image ViT block',25,5200,650,1530,kind='attention',dashed=True)
 p.text(30,70,f'Input: 64 × 64 × {E}',17);p.dot(230,110)
 items=[('LayerNorm',f'64 × 64 × {E}','norm'),('Window partition when local','Pad 64 to 70; 25 windows of 14 × 14','plain'),('Image self-attention',f'{h} heads; local 196 / global 4,096 tokens','attention'),('Reverse windows and crop','64 × 64 spatial grid','plain')];ids=chain(p,'vb1',items,w=400,y=150,gap=115);p.wire([(230,110),(230,150)],end=ids[0]);p.sum('vsum1',230,680);p.connect(ids[-1],'vsum1');p.wire([(230,110),(600,110),(600,680),(243,680)],end='vsum1')
 p.dot(230,745);p.wire([(230,693),(230,745)],start='vsum1',arrow=False);ids=chain(p,'vb2',[('LayerNorm',f'64 × 64 × {E}','norm'),('Linear',f'{E} to {M}','linear'),('GELU',f'64 × 64 × {M}','activation'),('Linear',f'{M} to {E}','linear')],w=400,y=790,gap=115);p.wire([(230,745),(230,790)],end=ids[0]);p.sum('vsum2',230,1300);p.connect(ids[-1],'vsum2');p.wire([(230,745),(600,745),(600,1300),(243,1300)],end='vsum2')
 # Both local and global attention equations, with concrete token counts.
 attention(d,'localvision','Local image attention',705,5200,E,h,196,D,relative=True,height=1530)
 attention(d,'globalvision','Global image attention',1380,5200,E,h,4096,D,relative=True,height=1530)
 p=d.panel('relative','Decomposed relative position',2055,5200,655,1530,kind='attention',dashed=True)
 p.box('relq',180,65,290,'Query spatial grid',detail=f'Head width {D}')
 for axis,xx in [('h',25),('w',355)]:
  p.box('table'+axis,xx,205,270,'Relative '+axis.upper()+' table',detail=f'27 / 127 distances × {D}')
  p.box('gather'+axis,xx,310,270,'Gather offsets',detail='14×14 / 64×64 coordinate pairs');p.connect('table'+axis,'gather'+axis)
  p.box('einsum'+axis,xx,470,270,'Einsum query × '+axis.upper(),detail='Query-dependent axis bias',kind='attention');p.connect('gather'+axis,'einsum'+axis)
 p.wire([(180,89.5),(10,89.5),(10,494.5),(25,494.5)],start='relq',end='einsumh')
 p.wire([(470,89.5),(645,89.5),(645,494.5),(625,494.5)],start='relq',end='einsumw')
 p.sum('reladd',325,665);p.connect('einsumh','reladd',via=[(160,665)],to_port='left');p.connect('einsumw','reladd',via=[(490,665)],to_port='right')
 p.text(25,750,'Broadcast axis biases before addition.',16)
 p.text(25,805,'Per-head bias: 196×196 locally;',16);p.text(25,845,'4,096×4,096 in global blocks.',16)
 p.text(25,960,'Variant      E      n      h      M',17,weight=700)
 for j,(key,v) in enumerate(SAM_SPECS.items()):p.text(25,1010+j*50,f'{key}: {v["width"]}, {v["depth"]}, {v["heads"]}, {v["mlp"]}',17)
 p.text(25,1210,'Global indices (1-based):',17,weight=700)
 for j,(key,v) in enumerate(SAM_SPECS.items()):p.text(25,1260+j*48,key+': '+', '.join(str(k+1) for k in v['global']),17)
 b.save(d,size+'-box-segment',size,kind='family' if sym else 'concrete',verification='source' if sym else 'meta',task='segment',input='Image: 1 × 3 × 1024 × 1024; box: 1 × 1 × 4')

def main():
 a=setup();b=Book(a,'sam','SAM',sourcefile='model.py');b.sourcefile='libreyolo/models/sam/model.py';import torch,transformers
 from transformers import SamConfig,SamVisionConfig,SamModel
 b.evidence['configuration_sources']=SAM_CONFIG_SOURCES;b.evidence['backend']={'package':'transformers','version':transformers.__version__,'license':'Apache-2.0','source':'transformers/models/sam/modeling_sam.py'}
 for size,spec in SAM_SPECS.items():
  vc=SamVisionConfig(hidden_size=spec['width'],num_hidden_layers=spec['depth'],num_attention_heads=spec['heads'],mlp_dim=spec['mlp'],global_attn_indexes=spec['global'])
  with torch.device('meta'):model=SamModel(SamConfig(vision_config=vc)).eval()
  with torch.inference_mode():out=model(pixel_values=torch.zeros(1,3,1024,1024,device='meta'),input_boxes=torch.zeros(1,1,4,device='meta'),multimask_output=True)
  b.evidence[size]={'device':'meta','vision_config':vc.to_dict(),'pred_masks':list(out.pred_masks.shape),'iou_scores':list(out.iou_scores.shape)};draw_sam(b,size,spec)
 draw_sam(b,'family',SAM_SPECS['base']);b.finish()
if __name__=='__main__':main()
