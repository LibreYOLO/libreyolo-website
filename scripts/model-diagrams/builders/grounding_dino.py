from clip import *

def post_block(d,id,title,x,y,N,E,hidden,heads,attn_name='Self-attention',activation='ReLU',height=1350):
 p=d.panel(id,title,x,y,650,height,kind='attention',dashed=True)
 p.text(25,65,f'Input {N} × {E}',17);p.dot(230,105)
 p.box(id+'a',30,150,400,attn_name,detail=f'{heads} heads, {E} channels',kind='attention');p.wire([(230,105),(230,150)],end=id+'a');p.sum(id+'s1',230,300);p.connect(id+'a',id+'s1');p.wire([(230,105),(600,105),(600,300),(243,300)],end=id+'s1')
 ids=chain(p,id+'mlp',[('LayerNorm',f'{N} × {E}','norm'),('Linear',f'{E} to {hidden}','linear'),(activation,f'{N} × {hidden}','activation'),('Linear',f'{hidden} to {E}','linear')],w=400,y=385,gap=130);p.connect(id+'s1',ids[0]);p.sum(id+'s2',230,940);p.connect(ids[-1],id+'s2');p.connect(ids[0],id+'s2',from_port='right',to_port='right',via=[(570,409.5),(570,940)])
 p.box(id+'n2',30,1040,400,'LayerNorm',detail=f'{N} × {E}',kind='norm');p.connect(id+'s2',id+'n2');return p

def deformable(d,x,y):
 p=d.panel('deformable','Multi-scale deformable attention',x,y,2975,1720,kind='attention',dashed=True)
 p.box('dq',25,65,560,'Query + positional embedding',detail='Encoder 13,294 / decoder 900 queries × 256')
 p.box('dv',1110,65,710,'Value features',detail='13,294 × 256 from four image levels')
 p.box('dr',2050,65,850,'Reference coordinates',detail='Encoder: normalized XY; decoder: normalized XYWH')
 p.box('offsets',25,240,560,'Linear sampling offsets',detail='256 to 256; 8 heads × 4 levels × 4 points × 2',kind='linear');p.connect('dq','offsets')
 p.box('weights',650,240,410,'Linear attention weights',detail='256 to 128',kind='linear');p.connect('dq','weights',from_port='right',to_port='top',via=[(855,89.5)])
 p.box('weightsoft',650,390,410,'Softmax levels and points',detail='8 heads × 16 probabilities per query',kind='activation');p.connect('weights','weightsoft')
 p.box('valueproj',1110,240,710,'Linear value projection',detail='256 to 256; reshape 13,294 × 8 × 32',kind='linear');p.connect('dv','valueproj')
 p.box('coords',2050,390,850,'Compute sampling positions',detail='XY reference + normalized offsets; decoder offsets scaled by box WH')
 p.connect('dr','coords');p.connect('offsets','coords',from_port='bottom',to_port='left',via=[(305,345),(1900,345),(1900,414.5)])
 p.box('value-split',1110,580,710,'Split values by four levels',detail='100×100, 50×50, 25×25 and 13×13, each 256 channels',kind='split');p.connect('valueproj','value-split')
 for j,(h,w) in enumerate([(100,100),(50,50),(25,25),(13,13)]):
  xx=25+j*740;p.box('sample'+str(j),xx,880,680,'Bilinear grid_sample',detail=f'{h} × {w} map; 4 samples per query/head',kind='attention')
  # Named value/coordinate continuations preserve distinct level inputs without a crowded global bus.
  p.text(xx,790,f'Values: level {j+1}; coordinates: level {j+1}',16)
  p.box('weighted'+str(j),xx,1070,680,'Multiply sample values × attention weights',detail='4 × 32 values per query/head',kind='plain');p.connect('sample'+str(j),'weighted'+str(j))
  p.text(xx,1025,f'Weights: level {j+1}, four of the 16 probabilities',15)
 p.box('sumlevels',25,1370,2900,'Sum all four levels and four sample points',detail='8 heads × 32 channels; merge to 256',kind='plain')
 for j in range(4):p.wire([(365+j*740,1119),(365+j*740,1270+j*20),(360+j*740,1270+j*20),(360+j*740,1370)],start='weighted'+str(j),end='sumlevels')
 p.box('dout',900,1530,1200,'Linear output projection',detail='256 to 256',kind='linear');p.connect('sumlevels','dout',via=[(1475,1480),(1500,1480)])
 return p

def fusion(d,x,y):
 p=d.panel('fusion','Bidirectional vision-text fusion',x,y,1490,1630,kind='attention',dashed=True)
 p.box('fv',25,65,610,'LayerNorm vision',detail='13,294 × 256',kind='norm');p.box('ft',835,65,610,'LayerNorm text',detail='16 × 256',kind='norm')
 p.box('vvalue',355,220,280,'Linear vision values VV',detail='256 to 1,024',kind='linear',font_size=13);p.connect('fv','vvalue',via=[(330,175),(495,175)]);p.text(355,310,'VV: 4 × 13,294 × 256',14)
 p.box('tvalue',1165,220,280,'Linear text values TV',detail='256 to 1,024',kind='linear',font_size=13);p.connect('ft','tvalue',via=[(1140,175),(1305,175)]);p.text(1165,310,'TV: 4 × 16 × 256',14)
 p.box('fq',25,220,280,'Linear vision Q',detail='256 to 1,024; 4 heads × 256; scale 1/16',kind='linear');p.connect('fv','fq',via=[(330,155),(165,155)])
 p.box('fk',835,220,280,'Linear text K',detail='256 to 1,024; 4 heads × 256',kind='linear');p.connect('ft','fk',via=[(1140,155),(975,155)])
 p.box('fscores',395,385,700,'MatMul vision Q × text K-transpose',detail='4 × 13,294 × 16',kind='attention');p.wire([(165,269),(165,335),(570,335),(570,385)],start='fq',end='fscores');p.wire([(975,269),(975,345),(950,345),(950,385)],start='fk',end='fscores')
 p.box('fstab',395,520,700,'Subtract max and clamp',detail='Clamp [-50,000,50,000]');p.connect('fscores','fstab')
 p.box('vsoft',25,715,610,'Mask text padding, softmax over text',detail='4 × 13,294 × 16',kind='activation');p.box('tsoft',835,715,610,'Transpose, stabilize, mask vision, softmax',detail='4 × 16 × 13,294',kind='activation',font_size=15)
 p.dot(745,620);p.wire([(745,569),(745,620)],start='fstab',arrow=False);p.wire([(745,620),(330,620),(330,715)],start='fstab',end='vsoft');p.wire([(745,620),(1140,620),(1140,715)],start='fstab',end='tsoft')
 p.box('fvvalues',25,880,610,'MatMul vision probabilities × TV',detail='TV is the text-value projection shown above',kind='attention')
 p.box('ftvalues',835,880,610,'MatMul text probabilities × VV',detail='VV is the vision-value projection shown above',kind='attention');p.connect('vsoft','fvvalues');p.connect('tsoft','ftvalues')
 for id,xx,N in [('v',25,13294),('t',835,16)]:
  ids=chain(p,'f'+id+'out',[('Linear output','1,024 to 256','linear'),('Multiply learned channel scale','256 weights, initialized 1e-4','plain')],x=xx,w=610,y=1040,gap=130);p.connect('fvvalues' if id=='v' else 'ftvalues',ids[0]);p.sum('fusionadd'+id,xx+305,1430);p.connect(ids[-1],'fusionadd'+id);p.text(xx,1530,'Residual starts at the normalized '+('vision' if id=='v' else 'text')+' input.',15)
  if id=='v':p.wire([(25,89.5),(10,89.5),(10,1430),(317,1430)],start='fv',end='fusionaddv')
  else:p.wire([(1445,89.5),(1475,89.5),(1475,1430),(1153,1430)],start='ft',end='fusionaddt')
 return p

def draw(b,size,cfg):
 sym=size=='family';bc=cfg.backbone_config;C=[f'C{i+1}' for i in range(4)] if sym else [bc.embed_dim*2**i for i in range(4)];depth=[f'n{i+1}' for i in range(4)] if sym else bc.depths;heads=[f'h{i+1}' for i in range(4)] if sym else bc.num_heads;ws='w' if sym else bc.window_size;WN='w²' if sym else ws*ws;SHIFT='floor(w/2)' if sym else ws//2
 d=b.diagram('Grounding DINO '+size,'Open-vocabulary detection, 800 × 800 square image, 16 text tokens, 900 object queries. Tensor sizes exclude batch.',3040,10240)
 p=d.panel('image','Image backbone and feature projection',25,220,740,1780)
 ops=[('Image','3 × 800 × 800','plain'),('Swin patch embedding','4×4 /4 Conv, channel LayerNorm','conv')]
 for i in range(4):ops.append((f'Swin stage {i+1}',f'{200//2**i} × {200//2**i} × {C[i]}, n={depth[i]}','attention'))
 ops +=[('Three 1×1 projections + GroupNorm',f'{C[1]}/{C[2]}/{C[3]} to 256; groups=32','conv'),('Fourth level from raw stage 4',f'Conv3×3 /2, {C[3]} to256; GroupNorm32','conv'),('Flatten and concat image levels','13,294 × 256','concat'),('Sine position + learned level embedding','13,294 × 256','plain')];chain(p,'image',ops,w=630,gap=130)
 p.text(25,1550,'Feature sizes: 100×100, 50×50, 25×25, 13×13.',17)
 p=d.panel('text','BERT text backbone',805,220,740,1780)
 ops=[('Input IDs + token types + phrase position IDs','16 tokens; vocabulary 30,522','plain'),('Sum word, position and token-type embeddings','16 × 768','plain'),('LayerNorm','16 × 768','norm'),('BERT encoder blocks, repeats=12','768 width, 12 heads, MLP 3,072','attention'),('Linear text projection','768 to 256','linear')];chain(p,'bertflow',ops,w=630,gap=180)
 p.text(25,1190,'Native phrase-mask values are added as 0/1 bias',17);p.text(25,1230,'inside this BERT path, preserving library behavior.',17)
 p.text(25,1320,'Padding and token masks are carried separately.',17)
 p=d.panel('joint','Multimodal encoder and decoder',1585,220,1425,1780)
 p.box('jointinput',25,65,1350,'Image and projected text states',detail='13,294 × 256 and 16 × 256')
 p.box('encoder',25,230,1350,'Encoder layer, repeats=6',detail='Bidirectional fusion; parallel text enhancement and deformable vision layer',kind='attention');p.connect('jointinput','encoder')
 p.box('proposal',25,430,640,'Linear + LayerNorm proposal features',detail='13,294 × 256');p.wire([(500,279),(500,350),(345,350),(345,430)],start='encoder',end='proposal')
 p.box('contrast',735,430,640,'Contrastive token scoring',detail='13,294 × 16, padded to256 token slots',kind='attention');p.wire([(900,279),(900,385),(1055,385),(1055,430)],start='encoder',end='contrast')
 p.box('topk',735,610,640,'Top 900 by max valid-token score',detail='900 proposal indices');p.connect('contrast','topk')
 p.box('boxproposal',25,610,640,'Proposal box MLP + grid box logits',detail='13,294 × 4');p.connect('proposal','boxproposal')
 p.box('refs',25,800,640,'Gather selected boxes, sigmoid',detail='900 × 4 initial reference boxes');p.connect('boxproposal','refs');p.connect('topk','refs',from_port='bottom',to_port='right',via=[(1055,824.5)])
 p.box('content',735,800,640,'Learned query content',detail='900 × 256')
 p.box('decoder',25,1030,1350,'Decoder layer, repeats=6',detail='Query self-attention, text cross-attention, deformable image attention, MLP',kind='attention');p.connect('refs','decoder',via=[(345,950),(350,950)]);p.connect('content','decoder',via=[(1055,970),(1050,970)])
 p.box('finalout',25,1270,1350,'Final-layer contrastive logits and refined boxes',detail='900 × 256 token logits; 900 × 4 normalized cxcywh');p.connect('decoder','finalout')
 p.text(25,1460,'Box deltas refine inverse-sigmoid references at every decoder layer.',17)
 p.text(25,1520,'Postprocess applies sigmoid, threshold and phrase-to-class alignment.',17)
 p.text(25,1580,'No NMS in the learned detector.',17)
 fusion(d,25,2070)
 post_block(d,'enhancer','Text enhancer',1550,2070,16,256,1024,4,height=1630)
 post_block(d,'visualencoder','Deformable vision layer',2230,2070,13294,256,2048,8,attn_name='Multi-scale deformable attention',height=1630)
 p=d.panel('decb','Decoder layer',25,3770,960,1930,kind='attention',dashed=True)
 prev=None;yy=80
 for j,label in enumerate(['Query self-attention','Text cross-attention','Deformable image attention']):
  p.dot(330,yy);p.box('da'+str(j),30,yy+45,600,label,detail=['Q/K/V=900','Q=900, K/V=16','Q=900, four image levels'][j],kind='attention')
  if prev:p.connect(prev,'da'+str(j))
  else:p.wire([(330,yy),(330,yy+45)],end='da'+str(j))
  p.sum('das'+str(j),330,yy+160);p.connect('da'+str(j),'das'+str(j));p.wire([(330,yy),(870,yy),(870,yy+160),(343,yy+160)],end='das'+str(j));p.box('dan'+str(j),30,yy+235,600,'LayerNorm',detail='900 × 256',kind='norm');p.connect('das'+str(j),'dan'+str(j));prev='dan'+str(j);yy+=365
 p.dot(330,yy);p.wire([p.port(prev),(330,yy)],start=prev,arrow=False)
 ids=chain(p,'dff',[('Linear','256 to 2,048','linear'),('ReLU','900 × 2,048','activation'),('Linear','2,048 to 256','linear')],w=600,y=yy+45,gap=125);p.wire([(330,yy),(330,yy+45)],end=ids[0]);p.sum('dffsum',330,1630);p.connect(ids[-1],'dffsum');p.wire([(330,yy),(870,yy),(870,1630),(343,1630)],end='dffsum');p.box('dffnorm',30,1730,600,'LayerNorm',detail='900 × 256',kind='norm');p.connect('dffsum','dffnorm')
 attention(d,'queries','Query self-attention',1020,3770,256,8,900,32,height=1930)
 attention(d,'textcross','Query-to-text attention',1700,3770,256,8,900,32,K=16,height=1930)
 p=d.panel('reference','Reference / box MLPs',2390,3770,620,1930,kind='bottleneck',dashed=True)
 chain(p,'boxhead',[('Normalized query state','900 × 256','plain'),('Linear','256 to256','linear'),('ReLU','256','activation'),('Linear','256 to256','linear'),('ReLU','256','activation'),('Linear','256 to4','linear'),('Add inverse-sigmoid reference','900 ×4','plain'),('Sigmoid refined box','900 ×4','activation')],w=510,gap=130)
 p.text(25,1320,'Proposal MLP uses the same 256/256/4 dimensions.',16)
 p.text(25,1380,'Position MLP: sine-encoded box512,',16);p.text(25,1420,'Linear512 to256, ReLU, Linear256 to256.',16)
 deformable(d,25,5770)
 post_block(d,'bertblock','BERT encoder block',25,7560,16,768,3072,12,activation='GELU',height=1420)
 attention(d,'bertattn','BERT self-attention',705,7560,768,12,16,64,relative=True,height=1420)
 p=d.panel('swinblock','Swin block and patch merging',1385,7560,1625,1420,kind='attention',dashed=True)
 widths='/'.join(map(str,C));mlps='/'.join((f'M{i+1}' for i in range(4))) if sym else '/'.join(str(x*4) for x in C)
 p.dot(350,75)
 items=[('LayerNorm','Stage channels '+widths,'norm'),('Pad to window multiple',f'Window {ws}; padding precedes shift','plain'),('Cyclic shift on even blocks',f'Shift -{SHIFT} in H and W','plain'),('Window self-attention',f'{WN} tokens; heads '+ '/'.join(map(str,heads)),'attention'),('Reverse window partition','Restore padded spatial grid','plain'),('Reverse cyclic shift',f'Shift +{SHIFT} in H and W','plain'),('Crop padding','Restore stage H × W','plain')]
 ids=chain(p,'swinstage',items,x=25,w=650,y=115,gap=70);p.wire([(350,75),(350,115)],end=ids[0]);p.sum('swinres1',350,625);p.connect(ids[-1],'swinres1');p.wire([(350,75),(770,75),(770,625),(363,625)],end='swinres1')
 p.dot(350,690);p.wire([(350,638),(350,690)],start='swinres1',arrow=False)
 ids=chain(p,'swinmlp',[('LayerNorm','Stage channels '+widths,'norm'),('Linear','Outputs '+mlps,'linear'),('GELU','MLP channels '+mlps,'activation'),('Linear','Outputs '+widths,'linear')],x=25,w=650,y=735,gap=105);p.wire([(350,690),(350,735)],end=ids[0]);p.sum('swinres2',350,1190);p.connect(ids[-1],'swinres2');p.wire([(350,690),(770,690),(770,1190),(363,1190)],end='swinres2')
 chain(p,'patchmerge',[('Input for next Swin stage','2×2 neighboring patches','plain'),('Reshape and concatenate neighborhood','Four times current channel width','plain'),('LayerNorm','Concatenated neighborhood channels','norm'),('Linear reduction (no bias)','Neighborhood channels to next stage width','linear')],x=860,w=710,y=100,gap=140)
 p.text(860,750,'Resolved merge widths:',18,weight=700)
 for j in range(3):p.text(860,815+j*65,('family C'+str(j+1)+' input; next C'+str(j+2)) if sym else f'{C[j]} input; {C[j]*4} concatenated; {C[j+1]} output',17)
 p.text(860,1070,'Output grids: 100×100, 50×50, 25×25.',17)
 # Additional explicit local-attention and stem details.
 attention(d,'swinattn','Swin window self-attention',25,9050,'/'.join(map(str,C)),'/'.join(map(str,heads)),WN,32,relative=True,height=1350)
 attention(d,'enhancerself','Text-enhancer self-attention',705,9050,256,4,16,64,relative=True,height=1350)
 p=d.panel('values','Variant values',1385,9050,1625,1350)
 for j,(key,v) in enumerate(b.evidence['resolved_configs'].items()):
  bb=v['backbone_config'];p.text(25,80+j*260,key,22,weight=700);p.text(25,140+j*260,'Swin channels: '+', '.join(str(bb['embed_dim']*2**i) for i in range(4)),18);p.text(25,190+j*260,'Stage repeats: '+', '.join(map(str,bb['depths']))+'; window '+str(bb['window_size']),18);p.text(25,240+j*260,'Heads: '+', '.join(map(str,bb['num_heads'])),18)
 p.text(25,700,'Both sizes: 256 model width, 6 encoder layers, 6 decoder layers, 900 queries.',18)
 p.text(25,760,'Deformable attention: 8 heads, 4 levels, 4 points per level.',18)
 p.text(25,820,'Vision/text fusion: 1,024 internal width, 4 heads; text enhancer: 4 heads, MLP1,024.',17)
 p.text(25,880,'Backbone input here is square 800px. Other aspect ratios change spatial token counts.',17)
 d.height=10630;d.root.set('height','10630');d.root.set('viewBox','0 0 3040 10630')
 for group in d.root.iter():
  if group.get('id')=='bertattnmask':
   for text in group.iter():
    if text.tag.endswith('text') and text.text=='Position bias':text.text='Phrase-mask bias'
 b.save(d,size+'-detect',size,kind='family' if sym else 'concrete',verification='source' if sym else 'cpu',task='detect',input='Image: 1 ×3 ×800 ×800; text: 1 ×16')

def main():
 a=setup();b=Book(a,'grounding_dino','Grounding DINO');path=WEBSITE/'scripts/model-diagrams/evidence/grounding_dino.json';raw=json.loads(path.read_text());source=raw.get('configurations',raw.get('shapes',{}).get('configurations'));b.evidence['configurations']=source
 from transformers import GroundingDinoConfig
 configs={k:GroundingDinoConfig(**v['config']) for k,v in source.items()};b.evidence['resolved_configs']={k:c.to_dict() for k,c in configs.items()};b.evidence['cpu_tiny']=raw.get('cpu_tiny',raw.get('shapes',{}).get('cpu_tiny'))
 import torch
 m=load(a.source,'grounding_dino')
 with torch.device('meta'):model=m.GroundingDinoDetectionModel(configs['b']).eval()
 with torch.inference_mode():features=model.backbone_conv(torch.zeros(1,3,800,800,device='meta'));projected=[model.input_proj_vision[i](f) for i,f in enumerate(features)]+[model.input_proj_vision[3](features[-1])]
 b.evidence['base_meta']={'scope':'Swin backbone and all four feature projections; multimodal head config matches the CPU-tested tiny graph','backbone':[list(t.shape) for t in features],'projected':[list(t.shape) for t in projected]}
 for size,cfg in configs.items():draw(b,size,cfg)
 draw(b,'family',configs['t']);b.finish()
if __name__=='__main__':main()
