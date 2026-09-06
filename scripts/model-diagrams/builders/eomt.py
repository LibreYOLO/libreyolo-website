from clip import *
def draw(b,size,task,S,Q,cfg,id=None):
 sym=size=='family';E='E' if sym else cfg['hidden_size'];depth='n' if sym else cfg['num_hidden_layers'];late='b' if sym else cfg['num_blocks'];h='h' if sym else cfg['num_attention_heads'];M='4E' if sym else 4*E;hd='E/h' if sym else E//h;P=(S//16)**2;T=P+5;N=T+Q;K={'semantic':150,'segment':80,'panoptic':133}[task];id=id or size+'-'+task+('-1280' if S==1280 else '')
 d=b.diagram('EoMT '+size+' '+task+(f' {S}px' if S==1280 else ''),f'{task.capitalize()}, {S} × {S} input, {Q} queries, {K} classes. Query tokens enter only the final {late} encoder blocks.',2660,4930)
 p=d.panel('encoder','Encoder-only mask transformer',25,220,610,1930)
 items=[('Normalize RGB','ImageNet mean/std','plain'),('Conv2d 16×16 / 16',f'{E} × {S//16} × {S//16}','conv2d'),('Flatten patch tokens',f'{P} × {E}','plain'),('Add patch position embedding',f'{P} × {E}','plain')];ids=chain(p,'patch',items,w=460,gap=100)
 p.box('prefix',30,565,460,'Concat CLS + four register tokens + patches',detail=f'{T} × {E}',kind='concat');p.connect(ids[-1],'prefix')
 p.box('plain',30,710,460,'Plain encoder blocks',detail=f'{T} × {E}, repeats='+('n-b' if sym else str(depth-late)),kind='attention');p.connect('prefix','plain')
 p.box('queries',335,840,235,'Learned queries',detail=f'{Q} × {E}',font_size=14);p.box('qcat',30,980,460,'Prepend query tokens',detail=f'{N} × {E}',kind='concat');p.connect('plain','qcat');p.connect('queries','qcat',via=[(550,864.5),(550,1004.5)],from_port='right',to_port='right')
 ids=chain(p,'tail',[('Mask-guided encoder blocks',f'{N} × {E}, repeats={late}','attention'),('Final LayerNorm',f'{N} × {E}','norm'),('Prediction heads',f'{Q} class rows and {Q} mask maps','aggregate')],w=460,y=1120,gap=145);p.connect('qcat',ids[0])
 p.text(30,1670,'No DETR decoder or cross-attention module.',16)
 p.text(30,1710,'The shared encoder attends jointly to',16);p.text(30,1745,'queries, five prefix tokens and image patches.',16)
 p=d.panel('block','Encoder block',670,220,600,1930,kind='attention',dashed=True)
 p.text(30,65,f'Tokens: {T} before queries, {N} after.',17);p.dot(230,110)
 ids=chain(p,'ba',[('LayerNorm',str(E)+' channels','norm'),('Self-attention',str(h)+' heads, head width '+str(hd),'attention'),('Multiply layer scale',str(E)+' learned channel weights','plain')],w=400,y=160,gap=130);p.wire([(230,110),(230,160)],end=ids[0]);p.sum('badd1',230,600);p.connect(ids[-1],'badd1');p.wire([(230,110),(550,110),(550,600),(243,600)],end='badd1')
 p.dot(230,675);p.wire([(230,613),(230,675)],start='badd1',arrow=False)
 ids=chain(p,'bm',[('LayerNorm',str(E)+' channels','norm'),('Linear',f'{E} to {M}','linear'),('GELU',str(M)+' channels','activation'),('Linear',f'{M} to {E}','linear'),('Multiply layer scale',str(E)+' learned channel weights','plain')],w=400,y=725,gap=125);p.wire([(230,675),(230,725)],end=ids[0]);p.sum('badd2',230,1435);p.connect(ids[-1],'badd2');p.wire([(230,675),(550,675),(550,1435),(243,1435)],end='badd2')
 attention(d,'plainattn','Plain encoder self-attention',1305,220,E,h,T,hd,height=1930)
 attention(d,'lateattn','Joint masked self-attention',1980,220,E,h,N,hd,relative=True,height=1930)
 # This attention's additive input is a predicted mask, not relative position.
 for elem in d.root.iter():
  if elem.tag.endswith('text') and elem.text=='Position bias':elem.text='Query mask'
 p=d.panel('feedback','Mask guidance before each final block',25,2220,1245,1040,kind='aggregate',dashed=True)
 items=[('Current joint token state',f'{N} × {E}','plain'),('LayerNorm',f'{N} × {E}','norm'),('Shared prediction heads',f'{Q} mask logits at {S//4} × {S//4}','aggregate'),('Bilinear resize mask logits',f'{Q} × {S//16} × {S//16}','plain'),('Threshold at zero',f'{Q} × {P} allowed query-to-patch entries','plain'),('Build additive attention mask',f'{h} × {N} × {N}; disallowed=-1e9','plain')];chain(p,'feedbackops',items,w=820,gap=120)
 p.text(30,845,'Other query/key pairs remain allowed. Masking is applied when attn_mask_probs for the block is positive.',16)
 p.text(30,885,'Fresh configurations use ones. Checkpoint buffer values can disable this guidance; the encoder path stays present.',16)
 p.text(30,925,'The next block consumes this mask and the current token state, then the prediction is recomputed.',16)
 p=d.panel('pred','Shared prediction heads',1305,2220,1250,1040,kind='pool',dashed=True)
 p.box('normalized',25,65,1200,'Normalized joint tokens',detail=f'{N} × {E}')
 p.box('selectq',25,210,510,'Select query tokens',detail=f'{Q} × {E}');p.box('selectpatch',655,210,570,'Remove queries, CLS and registers',detail=f'{P} × {E}');p.wire([(325,114),(325,160),(280,160),(280,210)],start='normalized',end='selectq');p.wire([(925,114),(925,160),(940,160),(940,210)],start='normalized',end='selectpatch')
 p.box('classhead',25,385,510,'Linear class predictor',detail=f'{E} to {K+1}, including no-object',kind='linear');p.connect('selectq','classhead')
 p.box('maskhead',25,550,510,'Mask embedding MLP',detail=f'{Q} × {E}',kind='linear');p.connect('selectq','maskhead',from_port='left',to_port='left',via=[(10,234.5),(10,574.5)])
 p.box('upscale',655,385,570,'Reshape and two upscale blocks',detail=f'{E} × {S//4} × {S//4}',kind='conv');p.connect('selectpatch','upscale')
 p.box('einsum',25,770,1200,'Einsum query embeddings × pixel embeddings',detail=f'{Q} × {S//4} × {S//4} mask logits',kind='attention');p.connect('maskhead','einsum',via=[(280,690),(300,690)]);p.connect('upscale','einsum',via=[(940,670),(940,770)])
 p.text(25,920,'The same heads serve mask guidance and the final predictions.',17)
 p=d.panel('maskmlp','Mask embedding MLP',25,3330,620,1120,kind='bottleneck',dashed=True)
 chain(p,'mlp',[('Linear',f'{E} to {E}','linear'),('GELU',f'{Q} × {E}','activation'),('Linear',f'{E} to {E}','linear'),('GELU',f'{Q} × {E}','activation'),('Linear',f'{E} to {E}','linear')],w=510,gap=130)
 p=d.panel('up','Two upscale blocks',680,3330,720,1120,kind='conv',dashed=True)
 items=[]
 for scale in [2,4]:
  z=S//16*scale;items.extend([('ConvTranspose2d 2×2 / 2',f'{E} to {E}; {z} × {z}','conv2d'),('GELU',f'{E} × {z} × {z}','activation'),('Depthwise Conv2d 3×3',f'g={E}, p=1, bias=False','conv2d'),('Channel LayerNorm',f'{E} × {z} × {z}','norm')])
 chain(p,'scale',items,w=610,gap=115)
 p=d.panel('task','LibreYOLO task output',1435,3330,1120,1120,kind='pool')
 p.box('maskresize',25,65,1040,'Bilinear resize mask logits',detail=f'{Q} × {S} × {S}, align_corners=False')
 if task=='semantic':items=[('Softmax class rows, drop no-object',f'{Q} × {K}','activation'),('Sigmoid mask logits',f'{Q} × {S} × {S}','activation'),('Sum class-weighted masks',f'{K} × {S} × {S}','attention'),('Argmax across classes',f'{S} × {S} semantic labels','plain')]
 elif task=='segment':items=[('Class probabilities and mask probabilities',f'{Q} candidates, {K} foreground classes','activation'),('Score and mask filtering','Keep nonempty instance masks','plain'),('Binarize masks and derive boxes','Variable number of masks and XYXY boxes','plain')]
 else:items=[('Score queries; exclude no-object','80 thing classes and 53 stuff classes','activation'),('Per-pixel score-weighted winner','Resolve overlaps between query masks','plain'),('Filter overlaps and merge same stuff class','Instance IDs for things, class regions for stuff','plain'),('Panoptic result','Segment IDs, categories and isthing metadata','plain')]
 ids=chain(p,'result',items,w=1040,y=220,gap=150);p.connect('maskresize',ids[0])
 if task=='semantic':
  for element in list(p.ops):
   if element.get('id','').startswith('result'):p.ops.remove(element)
  for wire in list(p.wires):
   if wire.get('data-from','').startswith('result') or wire.get('data-to','').startswith('result'):p.wires.remove(wire)
  p.box('class-in',610,225,455,'Class logits from class predictor',detail=f'{Q} × {K+1}',font_size=14)
  p.box('class-prob',610,365,455,'Softmax, discard no-object',detail=f'{Q} × {K}',kind='activation');p.connect('class-in','class-prob')
  p.box('mask-prob',25,365,455,'Sigmoid mask logits',detail=f'{Q} × {S} × {S}',kind='activation');p.connect('maskresize','mask-prob',via=[(545,170),(252.5,170)])
  p.box('semantic-sum',25,635,1040,'Einsum class probabilities × mask probabilities',detail=f'{K} × {S} × {S}',kind='attention')
  p.wire([(252.5,414),(252.5,550),(300,550),(300,635)],start='mask-prob',end='semantic-sum');p.wire([(837.5,414),(837.5,570),(820,570),(820,635)],start='class-prob',end='semantic-sum')
  p.box('semantic-label',25,825,1040,'Argmax across semantic classes',detail=f'{S} × {S} labels');p.connect('semantic-sum','semantic-label')
 p.text(25,990,'Postprocessing is outside the learned encoder and heads.',16)
 p=d.panel('table','Variant values',25,4520,2530,250)
 for x,label in [(25,'Size'),(300,'E'),(600,'n'),(900,'b'),(1200,'h'),(1500,'MLP width')]:p.text(x,65,label,18,weight=700)
 for j,(key,sp) in enumerate(b.evidence['specs'].items()):
  for x,v in zip([25,300,600,900,1200,1500],[key,sp['hidden_size'],sp['num_hidden_layers'],sp['num_blocks'],sp['num_attention_heads'],sp['hidden_size']*4]):p.text(x,110+j*38,str(v),18)
 b.save(d,id,size,kind='family' if sym else 'concrete',verification='source' if sym else 'meta',task=task,input=f'1 × 3 × {S} × {S}')

def main():
 a=setup();b=Book(a,'eomt','EoMT');m=load(a.source,'eomt');import torch,transformers
 b.evidence['specs']=m._SIZE_TO_CONFIG;b.evidence['backend']={'version':transformers.__version__,'license':'Apache-2.0','source':'transformers/models/eomt/modeling_eomt.py'}
 b.evidence['query_contract']='Published ADE20K semantic config uses 100 queries; all COCO instance/panoptic configs checked use 200. Small/base segment views correspond to supported things-only panoptic conversion. Semantic small/base are supported scratch configurations.'
 for task,S,Q,K in [('semantic',512,100,150),('segment',640,200,80),('panoptic',640,200,133)]:
  for size,cfg in m._SIZE_TO_CONFIG.items():
   with torch.device('meta'):model=m.LibreEoMTNet(size,nb_classes=K,image_size=S,num_queries=Q).eval()
   model.eomt.attn_mask_probs=torch.ones(cfg['num_blocks'],device='cpu')
   with torch.inference_mode():out=model(torch.zeros(1,3,S,S,device='meta'))
   b.evidence[size+'-'+task]={'device':'meta','input':[1,3,S,S],'outputs':{k:list(v.shape) for k,v in out.items()},'mask_schedule':'ones on CPU; matches fresh model defaults'};draw(b,size,task,S,Q,cfg)
  draw(b,'family',task,S,Q,m._SIZE_TO_CONFIG['s'])
 size='l';task='segment';S=1280;Q=200
 with torch.device('meta'):model=m.LibreEoMTNet(size,nb_classes=80,image_size=S,num_queries=Q).eval()
 model.eomt.attn_mask_probs=torch.ones(4,device='cpu')
 with torch.inference_mode():out=model(torch.zeros(1,3,S,S,device='meta'))
 b.evidence['l-segment-1280']={'device':'meta','input':[1,3,S,S],'outputs':{k:list(v.shape) for k,v in out.items()}};draw(b,'l','segment',S,Q,m._SIZE_TO_CONFIG['l']);b.finish()
if __name__=='__main__':main()
