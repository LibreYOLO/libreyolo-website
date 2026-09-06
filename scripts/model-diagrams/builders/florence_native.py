"""Florence-2 diagrams from pinned native Apache-2.0 Transformers and MIT adapters."""
import argparse,json,os,sys,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',Path(__file__).resolve().parents[4]/'libreyolo')));args=ap.parse_args()
WEB=Path(__file__).resolve().parents[3];SRC=args.source.resolve();sys.path.insert(0,str(SRC/'skills/libreyolo-make-diagram/scripts'))
from svg_diagram import Diagram
from wrap_svg import wrap
REV=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
def box(p,id,y,label,detail='',kind='plain',x=100,w=400,h=52,typ=''):
 return p.box(id,x,y,w,label,detail=detail,kind=kind,h=h,font_size=15,block_type=typ)
def seq(p,ops,x=100,w=400,start=70,step=100):
 prev=None
 for i,op in enumerate(ops):
  id,label,detail,kind,*rest=op;box(p,id,start+i*step,label,detail,kind,x,w,typ=rest[0] if rest else '')
  if prev:p.connect(prev,id)
  prev=id
 return prev
def residual(p,prefix,y,label,detail,kind='plain',postnorm=False,typ=''):
 box(p,prefix+'in',y,'Input',x=170,w=340);p.dot(340,y+78)
 box(p,prefix+'op',y+115,label,detail,kind,x=170,w=340,typ=typ);p.connect(prefix+'in',prefix+'op')
 p.sum(prefix+'add',340,y+235);p.connect(prefix+'op',prefix+'add');p.wire([(340,y+78),(70,y+78),(70,y+235),(327,y+235)],start=prefix+'in',end=prefix+'add')
 if postnorm:
  box(p,prefix+'norm',y+275,'LayerNorm',postnorm,'norm',x=170,w=340);p.connect(prefix+'add',prefix+'norm');return prefix+'norm'
 return prefix+'add'
def build(fam,r,symbolic=False):
 c=r['config'];v=c['vision_config'];t=c['text_config'];ground=fam.startswith('ground');title='Florence-2 grounding' if ground else 'Florence-2';slug='ground-florence2' if ground else 'florence-2'
 C=['C1','C2','C3','C4'] if symbolic else v['embed_dim'];D='D' if symbolic else t['d_model'];N='N' if symbolic else t['encoder_layers'];H='H' if symbolic else t['encoder_attention_heads'];F='F' if symbolic else t['encoder_ffn_dim'];name=title+(' family' if symbolic else ' '+r['size'])
 d=Diagram(name,'Image inference at 768 × 768. DaViT vision encoder, BART encoder-decoder and LibreYOLO output adapter.',width=2740,height=4530 if symbolic else 4260,source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{REV}/{r["source"]}',source_label=r['source']+'; Transformers 5.16.1',revision=REV,logo=WEB/'public/icon-128.png')
 p=d.panel('vision','DaViT stage sequence',40,220,640,1300)
 last=seq(p,[('image','Processor image tensor','1 × 3 × 768 × 768','plain')],w=440)
 for i in range(4):
  y=175+i*240;detail=f'{3 if i==0 else C[i-1]} to {C[i]}; k={7 if i==0 else 3}, s={4 if i==0 else 2}, p={3 if i==0 else 1}'
  box(p,'embed'+str(i),y,'Conv2d then LayerNorm' if i==0 else 'LayerNorm then Conv2d',detail,'conv2d',w=440);p.connect(last,'embed'+str(i))
  box(p,'stage'+str(i),y+110,'Spatial block then channel block',f'n={v["depths"][i]}; {C[i]} × {192//2**i} × {192//2**i}','attention',w=440,typ='davit-unit');p.connect('embed'+str(i),'stage'+str(i));last='stage'+str(i)
 p.text(24,1202,'Stage heads / channel groups: '+('G1, G2, G3, G4' if symbolic else ', '.join(map(str,v['num_heads']))),14)
 p.text(24,1235,'Both blocks are expanded below; channel width C follows the stage.',14)
 p=d.panel('project','Vision projector and source sequence',720,220,620,1300)
 seq(p,[('feature','Final feature map',f'1 × {C[-1]} × 24 × 24','plain'),('position','Add learned 2D positions','Row and column embedding concatenation','plain'),('flatten','Flatten spatial dimensions',f'1 × 576 × {C[-1]}','plain'),('time','Add cosine/sine temporal positions','One image frame; temporal index 0','plain')],w=420)
 box(p,'spatialmean',510,'Mean over 576 positions',f'1 × 1 × {C[-1]}','pool',x=20,w=275)
 box(p,'temporalmean',510,'Mean over one frame',f'1 × 576 × {C[-1]}','pool',x=325,w=275)
 p.dot(310,450);p.wire([p.port('time'),(310,450)],start='time',arrow=False)
 for id,x in [('spatialmean',157.5),('temporalmean',462.5)]:p.wire([(310,450),(x,450),(x,510)],start='time',end=id)
 box(p,'cat',630,'Concat token axis',f'577 × {C[-1]}','concat',w=420)
 for id,x,xx in [('spatialmean',157.5,225),('temporalmean',462.5,395)]:p.wire([p.port(id),(x,600),(xx,600),(xx,630)],start=id,end='cat')
 box(p,'projection',740,'Linear, no bias',f'{C[-1]} to {D}','linear',w=420);p.connect('cat','projection')
 box(p,'norm',850,'LayerNorm',f'577 × {D}','norm',w=420);p.connect('projection','norm')
 box(p,'scatter',975,'Replace image placeholders','Prompt tokens use vocabulary embeddings','concat',w=420);p.connect('norm','scatter')
 box(p,'srcpos',1090,'Add learned source positions',f'LayerNorm; sequence length L, width {D}','norm',w=420);p.connect('scatter','srcpos')
 p=d.panel('language','BART encoder and decoder',1380,220,640,1300)
 seq(p,[('src','Image and prompt sequence',f'L × {D}; vocabulary51328','plain'),('enc','Encoder layers',f'n={N}; {H} heads; FFN={F}','attention','bart-encoder'),('memory','Encoder memory',f'L × {D}; supplies cross-attention K/V','plain')],w=430)
 box(p,'target',405,'Generated prefix embedding',f'Vocabulary51328; width {D}','linear',w=430)
 box(p,'tgtpos',510,'Add learned target positions','LayerNorm before decoder stack','norm',w=430);p.connect('target','tgtpos')
 box(p,'dec',625,'Decoder layers',f'n={N}; {H} heads; FFN={F}','attention',w=430,typ='bart-decoder');p.connect('tgtpos','dec');p.connect('memory','dec',from_port='left',to_port='left',via=[(40,296),(40,651)])
 box(p,'lm',750,'Vocabulary Linear',f'{D} to 51328; shared embedding weights','linear',w=430);p.connect('dec','lm')
 box(p,'beam',865,'Beam search','3 beams; autoregressive token selection','plain',w=430);p.connect('lm','beam')
 p.wire([p.port('beam','right'),(600,891),(600,431),p.port('target','right')],start='beam',end='target')
 box(p,'reply',1020,'Generated token IDs','Decode with processor','plain',w=430);p.connect('beam','reply')
 p.text(24,1190,'Dropout is disabled in inference. Cached decoding reuses K/V.',14)
 p=d.panel('output','LibreYOLO adapter',2060,220,640,1300)
 ops=[('query','Task token and query','Phrase grounding' if ground else 'Open vocabulary detection','plain'),('decode','Processor post-process','Decode tags and pixel-coordinate boxes','plain'),('parse','Extract bboxes and labels','Original image size supplies geometry','plain')]
 if ground:ops += [('filter','Remove whole-image boxes','Match normalized query text','plain'),('choose','Choose tightest matching box','Fallback to tightest remaining candidate','plain'),('center','Box center point','x=(x1+x2)/2; y=(y1+y2)/2','plain'),('res','Results.points','One pixel point; synthetic score1','plain')]
 else:ops += [('normbox','Normalize pixel xyxy','Divide x by W and y by H','plain'),('filters','Class/confidence filtering','Synthetic score1; valid finite boxes','plain'),('dedup','Same-class deduplication','Rounded duplicates and IoU threshold','plain'),('res','Results.boxes','Return pixel xyxy; apply max_det','plain')]
 seq(p,ops,w=440,step=140)
 p.text(24,1180,'Task: '+('<CAPTION_TO_PHRASE_GROUNDING>' if ground else '<OPEN_VOCABULARY_DETECTION>'),13)
 p.text(24,1220,'No pretrained inference was run for this diagram.',14)
 # DaViT repeated residual subgraph; each primitive path is visible.
 p=d.panel('davit-unit','DaViT spatial / channel block',40,1560,640,1670,kind='attention',dashed=True,block_type='davit-unit')
 p.text(24,65,'Use spatial attention first, then channel attention; same skeleton.',14)
 prev=None
 for i,(lab,det,k,typ) in enumerate([('Depthwise Conv2d','C channels; kernel3, stride1, padding1','conv',''),('LayerNorm + selected attention','See separate spatial and channel insets','attention','davit-attention'),('Depthwise Conv2d','C channels; kernel3, stride1, padding1','conv',''),('LayerNorm + MLP','Linear C to 4C; GELU; Linear 4C to C','bottleneck','vision-mlp')]):
  y=105+i*350;end=residual(p,'dv'+str(i),y,lab,det,k,typ=typ)
  if prev:p.connect(prev,'dv'+str(i)+'in')
  prev=end
 p.text(24,1580,'LayerNorm acts on channels; reshape between grids and tokens.',14)
 p.text(24,1620,'C by stage: '+', '.join(map(str,C))+'. FFN widths: '+', '.join(str(4*x) if isinstance(x,int) else '4'+x for x in C)+'.',13)
 p=d.panel('spatial','Spatial attention primitives',720,1560,620,1670,kind='attention',dashed=True,block_type='davit-attention')
 seq(p,[('sln','LayerNorm','C channels','norm'),('window','Pad and partition windows','12 × 12 positions per window','plain'),('sqkv','Q/K/V Linear','C to 3C; split 3, heads, 32','linear'),('sqk','Q × transpose(K)','144 × 144 per head; scale1/sqrt(32)','attention'),('ssm','Softmax over keys','144 spatial positions','attention'),('sav','Attention weights × V','144 × 32 per head','attention'),('sjoin','Concat heads','Width C','concat'),('sout','Output Linear','C to C','linear'),('unwindow','Merge windows and crop padding','Restore original H × W × C','plain')],w=420,step=160)
 p=d.panel('channel','Channel attention primitives',1380,1560,640,1670,kind='attention',dashed=True,block_type='davit-attention')
 seq(p,[('cln','LayerNorm','C channels','norm'),('cqkv','Q/K/V Linear','C to 3C; split into G groups','linear'),('ctrans','Transpose each grouped tensor','32 channels × S spatial positions','plain'),('cqk','Q × transpose(K)','32 × 32 per group; scale1/sqrt(S)','attention'),('csm','Softmax over channel keys','32 channels per group','attention'),('cav','Attention weights × V','32 channels × S positions','attention'),('cjoin','Restore token and channel order','S × C; S = H × W','concat'),('cout','Output Linear','C to C','linear')],w=440,step=175)
 p=d.panel('bart-decoder','BART decoder layer',2060,1560,640,1670,kind='attention',dashed=True,block_type='bart-decoder')
 prev=None
 for i,(label,detail,typ) in enumerate([('Causal self-attention',f'Q/K/V from target prefix; {H} heads','bart-attention'),('Cross-attention',f'Q from target; K/V from encoder memory','bart-attention'),('MLP',f'Linear {D} to {F}; GELU; Linear {F} to {D}','text-mlp')]):
  y=80+i*490;end=residual(p,'bd'+str(i),y,label,detail,'attention' if i<2 else 'bottleneck',postnorm=f'{D} channels',typ=typ)
  if prev:p.connect(prev,'bd'+str(i)+'in')
  prev=end
 p.text(24,1605,'Each sublayer uses residual addition followed by LayerNorm.',14)
 p=d.panel('bart-encoder','BART encoder layer',40,3270,640,850,kind='attention',dashed=True,block_type='bart-encoder')
 a=residual(p,'be0',70,'Bidirectional self-attention',f'Q/K/V from source; {H} heads','attention',postnorm=f'{D} channels',typ='bart-attention')
 residual(p,'be1',455,'MLP',f'Linear {D} to {F}; GELU; Linear {F} to {D}','bottleneck',postnorm=f'{D} channels',typ='text-mlp');p.connect(a,'be1in')
 p=d.panel('bart-attention','BART attention primitives',720,3270,1280,850,kind='attention',dashed=True,block_type='bart-attention')
 # Three explicit projection and value paths.
 for ident,xx,label in [('q',30,'Q from sublayer input'),('k',430,'K from input or memory'),('v',830,'V from input or memory')]:
  box(p,ident,80,label,f'Linear {D} to {D}; reshape {H} heads','linear',x=xx,w=350)
 box(p,'qk',230,'Q × transpose(K)',f'Head width {32 if False else ("D/H" if symbolic else t["d_model"]//t["encoder_attention_heads"])}; inverse sqrt scaling','attention',x=190,w=460)
 p.wire([p.port('q'),(205,190),(300,190),(300,230)],start='q',end='qk');p.wire([p.port('k'),(605,190),(530,190),(530,230)],start='k',end='qk')
 box(p,'mask',350,'Add mask; softmax over keys','Causal for decoder self-attention only','attention',x=190,w=460);p.connect('qk','mask')
 box(p,'av',490,'Attention weights × V','Encoder memory used only in cross-attention','attention',x=400,w=460);p.connect('mask','av',via=[(420,455),(540,455)])
 p.wire([p.port('v'),(1005,455),(750,455),(750,490)],start='v',end='av')
 box(p,'out',635,'Concat heads; output Linear',f'{D} to {D}','linear',x=400,w=460);p.connect('av','out')
 p.text(24,780,'No rotary positions. Learned positions enter embeddings before the encoder and decoder.',16)
 p=d.panel('mlps','MLP primitives',2040,3270,660,850,kind='bottleneck',dashed=True,block_type='text-mlp')
 seq(p,[('mi','LayerNorm (DaViT only)','BART normalizes after the residual sum','norm'),('m1','Linear',f'DaViT C to 4C; BART {D} to {F}','linear'),('ma','GELU','Elementwise activation','activation'),('m2','Linear',f'DaViT 4C to C; BART {F} to {D}','linear')],w=460,step=175)
 if symbolic:
  p=d.panel('family-table','Resolved family parameters',40,4140,2660,250)
  p.text(24,65,'Size                C1 / C2 / C3 / C4                   G1 / G2 / G3 / G4                 D            N           H           F',20)
  p.text(24,112,'base                128 / 256 / 512 / 1024              4 / 8 / 16 / 32                      768          6          12         3072',20)
  p.text(24,157,'large               256 / 512 / 1024 / 2048             8 / 16 / 32 / 64                     1024        12          16         4096',20)
  p.text(24,213,'C: stage width; G: spatial heads/channel groups; D: BART width; N: layers per stack; H: BART heads; F: BART MLP width.',17)
 d.text(50,d.height-75,'Checked: pinned config and native meta module geometry. Runtime token counts depend on the processor; no weight-based execution.',15)
 ident='family' if symbolic else r['size']+('-point' if ground else '-detect');out=WEB/'public/diagrams/models'/slug;out.mkdir(parents=True,exist_ok=True);svg=out/(ident+'.svg');d.save(svg);wrap(svg,out/(ident+'.html'))
 return dict(id=ident,label=name,kind='family' if symbolic else 'concrete',task='point' if ground else 'detect',size='family' if symbolic else r['size'],svg=f'/diagrams/models/{slug}/{ident}.svg',html=f'/diagrams/models/{slug}/{ident}.html',verification='meta',input='768 × 768')
if __name__ == "__main__":
 for fam,slug in [('florence2','florence-2'),('ground_florence2','ground-florence2')]:
  rows=[json.loads(p.read_text()) for p in sorted((WEB/'scripts/model-diagrams/evidence/vlm-configs').glob(fam+'-*.json'))];views=[build(fam,r) for r in rows];views.append(build(fam,rows[0],True));out=WEB/'public/diagrams/models'/slug
  (out/'manifest.json').write_text(json.dumps(dict(family=fam,slug=slug,title='Florence-2 grounding' if fam.startswith('ground') else 'Florence-2',source_revision=REV,default_view=views[0]['id'],views=views),indent=2)+'\n')
  (WEB/'scripts/model-diagrams/evidence'/f'{fam}.json').write_text(json.dumps(dict(source_revision=REV,backend='Transformers5.16.1 native Apache-2.0',configs=[{k:r[k] for k in ['repo','resolved_revision','size','source','meta']} for r in rows],checks=['Pinned configuration and meta module construction','Native DaViT and BART forward-path source inspection'],limits=['No pretrained execution','Nominal square input; processor owns resizing']),indent=2)+'\n')
  print(fam,len(views),'views')
