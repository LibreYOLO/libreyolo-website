"""Original diagrams from LibreYOLO adapters, pinned HF config metadata and
Apache-2.0 Transformers 5.16.1 native classes. No model code/weights are vendored.
"""
import argparse,json,os,sys,subprocess,copy
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',Path(__file__).resolve().parents[4]/'libreyolo')));P.add_argument('--family');args=P.parse_args()
WEB=Path(__file__).resolve().parents[3];SRC=args.source.resolve();sys.path.insert(0,str(SRC/'skills/libreyolo-make-diagram/scripts'))
from svg_diagram import Diagram
from wrap_svg import wrap
REV=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
CONFIG=WEB/'scripts/model-diagrams/evidence/vlm-configs'
SPECS={
 'qwen3vl':('qwen3-vl','Qwen3-VL','qwen3','detect','2b'),
 'ground_qwen3vl':('ground-qwen3vl','Qwen3-VL grounding','qwen3','point','2b'),
 'internvl3':('internvl3','InternVL3','internvl','detect','1b'),
 'smolvlm2':('smolvlm2','SmolVLM2','smol','detect','500m'),
 'lfm2vl':('lfm2-vl','LFM2.5-VL','lfm','detect','450m'),
 'northmicrovision':('northmicrovision','North Micro Vision','north','detect','2.4b'),
 'showui':('showui','ShowUI','qwen2','point','2b'),
}

def mul(x,n):return x*n if isinstance(x,int) else f'{n} × {x}'
def chain(p,ids):
 for a,b in zip(ids,ids[1:]):p.connect(a,b)
def box(p,id,y,label,detail='',kind='plain',x=110,w=340,h=49,typ=''):
 return p.box(id,x,y,w,label,detail=detail,kind=kind,h=h,font_size=15,block_type=typ,description=(label+'. '+detail).strip())
def sig(p,id,x,y,name):return p.box(id,x,y,56,name,h=26,center=True,font_size=13,block_type='signal-'+name,description='Matching connector names carry the same tensor.')
def linked(p,a,b,from_port='bottom',to_port='top',via=()):p.connect(a,b,from_port=from_port,to_port=to_port,via=via)

def parameters(r):
 c=r['config'];v=c['vision_config'];t=c['text_config'];vd=v.get('hidden_size',v.get('embed_dim'));vl=v.get('num_hidden_layers',v.get('depth'));vh=v.get('num_attention_heads',v.get('num_heads'));vf=v.get('intermediate_size',vd*int(v.get('mlp_ratio',4)))
 if c['model_type']=='qwen2_vl':vd=v['embed_dim'];vf=vd*int(v['mlp_ratio'])
 td=t['hidden_size'];th=t['num_attention_heads'];kv=t.get('num_key_value_heads',th);ff=t['intermediate_size']
 if r['family']=='lfm2vl':
  m=r.get('meta',{}).get('modules',{});ff=next((val[1] for k,val in m.items() if '0.feed_forward.w1' in k),ff)
 patch=v.get('patch_size',16);patch=patch[0] if isinstance(patch,list) else patch
 return dict(V=vd,Lv=vl,Hv=vh,Fv=vf,D=td,Lt=t['num_hidden_layers'],Ht=th,Hkv=kv,Ft=ff,Vocab=t['vocab_size'],dv=vd//vh,dt=t.get('head_dim',td//th),patch=patch,scale=c.get('scale_factor',c.get('downsample_factor',int(1/c.get('downsample_ratio',.5)))),deep=v.get('deepstack_visual_indexes',[]))

def residual_definition(d,id,title,x,y,dim,attn_name,mlp_name,norm,*,layerscale=False,parallel=False):
 p=d.panel(id,title,x,y,510,1050,kind='attention',dashed=True,block_type=id)
 box(p,id+'-in',65,'Input',f'S × {dim}',x=160,w=290)
 box(p,id+'-n1',170,norm,str(dim)+' channels','norm',x=160,w=290)
 p.dot(305,140);p.connect(id+'-in',id+'-n1')
 if parallel:
  box(p,id+'-att',270,attn_name,kind='attention',x=25,w=210,typ='text-attn')
  box(p,id+'-mlp',270,mlp_name,kind='bottleneck',x=275,w=210,typ='text-mlp')
  p.dot(305,242);p.wire([p.port(id+'-n1'),(305,242)],start=id+'-n1',arrow=False)
  p.wire([(305,242),(130,242),(130,270)],start=id+'-n1',end=id+'-att');p.wire([(305,242),(380,242),(380,270)],start=id+'-n1',end=id+'-mlp')
  p.sum(id+'-sum',305,430);p.wire([p.port(id+'-att'),(130,430),(292,430)],start=id+'-att',end=id+'-sum');p.wire([p.port(id+'-mlp'),(380,430),(318,430)],start=id+'-mlp',end=id+'-sum')
  p.sum(id+'-sum2',305,540);p.connect(id+'-sum',id+'-sum2');p.wire([(305,140),(12,140),(12,540),(292,540)],start=id+'-in',end=id+'-sum2')
  p.text(24,620,'Attention and MLP use the same normalized input.',14);p.text(24,655,'One shared LayerNorm, parallel sublayers.',14)
 else:
  box(p,id+'-att',265,attn_name,kind='attention',x=160,w=290,typ='vision-attn' if id=='vision-layer' else 'text-attn')
  chain(p,[id+'-n1',id+'-att'])
  prev=id+'-att'
  if layerscale:box(p,id+'-scale1',345,'LayerScale','Learned per-channel multiplier','linear',x=160,w=290);p.connect(prev,id+'-scale1');prev=id+'-scale1'
  p.sum(id+'-sum1',305,440);p.connect(prev,id+'-sum1');p.wire([(305,140),(45,140),(45,440),(292,440)],start=id+'-in',end=id+'-sum1')
  p.dot(305,488)
  box(p,id+'-n2',535,norm,str(dim)+' channels','norm',x=160,w=290);p.connect(id+'-sum1',id+'-n2')
  box(p,id+'-mlp',640,mlp_name,kind='bottleneck',x=160,w=290,typ='vision-mlp' if id=='vision-layer' else 'text-mlp');p.connect(id+'-n2',id+'-mlp');prev=id+'-mlp'
  if layerscale:box(p,id+'-scale2',725,'LayerScale',kind='linear',x=160,w=290);p.connect(prev,id+'-scale2');prev=id+'-scale2'
  p.sum(id+'-sum2',305,835);p.connect(prev,id+'-sum2');p.wire([(305,488),(85,488),(85,835),(292,835)],start=id+'-sum1',end=id+'-sum2')
 box(p,id+'-out',920,'Output',f'S × {dim}',x=160,w=290);p.connect(id+'-sum2',id+'-out')
 p.text(24,1016,'S is the token count of this sublayer.',13)
 return p

def attention_definition(d,id,title,x,y,D,H,KV,HD,*,rotary=False,qknorm=False,causal=False):
 p=d.panel(id,title,x,y,510,1050,kind='attention',dashed=True,block_type=id)
 p.text(20,66,f'Input width {D}; Q heads {H}; K/V heads {KV}.',14)
 for name,xx,width in [('q',20,mul(HD,H) if isinstance(H,int) and isinstance(HD,int) else f'{H} × {HD}'),('k',180,mul(HD,KV) if isinstance(KV,int) and isinstance(HD,int) else f'{KV} × {HD}'),('v',340,mul(HD,KV) if isinstance(KV,int) and isinstance(HD,int) else f'{KV} × {HD}')]:
  box(p,id+'-'+name,110,name.upper()+' Linear',f'{D} to {width}','linear',x=xx,w=145,h=60)
  box(p,id+'-'+name+'reshape',205,'Split heads',f'{HD} per head','split',x=xx,w=145,h=60);p.connect(id+'-'+name,id+'-'+name+'reshape')
 if qknorm:
  for name,xx in [('q',20),('k',180)]:box(p,id+'-'+name+'norm',300,'RMSNorm',f'{HD} per head','norm',x=xx,w=145);p.connect(id+'-'+name+'reshape',id+'-'+name+'norm')
 q=id+'-qnorm' if qknorm else id+'-qreshape';k=id+'-knorm' if qknorm else id+'-kreshape'
 if rotary:
  for name,xx,src in [('q',20,q),('k',180,k)]:box(p,id+'-'+name+'rot',385,'Rotary position','Sliding layers only' if rotary=='sliding' else 'cos/sin half rotation','plain',x=xx,w=145,h=60);p.connect(src,id+'-'+name+'rot')
  q=id+'-qrot';k=id+'-krot'
 if causal:
  box(p,id+'-kalign',455,'Align K heads',f'{KV} to {H}','plain',x=180,w=145,h=38);p.connect(k,id+'-kalign');k=id+'-kalign'
  box(p,id+'-valign',385,'Align V heads',f'{KV} to {H}','plain',x=340,w=145,h=60);p.connect(id+'-vreshape',id+'-valign')
 vsource=id+'-valign' if causal else id+'-vreshape'
 box(p,id+'-qk',550,'Q × transpose(K)',f'Scale by 1/sqrt({HD})','attention',x=65,w=275)
 p.wire([p.port(q),(92.5,522),(145,522),(145,550)],start=q,end=id+'-qk');p.wire([p.port(k),(252.5,522),(260,522),(260,550)],start=k,end=id+'-qk')
 box(p,id+'-soft',650,'Softmax over keys','Causal mask; cached K/V optional' if causal else 'Within the image; padding mask','attention',x=65,w=275);p.connect(id+'-qk',id+'-soft')
 box(p,id+'-av',755,'Attention weights × V',f'{H} query heads, weighted values','attention',x=95,w=345)
 p.wire([p.port(id+'-soft'),(202.5,730),(205,730),(205,755)],start=id+'-soft',end=id+'-av')
 p.wire([p.port(vsource),(412.5,705),(340,705),(340,755)],start=vsource,end=id+'-av')
 box(p,id+'-join',850,'Concat heads',kind='concat',x=95,w=345);p.connect(id+'-av',id+'-join')
 box(p,id+'-out',940,'Output Linear',f'{mul(HD,H) if isinstance(H,int) and isinstance(HD,int) else str(H)+" × "+str(HD)} to {D}','linear',x=95,w=345);p.connect(id+'-join',id+'-out')
 p.text(20,1002,'RoPE: x*cos + rotate_half(x)*sin.' if rotary else 'Position information enters through embeddings.',13)
 p.text(20,1025,'Full layers skip RoPE.' if rotary=='sliding' else 'Cached decode appends past K/V before head alignment.' if causal else 'Distinct projections share the same normalized input.',13)
 return p

def mlp_definition(d,id,title,x,y,D,F,act,gated):
 p=d.panel(id,title,x,y,510,930,kind='bottleneck',dashed=True,block_type=id)
 box(p,id+'-input',65,'Input',f'S × {D}',x=135,w=240)
 if gated:
  box(p,id+'-gate',185,'Gate Linear',f'{D} to {F}','linear',x=25,w=205)
  box(p,id+'-up',185,'Up Linear',f'{D} to {F}','linear',x=280,w=205)
  p.dot(255,145);p.wire([p.port(id+'-input'),(255,145)],start=id+'-input',arrow=False)
  p.wire([(255,145),(127.5,145),(127.5,185)],start=id+'-input',end=id+'-gate');p.wire([(255,145),(382.5,145),(382.5,185)],start=id+'-input',end=id+'-up')
  box(p,id+'-act',300,act,kind='activation',x=25,w=205);p.connect(id+'-gate',id+'-act')
  box(p,id+'-multiply',430,'Elementwise multiply',f'S × {F}','plain',x=110,w=300)
  p.wire([p.port(id+'-act'),(127.5,400),(180,400),(180,430)],start=id+'-act',end=id+'-multiply');p.wire([p.port(id+'-up'),(382.5,400),(330,400),(330,430)],start=id+'-up',end=id+'-multiply')
  prev=id+'-multiply'
 else:
  box(p,id+'-up',185,'Linear',f'{D} to {F}','linear',x=135,w=240);p.connect(id+'-input',id+'-up')
  box(p,id+'-act',330,act,kind='activation',x=135,w=240);p.connect(id+'-up',id+'-act');prev=id+'-act'
 box(p,id+'-down',580,'Down Linear',f'{F} to {D}','linear',x=110,w=300);p.connect(prev,id+'-down')
 box(p,id+'-output',740,'Output',f'S × {D}',x=110,w=300);p.connect(id+'-down',id+'-output')
 p.text(20,868,'Residual addition is in the enclosing layer diagram.',13)
 p.text(20,898,'S is the token count of this sublayer.',13)
 return p

def build(family,r,*,symbolic=False,family_rows=()):
 slug,title,kind,task,default=SPECS[family];c=r['config'];v=c['vision_config'];tc=c['text_config'];vals=parameters(r);z=dict(vals)
 if symbolic:
  for key in ['V','Lv','Hv','Fv','D','Lt','Ht','Hkv','Ft','Vocab','dv','dt']:z[key]=key
 V,Lv,Hv,Fv,D,Lt,Ht,KV,Ft,Vocab,dv,dt=[z[k] for k in ['V','Lv','Hv','Fv','D','Lt','Ht','Hkv','Ft','Vocab','dv','dt']]
 deep=vals['deep'] if kind in ('qwen3','north') else []
 name=title+' family' if symbolic else title+' '+r['size'].upper()
 d=Diagram(name,f'{task.capitalize()}. P = image patches; L = combined tokens (runtime axes). Model widths below are '+('variables in the table.' if symbolic else 'resolved.'),width=2200,height=4050 if symbolic else 3710,source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{REV}/{r["source"]}',source_label=r['source']+'; Transformers 5.16.1',revision=REV,logo=WEB/'public/icon-128.png')
 a=d.panel('image','Image encoder',40,220,650,1100)
 box(a,'image',70,'Image processor',f'Nominal runner size {r["input"]}; native tiling/resize',x=95,w=450)
 patch='Linear patch projection' if kind=='lfm' else ('Conv3d patch embedding' if kind in ('qwen3','qwen2','north') else 'Conv2d patch embedding')
 box(a,'patch',175,patch,f'patch={vals["patch"]}; output P × {V}','conv2d',x=95,w=450);a.connect('image','patch')
 pos='Learned absolute positions + CLS' if kind=='internvl' else ('Learned/interpolated 2D positions' if kind!='qwen2' else '2D rotary position information')
 box(a,'pos',280,pos,f'P × {V}'+('; CLS adds one token' if kind=='internvl' else ''),'plain',x=95,w=450);a.connect('patch','pos')
 previous='pos'
 if deep:
  begin=0
  for i,end in enumerate(deep+[vals['Lv']-1]):
   label=f'Vision blocks {begin} to {end}' if not symbolic else ('Vision blocks to tap '+str(i+1) if i<3 else 'Remaining vision blocks')
   box(a,'vision'+str(i),390+i*130,label,f'n={end-begin+1}; width {V}' if not symbolic else 'Total depth Lv; tap indices in table','attention',x=95,w=450,typ='vision-layer');a.connect(previous,'vision'+str(i));previous='vision'+str(i)
   if i<3:
    sig(a,'tap'+str(i),570,401+i*130,'V'+str(i));a.connect(previous,'tap'+str(i),from_port='right',to_port='left')
   begin=end+1
 else:
  box(a,'vision',400,'Vision transformer blocks',f'n={Lv}; width {V}; heads={Hv}','attention',x=95,w=450,typ='vision-layer');a.connect(previous,'vision');previous='vision'
  if kind in ('smol','lfm'):
   box(a,'final-vnorm',555,'Final LayerNorm',f'{V} channels','norm',x=95,w=450);a.connect(previous,'final-vnorm');previous='final-vnorm'
  if kind=='internvl':
   box(a,'dropcls',555,'Drop CLS; restore spatial grid','1024 patch positions per 448-square tile','plain',x=95,w=450);a.connect(previous,'dropcls');previous='dropcls'
 sig(a,'v-output',570,930,'V');a.wire([a.port(previous),(320,943),(570,943)],start=previous,end='v-output')
 a.text(25,1015,'V carries final visual features to the projector.',14)
 a.text(25,1044,'Patch counts vary with processor output, not model size.',14)
 if deep:a.text(25,1072,'V0/V1/V2 are the three intermediate visual taps.',14)
 b=d.panel('fusion','Projection and multimodal sequence',730,220,700,1100)
 sig(b,'v-input',35,70,'V');box(b,'projector',140,'Visual projector',f'{V} channels become {D} channels','bottleneck',x=35,w=300,typ='projector');b.connect('v-input','projector',via=[(63,116),(185,116)])
 box(b,'v-tokens',270,'Visual token sequence',f'Patch grouping factor {vals["scale"]} per axis','plain',x=35,w=300);b.connect('projector','v-tokens')
 box(b,'prompt',70,'Prompt and image placeholders','One query per class' if kind=='north' else ('Click instruction' if task=='point' else 'Vocabulary and detection request'),x=385,w=280)
 box(b,'tokenize',170,'Tokenizer','Runtime text length','plain',x=385,w=280);b.connect('prompt','tokenize')
 box(b,'embed',270,'Token embedding',f'Vocabulary {Vocab}; width {D}','linear',x=385,w=280);b.connect('tokenize','embed')
 box(b,'scatter',405,'Replace image-token placeholders',f'L × {D}; image and text in one sequence','concat',x=130,w=440)
 b.wire([b.port('v-tokens'),(185,365),(235,365),(235,405)],start='v-tokens',end='scatter');b.wire([b.port('embed'),(525,365),(450,365),(450,405)],start='embed',end='scatter')
 sig(b,'sequence-output',610,417,'S');b.connect('scatter','sequence-output',from_port='right',to_port='left')
 if deep:
  for i in range(3):
   yy=550+i*135;sig(b,'dv'+str(i),35,yy+10,'V'+str(i));box(b,'dmerge'+str(i),yy,'Intermediate patch merger',f'Post-shuffle LayerNorm; output width {D}','bottleneck',x=140,w=390,typ='projector');b.connect('dv'+str(i),'dmerge'+str(i),from_port='right',to_port='left');sig(b,'deep'+str(i),610,yy+10,'D'+str(i));b.connect('dmerge'+str(i),'deep'+str(i),from_port='right',to_port='left')
  b.text(25,1000,'D0/D1/D2 add to visual positions in text layers 0/1/2.',14)
 else:
  b.text(25,650,'Visual features replace placeholders before language layers.',15)
  b.text(25,695,'Text and image masks preserve their positions.',15)
 b.text(25,1050,'S denotes the multimodal token sequence.',14)
 g=d.panel('generation','Language model and generated output',1470,220,690,1100)
 sig(g,'sequence-input',20,70,'S');last=None
 if deep:
  for i in range(3):
   yy=70+i*150;box(g,'decoder'+str(i),yy,'Text decoder layer '+str(i),f'Width {D}','attention',x=135,w=470,typ='text-layer')
   if last:g.connect(last,'decoder'+str(i))
   else:g.connect('sequence-input','decoder0',from_port='right',to_port='left')
   g.sum('deep-add'+str(i),370,yy+95);g.connect('decoder'+str(i),'deep-add'+str(i));sig(g,'deep-in'+str(i),20,yy+82,'D'+str(i));g.wire([(76,yy+95),(357,yy+95)],start='deep-in'+str(i),end='deep-add'+str(i));last='deep-add'+str(i)
  box(g,'remaining',540,'Remaining text decoder layers',f'n={vals["Lt"]-3}' if not symbolic else 'n=Lt-3','attention',x=135,w=470,typ='text-layer');g.connect(last,'remaining');last='remaining'
 else:
  box(g,'remaining',170,'Text decoder layers',f'n={Lt}; width {D}','attention',x=135,w=470,typ='text-layer');g.connect('sequence-input','remaining',via=[(48,125),(370,125)]);last='remaining'
  if kind=='lfm':
   layers=tc['layer_types'];g.text(25,285,'Layer order (zero-based):',15)
   for start in range(0,len(layers),8):g.text(25,320+32*(start//8),'; '.join(f'{i}:{"A" if val=="full_attention" else "C"}' for i,val in enumerate(layers[start:start+8],start)),14)
   g.text(25,460,'A: causal attention; C: gated short convolution.',14)
  if kind=='north':g.text(25,440,'Sliding attention window 4096; every fourth layer is full.',14)
 box(g,'norm',640,'Final '+('LayerNorm' if kind=='north' else 'RMSNorm'),f'{D} channels','norm',x=135,w=470);g.connect(last,'norm')
 box(g,'lmhead',735,'Language head Linear',f'{D} to {Vocab}'+('; logits × 0.25' if kind=='north' else ''),'linear',x=135,w=470);g.connect('norm','lmhead')
 max_new=64 if kind=='qwen2' else (128 if task=='point' else 1024)
 box(g,'generate',835,'Greedy token generation',f'max_new_tokens={max_new}; do_sample=False','plain',x=135,w=470);g.connect('lmhead','generate')
 sig(g,'output-text',610,930,'O');g.wire([g.port('generate'),(370,943),(610,943)],start='generate',end='output-text')
 g.text(25,995,'North: sliding window4096; every fourth layer uses full attention.',13) if kind=='north' else None
 g.text(25,1025,'O is the generated reply; prompt tokens are removed.',14)
 g.text(25,1055,'Generation is repeated per vocabulary class.' if kind=='north' else 'The adapter decodes O into task geometry below.',14)
 # Four detailed computation panels.
 residual_definition(d,'vision-layer','Vision transformer layer',40,1390,V,'Vision attention','Vision MLP','LayerNorm',layerscale=kind=='internvl')
 residual_definition(d,'text-layer','Text decoder layer',580,1390,D,'Attention / short Conv' if kind=='lfm' else 'Causal attention','Gated MLP','LayerNorm' if kind=='north' else 'RMSNorm',parallel=kind=='north')
 attention_definition(d,'vision-attn','Vision attention',1120,1390,V,Hv,Hv,dv,rotary=kind in ('qwen3','qwen2','north'))
 attention_definition(d,'text-attn','Text attention',1660,1390,D,Ht,KV,dt,rotary='sliding' if kind=='north' else True,qknorm=kind in ('qwen3','lfm'),causal=True)
 mlp_definition(d,'vision-mlp','Vision MLP',40,2510,V,Fv,v.get('hidden_act','GELU'),False)
 mlp_definition(d,'text-mlp','Text MLP',580,2510,D,Ft,'SiLU',True)
 pr=d.panel('projector','Visual projector',1120,2510,510,930,kind='bottleneck',dashed=True,block_type='projector')
 factor=vals['scale'];packed=mul(V,factor*factor)
 y=70;ids=[]
 def op(id,label,detail='',kind='plain'):
  nonlocal y
  box(pr,id,y,label,detail,kind,x=65,w=385);ids.append(id);y+=100
 if kind in ('qwen3','north','qwen2'):
  op('prnorm','LayerNorm',f'{V} channels (main merger)','norm');op('pack','Group spatial neighbors',f'{factor} × {factor}; packed width {packed}','plain');op('prfc1','Linear',f'{packed} to {packed}','linear');op('pract','GELU',kind='activation');op('prfc2','Linear',f'{packed} to {D}','linear')
  pr.text(20,675,'Deep-tap mergers normalize after grouping.' if deep else 'Rotary vision positions are applied before merging.',13)
 elif kind=='internvl':
  op('pack','Pixel shuffle / spatial grouping',f'2 × 2; width {packed}','plain');op('prnorm','LayerNorm',f'{packed} channels','norm');op('prfc1','Linear',f'{packed} to {D}','linear');op('pract','GELU',kind='activation');op('prfc2','Linear',f'{D} to {D}','linear')
 elif kind=='smol':
  op('pack','Pixel shuffle / spatial grouping',f'{factor} × {factor}; width {packed}','plain');op('prfc1','Linear without bias',f'{packed} to {D}','linear');pr.text(20,445,'No connector activation or hidden layer.',14)
 elif kind=='lfm':
  op('pack','Pixel unshuffle',f'2 × 2; width {packed}','plain')
  if c.get('projector_use_layernorm'):op('prnorm','LayerNorm',f'{packed} channels','norm')
  ph=c['projector_hidden_size'];op('prfc1','Linear',f'{packed} to {ph}','linear');op('pract','GELU',kind='activation');op('prfc2','Linear',f'{ph} to {D}','linear')
 chain(pr,ids);pr.text(20,825,'Output width equals the language-model embedding.',14)
 po=d.panel('post','LibreYOLO output adapter',1660,2510,510,930,kind='plain',dashed=True)
 sig(po,'reply',20,62,'O');box(po,'decode',110,'Decode generated tokens','Retain reply text','plain',x=70,w=380);po.connect('reply','decode',via=[(48,100),(260,100)])
 if task=='point':
  post=[('parse','Extract click coordinates','Bare [x,y] or parsed point','plain'),('label','Attach active query label','One requested click target','plain'),('scale','Scale to original image','0 to1 coordinates' if kind=='qwen2' else '0 to1000 coordinates','plain'),('filter','Validate and filter','Confidence / classes; at most one point','plain'),('result','Results.points','Pixel coordinates + synthetic score1','plain')]
 else:
  divisor=1000 if kind in ('qwen3','north','internvl') or (kind=='lfm' and r['size']=='3b') else 1
  post=[('parse','Parse '+('bare boxes' if kind=='north' else 'JSON boxes'),'Flatten nested boxes' if kind=='internvl' else ('Assign the queried class' if kind=='north' else 'Resolve label and box-key aliases'),'plain'),('scale','Normalize coordinates',f'Divide by {divisor}; xyxy corners','plain'),('filter','Validate / class / confidence','Synthetic per-box score1; clamp valid boxes','plain'),('dedup','Same-class deduplication','Rounded boxes and IoU filter before max_det','plain'),('result','Results.boxes','Scale xyxy to original W,H; cap max_det','plain')]
 prev='decode'
 for i,(id,label,detail,kind2) in enumerate(post):box(po,id,235+i*112,label,detail,kind2,x=70,w=380);po.connect(prev,id);prev=id
 po.text(20,875,'Scores are adapter placeholders, not calibrated confidence.',12)
 # LFM short-convolution definition uses its own extra row.
 if kind=='lfm':
  # Add enough space before provenance and family table.
  d.height+=760;d.root.set('height',str(d.height));d.root.set('viewBox',f'0 0 2200 {d.height}');next(e for e in d.root if e.tag.endswith('}rect')).set('height',str(d.height))
  conv=d.panel('shortconv','LFM gated short convolution',40,3500,2120,600,kind='conv',dashed=True,block_type='text-attn')
  conv.text(20,66,'Selected instead of attention at C entries in the layer-order list. Input/output width '+str(D)+'.',16)
  for id,x,yy,lab,detail,k in [('sc-in',30,120,'Linear in projection',f'{D} to {mul(D,3)}','linear'),('sc-split',365,120,'Split B, C, x',f'Three streams of width {D}','split'),('sc-bx',695,120,'B × x','Elementwise gating','plain'),('sc-conv',1030,120,'Depthwise causal Conv1d',f'kernel3, groups={D}; crop causal length','conv'),('sc-gate',1370,120,'C × convolution','Elementwise gating','plain'),('sc-out',1730,120,'Linear output',f'{D} to {D}','linear')]:box(conv,id,yy,lab,detail,k,x=x,w=300)
  for aa,bb in zip(['sc-in','sc-split','sc-bx','sc-conv','sc-gate'],['sc-split','sc-bx','sc-conv','sc-gate','sc-out']):conv.connect(aa,bb,from_port='right',to_port='left')
  conv.wire([(515,169),(515,265),(1520,265),(1520,169)],start='sc-split',end='sc-gate');conv.text(740,290,'C gate bypass',14)
  conv.text(20,390,'Cache keeps the causal convolution state during single-token decoding.',16)
  conv.text(20,430,'The following gated MLP uses the effective rounded width shown in the Text MLP inset.',16)
 if symbolic:
  tb=d.panel('parameters','Family parameters',40,d.height-420,2120,315)
  cols=[('Size',20),('V',200),('Lv',350),('Hv',500),('Fv',650),('D',850),('Lt',1010),('Ht / KV',1160),('dt',1270),('Ft',1360),('Vocab',1570),('Tap indices',1780)]
  for label,xx in cols:tb.text(xx,65,label,15,weight=700)
  for i,row in enumerate(family_rows):
   pp=parameters(row);values=[row['size'],pp['V'],pp['Lv'],pp['Hv'],pp['Fv'],pp['D'],pp['Lt'],str(pp['Ht'])+' / '+str(pp['Hkv']),pp['dt'],pp['Ft'],pp['Vocab'],','.join(map(str,pp['deep'])) or 'none']
   for (_,xx),value in zip(cols,values):tb.text(xx,110+38*i,str(value),15)
  tb.text(20,285,'dv = V/Hv; dt is the language attention head width. P and L are runtime sequence dimensions.',14)
 d.text(50,d.height-83,'Scope: native backend architecture from pinned config metadata; no pretrained execution. The image path is shown.',14)
 out=WEB/'public/diagrams/models'/slug;out.mkdir(parents=True,exist_ok=True)
 ident=('family' if symbolic else r['size'].replace('.','-')+'-'+task)
 if symbolic and kind=='lfm':ident='family-16-layer'
 svg=out/(ident+'.svg');d.save(svg);wrap(svg,out/(ident+'.html'))
 return dict(id=ident,label=name,task=task,size='family' if symbolic else r['size'],kind='family' if symbolic else 'concrete',svg='/diagrams/models/'+slug+'/'+ident+'.svg',html='/diagrams/models/'+slug+'/'+ident+'.html',input=str(r['input'])+' nominal; processor-dependent',verification='meta' if r.get('meta',{}).get('status')=='meta' else 'source')

if __name__ == "__main__":
 families=[args.family] if args.family else list(SPECS)
 for family in families:
  rows=[json.loads(p.read_text()) for p in CONFIG.glob(family+'-*.json')]
  # Match exact family, since the evidence directory also contains grounder variants.
  rows=[r for r in rows if r['family']==family]
  slug,title,kind,task,default=SPECS[family];rows.sort(key=lambda r:(r['size']!=default,r['size']))
  views=[build(family,r) for r in rows]
  group=[r for r in rows if kind!='lfm' or len(r['config']['text_config']['layer_types'])==16]
  if len(group)>1:views.append(build(family,group[0],symbolic=True,family_rows=group))
  out=WEB/'public/diagrams/models'/slug
  manifest=dict(family=family,slug=slug,title=title,source_revision=REV,default_view=views[0]['id'],views=views)
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  evidence={'source_revision':REV,'backend':'Transformers5.16.1 Apache-2.0 native source','families':[{k:r[k] for k in ['size','repo','resolved_revision','source','meta']} for r in rows], 'scope':'Image inference path, configuration and native module-shape verification; no weights or remote Python. Runtime token counts are explicit symbols. Adapter parser rules read from LibreYOLO source.'}
  (WEB/'scripts/model-diagrams/evidence'/(family+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')
  print(family,len(views),'views',flush=True)
