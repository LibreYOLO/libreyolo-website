"""SwinIR S/M/L, including shifted windows and all distinct reconstruction heads."""
from quicksrnet import *
CFG={'s':(60,4,6,'pixelshuffledirect','1conv'),'m':(180,6,6,'nearest+conv','1conv'),'l':(240,9,8,'nearest+conv','3conv')}

def window_attention(d,prefix,D,heads,window,x,y,w=570,h=1330):
 p=d.panel(prefix+'attention','Window attention',x,y,w,h,kind='attention',dashed=True,block_type=prefix+'att')
 token=window*window;dim=D//heads if isinstance(D,int) else 'D/A';trip=3*D if isinstance(D,int) else '3D'
 rows=[('in','Window tokens',f'{token} × {D}','plain'),('qkv','Linear QKV',f'{D} to{trip}; split into{heads} heads','linear'),('qk','Q × transpose(K)',f'Scale1/sqrt({dim}); {token} × {token} per head','attention'),('bias','Add relative position bias',f'{(2*window-1)**2} × {heads} learned table; indexed to{token}²','attention'),('mask','Add shifted-window mask','0 within region, -100 across artificial wrap boundary','attention'),('softmax','Softmax over keys','','attention'),('av','Attention weights × V',f'{heads} heads, width{dim}','attention'),('concat','Concat heads',f'{token} × {D}','concat'),('out','Output Linear',f'{D} to{D}','linear')]
 for j,(id,label,detail,kind) in enumerate(rows):op(p,prefix+id,65,65+j*132,w-130,label,detail,kind)
 chain(p,[prefix+r[0] for r in rows])
 p.text(20,h-55,'V is the value slice of the same QKV projection; it bypasses QK/bias/mask.',12)
 # Named V continuation provides a separate input to the final weighted product.
 op(p,prefix+'V',15,65+6*132+12,40,'V',h=26);p.connect(prefix+'V',prefix+'av',from_port='right',to_port='left')

def build(a,size,ev):
 sym=size=='family';D,N,A,up,res=('D','N','A','variant','variant') if sym else CFG[size]
 d=diagram(a,'SwinIR family' if sym else 'SwinIR '+size.upper(),'RGB super-resolution ×4,input3 × 64 × 64,native eval. Shapes exclude batch.','swinir',1800,4200)
 p=d.panel('main','Shallow and deep features',40,230,560,1580)
 rows=[('input','Input RGB','3 × 64 × 64; pad to window multiple8 (none here)','plain'),('norm','Subtract RGB mean','[0.4488,0.4371,0.4040]; img_range=1','norm'),('conv','Conv2d3×3',f'3 to{D};s1,p1','conv2d'),('token','Flatten + LayerNorm',f'4096 × {D}; patch_size1','norm'),('rstb',f'RSTB residual group,n={N}',f'Six Swin blocks/group; {A} heads; window8','attention'),('ln','Final LayerNorm',f'{D} channels; epsilon1e-5','norm'),('reshape','Restore feature grid',f'{D} × 64 × 64','plain'),('bodyconv','Body convolution residual module',f'{D} to{D}; '+res,'conv')]
 for j,(id,label,detail,kind) in enumerate(rows):op(p,id,75,65+j*145,410,label,detail,kind,block='rstb' if id=='rstb' else 'resconv' if id=='bodyconv' else '')
 chain(p,[r[0] for r in rows]);p.sum('bodyadd',280,1295);p.connect('bodyconv','bodyadd');p.connect('conv','bodyadd',from_port='left',to_port='left',via=[(25,379.5),(25,1295)])
 op(p,'deep','75' if False else 75,1400,410,'Deep feature for upsampling',f'{D} × 64 × 64');p.connect('bodyadd','deep')
 q=d.panel('rstbdef','Residual Swin Transformer Block',640,230,540,1580,kind='attention',dashed=True,block_type='rstb')
 rows=[('ri','Token input',f'4096 × {D}','plain'),('six','Swin transformer block,n=6','Alternating shift0,4,0,4,0,4','attention'),('ru','Reshape to image',f'{D} × 64 × 64','plain'),('rc','Residual convolution module',f'{D} to{D}; '+res,'conv'),('rt','Flatten (no LayerNorm)',f'4096 × {D}','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(q,id,75,65+j*225,390,label,detail,kind,block='swin' if id=='six' else 'resconv' if id=='rc' else '')
 chain(q,[r[0] for r in rows]);q.sum('rsum',270,1290);q.connect('rt','rsum');q.connect('ri','rsum',from_port='left',to_port='left',via=[(25,89.5),(25,1290)])
 q.text(25,1425,'No spatial downsampling or patch merging inside RSTB.',14)
 q.text(25,1480,'Initial patch embed has LayerNorm; RSTB re-embed does not.',13)
 h=d.panel('uphead','Reconstruction and output',1220,230,540,1580)
 if up=='pixelshuffledirect':
  rows=[('last','Conv2d3×3',f'{D} to48;s1,p1','conv2d'),('shuffle','PixelShuffle×4','48×64² becomes3×256²','aggregate')]
 elif up=='nearest+conv':
  rows=[('pre','Conv2d3×3',f'{D} to64;s1,p1','conv2d'),('a0','LeakyReLU','Negative slope0.01','activation'),('u1','Nearest resize×2','64 × 128 × 128','pool'),('c1','Conv2d3×3 + LeakyReLU','64 to64;s1,p1;slope0.2','conv'),('u2','Nearest resize×2','64 × 256 × 256','pool'),('c2','Conv2d3×3 + LeakyReLU','64 to64;s1,p1;slope0.2','conv'),('hr','Conv2d3×3 + LeakyReLU','64 to64;s1,p1;slope0.2','conv'),('last','Conv2d3×3','64 to3;s1,p1','conv2d')]
 else:rows=[('choice','Head selected by size','S:direct PixelShuffle; M/L:nearest+conv','aggregate')]
 for j,(id,label,detail,kind) in enumerate(rows):op(h,'head'+id,65,65+j*125,410,label,detail,kind)
 chain(h,['head'+r[0] for r in rows]);prev='head'+rows[-1][0]
 op(h,'unmean',65,1190,410,'Add RGB mean back','img_range=1;3 × 256 × 256','norm');h.connect(prev,'unmean')
 op(h,'crop',65,1340,410,'Crop to4× original input canvas','3 × 256 × 256','plain');h.connect('unmean','crop')
 h.text(20,1485,'No input-image residual in these registered SR presets.',13)
 s=d.panel('swindef','Swin transformer block',40,1870,560,1980,kind='attention',dashed=True,block_type='swin')
 rows=[('sx','Token input',f'4096 × {D}','plain'),('sn1','LayerNorm',f'{D} channels; epsilon1e-5','norm'),('roll','Cyclic shift by(-4,-4) or identity','Shift4 on odd blocks;0 on even blocks','plain'),('windows','Partition8×8 windows',f'64 windows,64 tokens/window,width{D}','split'),('att','Window attention',f'{A} heads','attention'),('reverse','Reverse windows and undo cyclic shift',f'4096 × {D}','aggregate')]
 for j,(id,label,detail,kind) in enumerate(rows):op(s,id,75,65+j*150,410,label,detail,kind,block='wa-att' if id=='att' else '')
 chain(s,[r[0] for r in rows]);s.sum('sadd1',280,1015);s.connect('reverse','sadd1');s.connect('sx','sadd1',from_port='left',to_port='left',via=[(25,89.5),(25,1015)])
 ff=2*D if isinstance(D,int) else '2D'
 for j,(id,label,detail,kind) in enumerate([('sn2','LayerNorm',f'{D} channels','norm'),('fc1','Linear MLP',f'{D} to{ff}','linear'),('gelu','GELU','','activation'),('fc2','Linear MLP',f'{ff} to{D}','linear')]):op(s,id,75,1130+j*150,410,label,detail,kind)
 chain(s,['sadd1','sn2','fc1','gelu','fc2']);s.sum('sadd2',280,1760);s.connect('fc2','sadd2');s.dot(280,1070);s.wire([(280,1070),(25,1070),(25,1760),(267,1760)],start='sadd1',end='sadd2')
 s.text(25,1900,'Dropout/DropPath are identities in eval. No absolute positions.',13)
 window_attention(d,'wa-',D,A,8,640,1870,w=540,h=1440)
 c=d.panel('resconv','Convolutional residual module',1220,1870,540,1150,kind='conv',dashed=True,block_type='resconv')
 if res=='3conv':
  rows=[('r1','Conv2d3×3','240 to60;s1,p1','conv2d'),('a1','LeakyReLU','slope0.2','activation'),('r2','Conv2d1×1','60 to60','conv2d'),('a2','LeakyReLU','slope0.2','activation'),('r3','Conv2d3×3','60 to240;s1,p1','conv2d')]
 else:rows=[('r1','Conv2d3×3',f'{D} to{D};s1,p1,bias=True','conv2d')]
 for j,(id,label,detail,kind) in enumerate(rows):op(c,'res'+id,70,65+j*155,400,label,detail,kind)
 chain(c,['res'+r[0] for r in rows])
 c.text(25,985,'Same module choice in every RSTB and after the full body.',13)
 t=d.panel('variants','Concrete configurations',1220,3080,540,770)
 t.text(25,70,'S: D60,N4,A6;24 transformer blocks.',15)
 t.text(25,120,'M: D180,N6,A6;36 transformer blocks.',15)
 t.text(25,170,'L: D240,N9,A8;54 transformer blocks.',15)
 t.text(25,270,'S/M use1conv residual; L uses3conv.',15)
 t.text(25,320,'S usesdirect PixelShuffle4.',15)
 t.text(25,370,'M/L use two nearest×2 resize/conv stages.',15)
 t.text(25,480,'All MLP hidden widths are2D:120,360,480.',15)
 t.text(25,530,'All windows are8×8 withshift4 in alternating blocks.',14)
 t.text(25,640,'Family view shows structural alternatives, not width-only scaling.',13)
 rec=ev['records'].get(size,{})
 return finish_view(a,d,'swinir','family' if sym else size+'-restore','Family with explicit head/residual alternatives' if sym else size,'restore',size,'family' if sym else 'concrete','3×64×64',rec.get('device','source'))

def main():
 a=environment('Build all SwinIR registered SR configurations')
 if a.verify:
  nn=nn_module(a,'swinir');records={}
  for size,(D,N,A,up,res) in CFG.items():records[size]=cpu_probe(nn.SwinIR(img_size=64,in_chans=3,embed_dim=D,depths=[6]*N,num_heads=[A]*N,window_size=8,mlp_ratio=2,upscale=4,upsampler=up,resi_connection=res),(1,3,64,64),['conv_first','patch_embed','layers.0','norm','conv_after_body','upsample','conv_last'])
  write_evidence(a,'swinir',records,['libreyolo/models/swinir/model.py:SWINIR_SIZE_CONFIGS and _init_model','libreyolo/models/swinir/nn.py:SwinIR,RSTB,SwinTransformerBlock,WindowAttention'],['All three64 input configurations run onCPU and produce3×256×256.','MLP ratio is2, not classdefault4. Sdirect pixelshuffle; Mnearest+conv; Lnearest+conv and3conv residual.','conv_before_upsample LeakyReLU hasdefault slope0.01; later nearest-conv activations use0.2.'])
 ev=read_evidence('swinir');views=[build(a,s,ev) for s in CFG]+[build(a,'family',ev)];manifest(a,'swinir','swinir','SwinIR',views)
if __name__=='__main__':main()
