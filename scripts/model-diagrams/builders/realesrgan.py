"""Real-ESRGAN RRDBNet x4/x2 and structurally distinct SRVGG x4t."""
from quicksrnet import *

def rrdb_view(a,size,verification):
 symbolic=size=='family';u='U' if symbolic else 2 if size=='x2' else 1
 h='64/U' if symbolic else 64//u
 cin='3U²' if symbolic else 3*u*u
 outs='256/U' if symbolic else 256//u
 d=diagram(a,'Real-ESRGAN RRDBNet family' if symbolic else 'Real-ESRGAN '+size,'RGB super-resolution, native eval, 3 × 64 × 64 input. Shapes exclude batch.','realesrgan',1900,2050)
 p=d.panel('net','RRDBNet generator',40,230,590,1640)
 rows=[('input','Input image','3 × 64 × 64','plain'),('unshuffle','PixelUnshuffle '+str(u) if u!=1 else 'Identity input',f'{cin} × {h} × {h}','aggregate'),('first','Conv2d 3×3',f'{cin} to 64; s=1, p=1','conv2d'),('body','RRDB, n=23',f'64 × {h} × {h}','bottleneck'),('bodyconv','Conv2d 3×3','64 to 64; s=1, p=1','conv2d')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,105,65+i*100,380,label,detail,kind,block='rrdb' if id=='body' else '')
 chain(p,[r[0] for r in rows]);p.sum('trunksum',295,590);p.connect('bodyconv','trunksum')
 p.connect('first','trunksum',from_port='left',to_port='left',via=[(45,289.5),(45,590)])
 for j in range(2):
  sy=655+255*j;hw=f'{128*2**j}/U' if symbolic else 128*2**j//u
  op(p,f'up{j}',105,sy,380,'Nearest-neighbor resize ×2',f'64 × {hw} × {hw}','pool')
  op(p,f'uc{j}',105,sy+80,380,'Conv2d 3×3','64 to 64; s=1, p=1','conv2d')
  op(p,f'ua{j}',105,sy+160,380,'LeakyReLU','Negative slope 0.2','activation')
  p.connect('trunksum' if j==0 else 'ua0',f'up{j}');chain(p,[f'up{j}',f'uc{j}',f'ua{j}'])
 for i,(id,label,detail,kind) in enumerate([('hr','Conv2d 3×3','64 to 64; s=1, p=1','conv2d'),('ha','LeakyReLU','Negative slope 0.2','activation'),('last','Conv2d 3×3','64 to 3; s=1, p=1','conv2d'),('out','Restored RGB',f'3 × {outs} × {outs}','plain')]):op(p,id,105,1165+i*90,380,label,detail,kind)
 chain(p,['ua1','hr','ha','last','out'])
 p.text(25,1573,'All convolutions have bias; no normalization layers.',14)
 q=d.panel('rrdbdef','RRDB: residual in residual',670,230,470,800,kind='bottleneck',dashed=True,block_type='rrdb')
 for i,(id,label,detail,kind) in enumerate([('rx','Input','64 channels','plain'),('r1','ResidualDenseBlock','64 channels','bottleneck'),('r2','ResidualDenseBlock','64 channels','bottleneck'),('r3','ResidualDenseBlock','64 channels','bottleneck'),('scale','Multiply 0.2','Residual scaling','linear')]):op(q,id,85,65+i*110,300,label,detail,kind,block='rdb' if id.startswith('r') and id!='rx' else '')
 chain(q,['rx','r1','r2','r3','scale']);q.sum('radd',235,665);q.connect('scale','radd');q.connect('rx','radd',from_port='left',to_port='left',via=[(30,89.5),(30,665)])
 q.text(25,748,'Each inner dense block also scales its residual by 0.2.',13)
 q=d.panel('rearrange','Input rearrangement and variants',670,1070,470,800,kind='aggregate',dashed=True)
 op(q,'ur',70,65,330,'PixelUnshuffle ×2 (x2 only)','3 × 64 × 64 becomes 12 × 32 × 32','aggregate')
 op(q,'reshape',70,165,330,'Reshape and permute','2×2 spatial positions become channels','split');q.connect('ur','reshape')
 q.text(25,330,'x4: U=1, no unshuffle; trunk at 64×64.',14)
 q.text(25,370,'x2: U=2, pixel unshuffle; trunk at 32×32.',14)
 q.text(25,430,'Both use 23 RRDBs, width 64, growth width 32.',14)
 q.text(25,470,'Both execute two nearest-neighbor ×2 upsamplings.',14)
 q.text(25,530,'Net upscale = 4/U; output is 256/U pixels per side.',14)
 q.text(25,605,'x4t uses SRVGG, shown in a separate concrete view.',14)
 q.text(25,706,'Network output is not clamped by RRDBNet.forward.',14)
 r=d.panel('rdbdef','ResidualDenseBlock',1180,230,680,1640,kind='bottleneck',dashed=True,block_type='rdb')
 r.text(25,62,'X is the 64-channel input. F1...F4 each have 32 channels.',14)
 # Every dense input has its own visible continuation port.
 for j in range(5):
  yy=120+j*235;names=['X']+[f'F{i}' for i in range(1,j+1)];inc=64+32*j
  for i,name in enumerate(names):
   tx=165+i*75
   op(r,f'dense{j}-in{i}',tx,yy,55,name,kind='plain',h=26,block='signal-'+name)
   r.wire([(tx+27.5,yy+26),(tx+27.5,yy+60)],start=f'dense{j}-in{i}',end=f'cat{j}')
  op(r,f'cat{j}',140,yy+60,470,'Identity X' if j==0 else 'Concat channel inputs',f'{inc} channels','plain' if j==0 else 'concat')
  op(r,f'conv{j}',140,yy+132,470,'Conv2d 3×3',f'{inc} to {32 if j<4 else 64}; s=1, p=1','conv2d');r.connect(f'cat{j}',f'conv{j}')
  if j<4:
   op(r,f'act{j}',140,yy+202,305,'LeakyReLU, slope 0.2',kind='activation',h=30)
   r.connect(f'conv{j}',f'act{j}',via=[(375,yy+190),(292.5,yy+190)])
   op(r,f'F{j+1}',520,yy+204,65,f'F{j+1}',kind='plain',h=26,block='signal-F'+str(j+1));r.connect(f'act{j}',f'F{j+1}',from_port='right',to_port='left')
 op(r,'dense-scale',140,1350,470,'Multiply 0.2','64-channel dense residual','linear');r.connect('conv4','dense-scale')
 r.sum('dense-add',375,1460);r.connect('dense-scale','dense-add')
 op(r,'dense-X',140,1447,55,'X',kind='plain',h=26,block='signal-X');r.connect('dense-X','dense-add',from_port='right',to_port='left')
 r.text(25,1560,'Named X/F connectors denote the same tensor at each use.',14)
 d.text(50,1940,'Inference generator only. GAN discriminator, degradation synthesis and adversarial training are outside this graph.',15)
 return finish_view(a,d,'real-esrgan','rrdb-family' if symbolic else size+'-restore','RRDBNet shared topology' if symbolic else size,'restore',size,'family' if symbolic else 'concrete','3×64×64',verification)

def srvgg_view(a,verification):
 d=diagram(a,'Real-ESRGAN x4t','SRVGGNetCompact RGB super-resolution, native eval. Input 3 × 64 × 64. Shapes exclude batch.','realesrgan',1400,1460)
 p=d.panel('srvgg','Generator',40,230,660,1090)
 rows=[('input','RGB input','3 × 64 × 64','plain'),('stem','Conv2d 3×3','3 to 64; stride 1, padding 1','conv2d'),('act','PReLU','64 learned activation parameters','activation'),('body','Conv/PReLU block, n=32','64 × 64 × 64','conv'),('last','Conv2d 3×3','64 to 48; stride 1, padding 1','conv2d'),('shuffle','PixelShuffle ×4','48 × 64 × 64 becomes 3 × 256 × 256','aggregate')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,150,65+i*100,420,label,detail,kind,block='srvggblock' if id=='body' else '')
 chain(p,[n[0] for n in rows]);p.sum('add',360,760);p.connect('shuffle','add')
 op(p,'nearest',15,655,245,'Nearest resize input ×4','3 × 256 × 256','pool')
 p.connect('input','nearest',from_port='left',to_port='top',via=[(85,89.5),(85,630),(137.5,630)]);p.connect('nearest','add',via=[(137.5,760)],to_port='left')
 op(p,'out',150,860,420,'Restored RGB','3 × 256 × 256','plain');p.connect('add','out')
 p.text(25,1013,'Residual is added after pixel shuffle; no output clamp here.',14)
 q=d.panel('srvggdef','Repeated Conv/PReLU block',740,230,620,650,kind='conv',dashed=True,block_type='srvggblock')
 for i,(id,label,detail,kind) in enumerate([('bi','Input','64 × 64 × 64','plain'),('bc','Conv2d 3×3','64 to 64; stride 1, padding 1, bias=True','conv2d'),('ba','PReLU','One learned slope per channel (64)','activation'),('bo','Output','64 × 64 × 64','plain')]):op(q,id,85,80+i*130,450,label,detail,kind)
 chain(q,['bi','bc','ba','bo'])
 q=d.panel('scope','Concrete configuration',740,930,620,390)
 q.text(25,75,'32 intermediate convolutions, one stem and one output conv.',15)
 q.text(25,121,'All feature convolutions have 64 channels and use bias.',15)
 q.text(25,167,'Final convolution produces 3 × 4² = 48 channels.',15)
 q.text(25,213,'PixelShuffle rearranges channels into four spatial offsets',15)
 q.text(25,241,'per axis, producing a 256×256 RGB image.',15)
 q.text(25,305,'This topology is structurally different from x4/x2 RRDBNet.',15)
 return finish_view(a,d,'real-esrgan','x4t-restore','x4t (SRVGG)','restore','x4t','concrete','3×64×64',verification)

def main():
 a=environment('Build all Real-ESRGAN registered sizes')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'realesrgan');records={}
  for size,scale in [('x4',4),('x2',2)]:records[size]=cpu_probe(nn.RRDBNet(scale=scale,num_feat=64,num_block=23,num_grow_ch=32),(1,3,64,64),['conv_first','body','conv_body','conv_up1','conv_up2','conv_last'])
  records['x4t']=cpu_probe(nn.SRVGGNetCompact(num_feat=64,num_conv=32,upscale=4,act_type='prelu'),(1,3,64,64),['body.0','body.66','upsampler'])
  write_evidence(a,'realesrgan',records,['libreyolo/models/realesrgan/model.py:REALESRGAN_SIZE_CONFIGS','libreyolo/models/realesrgan/nn.py:RRDBNet, RRDB, ResidualDenseBlock, SRVGGNetCompact'],['Input 64×64 is the configured representative; inference may accept native-resolution images.','x4/x2 share RRDB trunk but have explicit differing input rearrangement. x4t is separately drawn SRVGG topology.'])
 ev=read_evidence('realesrgan');v=lambda k:'cpu' if k in ev['records'] else 'source'
 views=[rrdb_view(a,k,v(k)) for k in ['x4','x2']]+[srvgg_view(a,v('x4t')),rrdb_view(a,'family','source')];manifest(a,'realesrgan','real-esrgan','Real-ESRGAN',views)
if __name__=='__main__':main()
