"""ViTMatte-S guided 4-channel model with ViTDet local/global attention and trimap constraints."""
from quicksrnet import *

def main():
 a=environment('Build ViTMatte Small guided matting architecture')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'vitmatte');m=nn.LibreViTMatteModel();x=torch.rand(1,4,512,512);x[:,3,:,:170]=0;x[:,3,:,170:342]=0.5;x[:,3,:,342:]=1
  rec=cpu_probe(m,(1,4,512,512),['backbone.embeddings','backbone.encoder.layer.2','backbone','decoder.convstream','decoder.fusion_blocks.0','decoder.fusion_blocks.3','decoder.matting_head','decoder'],input_tensor=x)
  with torch.inference_mode():alpha=m(x)
  rec['trimap_constraints']={'background_exact_zero':bool((alpha[:,:,:,:170]==0).all()),'foreground_exact_one':bool((alpha[:,:,:,342:]==1).all())}
  write_evidence(a,'vitmatte',{'s':rec},['libreyolo/models/vitmatte/nn.py:constants,VitDetLayer,VitDetAttention,VitDetResBottleneckBlock,VitMatteDetailCaptureModule,constrain_alpha_to_trimap','libreyolo/models/vitmatte/model.py:INPUT_SIZES'],['CPU full512 check including background/unknown/foreground trimapstrips.','Local attention isunshifted14×14; full attention andCNN residual bottlenecks occur atzero-based2,5,8,11.','Relativeposition termsdependonqueries andseparate height/width tables; notSwin staticrelativebias.','Single sigmoid inside decoder; public callers receive alpha probabilities.'])
 ev=read_evidence('vitmatte');d=diagram(a,'ViTMatte Small','Guided alpha matting,input normalized RGB3 channels + trimap1 channel at512 × 512,native eval.','vitmatte',1900,4230)
 p=d.panel('vitdet','ViTDet-S backbone',40,230,570,1660)
 rows=[('input','RGB + trimap input','4 × 512 × 512; trimap0,0.5,1','plain'),('patch','Conv2d16×16,stride16','4 to384;384 × 32 × 32','conv2d'),('pos','Add bicubic absolute positions','Learned14×14 grid resized32×32; discardCLS slot','aggregate')]
 for j,(id,label,detail,kind) in enumerate(rows):op(p,id,65,65+j*180,440,label,detail,kind)
 chain(p,[r[0] for r in rows]);prev='pos'
 for j in range(4):
  yy=650+j*210;op(p,f'local{j}',65,yy,440,'Window transformer blocks,n=2',f'Indices{3*j},{3*j+1};384channels;6heads','attention',block='transformer');p.connect(prev,f'local{j}')
  op(p,f'global{j}',65,yy+100,440,'Global transformer + CNN bottleneck',f'Index{3*j+2};384 × 32 × 32','attention',block='global');p.connect(f'local{j}',f'global{j}');prev=f'global{j}'
 p.text(20,1565,'Window blocks pad32 to42,partition nine14×14 windows,',14);p.text(20,1595,'then crop back. There is no cyclic window shift orCLS token.',14)
 q=d.panel('detail','Convolution detail stream',650,230,570,1660)
 rows=[('raw','Input RGB + trimap','4 × 512 × 512','plain'),('c1','ConvBNReLU3×3,stride2','4 to48;256×256','conv'),('c2','ConvBNReLU3×3,stride2','48 to96;128×128','conv'),('c3','ConvBNReLU3×3,stride2','96 to192;64×64','conv')]
 for j,(id,label,detail,kind) in enumerate(rows):op(q,id,65,65+j*285,440,label,detail,kind,block='convbn')
 chain(q,[r[0] for r in rows]);q.text(25,1330,'Retain raw input plus all three detail levels as decoder skips.',14)
 q.text(25,1400,'All three convolutions usepadding1 andbias=False.',14)
 q.text(25,1470,'BatchNorm epsilon1e-5; ReLU follows normalization.',14)
 r=d.panel('fusion','Detail fusion and alpha',1260,230,600,1660)
 for j,(cin,skip,co,hw) in enumerate([(384,192,256,64),(256,96,128,128),(128,48,64,256),(64,4,32,512)]):
  yy=65+j*275
  op(r,f'up{j}',65,yy,470,'Bilinear resize×2',f'{cin} × {hw} × {hw};align_cornersFalse','pool')
  op(r,f'cat{j}',65,yy+85,470,'Concat detail skip thenupsampledfeature',f'{skip}+{cin}={skip+cin} channels','concat');r.connect(f'up{j}',f'cat{j}')
  op(r,f'fc{j}',65,yy+170,470,'ConvBNReLU3×3,stride1',f'{skip+cin} to{co};padding1','conv',block='convbn');r.connect(f'cat{j}',f'fc{j}')
  if j:r.connect(f'fc{j-1}',f'up{j}')
 op(r,'head',65,1220,470,'Matting head','Conv3×3:32to16;BN;ReLU;Conv1×1:16to1','conv',block='mattehead');r.connect('fc3','head')
 op(r,'sigmoid',65,1350,470,'Sigmoid','1 × 512 × 512 alpha probabilities','activation');r.connect('head','sigmoid')
 op(r,'known',65,1480,470,'Apply exact known trimap labels','Background0 forcesalpha0;foreground1 forcesalpha1','aggregate');r.connect('sigmoid','known')
 t=d.panel('transformer','Transformer layer',40,1950,600,1540,kind='attention',dashed=True,block_type='transformer')
 rows=[('x','Input','384channels at32×32','plain'),('ln1','LayerNorm','384channels,epsilon1e-6','norm'),('att','Attention on local windows orfull grid','Local196tokens;global1024tokens;6heads','attention')]
 for j,(id,label,detail,kind) in enumerate(rows):op(t,'t'+id,85,65+j*165,440,label,detail,kind,block='att' if id=='att' else '')
 chain(t,['tx','tln1','tatt']);t.sum('tadd1',305,640);t.connect('tatt','tadd1');t.connect('tx','tadd1',from_port='left',to_port='left',via=[(25,89.5),(25,640)])
 rows=[('ln2','LayerNorm','384channels,epsilon1e-6','norm'),('fc1','Linear','384 to1536','linear'),('gelu','GELU','','activation'),('fc2','Linear','1536 to384','linear')]
 for j,(id,label,detail,kind) in enumerate(rows):op(t,'t'+id,85,755+j*135,440,label,detail,kind)
 chain(t,['tadd1','tln2','tfc1','tgelu','tfc2']);t.sum('tadd2',305,1360);t.connect('tfc2','tadd2');t.dot(305,700);t.wire([(305,700),(25,700),(25,1360),(292,1360)],start='tadd1',end='tadd2')
 t.text(25,1470,'Global layers subsequently run the residual CNN bottleneck.',14)
 att=d.panel('att','Attention and decomposed positions',680,1950,580,1540,kind='attention',dashed=True,block_type='att')
 rows=[('qkv','LinearQKV','384 to1152;split into6 heads of64','linear'),('qk','Q × transpose(K) /8','196² local or1024² global logits/head','attention'),('relh','Add Q-dependent height relative term','einsum(Q,relative_height)','attention'),('relw','Add Q-dependent width relative term','einsum(Q,relative_width)','attention'),('soft','Softmax over keys','','attention'),('av','Weights × V','64 channels/head','attention'),('cat','Concat heads + Linear','384 to384','linear')]
 for j,(id,label,detail,kind) in enumerate(rows):op(att,id,70,65+j*175,440,label,detail,kind)
 chain(att,[r[0] for r in rows]);att.text(25,1360,'Each relative table is27×64 for14×14 window attention;',14);att.text(25,1400,'63×64 for32×32 global attention. Tables resize if needed.',14)
 c=d.panel('global','CNN residual after each global layer',1300,1950,560,1540,kind='bottleneck',dashed=True,block_type='global')
 rows=[('in','Global transformer output','384 × 32 × 32','plain'),('c1','Conv1×1','384 to192;biasFalse','conv2d'),('n1','ChannelLayerNorm + GELU','192channels,epsilon1e-6','norm'),('c2','Conv3×3','192 to192;padding1,biasFalse','conv2d'),('n2','ChannelLayerNorm + GELU','192channels,epsilon1e-6','norm'),('c3','Conv1×1','192 to384;biasFalse','conv2d'),('n3','ChannelLayerNorm','384channels,epsilon1e-6','norm')]
 for j,(id,label,detail,kind) in enumerate(rows):op(c,'b'+id,75,65+j*155,410,label,detail,kind)
 chain(c,['b'+r[0] for r in rows]);c.sum('badd',280,1270);c.connect('bn3','badd');c.connect('bin','badd',from_port='left',to_port='left',via=[(25,89.5),(25,1270)])
 c.text(25,1430,'No additional finalbackbone normalization after layer11.',13)
 z=d.panel('convbn','Convolutional primitives',40,3550,1820,510,kind='conv',dashed=True,block_type='convbn')
 for j,(id,label,kind) in enumerate([('conv','Conv2d3×3','conv2d'),('bn','BatchNorm2d','norm'),('relu','ReLU','activation')]):op(z,id+'primitive',45+j*570,75,500,label,'Numericchannels/stride at occurrence' if j==0 else '',kind)
 z.connect('convprimitive','bnprimitive',from_port='right',to_port='left');z.connect('bnprimitive','reluprimitive',from_port='right',to_port='left')
 z.text(25,250,'Matting head differs:Conv3×3(32to16,biasTrue),BatchNorm16,ReLU,Conv1×1(16to1,biasTrue).',16)
 z.text(25,315,'Known trimap pixels are constrained after the single decoder sigmoid; unknown0.5 pixels keep predictedalpha.',15)
 rec=ev['records'].get('s',{});view=finish_view(a,d,'vitmatte','s-matte','Small','matte','s','concrete','normalizedRGB3 + trimap1,512×512',rec.get('device','source'));manifest(a,'vitmatte','vitmatte','ViTMatte',[view])
if __name__=='__main__':main()
