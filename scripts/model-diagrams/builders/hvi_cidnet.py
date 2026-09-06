"""HVI-CIDNet actual color/intensity streams, including overwritten intermediate path."""
from quicksrnet import *
from nafnet import product

def main():
 a=environment('Build HVI-CIDNet tiny native graph')
 if a.verify:
  nn=nn_module(a,'hvi_cidnet');record=cpu_probe(nn.CIDNet(),(1,3,256,256),['HVE_block0','IE_block0','HVE_block1','HVE_block2','HVE_block3','I_LCA2','HV_LCA2','I_LCA5','HVD_block3','ID_block2','HVD_block0','ID_block0'],input_range=(0,1))
  write_evidence(a,'hvi_cidnet',{'t':record},['libreyolo/models/hvi_cidnet/nn.py:CIDNet.forward, HV_LCA, I_LCA, CAB, IEL, RGB_HVI','libreyolo/models/hvi_cidnet/model.py:INPUT_SIZES'],['CPU actual256 forward. Defaultchannels36,36,72,144; heads2,4,8; noNormDownsample/NormUpsample LayerNorm.','LCA2 results feed skip2 only; deepest downsamplers read their pre-LCA2 inputs.','I_LCA5 executes but its output is overwritten by ID_block2(i_dec3,...); this source quirk is explicitly shown, not repaired.','HV_LCA has no final IEL residual; I_LCA does. Color stem consumes full3-channelHVI, not just2-channelHV.'])
 ev=read_evidence('hvi_cidnet');d=diagram(a,'HVI-CIDNet Tiny','Low-light RGB restoration,input3 × 256 × 256,native eval. Shapes exclude batch.','hvi_cidnet',1900,4990)
 t=d.panel('color','Learned HVI color transform',40,230,590,2020)
 rows=[('rgb','Input RGB','3 × 256 × 256','plain'),('hsv','Channel max/min and piecewise hue','I=max(R,G,B); S=(max-min)/(max+1e-8)','aggregate'),('density','Color-sensitive factor Ck','(sin(Iπ/2)+1e-8)^k; learnedk initially0.2','activation'),('hvi','Concat horizontal,vertical,intensity','H=Ck*S*cos(2πh); V=Ck*S*sin(2πh); I','concat')]
 for i,(id,label,detail,kind) in enumerate(rows):op(t,id,65,65+i*200,460,label,detail,kind)
 chain(t,[r[0] for r in rows]);t.text(25,915,'Hue is HSV sector hue from the maximal RGB channel.',14)
 t.text(25,955,'For achromatic pixels h=0; for I=0,saturation=0.',14)
 t.text(25,1020,'HVI has3 channels. Intensity stream selects channel2.',14)
 op(t,'residualout',65,1130,460,'Concat predicted HV andI residuals','2 channels +1 channel =3 × 256 × 256','concat')
 t.sum('hviadd',295,1310);t.connect('residualout','hviadd');t.connect('hvi','hviadd',from_port='left',to_port='left',via=[(25,689.5),(25,1310)])
 op(t,'inverse',65,1420,460,'PHVIT inverse color transform','Clamp,HVI-to-HSV reconstruction,HSV-to-RGB','aggregate',block='phvit');t.connect('hviadd','inverse')
 op(t,'output',65,1620,460,'Restored RGB','3 × 256 × 256','plain');t.connect('inverse','output')
 t.text(25,1840,'Default saturation_scale=intensity_scale=1.',14)
 # Explicit source-accurate two stream paths, with named contexts and skip taps.
 for intensity,x in [(False,670),(True,1285)]:
  prefix='I' if intensity else 'HV';p=d.panel(prefix+'stream',prefix+' encoder and decoder',x,230,575,2020)
  rows=[('stem','ReplicationPad1;Conv3×3',f'{1 if intensity else 3} to36;256²; skipJ0','conv'),('down1','Downsample1','36 to36;128²; paired contextLCA1','aggregate'),('lca1',prefix+'_LCA1','36 channels,2 heads; resultskipJ1','attention'),('down2','Downsample2','36 to72;64²; paired contextLCA2','aggregate'),('lca2',prefix+'_LCA2 (skip-only)','72 channels,4 heads; resultskipJ2','attention'),('down3','Downsample3 from pre-LCA2 feature','72 to144;32²','aggregate'),('lca3',prefix+'_LCA3','144 channels,8 heads','attention'),('lca4',prefix+'_LCA4','144 channels,8 heads; usespairedLCA3 outputs','attention'),('up3','Upsample3 + skipJ2','144 to72;64²','aggregate'),('lca5',prefix+'_LCA5'+(' (result unused)' if intensity else ''),'72 channels,4 heads','attention'),('up2','Upsample2 + skipJ1','72 to36;128²; '+('reads pre-LCA5 feature' if intensity else 'reads LCA5 output'),'aggregate'),('lca6',prefix+'_LCA6','36 channels,2 heads','attention'),('up1','Upsample1 + skipJ0','36 to36;256²','aggregate'),('end','ReplicationPad1;Conv3×3',f'36 to{1 if intensity else 2};256²','conv')]
  prev=None
  for j,(id,label,detail,kind) in enumerate(rows):
   yy=65+j*132;nid=prefix+id;op(p,nid,65,yy,455,label,detail,kind,block='lca' if id.startswith('lca') else 'down' if id.startswith('down') else 'up' if id.startswith('up') else '')
   if prev and id!='down3' and not(intensity and id=='up2'):p.connect(prev,nid)
   if id=='down3':p.connect(prefix+'down2',nid,from_port='right',to_port='right',via=[(550,485.5),(550,yy+24.5)])
   if intensity and id=='up2':p.connect(prefix+'up3',nid,from_port='right',to_port='right',via=[(550,1145.5),(550,yy+24.5)])
   prev=nid
  p.text(20,1960,'Contexts are the paired stream features before each LCA update.',12)
 # Cross-attention details.
 c=d.panel('cab','CAB channel cross-attention',40,2320,590,1480,kind='attention',dashed=True,block_type='cab')
 c.text(20,65,'C/heads:36/2,72/4,144/8. Every head has18 channels.',14)
 for name,x in [('Q',20),('KV',320)]:
  op(c,name+'proj',x,135,250,'Conv2d1×1 from'+('X' if name=='Q' else 'context'), '36/72/144 to'+('same' if name=='Q' else '72/144/288'),'conv2d')
  op(c,name+'dw',x,265,250,'DepthwiseConv3×3','Groups=outputchannels;s1,p1','conv2d');c.connect(name+'proj',name+'dw')
 op(c,'qknorm',65,430,460,'Split K,V; L2-normalize Q and K','Normalize over spatial length, not channels','norm')
 c.wire([(145,314),(145,385),(180,385),(180,430)],start='Qdw',end='qknorm');c.wire([(445,314),(445,385),(410,385),(410,430)],start='KVdw',end='qknorm')
 op(c,'qk',65,580,460,'Q × transpose(K) × learned temperature','18 × 18 channel-attention matrix per head','attention');c.connect('qknorm','qk')
 op(c,'soft',65,730,460,'Softmax over key channels','18 × 18 perhead','attention');c.connect('qk','soft')
 op(c,'val',65,880,460,'Attention × V','Restore36/72/144 spatial channels','attention');c.connect('soft','val')
 op(c,'project',65,1030,460,'Conv2d1×1','36/72/144 to samechannels','conv2d');c.connect('val','project')
 c.text(25,1210,'V is the unchanged value half of the KV depthwise projection.',13)
 c.text(25,1260,'All CAB convolutions use bias=False.',14)
 l=d.panel('lca','LCA residual structure',670,2320,590,1480,kind='attention',dashed=True,block_type='lca')
 op(l,'x',65,65,460,'Feature X and paired context','36,72 or144 channels')
 op(l,'norm',65,210,460,'Shared channel LayerNorm on both','epsilon1e-6; own module per LCA','norm');l.connect('x','norm')
 op(l,'cabop',65,365,460,'CAB','Context supplies K/V; X supplies Q','attention',block='cab');l.connect('norm','cabop');l.sum('lcaadd',295,540);l.connect('cabop','lcaadd');l.connect('x','lcaadd',from_port='left',to_port='left',via=[(20,89.5),(20,540)])
 op(l,'ln2',65,660,460,'Channel LayerNorm','Same LCA norm module','norm');l.connect('lcaadd','ln2')
 op(l,'iel',65,810,460,'IEL intensity-enhancement layer','Hidden widths95,191,383 respectively','aggregate',block='iel');l.connect('ln2','iel')
 op(l,'hvout',20,1035,250,'HV_LCA output','IEL result; no secondskip')
 l.sum('iadd',445,1059.5);l.wire([(295,859),(295,980)],start='iel',arrow=False);l.wire([(295,980),(145,980),(145,1035)],start='iel',end='hvout');l.wire([(295,980),(445,980),(445,1046.5)],start='iel',end='iadd');l.dot(295,980)
 l.dot(295,605);l.wire([(295,605),(560,605),(560,1059.5),(458,1059.5)],start='lcaadd',end='iadd')
 l.text(330,1130,'I_LCA output',14);l.text(25,1260,'HV and I differ in the second residual connection.',14)
 i=d.panel('ieldef','IEL',1300,2320,560,1480,kind='aggregate',dashed=True,block_type='iel')
 op(i,'inproj',65,65,430,'Conv2d1×1','36/72/144 to190/382/766','conv2d')
 op(i,'dw',65,205,430,'DepthwiseConv3×3','190/382/766 channels/groups;s1,p1','conv2d');i.connect('inproj','dw')
 op(i,'split',65,345,430,'Split equal halves','Each half95/191/383 channels','split');i.connect('dw','split')
 for side,x in [('left',15),('right',300)]:
  op(i,side+'dw',x,500,245,'DepthwiseConv3×3','95/191/383 groups;s1,p1','conv2d')
  i.wire([(170 if side=='left' else 390,394),(170 if side=='left' else 390,455),(x+122.5,455),(x+122.5,500)],start='split',end=side+'dw')
  op(i,side+'tanh',x,645,245,'Tanh','','activation');i.connect(side+'dw',side+'tanh');i.sum(side+'add',x+122.5,840);i.connect(side+'tanh',side+'add')
  op(i,side+'identity',x,750,65,'Half',h=26);i.connect(side+'identity',side+'add',via=[(x+32.5,840)],to_port='left')
 product(i,'ielmul',280,1040);i.wire([(137.5,853),(137.5,1040),(267,1040)],start='leftadd',end='ielmul');i.wire([(422.5,853),(422.5,1040),(293,1040)],start='rightadd',end='ielmul')
 op(i,'outproj',65,1190,430,'Conv2d1×1','95/191/383 to36/72/144','conv2d');i.connect('ielmul','outproj')
 i.text(20,1370,'hidden=floor(2.66*C); two residual Tanh branches multiply.',13)
 # Spatial and inverse-color primitives.
 u=d.panel('sampling','Downsample and upsample',40,3860,890,910,kind='conv',dashed=True)
 steps=[('downc','Downsample:Conv3×3','36to36,36to72,72to144;s1,p1;biasFalse','conv2d'),('downr','Bilinear resize×0.5','align_corners=True','pool'),('downa','PReLU','One learned slope; no optionalLayerNorm','activation')]
 for j,(id,label,detail,kind) in enumerate(steps):op(u,id,25,65+j*180,390,label,detail,kind)
 chain(u,[s[0] for s in steps]);steps=[('upc','Upsample:Conv3×3','144to72,72to36,36to36;s1,p1','conv2d'),('upr','Bilinear resize×2','align_corners=True','pool'),('upcat','Concat with matching skip','144,72,72 channels respectively','concat'),('upmix','Conv1×1 thenPReLU','144to72,72to36,72to36; noLayerNorm','conv')]
 for j,(id,label,detail,kind) in enumerate(steps):op(u,id,475,65+j*180,390,label,detail,kind)
 chain(u,[s[0] for s in steps]);u.text(25,825,'All sampling convolutions have bias=False; nearest-neighbor upsampling is not used.',14)
 inv=d.panel('inverse-color','PHVIT inverse transform',970,3860,890,910,kind='aggregate',dashed=True,block_type='phvit')
 rows=[('clamp','Clamp H,V to[-1,1], I to[0,1]','Recompute Ck from I and stored density k','activation'),('divide','Divide H,V by Ck+1e-8','Clamp normalized H,V again to[-1,1]','linear'),('hue','atan2(V+eps,H+eps)/(2π) modulo1','Hue; saturation=sqrt(H²+V²+eps),clamped[0,1]','activation'),('sector','HSV sector conversion','Six sector choices using I, I(1-S), q andt','aggregate'),('rgbout','RGB output × intensity_scale','Default intensity_scale=saturation_scale=1','plain')]
 for j,(id,label,detail,kind) in enumerate(rows):op(inv,id,85,65+j*155,720,label,detail,kind)
 chain(inv,[s[0] for s in rows]);inv.text(25,860,'The inverse uses the same learned density k captured during the forward HVI transform.',14)
 rec=ev['records'].get('t',{});view=finish_view(a,d,'hvi-cidnet','t-restore','Tiny','restore','t','concrete','3×256×256',rec.get('device','source'));manifest(a,'hvi_cidnet','hvi-cidnet','HVI-CIDNet',[view])
if __name__=='__main__':main()
