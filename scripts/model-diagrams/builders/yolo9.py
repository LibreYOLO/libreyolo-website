"""YOLO9, YOLO9 E2E and P2: source-derived concrete and shared-topology views."""
from pathlib import Path
import argparse, os, sys, json, importlib.util, subprocess, types, math
WEBSITE=Path(__file__).resolve().parents[3]

def load_networks(src):
 for name in ['diagram_models','diagram_models.yolo9','diagram_models.yolo9_e2e','diagram_models.yolo9_p2']:
  package=types.ModuleType(name);package.__path__=[];sys.modules[name]=package
 result={}
 for family in ['yolo9','yolo9_e2e','yolo9_p2']:
  name=f'diagram_models.{family}.nn';spec=importlib.util.spec_from_file_location(name,src/f'libreyolo/models/{family}/nn.py');mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);result[family]=mod
 return result

def box(p,id,x,y,w,label,detail='',kind='plain',block='',h=46):
 return p.box(id,x,y,w,label,detail=detail,kind=kind,block_type=block,h=h,font_size=15,description=label+('. '+detail if detail else ''))
def seq(p,ids):
 for a,b in zip(ids,ids[1:]):p.connect(a,b)
def fanin(p,starts,end,ports,base_y,spacing=24):
 """Noncrossing nested left-side lanes; port order follows source order."""
 for j,start in enumerate(starts):
  sx,sy=p.port(start,'bottom');lx=35+j*34;yy=base_y-j*spacing;px=ports[j];_,ey=p.port(end,'top')
  p.wire([(sx,sy),(sx,sy+14),(lx,sy+14),(lx,yy),(px,yy),(px,ey)],start=start,end=end)
def diagram_main(args,src,networks,evidence,variant,kind='yolo9',family_group=None):
 from svg_diagram import Diagram
 symbolic=family_group is not None;size=variant;cfg=(networks['yolo9_p2'].YOLO9_P2_CONFIGS if kind=='yolo9_p2' else networks['yolo9'].YOLO9_CONFIGS)[size];p2=kind=='yolo9_p2';e2e=kind=='yolo9_e2e';n=cfg['repeat_num'];down='ADown' if cfg['down_type']=='adown' else 'AConv';first='ELAN' if cfg['first_block']=='elan' else 'RepNCSPELAN';nc=80
 prefix={'yolo9':'','yolo9_e2e':'e2e-','yolo9_p2':'p2-'}[kind];viewid=prefix+('family-ts' if symbolic else size);title='YOLO9'+(' E2E' if e2e else ' P2' if p2 else '')+(' T/S family' if symbolic else '-'+size.upper());rev=evidence['source_revision'];out=WEBSITE/'public/diagrams/models/yolov9';out.mkdir(parents=True,exist_ok=True)
 H=640;levels=[2,3,4,5] if p2 else [3,4,5];N=sum((H//(2**l))**2 for l in levels);Chead=list(cfg['head_channels']);classwidth=max(Chead[0],min(nc,100));partset=sorted(set([cfg.get('first_block_part',0)]*(first!='ELAN')+[x[2] for x in cfg['stages']]+[cfg[k][1] for k in ['neck_elan_up1','neck_elan_up2','neck_elan_down1','neck_elan_down2']]+([cfg['neck_elan_up3'][1]] if p2 else [])));partset=[p for p in partset if p]
 parts='P' if symbolic else ' / '.join(map(str,partset));halves='P/2' if symbolic else ' / '.join(str(p//2) for p in partset);quarters='P/4' if symbolic else ' / '.join(str(p//4) for p in partset);double='2P' if symbolic else ' / '.join(str(p*2) for p in partset)
 internals=['Conv',down,'RepConvN','RepNBottleneck','RepNCSPELAN','RepNCSP','SPPELAN','Decode']+(['ELAN'] if first=='ELAN' else []);extra=800 if symbolic else 0;HEIGHT=3490+extra
 d=Diagram(title,'Detection; 640 × 640 RGB; 80 classes; batch 1. Unfused PyTorch eval; tensor sizes exclude batch.',width=2340,height=HEIGHT,revision=rev,source_label=f'models/{kind}/nn.py; models/yolo9/nn.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/{kind}/nn.py',logo=WEBSITE/'public/icon-128.png')
 if symbolic:d.text(50,218,'B, N and P labels identify stages. C(stage) is output channels; P(stage) is internal part width. Values are tabulated below.',14)
 else:d.text(50,218,'Matching tensor labels continue branches between panels; all block definitions and head operations are visible.',14)
 bp=d.panel('backbone','Backbone',40,240,420,930);np=d.panel('neck','Top-down and bottom-up neck',490,240,780,930);hp=d.panel('head','One-to-one head' if e2e else 'DDetect head',1300,240,1000,930)
 dims=lambda name,c,level: (f'C({name})' if symbolic and name!='input' else str(c))+f' × {640//2**level} × {640//2**level}'
 b2=cfg['first_block_out'];b3,b4,b5=[st[1] for st in cfg['stages']];spp=cfg['spp_out']
 backbone=[('input','Input',3,0,'plain',''),('conv0','Conv 3×3, s=2',cfg['conv0_out'],1,'conv','Conv'),('conv1','Conv 3×3, s=2',cfg['conv1_out'],2,'conv','Conv'),('B2',first+' (B2)',b2,2,'aggregate',first)]
 for j,(dc,oc,pc) in enumerate(cfg['stages'],3):backbone.extend([(f'D{j}',down,dc,j,'pool',down),(f'B{j}',f'RepNCSPELAN (B{j})',oc,j,'aggregate','RepNCSPELAN')])
 backbone.append(('SPP','SPPELAN',spp,5,'spp','SPPELAN'))
 partmap={f'B{i}':cfg['stages'][i-3][2] for i in [3,4,5]};
 if first!='ELAN':partmap['B2']=cfg['first_block_part']
 for i,(id,l,c,level,k,bl) in enumerate(backbone):
  dt=dims(id,c,level);dt+=('; part='+('P('+id+')' if symbolic else str(partmap[id]))) if id in partmap else ''
  box(bp,id,55,70+i*73,310,l,dt,k,bl)
 seq(bp,[x[0] for x in backbone]);bp.text(25,908,'SPP and B2/B3/B4 feed the neck.',14)
 for id in ['SPP','B3','B4']+(['B2'] if p2 else []):
  x,y=bp.port(id,'right');bp.wire([(x,y),(395,y)],start=id,end='tensor-'+id);bp.text(370,y-8,id,12,weight=600)
 # Top-down tensor sequence follows Neck9/Neck9P2.forward exactly.
 td=[('U4','Upsample ×2',spp,4,'norm',''),('CatB4','Concat with B4',spp+b4,4,'concat',''),('N4','RepNCSPELAN (N4)',cfg['neck_elan_up1'][0],4,'aggregate','RepNCSPELAN'),('U3','Upsample ×2',cfg['neck_elan_up1'][0],3,'norm',''),('CatB3','Concat with B3',cfg['neck_elan_up1'][0]+b3,3,'concat',''),('N3' if p2 else 'P3','RepNCSPELAN ('+('N3' if p2 else 'P3')+')',cfg['neck_elan_up2'][0],3,'aggregate','RepNCSPELAN')]
 if p2:td.extend([('U2','Upsample ×2',cfg['neck_elan_up2'][0],2,'norm',''),('CatB2','Concat with B2',cfg['neck_elan_up2'][0]+b2,2,'concat',''),('P2','RepNCSPELAN (P2)',cfg['neck_elan_up3'][0],2,'aggregate','RepNCSPELAN')])
 bu=[]
 if p2:bu.extend([('ND3',down,cfg['neck_down0_out'],3,'pool',down),('CatN3','Concat with N3',cfg['neck_down0_out']+cfg['neck_elan_up2'][0],3,'concat',''),('P3','RepNCSPELAN (P3)',cfg['neck_elan_down0'][0],3,'aggregate','RepNCSPELAN')])
 bu.extend([('ND4',down,cfg['neck_down1_out'],4,'pool',down),('CatN4','Concat with N4',cfg['neck_down1_out']+cfg['neck_elan_up1'][0],4,'concat',''),('P4','RepNCSPELAN (P4)',cfg['neck_elan_down1'][0],4,'aggregate','RepNCSPELAN'),('ND5',down,cfg['neck_down2_out'],5,'pool',down),('CatSPP','Concat with SPP',cfg['neck_down2_out']+spp,5,'concat',''),('P5','RepNCSPELAN (P5)',cfg['neck_elan_down2'][0],5,'aggregate','RepNCSPELAN')])
 partmap.update({'N4':cfg['neck_elan_up1'][1],('N3' if p2 else 'P3'):cfg['neck_elan_up2'][1],'P4':cfg['neck_elan_down1'][1],'P5':cfg['neck_elan_down2'][1]})
 if p2:partmap.update({'P2':cfg['neck_elan_up3'][1],'P3':cfg['neck_elan_down0'][1]})
 for x,stages in [(70,td),(480,bu)]:
  for i,(id,l,c,level,k,bl) in enumerate(stages):
   dt=dims(id,c,level);dt+=('; part='+('P('+id+')' if symbolic else str(partmap[id]))) if id in partmap else ''
   box(np,id,x,90+i*82,270,l,dt,k,bl)
  seq(np,[v[0] for v in stages])
 last=td[-1][0];sy=np.port(last,'right')[1];np.wire([(340,sy),(397,sy),(397,60),(615,60),(615,90)],start=last,end=bu[0][0]);np.text(70,72,'SPP from backbone',13);np.wire([(205,76),(205,90)],start='SPP',end='U4')
 for dest,source in [('CatB4','B4'),('CatB3','B3')]+([('CatB2','B2')] if p2 else [])+[('CatN4','N4'),('CatSPP','SPP')]+([('CatN3','N3')] if p2 else []):
  px,py=np.port(dest,'left');np.text(px-50,py-10,source,12,weight=600);np.wire([(px-28,py),(px,py)],start=source,end=dest)
 for id in ['N4']+(['N3'] if p2 else []):
  px,py=np.port(id,'right');np.wire([(px,py),(px+32,py)],start=id,end='tensor-'+id);np.text(px+5,py-10,id,12,weight=600)
 np.text(25,882,'Nearest upsampling. Concat uses channels.',14);np.text(25,910,f'RepNCSP repeat count n = {n}.',14)
 # Every scale has separate box/class towers with exact hidden widths.
 for j,(level,c) in enumerate(zip(levels,Chead)):
  y=72+j*184;pre='H'+str(level);grid=640//2**level
  hp.text(20,y-12,f'P{level} from neck; stride {2**level}',14,weight=600)
  box(hp,pre+'in',20,y,135,'Input',dims('P'+str(level),c,level),'plain')
  for tower,ty,width,outwidth in [('box',y,64,64),('class',y+70,classwidth,80)]:
   for q in range(3):
    label='Conv 3×3' if q<2 else 'Conv2d 1×1';detail=(('C(class)' if symbolic and tower=='class' else str(width))+' ch') if q<2 else str(outwidth)+' ch';detail+=('; g=4' if tower=='box' and q>0 else '')
    box(hp,f'{pre}{tower}{q}',210+q*240,ty,210,label,detail,'conv' if q<2 else 'conv2d','Conv' if q<2 else '')
   for q in range(2):hp.connect(f'{pre}{tower}{q}',f'{pre}{tower}{q+1}',from_port='right',to_port='left')
   hp.wire([(155,y+23),(182,y+23),(182,ty+23),(210,ty+23)],start=pre+'in',end=pre+tower+'0')
  box(hp,pre+'raw',210,y+132,690,'Concat box distributions and class logits',f'144 × {grid} × {grid}; {grid*grid:,} locations','concat',h=38)
  hp.wire([(900,y+23),(966,y+23),(966,y+158),(900,y+158)],start=pre+'box2',end=pre+'raw');hp.wire([(900,y+93),(940,y+93),(940,y+138),(900,y+138)],start=pre+'class2',end=pre+'raw')
 hp.text(20,834,f'Raw location count: {N:,}. Decode gives 1 × 84 × {N:,}.',16,weight=600);hp.text(20,865,'Final 1×1 layers have bias, no normalization or activation.',15);hp.text(20,895,'Only one-to-one towers run in eval; top-K is postprocessing.' if e2e else 'Confidence filtering and NMS run after neural-network decoding.',15)
 # Recursive block diagrams with resolved widths. Every module name has a graph.
 topdefs=1220
 for idx,name in enumerate(internals):
  col=idx%3;row=idx//3;p=d.panel('def-'+name,name,40+col*770,topdefs+row*705,740,675,kind='pool' if name in ['AConv','ADown','Decode'] else 'aggregate' if name in ['ELAN','RepNCSPELAN'] else 'conv' if name=='Conv' else 'bottleneck',dashed=True,block_type=name)
  def b(id,y,l,dt='',k='plain',x=200,w=380,block=''):return box(p,name+id,x,y,w,l,dt,k,block,h=44)
  def sq(ids):seq(p,[name+i for i in ids])
  if name=='Conv':
   for id,y,l,dt,k in [('conv',90,'Conv2d','k, stride, output channels, groups given by occurrence','conv2d'),('bn',205,'BatchNorm2d','eps=0.001; momentum=0.03','norm'),('act',320,'SiLU','x × sigmoid(x)','activation')]:b(id,y,l,dt,k)
   sq(['conv','bn','act']);p.text(25,510,'Convolution has no bias. Padding is k//2 for shown odd kernels.',15);p.text(25,545,'Unmarked strides and group counts are 1.',15)
  elif name=='AConv':
   b('avg',100,'AvgPool2d','k=2, s=1, p=0','pool');b('conv',260,'Conv 3×3','s=2, p=1; output channels from occurrence','conv',block='Conv');sq(['avg','conv']);p.text(25,430,'Square sizes: 160 to 159 to 80; 80 to 79 to 40;',15);p.text(25,460,'40 to 39 to 20.'+(' P2 return: 160 to 159 to 80.' if p2 else ''),15)
   inputs=[b2,b3,b4]+([cfg['neck_elan_up3'][0]] if p2 else [])+[cfg['neck_elan_up2'][0],cfg['neck_elan_down1'][0]];outputs=[x[0] for x in cfg['stages']]+([cfg['neck_down0_out']] if p2 else [])+[cfg['neck_down1_out'],cfg['neck_down2_out']];p.text(25,520,'Input channels: '+('C(stage)' if symbolic else ' / '.join(map(str,inputs))),15);p.text(25,550,'Output channels: '+('C(down)' if symbolic else ' / '.join(map(str,outputs))),15)
  elif name=='ADown':
   b('avg',70,'AvgPool2d','k=2, s=1, p=0','pool');b('split',150,'Split channels','128 / 256 channels per half','split');b('left',290,'Conv 3×3','s=2, p=1; 128 / 256 ch','conv',x=35,w=290,block='Conv');b('pool',240,'MaxPool2d','k=3, s=2, p=1','pool',x=410,w=290);b('right',335,'Conv 1×1','128 / 256 ch','conv',x=410,w=290,block='Conv');b('cat',495,'Concat','256 / 512 output channels','concat');sq(['avg','split']);p.connect(name+'split',name+'left',from_port='left',via=[(180,172)]);p.connect(name+'split',name+'pool',from_port='right',via=[(555,172)]);sq(['pool','right']);p.wire([(180,334),(180,450),(275,450),(275,495)],start=name+'left',end=name+'cat');p.wire([(555,379),(555,474),(505,474),(505,495)],start=name+'right',end=name+'cat');p.text(25,600,'Square sizes: 160 to 80; 80 to 40; 40 to 20.',15)
  elif name=='RepConvN':
   b('input',70,'Input',quarters+' channels');b('c3',175,'Conv2d 3×3','p=1; no bias','conv2d',x=30,w=300);b('c1',175,'Conv2d 1×1','p=0; no bias','conv2d',x=410,w=300);b('bn3',290,'BatchNorm2d','eps=0.001','norm',x=30,w=300);b('bn1',290,'BatchNorm2d','eps=0.001','norm',x=410,w=300)
   p.connect(name+'input',name+'c3',via=[(390,140),(180,140)]);p.connect(name+'input',name+'c1',via=[(390,140),(560,140)]);sq(['c3','bn3']);sq(['c1','bn1']);p.sum(name+'add',390,445);p.connect(name+'bn3',name+'add',via=[(180,445)],to_port='left');p.connect(name+'bn1',name+'add',via=[(560,445)],to_port='right');b('act',525,'SiLU',quarters+' channels','activation');p.connect(name+'add',name+'act');p.text(25,630,'The optional identity BatchNorm branch is disabled.',15)
  elif name=='RepNBottleneck':
   for id,y,l,k,bl in [('in',70,'Input','plain',''),('rep',190,'RepConvN','bottleneck','RepConvN'),('conv',310,'Conv 3×3','conv','Conv')]:b(id,y,l,quarters+' channels',k,block=bl)
   sq(['in','rep','conv']);p.sum(name+'add',390,475);p.connect(name+'conv',name+'add');p.connect(name+'in',name+'add',from_port='left',via=[(75,92),(75,475)],to_port='left');p.text(80,440,'identity',14);p.text(25,590,'Equal input/output widths; the residual is enabled.',15)
  elif name=='RepNCSPELAN':
   p.text(20,63,'Part widths by occurrence: '+parts,14)
   for id,y,l,dt,k,bl in [('cv1',85,'Conv 1×1',parts+' channels','conv','Conv'),('split',150,'Split',halves+' channels per half','split',''),('csp1',215,f'RepNCSP (n={n})',halves+' channels','bottleneck','RepNCSP'),('cv2',280,'Conv 3×3',halves+' channels','conv','Conv'),('csp2',345,f'RepNCSP (n={n})',halves+' channels','bottleneck','RepNCSP'),('cv3',410,'Conv 3×3',halves+' channels','conv','Conv'),('cat',565,'Concat four inputs',double+' channels','concat',''),('cv4',620,'Conv 1×1','Output width of its stage','conv','Conv')]:b(id,y,l,dt,k,x=235,w=470,block=bl)
   sq(['cv1','split','csp1','cv2','csp2','cv3'])
   for j,(id,side,tap) in enumerate([('split','left',172),('split','bottom',205),('cv2','bottom',333),('cv3','bottom',466)]):
    sx,sy=p.port(name+id,side);xx=30+j*40;yy=550-j*24;px=260+j*130;p.wire([(sx,sy),(sx,tap),(xx,tap),(xx,yy),(px,yy),(px,565)],start=name+id,end=name+'cat')
   sq(['cat','cv4'])
  elif name=='RepNCSP':
   b('input',70,'Input',halves+' channels');b('left',160,'Conv 1×1',quarters+' channels','conv',x=30,w=300,block='Conv');b('right',160,'Conv 1×1',quarters+' channels','conv',x=410,w=300,block='Conv');p.connect(name+'input',name+'left',via=[(390,137),(180,137)]);p.connect(name+'input',name+'right',via=[(390,137),(560,137)])
   prev='left'
   for j in range(n):b('bot'+str(j),245+j*72,'RepNBottleneck',quarters+' channels','bottleneck',x=30,w=300,block='RepNBottleneck');sq([prev,'bot'+str(j)]);prev='bot'+str(j)
   b('cat',515,'Concat',halves+' channels','concat');b('out',590,'Conv 1×1',halves+' channels','conv',block='Conv');sy=p.port(name+prev,'bottom')[1];p.wire([(180,sy),(180,475),(280,475),(280,515)],start=name+prev,end=name+'cat');p.wire([(560,204),(560,495),(500,495),(500,515)],start=name+'right',end=name+'cat');sq(['cat','out'])
  elif name=='SPPELAN':
   p.text(20,63,'20 × 20 spatial grid throughout.',14)
   sh='C(SPP)/2' if symbolic else str(spp//2);sc='C(SPP)' if symbolic else str(spp)
   for id,y,l,dt,k in [('conv',85,'Conv 1×1',sh+' channels','conv'),('p1',170,'MaxPool2d',f'k=5, s=1, p=2; {sh} ch','pool'),('p2',255,'MaxPool2d',f'k=5, s=1, p=2; {sh} ch','pool'),('p3',340,'MaxPool2d',f'k=5, s=1, p=2; {sh} ch','pool'),('cat',535,'Concat four taps',('2C(SPP)' if symbolic else str(2*spp))+' channels','concat'),('out',610,'Conv 1×1',sc+' channels','conv')]:b(id,y,l,dt,k,x=235,w=470,block='Conv' if k=='conv' else '')
   sq(['conv','p1','p2','p3']);fanin(p,[name+i for i in ['conv','p1','p2','p3']],name+'cat',[260,390,520,650],510);sq(['cat','out'])
  elif name=='ELAN':
   cc='C(B2)' if symbolic else str(b2);half='C(B2)/2' if symbolic else str(b2//2)
   for id,y,l,dt,k in [('in',80,'Conv 1×1',cc+' channels','conv'),('split',155,'Split',half+' channels per half','split'),('c1',255,'Conv 3×3',half+' channels','conv'),('c2',355,'Conv 3×3',half+' channels','conv'),('cat',535,'Concat four inputs',('2C(B2)' if symbolic else str(2*b2))+' channels','concat'),('out',610,'Conv 1×1',cc+' channels','conv')]:b(id,y,l,dt,k,x=235,w=470,block='Conv' if k=='conv' else '')
   sq(['in','split','c1','c2'])
   for j,(id,side,tap) in enumerate([('split','left',177),('split','bottom',213),('c1','bottom',313),('c2','bottom',413)]):
    sx,sy=p.port(name+id,side);xx=30+j*40;yy=510-j*24;px=260+j*130;p.wire([(sx,sy),(sx,tap),(xx,tap),(xx,yy),(px,yy),(px,535)],start=name+id,end=name+'cat')
   sq(['cat','out'])
  elif name=='Decode':
   b('raw',65,'Flatten spatial axes and join scales',f'64 × {N:,} box logits; 80 × {N:,} class logits','split');b('soft',190,'Reshape; softmax over 16 bins',f'4 × 16 × {N:,}','activation',x=25,w=435);b('sig',190,'Sigmoid',f'80 × {N:,}','activation',x=520,w=190);b('mean',310,'Weighted sum over bins 0...15',f'4 × {N:,} distances','linear',x=25,w=435);b('grid',425,'Grid centers minus/plus distances','xyxy corners; scale by feature stride','plain',x=25,w=435);b('join',570,'Concat boxes and class scores',f'1 × 84 × {N:,}','concat');p.connect(name+'raw',name+'soft',via=[(390,150),(242.5,150)]);p.connect(name+'raw',name+'sig',via=[(390,150),(615,150)]);sq(['soft','mean','grid']);p.wire([(242.5,469),(242.5,525),(285,525),(285,570)],start=name+'grid',end=name+'join');p.wire([(615,234),(615,549),(495,549),(495,570)],start=name+'sig',end=name+'join');p.text(20,655,'Top-K selection follows decode.' if e2e else 'Confidence filtering and NMS follow decode.',14)
 # Every stage part width is explicitly mapped; concrete values do not rely on multipliers.
 tabley=3340
 d.text(50,tabley,('For T/S, each RepNCSPELAN part width P(stage) equals its stage output channels C(stage).' if symbolic else 'RepNCSPELAN part widths: '+', '.join(f'{key}={value}' for key,value in partmap.items())),15)
 if symbolic:
  tp=d.panel('variants','Resolved T/S family values',40,tabley+35,2260,700)
  xcolumns=[25,520,1120,1740]
  for x,txt in zip(xcolumns,['Stage or quantity','T','S','Meaning']):tp.text(x,50,txt,16,weight=600)
  rowsdata=[]
  for id,l,c,level,k,bl in backbone+td+bu:
   if id=='input':continue
   if id in [r[0] for r in rowsdata]:continue
   vals=[]
   for sz in ['t','s']:
    cf=(networks['yolo9_p2'].YOLO9_P2_CONFIGS if p2 else networks['yolo9'].YOLO9_CONFIGS)[sz]
    mp={'conv0':cf['conv0_out'],'conv1':cf['conv1_out'],'B2':cf['first_block_out'],'B3':cf['stages'][0][1],'B4':cf['stages'][1][1],'B5':cf['stages'][2][1],'D3':cf['stages'][0][0],'D4':cf['stages'][1][0],'D5':cf['stages'][2][0],'SPP':cf['spp_out'],'N4':cf['neck_elan_up1'][0],'N3':cf['neck_elan_up2'][0],'P3':cf['neck_elan_up2'][0],'P4':cf['neck_elan_down1'][0],'P5':cf['neck_elan_down2'][0]}
    mp.update({'U4':cf['spp_out'],'CatB4':cf['spp_out']+cf['stages'][1][1],'U3':cf['neck_elan_up1'][0],'CatB3':cf['neck_elan_up1'][0]+cf['stages'][0][1],'ND4':cf['neck_down1_out'],'CatN4':cf['neck_down1_out']+cf['neck_elan_up1'][0],'ND5':cf['neck_down2_out'],'CatSPP':cf['neck_down2_out']+cf['spp_out']})
    if p2:mp.update({'P2':cf['neck_elan_up3'][0],'U2':cf['neck_elan_up2'][0],'CatB2':cf['neck_elan_up2'][0]+cf['first_block_out'],'ND3':cf['neck_down0_out'],'CatN3':cf['neck_down0_out']+cf['neck_elan_up2'][0]})
    vals.append(mp[id])
   rowsdata.append((id,*vals,'output channels'))
  for j,row in enumerate(rowsdata):
   yy=80+j*20
   for x,txt in zip(xcolumns,row):tp.text(x,yy,str(txt),14)
  tp.text(25,677,'P(stage) equals output channels for these T/S blocks; class tower widths: '+('80 / 80' if p2 else '80 / 128')+'.',15)
  d.text(50,HEIGHT-92,'Shared topology: T/S only for this execution path. M and C use distinct block/downsampling choices.',14)
 else:d.text(50,HEIGHT-92,'Eval graph only. Optional PGI or dual-assignment training branches are not executed. Shape checks use random weights.',14)
 path=out/f'{viewid}.svg';d.save(path);scripts=src/'skills/libreyolo-make-diagram/scripts';subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{viewid}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));evidence.setdefault('routes',{})[viewid]=routes
 return dict(id=viewid,label=title,task='detect',size='t/s' if symbolic else size,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/yolov9/{viewid}.svg',html=f'/diagrams/models/yolov9/{viewid}.html',input='1 × 3 × 640 × 640',verification='cpu')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();sys.path.insert(0,str(src/'skills/libreyolo-make-diagram/scripts'));networks=load_networks(src)
 import torch
 torch.set_num_threads(4);ep=WEBSITE/'scripts/model-diagrams/evidence/yolo9.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];evidence=json.loads(ep.read_text()) if args.diagram_only else dict(family='yolo9',source_revision=rev,source_files=['libreyolo/models/'+x+'/nn.py' for x in networks],shapes={})
 views=[]
 for kind,klass,sizes in [('yolo9','LibreYOLO9Model','tsmc'),('yolo9_e2e','LibreYOLO9E2EModel','tsmc'),('yolo9_p2','LibreYOLO9P2Model','ts')]:
  for size in sizes:
   if not args.diagram_only:
    model=getattr(networks[kind],klass)(config=size,nb_classes=80,img_size=640).eval();obs={}
    def hook(name):
     def record(mod,ins,out):
      def shape(v):
       if isinstance(v,torch.Tensor):return list(v.shape)
       if isinstance(v,(list,tuple)):return [shape(x) for x in v]
       if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
       return None
      obs[name]={'input':shape(ins),'output':shape(out)}
     return record
    for name,mod in model.named_modules():
     if name and (name.count('.')<=1 or name.startswith(('head.cv2.','head.cv3.','head.one2one_cv2.','head.one2one_cv3.'))):mod.register_forward_hook(hook(name))
    with torch.inference_mode():y=model(torch.zeros(1,3,640,640))
    N=34000 if kind=='yolo9_p2' else 8400;assert y['predictions'].shape==(1,84,N);evidence['shapes'][kind+'-'+size]=obs
   views.append(diagram_main(args,src,networks,evidence,size,kind));print(kind,size,evidence['routes'][views[-1]['id']]['total_findings'],flush=True)
  views.append(diagram_main(args,src,networks,evidence,'t',kind,family_group=['t','s']))
 (WEBSITE/'public/diagrams/models/yolov9/manifest.json').write_text(json.dumps(dict(family='yolo9',slug='yolov9',title='YOLOv9',source_revision=rev,default_view='t',views=views),indent=2)+'\n')
 evidence.update(verification='CPU every concrete size/path, random weights',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/yolo9.py --source /path/to/libreyolo');ep.write_text(json.dumps(evidence,indent=2)+'\n');print('YOLO9 ready',len(views),'views')
if __name__=='__main__':main()
