"""EfficientDet D0-D4, native EfficientNet/BiFPN/head geometry."""
from pathlib import Path
import argparse,os,sys,json,subprocess,math
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]
def config_values(nn,cfg):
 ci=nn.round_channels(32,cfg.channel_multiplier);stem=ci;rows=[];div=2
 for typ,repeats,k,s,exp,base,se in nn._BACKBONE_STAGES:
  co=nn.round_channels(base,cfg.channel_multiplier);n=math.ceil(repeats*cfg.depth_multiplier);mid=ci if typ=='ds' else nn.make_divisible(ci*exp);later=co if typ=='ds' else nn.make_divisible(co*exp);reduced=max(1,round(mid*(se/exp)));reduced_later=max(1,round(later*(se/exp)));div*=s;rows.append(dict(type=typ,n=n,k=k,s=s,ci=ci,co=co,mid=mid,later=later,se=reduced,se_later=reduced_later,div=div));ci=co
 return stem,rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.efficientdet import nn as nn
 from libreyolo.models.efficientdet.config import SCALE_CONFIGS
 from libreyolo.postprocess.efficientdet import generate_anchors
 import torch
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/efficientdet';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/efficientdet.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];ev=json.loads(ep.read_text()) if args.diagram_only else dict(family='efficientdet',source_revision=rev,source_files=['libreyolo/models/efficientdet/nn.py','libreyolo/models/efficientdet/config.py','libreyolo/postprocess/efficientdet.py'],shapes={});views=[]
 for size in [*SCALE_CONFIGS,'family']:
  symbolic=size=='family';sz='d0' if symbolic else size;cfg=SCALE_CONFIGS[sz];H=cfg.image_size;F=cfg.fpn_channels;R=cfg.fpn_repeats;T=cfg.head_repeats;stem,rows=config_values(nn,cfg);grids=[H//2**l for l in range(3,8)];N=sum(x*x for x in grids)*9;f='F' if symbolic else str(F)
  if not symbolic and not args.diagram_only:
   model=nn.LibreEfficientDetModel(size,num_classes=90).eval();obs={}
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
    if name and (name.count('.')<=3 or name.startswith('fpn.cell.0.fnode.')):mod.register_forward_hook(hook(name))
   with torch.inference_mode():cl,bo=model(torch.zeros(1,3,H,H))
   assert [list(x.shape) for x in cl]==[[1,810,g,g] for g in grids];assert [list(x.shape) for x in bo]==[[1,36,g,g] for g in grids];assert generate_anchors(H).shape==(N,4);ev['shapes'][size]=obs
  d=Diagram('EfficientDet D0-D4 family' if symbolic else 'EfficientDet '+size.upper(),('Symbolic H/F/R/T and stage values are defined below.' if symbolic else f'{H} × {H} RGB; batch 1; unfused eval.')+' 90 sparse class slots map to 80 COCO classes.',width=2580,height=3620+(610 if symbolic else 0),revision=rev,source_label='models/efficientdet/nn.py; models/efficientdet/config.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/efficientdet/nn.py',logo=WEBSITE/'public/icon-128.png')
  b=d.panel('backbone','EfficientNet feature backbone',40,230,610,1150);n=d.panel('bifpn','Bidirectional feature pyramid cell',680,230,1150,1150);h=d.panel('heads','Class and box towers',1860,230,680,1150)
  box(b,'input',40,65,530,'Input','3 × H × H' if symbolic else f'3 × {H} × {H}');box(b,'stem',40,160,530,'Conv 3×3 s=2; BN; SiLU',('Cstem × H/2 square' if symbolic else f'{stem} × {H//2} × {H//2}'),'conv','ConvBNSiLU');b.connect('input','stem');prev='stem'
  for j,row in enumerate(rows):
   label=f'S{j}: '+('DepthwiseSeparable' if row['type']=='ds' else 'MBConv')+f' k={row["k"]}, s={row["s"]}, n='+('n'+str(j) if symbolic else str(row['n']));dt=(f'C{j} × H/{row["div"]} square; E{j}, SE{j}' if symbolic else f"{row['co']} × {H//row['div']} × {H//row['div']}; E={row['mid']}{'/'+str(row['later']) if row['n']>1 else ''}; SE={row['se']}{'/'+str(row['se_later']) if row['n']>1 else ''}")
   box(b,'S'+str(j),40,265+j*115,530,label,dt,'aggregate','MBConv' if row['type']=='ir' else 'DSConv');b.connect(prev,'S'+str(j));prev='S'+str(j)
  b.text(20,1100,'E/SE pairs give first/later widths only when n>1.',14);b.text(20,1130,'S2, S4 and S6 provide C3, C4 and C5.',14)
  # Eight-node cell: exact offsets from _bifpn_nodes(). Original inputs and
  # interior skip tensors use matching continuation labels where they cross.
  td=[('I6',6,2),('I5',5,2),('I4',4,2),('P3',3,2)];bu=[('P4',4,3),('P5',5,3),('P6',6,3),('P7',7,2)]
  for j,(id,level,inputs) in enumerate(td):box(n,id,130,105+j*225,370,'Fnode '+id, f'{f} × '+(f'H/{2**level} square' if symbolic else f'{H//2**level} × {H//2**level}')+f'; {inputs} inputs','aggregate','Fnode')
  for j,(id,level,inputs) in enumerate(bu):box(n,id,715,780-j*225,370,'Fnode '+id, f'{f} × '+(f'H/{2**level} square' if symbolic else f'{H//2**level} × {H//2**level}')+f'; {inputs} inputs','aggregate','Fnode')
  seq(n,[x[0] for x in td]);n.connect('P3','P4',from_port='right',to_port='left')
  for a,z in [('P4','P5'),('P5','P6'),('P6','P7')]:n.connect(a,z,from_port='top',to_port='bottom')
  for dest,source in [('I6','in6'),('I5','in5'),('I4','in4'),('P3','in3')]:
   px,py=n.port(dest,'left');n.text(25,py+5,source,13,weight=600);n.wire([(80,py),(px,py)],start=source,end=dest)
  n.text(200,80,'in7',13,weight=600);n.wire([(315,83),(315,105)],start='in7',end='I6')
  for dest,orig,skip in [('P4','in4','I4'),('P5','in5','I5'),('P6','in6','I6'),('P7','in7',None)]:
   px,py=n.port(dest,'left');n.text(610,py-30,orig,13,weight=600);n.wire([(662,py-20),(715,py-20)],start=orig,end=dest)
   if skip:n.text(530,py+35,skip,13,weight=600);n.wire([(580,py+30),(690,py+30),(690,py+20),(715,py+20)],start=skip,end=dest)
  for id in ['I6','I5','I4']:
   px,py=n.port(id,'right');n.wire([(px,py),(px+55,py)],start=id,end='continuation-'+id);n.text(px+12,py-10,id,12,weight=600)
  n.text(25,965,'Each incoming edge resamples to its destination resolution.',15);n.text(25,1000,'Outputs are P3/P4/P5/P6/P7. Repeat this cell '+('R' if symbolic else str(R))+' times, with independent weights.',15)
  n.text(25,1040,'Cell 1: in3/in4/in5 = C3/C4/C5; in6/in7 come from C5 downsampling.',15);n.text(25,1080,'Later cells receive five '+f+'-channel maps from the previous cell.',15)
  # Separate class/box tower parameters. Convs shared over levels; BN not shared.
  h.text(20,65,'Apply both towers independently at every P3...P7 level.',14)
  for prefix,x,target in [('cls',20,810),('box',365,36)]:
   box(h,prefix+'in',x,115,295,'Pyramid feature',f+' channels')
   box(h,prefix+'sep',x,235,295,'SeparableConv2d 3×3',f+' channels; pointwise bias=True','conv','Separable')
   box(h,prefix+'bn',x,355,295,'BatchNorm2d','Per level and per repeat; eps=.001','norm')
   box(h,prefix+'act',x,475,295,'SiLU','Repeat tower unit '+('T' if symbolic else str(T))+' times','activation')
   seq(h,[prefix+x for x in ['in','sep','bn','act']])
   box(h,prefix+'pred',x,665,295,'SeparableConv2d 3×3',f'{target} output channels; no BN/activation','conv','Separable');h.connect(prefix+'act',prefix+'pred')
  h.text(20,805,'Class: 9 anchors × 90 logits = 810 channels.',15);h.text(20,845,'Box: 9 anchors × 4 deltas = 36 channels.',15);h.text(20,890,'Every repeated conv is shared across levels.',15);h.text(20,930,'BatchNorm parameters/statistics differ by level.',15);h.text(20,985,'Five raw maps per tower; no sigmoid in network output.',14);h.text(20,1030,('Anchor rows from table.' if symbolic else f'{N:,} anchor rows after flattening.'),15)
  # Definitions arranged around actual module distinctions.
  for i,name in enumerate(['MBConv','DSConv','SqueezeExcite','Fnode','Resample','Separable','SamePadding','Decode']):
   x=40+(i%4)*640;y=1420+(i//4)*960;p=d.panel('def'+name,name,x,y,610,915,kind='aggregate' if name in ['MBConv','DSConv'] else 'attention' if name=='SqueezeExcite' else 'pool' if name in ['Resample','SamePadding','Decode'] else 'conv',dashed=True,block_type=name)
   def bb(id,yy,l,dt='',kind='plain',xx=200,ww=380,bl=''):return box(p,name+id,xx,yy,ww,l,dt,kind,bl,h=43)
   def sq(ids):seq(p,[name+id for id in ids])
   if name in ['MBConv','DSConv']:
    operations=[('in','Input','Channel widths are resolved by stage','plain','')]
    if name=='MBConv':operations.extend([('expand','Conv2d 1×1 expansion','E first/later width from stage','conv2d',''),('ebn','BatchNorm2d + SiLU','eps=.001; momentum=.1','norm','ConvBNSiLU')])
    operations.extend([('dw','Depthwise Conv2d','k/s from stage; groups=E; SAME padding','conv2d',''),('dbn','BatchNorm2d + SiLU','eps=.001; momentum=.1','norm','ConvBNSiLU'),('se','SqueezeExcite','SE first/later reduced width from stage','attention','SqueezeExcite'),('proj','Conv2d 1×1 projection','Stage output channels; no bias','conv2d',''),('pbn','BatchNorm2d','eps=.001; no activation','norm','')])
    for j,(id,l,dt,k,bl) in enumerate(operations):bb(id,70+j*80,l,dt,k,bl=bl)
    sq([o[0] for o in operations]);yy=80*len(operations)+95;p.sum(name+'sum',390,yy);p.connect(name+operations[-1][0],name+'sum');p.connect(name+'in',name+'sum',from_port='left',via=[(55,91.5),(55,yy)],to_port='left');p.text(20,865,'Add identity only when stride 1 and input/output widths match.',13)
   elif name=='SqueezeExcite':
    for j,(id,l,dt,k) in enumerate([('in','Input','Expanded feature','plain'),('mean','Spatial mean','1 ×1; channels preserved','pool'),('reduce','Conv2d 1×1','SE reduced width; bias=True','conv2d'),('act','SiLU','','activation'),('expand','Conv2d 1×1','Restore expanded width; bias=True','conv2d'),('gate','Sigmoid','','activation'),('multiply','Multiply with original input','Broadcast over spatial positions','activation')]):bb(id,70+j*105,l,dt,k)
    sq(['in','mean','reduce','act','expand','gate','multiply']);p.connect(name+'in',name+'multiply',from_port='left',to_port='left',via=[(55,91.5),(55,721.5)])
   elif name=='Fnode':
    bb('resample',70,'Resample each input','Two or three separate tensors','pool',bl='Resample')
    bb('params',165,'Learned edge weights','2 or 3 scalar parameters','plain',xx=20,ww=235);bb('weight',255,'ReLU weights','Nonnegative weights','activation',xx=20,ww=235);bb('normalize',345,'Normalize weights','Divide by sum(weights) +0.0001','linear',xx=20,ww=235);sq(['params','weight','normalize'])
    for id,yy,label,detail,kind,bl in [('multiply',450,'Multiply each input by its weight','One product for each input','linear',''),('sum',535,'Stack and sum products','Reduce incoming-edge axis','linear',''),('silu',620,'SiLU','Activation before convolution','activation',''),('sep',705,'SeparableConv2d 3×3',f+' channels; pointwise bias','conv','Separable'),('bn',790,'BatchNorm2d','eps=.001; momentum=.01','norm','')]:bb(id,yy,label,detail,kind,xx=300,ww=280,bl=bl)
    p.wire([(390,113),(390,140),(440,140),(440,450)],start=name+'resample',end=name+'multiply');p.wire([(137.5,388),(137.5,473),(300,473)],start=name+'normalize',end=name+'multiply');sq(['multiply','sum','silu','sep','bn'])
   elif name=='Resample':
    bb('projection',75,'Optional Conv2d 1×1','Cell 1 backbone edges project to '+f+' channels','conv2d');bb('bn',185,'BatchNorm2d','Projection has bias=True; eps=.001','norm');bb('choice',300,'Resolution transform','Per edge; target grid given by Fnode','split');sq(['projection','bn','choice'])
    box(p,name+'up',20,440,275,'Nearest interpolation','For coarser input','norm',h=43);box(p,name+'down',325,440,265,'MaxPool2d3×3, s=2','SAME pad with negative infinity','pool',h=43);p.connect(name+'choice',name+'up',via=[(390,390),(157.5,390)]);p.connect(name+'choice',name+'down',via=[(390,390),(457.5,390)]);p.text(20,585,'Equal grid: identity. Equal channels: no projection.',14);p.text(20,630,'Initial in6: C5 projection + pooling. in7: pool in6.',14);p.text(20,680,'Cell 1 projects each C3/C4/C5 edge independently.',14);p.text(20,730,'All later cell inputs already have '+f+' channels.',14);p.text(20,805,'Incoming C3/C4/C5 channels: '+('/'.join(str(rows[j]['co']) for j in [2,4,6]) if not symbolic else 'C2/C4/C6'),14)
   elif name=='Separable':
    bb('dw',80,'Depthwise Conv2d3×3','groups=input channels; p=1; no bias','conv2d');bb('pw',250,'Pointwise Conv2d 1×1','bias=True; output width from occurrence','conv2d');sq(['dw','pw']);p.text(20,440,'This core has no normalization or activation.',14);p.text(20,480,'BiFPN appends BN; head repeat appends BN and SiLU.',14);p.text(20,520,'Final predictions append neither.',14)
    bb('norm',645,'BatchNorm2d','eps=.001','norm');bb('act',760,'SiLU','Visible ConvBNSiLU suffix definition','activation');p.connect(name+'norm',name+'act')
   elif name=='SamePadding':
    bb('pad',80,'Pad before convolution/pooling','Left/top use floor(total padding /2)','plain');bb('kernel',245,'Conv2d or MaxPool2d','Output is ceil(input/stride)','conv2d');sq(['pad','kernel']);p.text(20,425,'For these even inputs and stride 2:',15);p.text(20,470,'k3: left/top 0, right/bottom 1.',15);p.text(20,510,'k5: left/top 1, right/bottom 2.',15);p.text(20,570,'Stride1 uses symmetric padding 1 (k3) or 2 (k5).',14);p.text(20,635,'Convolutions pad zero; max pooling pads negative infinity.',14)
   elif name=='Decode':
    for j,(id,l,dt,k) in enumerate([('flatten','Flatten five class/box levels',('N ×90 logits; N ×4 deltas' if symbolic else f'{N:,} ×90 logits; {N:,} ×4 deltas'),'split'),('topk','Top 5000 class logits','Select anchors and class slots before sigmoid','plain'),('anchors','Generate 9 anchors/location','Strides 8/16/32/64/128; center=stride/2','plain'),('decode','Decode ty,tx,th,tw','Shift centers; exp height/width','linear'),('sigmoid','Sigmoid selected logits; map COCO','90 slots; unused IDs become -1','activation'),('nms','Filter invalid/low-score boxes; class NMS','Default maximum 100 detections','pool')]):bb(id,60+j*130,l,dt,k)
    sq(['flatten','topk','anchors','decode','sigmoid','nms'])
  d.text(50,3390,'Anchor shapes: base size 4 × stride × 2^(scale_index/3), scale_index 0/1/2; width/height factors (1,1), (1.4,.7), (.7,1.4).',15)
  if symbolic:
   table=d.panel('variantvalues','Resolved family values',40,3435,2500,650);xs=[25,620,1010,1400,1790,2180]
   for x,v in zip(xs,['Quantity',*SCALE_CONFIGS]):table.text(x,55,v.upper(),16,weight=600)
   collected={k:config_values(nn,v) for k,v in SCALE_CONFIGS.items()};properties=[('Input H',lambda k,c,s,r:c.image_size),('FPN width F',lambda k,c,s,r:c.fpn_channels),('BiFPN repeats R / head repeats T',lambda k,c,s,r:f'{c.fpn_repeats} / {c.head_repeats}'),('Stem Cstem',lambda k,c,s,r:s),('Stage outputs C0...C6',lambda k,c,s,r:'/'.join(str(v['co']) for v in r)),('Stage repeats n0...n6',lambda k,c,s,r:'/'.join(str(v['n']) for v in r)),('First expanded widths E0...E6',lambda k,c,s,r:'/'.join(str(v['mid']) for v in r)),('Later expanded widths E0...E6',lambda k,c,s,r:'/'.join(str(v['later']) if v['n']>1 else 'n/a' for v in r)),('First SE widths SE0...SE6',lambda k,c,s,r:'/'.join(str(v['se']) for v in r)),('Later SE widths SE0...SE6',lambda k,c,s,r:'/'.join(str(v['se_later']) if v['n']>1 else 'n/a' for v in r)),('Total anchor rows N',lambda k,c,s,r:sum((c.image_size//2**l)**2 for l in range(3,8))*9)]
   for j,(label,fn) in enumerate(properties):
    table.text(25,100+j*46,label,15)
    for x,(k,cfg2) in zip(xs[1:],SCALE_CONFIGS.items()):s2,r2=collected[k];table.text(x,100+j*46,str(fn(k,cfg2,s2,r2)),13)
  else:d.text(50,3490,'Native fixed-resolution eval graph. All five scales and every first-cell resample shape passed a random-weight CPU forward.',15)
  path=out/f'{size}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));ev.setdefault('routes',{})[size]=routes;views.append(dict(id=size,label='EfficientDet family' if symbolic else 'EfficientDet '+size.upper(),task='detect',size='d0/d1/d2/d3/d4' if symbolic else size,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/efficientdet/{size}.svg',html=f'/diagrams/models/efficientdet/{size}.html',input='H from table' if symbolic else f'1 × 3 × {H} × {H}',verification='cpu'));print('efficientdet',size,routes['total_findings'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family='efficientdet',slug='efficientdet',title='EfficientDet',source_revision=rev,default_view='d0',views=views),indent=2)+'\n');ev.update(verification='CPU every D0-D4 size; no external weights; anchors also checked',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/efficientdet.py --source /path/to/libreyolo');ep.write_text(json.dumps(ev,indent=2)+'\n')
if __name__=='__main__':main()
