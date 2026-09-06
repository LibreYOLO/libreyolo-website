"""Native two-stage Faster/Mask R-CNN, including data-dependent RoI geometry."""
from pathlib import Path
import argparse,os,sys,json,subprocess,math
from yolo9 import box,seq
from retinanet import resnet_backbone,resnet_insets
WEBSITE=Path(__file__).resolve().parents[3]

def mobile_rows(model):
 import torch
 rows=[]
 for idx,mod in model.backbone.body.named_children():
  convs=[x for x in mod.modules() if isinstance(x,torch.nn.Conv2d)]
  if hasattr(mod,'block'):
   dw=next(x for x in convs if x.groups==x.in_channels and x.kernel_size[0]>1);se=next((x for x in mod.block if type(x).__name__=='SqueezeExcitation'),None);act='Hardswish' if any(isinstance(x,torch.nn.Hardswish) for x in mod.modules()) else 'ReLU';rows.append(dict(index=int(idx),ci=convs[0].in_channels,hidden=dw.in_channels,co=convs[-1].out_channels,k=dw.kernel_size[0],stride=dw.stride[0],se=se.fc1.out_channels if se is not None else 0,act=act,residual=mod.use_res_connect))
  else:rows.append(dict(index=int(idx),ci=convs[0].in_channels,hidden=0,co=convs[0].out_channels,k=convs[0].kernel_size[0],stride=convs[0].stride[0],se=0,act='Hardswish',residual=False))
 return rows

def mobile_insets(d,y,rows):
 p=d.panel('mobileir','MobileNetV3 inverted residual',40,y,780,1120,kind='aggregate',dashed=True,block_type='MobileIR');p.text(20,62,'Exact input, expanded and output widths are on each stage.',14)
 for id,yy,l,dt,k,bl in [('mi',90,'Input','Residual only at stages marked +','plain',''),('mexp',180,'Optional Conv2d 1×1','Skipped only in block 1; expanded width from stage','conv2d',''),('men',270,'FrozenBatchNorm2d + activation','eps=0.00001; ReLU or Hardswish from stage','norm','MobileNorm'),('mdw',360,'Depthwise Conv2d','k/stride from stage; groups = expanded channels','conv2d',''),('mdn',450,'FrozenBatchNorm2d + activation','eps=0.00001','norm','MobileNorm'),('mse',540,'Optional SqueezeExcitation','Present when stage gives SE width','attention','SqueezeExcitation'),('mproj',640,'Conv2d 1×1 projection','Output width from stage; no activation','conv2d',''),('mpn',730,'FrozenBatchNorm2d','eps=0.00001; no activation','norm','')]:box(p,id,330,yy,420,l,dt,k,bl)
 seq(p,['mi','mexp','men','mdw','mdn','mse','mproj','mpn']);p.sum('madd',540,915);p.connect('mpn','madd');p.connect('mi','madd',from_port='left',via=[(105,113),(105,915)],to_port='left');p.text(110,875,'identity for + stages',15);p.text(25,1020,'Other stages return the projection normalization directly.',15);p.text(25,1060,'All convs in this residual path are bias-free.',15)
 p=d.panel('se','SqueezeExcitation',850,y,780,1120,kind='attention',dashed=True,block_type='SqueezeExcitation')
 for id,yy,l,dt,k in [('sei',80,'Input','Expanded channels from stage','plain'),('gap',200,'AdaptiveAvgPool2d','1 × 1 spatial output','pool'),('ser',330,'Conv2d 1×1','SE reduced width printed on stage; bias=True','conv2d'),('sere',440,'ReLU','','activation'),('see',550,'Conv2d 1×1','Restore expanded channels; bias=True','conv2d'),('seh',660,'Hardsigmoid','clamp(x + 3, 0, 6) / 6','activation'),('sem',830,'Elementwise multiply','Gate the original expanded feature','activation')]:box(p,id,330,yy,420,l,dt,k)
 seq(p,['sei','gap','ser','sere','see','seh','sem']);p.connect('sei','sem',from_port='left',to_port='left',via=[(105,103),(105,853)]);p.text(25,985,'SE reduced widths used here: 24,32,120,168,240.',15);p.text(25,1020,'This channel gate preserves the spatial feature map.',15)


def roi_inset(d,y,norm='FrozenBatchNorm2d',eps='0.00001',mobile=False):
 p=d.panel('roiPrimitive','Normalization and RoIAlign',1660,y,780,1120,kind='conv',dashed=True,block_type='RoIAlign')
 box(p,'rnorm',120,80,540,norm,'eps='+eps+'; stored statistics in eval','norm');box(p,'ract',120,175,540,'ReLU or Hardswish','Hardswish(x) = x × clamp(x + 3, 0, 6) / 6','activation');p.connect('rnorm','ract');p.text(20,280,'RPN repeated unit (no normalization)',15)
 box(p,'rpnc',30,325,310,'Conv2d 3×3, p=1','256 channels; bias=True','conv2d');box(p,'rpna',430,325,310,'ReLU','Preserve spatial dimensions','activation');p.connect('rpnc','rpna',from_port='right',to_port='left')
 box(p,'levels',120,450,540,'Choose a feature level per proposal','floor(4 + log2(sqrt(box area) / 224)); clip level range','plain');box(p,'samples',120,565,540,'RoIAlign: scale box to feature coordinates','Aligned=False; bilinear samples; sampling_ratio=2','plain');box(p,'average',120,680,540,'Average 2 × 2 samples in every bin','7 × 7 bins for boxes; 14 × 14 bins for masks','pool');seq(p,['levels','samples','average']);p.text(20,860,'R is proposal count and N is final detection count.',16);p.text(20,900,'These dimensions depend on image values, even at fixed input size.',15);p.text(20,960,'MobileNet has two stride-32 maps; its RoI level range is 5...5.' if mobile else 'ResNet box/mask RoIAlign selects among P2...P5.',15)

def main(family='faster_rcnn'):
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.faster_rcnn.nn import LibreFasterRCNNModel,FASTER_RCNN_CONFIGS
 from libreyolo.models.mask_rcnn.nn import LibreMaskRCNNModel
 import torch
 torch.set_num_threads(4);slug=family.replace('_','-');out=WEBSITE/f'public/diagrams/models/{slug}';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/f'scripts/model-diagrams/evidence/{family}.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];evidence=json.loads(ep.read_text()) if args.diagram_only else dict(family=family,source_revision=rev,source_files=['libreyolo/models/faster_rcnn/nn.py']+(['libreyolo/models/mask_rcnn/nn.py'] if family=='mask_rcnn' else []),shapes={});views=[]
 variants=[('r50-segment','l',True),('r50-detect','l',False)] if family=='mask_rcnn' else [(s,s,False) for s in 'nsml']+[('family-mobile','n',False)]
 for viewid,size,masks in variants:
  symbolic=viewid=='family-mobile';mobile=size in 'ns';v2=size=='l';H=FASTER_RCNN_CONFIGS[size]['min_size'];cfg=FASTER_RCNN_CONFIGS[size];K=cfg['rpn_post_nms_top_n_test'];norm='BatchNorm2d' if v2 else 'FrozenBatchNorm2d';eps='0' if size=='m' else '0.00001';A=15 if mobile else 3;grids=[H//32,H//32,math.ceil(H/64)] if mobile else [H//4,H//8,H//16,H//32,math.ceil(H/64)];anchors=A*sum(x*x for x in grids)
  # Model construction uses no pretrained loaders; rows inspect native modules.
  torch.manual_seed(0);model=LibreMaskRCNNModel(return_masks=masks).eval() if family=='mask_rcnn' else LibreFasterRCNNModel(size).eval();rows=mobile_rows(model) if mobile else []
  if not symbolic and not args.diagram_only:
   obs={}
   def hook(name):
    def record(mod,ins,output):
     def shape(v):
      if isinstance(v,torch.Tensor):return list(v.shape)
      if isinstance(v,(list,tuple)):return [shape(x) for x in v]
      if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
      if hasattr(v,'tensors'):return {'tensors':list(v.tensors.shape),'image_sizes':v.image_sizes}
      return str(v)
     obs[name]={'input':shape(ins),'output':shape(output)}
    return record
   for name,mod in model.named_modules():
    if name and (name.count('.')<=3 or name.startswith(('roi_heads.','rpn.head.'))):mod.register_forward_hook(hook(name))
   with torch.inference_mode():pred=model(torch.zeros(1,3,H,H))
   assert isinstance(pred,list) and len(pred)==1;assert pred[0]['boxes'].shape[1]==4;assert len(pred[0]['boxes'])<=100
   if masks:assert pred[0]['masks'].shape==(len(pred[0]['boxes']),1,H,H)
   evidence['shapes'][viewid]=obs;evidence.setdefault('sample_outputs',{})[viewid]={k:list(v.shape) for k,v in pred[0].items()}
  title=('Mask R-CNN R50 segmentation' if masks else 'Mask R-CNN R50 detection') if family=='mask_rcnn' else ('Faster R-CNN MobileNet N/S family' if symbolic else 'Faster R-CNN '+size.upper());insety=3460 if masks else 2650;HEIGHT=insety+1270+(450 if symbolic else 0)
  d=Diagram(title,('Variable canvas H and proposal limit K are tabulated below.' if symbolic else f'{H} × {H} RGB canvas; batch 1; eval.')+' COCO: 91 internal slots mapped to 80 classes. R and N are data-dependent counts.',width=2480,height=HEIGHT,revision=rev,source_label=f'models/{family}/nn.py; models/faster_rcnn/nn.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/{family}/nn.py',logo=WEBSITE/'public/icon-128.png')
  pre=d.panel('preprocess','Input transform',40,230,2400,185)
  for j,(id,l,dt,k) in enumerate([('rgb','RGB image','Values in [0,1]','plain'),('normalize','Channel normalization','Mean .485/.456/.406; std .229/.224/.225','norm'),('resize','Bilinear resize',('Min/max edge from N/S table' if symbolic else f'Min edge {H}; max edge {cfg["max_size"]}'),'plain'),('pad','Pad right/bottom to multiple of 32','Canvas unchanged for the selected square input','plain')]):box(pre,id,25+j*595,85,545,l,dt,k)
  for a,b in zip(['rgb','normalize','resize'],['normalize','resize','pad']):pre.connect(a,b,from_port='right',to_port='left')
  if mobile:
   b=d.panel('backbone','MobileNetV3-Large features',40,455,610,1410);div=1
   for j,row in enumerate(rows):
    div*=row['stride'];grid=f'H/{div}' if symbolic else str(H//div);label=f'{row["index"]}: '+('IR' if row['hidden'] else 'Conv2d')+f' {row["k"]}×{row["k"]}, s={row["stride"]}'+(' +' if row['residual'] else '')
    detail=f'{row["ci"]} input; {row["co"]} × {grid} × {grid}'
    if row['hidden']:label+=f'; expand {row["hidden"]}'+(f'; SE {row["se"]}' if row['se'] else '')
    box(b,'M'+str(row['index']),40,65+j*73,530,label,detail,'aggregate' if row['hidden'] else 'conv','MobileIR' if row['hidden'] else 'MobileNorm')
   seq(b,['M'+str(r['index']) for r in rows]);b.text(20,1350,'Blocks 1...6 use ReLU; blocks 7...15 and stem/final use Hardswish.',14)
  else:resnet_backbone(d,40,455,610,1410,norm,eps,H)
  f=d.panel('fpn','Feature pyramid',680,455,930,1410);channels=[160,960] if mobile else [256,512,1024,2048];levels=list(range(len(channels)))
  for j in reversed(levels):
   yy=100+(len(channels)-1-j)*250;g=H//32 if mobile else H//(4*2**j);tag=('M13' if j==0 else 'M16') if mobile else 'C'+str(j+2)
   box(f,'lat'+str(j),25,yy,285,f'{tag}: Conv2d 1×1'+(' + BN' if v2 else ''),f'{channels[j]} input; 256 × '+(f'H/{32 if mobile else 4*2**j}' if symbolic else str(g))+' square','conv2d');box(f,'F'+str(j),615,yy,285,'Conv2d 3×3, p=1'+(' + BN' if v2 else ''),f'F{j}: 256 × '+(f'H/{32 if mobile else 4*2**j}' if symbolic else str(g))+' square','conv2d')
   if j==len(channels)-1:f.connect('lat'+str(j),'F'+str(j),from_port='right',to_port='left')
   else:
    f.sum('sum'+str(j),450,yy+23);f.connect('lat'+str(j),'sum'+str(j),from_port='right',to_port='left');f.connect('sum'+str(j),'F'+str(j),from_port='right',to_port='left');box(f,'up'+str(j),325,yy-115,250,'Nearest resize',f'256 × '+(f'H/{32 if mobile else 4*2**j}' if symbolic else str(g))+' square','norm');prev='lat'+str(j+1) if j+1==len(channels)-1 else 'sum'+str(j+1);sx,sy=f.port(prev,'bottom');f.wire([(sx,sy),(sx,yy-150),(450,yy-150),(450,yy-115)],start=prev,end='up'+str(j));f.connect('up'+str(j),'sum'+str(j))
  pooly=720 if mobile else 1130;box(f,'last',25,pooly,285,f'F{len(channels)-1} continuation',f'256 × {H//32} × {H//32}' if not symbolic else '256 × H/32 × H/32');box(f,'pooled',405,pooly,430,'MaxPool2d 1×1, s=2','pool: 256 × '+('ceil(H/64) square' if symbolic else f'{grids[-1]} × {grids[-1]}'),'pool');f.connect('last','pooled',from_port='right',to_port='left')
  f.text(25,1280,'RPN receives '+str(len(grids))+' feature maps. RoIAlign excludes the extra pooled map.',15);f.text(25,1320,'FPN normalization: '+('BatchNorm2d, eps=0.00001; no activation.' if v2 else 'none; all FPN convs have bias.'),15)
  if mobile:f.text(25,880,'Both M13 and M16 outputs have stride 32; resize preserves the grid.',15);f.text(25,925,'The selected RoI feature level range is 5...5.',15)
  r=d.panel('rpn','Region Proposal Network (RPN)',1640,455,800,1410);cap='K' if symbolic else str(K)
  box(r,'rinput',175,75,450,'Each FPN feature independently','256 channels; shared RPN weights')
  box(r,'rconv',175,170,450,'Conv2d 3×3 + ReLU',f'256 channels; repeat {2 if v2 else 1}; no norm','conv','RPNConv');r.connect('rinput','rconv')
  box(r,'obj',25,290,335,'Conv2d 1×1',f'{A} objectness logits/location','conv2d');box(r,'delta',425,290,335,'Conv2d 1×1',f'{A*4} box deltas/location','conv2d');r.connect('rconv','obj',via=[(400,250),(192.5,250)]);r.connect('rconv','delta',via=[(400,250),(592.5,250)])
  box(r,'topk',25,420,335,'Top-K per level, then sigmoid',cap+' pre-NMS proposals per level','activation');r.connect('obj','topk');box(r,'reshape',425,420,335,'Flatten anchor rows','Weights for box delta decode: 1,1,1,1','split');r.connect('delta','reshape')
  box(r,'anchors',25,610,335,'Generate anchors',('15 per location; all five sizes on each level' if mobile else '3 per location; one size per level'),'plain');box(r,'decode',425,610,335,'Decode anchor-relative deltas','Center shift; exp sizes; clip log-scale','linear');r.connect('reshape','decode');r.connect('anchors','decode',from_port='right',to_port='left')
  box(r,'filter',175,845,450,'Filter proposals and level-wise NMS',f'score ≥ {cfg["rpn_score_thresh"]}; NMS IoU 0.7','pool');r.wire([(192.5,466),(192.5,785),(280,785),(280,845)],start='topk',end='filter');r.wire([(592.5,656),(592.5,809),(520,809),(520,845)],start='decode',end='filter');box(r,'proposals',175,960,450,'Retain at most '+cap+' proposals','R × 4 boxes; R varies with image values','split');r.connect('filter','proposals');r.text(25,1080,'Anchor ratios: 0.5, 1, 2. Sizes: 32, 64, 128, 256, 512.',15);r.text(25,1120,('Raw anchor rows from N/S table.' if symbolic else f'Raw anchor rows on this canvas: {anchors:,}.'),15);r.text(25,1170,'Clip to canvas; remove boxes smaller than 0.001 pixels.',14);r.text(25,1210,'Box size exponent clamp: log(1000/16).',14)
  roi=d.panel('roi','Box RoI head and final detection selection',40,1905,2400,700)
  box(roi,'rfeatures',25,80,450,'Selected FPN features',('F0/F1 (both stride 32)' if mobile else 'F0/F1/F2/F3 (strides4/8/16/32)'));box(roi,'roiproposals',25,205,450,'RPN proposals','R × 4 boxes');box(roi,'align',575,125,450,'MultiScaleRoIAlign','R × 256 × 7 × 7; sampling_ratio=2','plain','RoIAlign');roi.wire([(475,103),(530,103),(530,137),(575,137)],start='rfeatures',end='align');roi.wire([(475,228),(545,228),(545,159),(575,159)],start='roiproposals',end='align')
  if v2:
   box(roi,'convroi',1130,125,530,'Conv3×3 + BatchNorm + ReLU ×4','R × 256 × 7 × 7; eps=0.00001','conv','RoIConv');roi.connect('align','convroi',from_port='right',to_port='left');box(roi,'fcroi',1750,125,570,'Flatten; Linear 12544 to 1024; ReLU','R × 1024','linear');roi.connect('convroi','fcroi',from_port='right',to_port='left')
  else:
   box(roi,'convroi',1130,125,530,'Flatten; Linear 12544 to 1024; ReLU','R × 1024','linear');box(roi,'fcroi',1750,125,570,'Linear 1024 to 1024; ReLU','R × 1024','linear');roi.connect('align','convroi',from_port='right',to_port='left');roi.connect('convroi','fcroi',from_port='right',to_port='left')
  box(roi,'cls',1750,280,570,'Linear 1024 to 91','R × 91 class logits','linear');box(roi,'reg',1120,280,540,'Linear 1024 to 364','R × 91 × 4 class-specific deltas','linear');roi.connect('fcroi','cls');roi.wire([(1750,148),(1700,148),(1700,250),(1390,250),(1390,280)],start='fcroi',end='reg')
  box(roi,'softmax',1750,395,570,'Softmax across 91 slots','Drop background slot 0','activation');box(roi,'boxes',1120,395,540,'Decode against proposals','Box coder weights 10,10,5,5; clip to canvas','linear');roi.connect('cls','softmax');roi.connect('reg','boxes');box(roi,'det',1470,550,850,'Score > 0.05; remove tiny boxes; class NMS 0.5; top 100','N ≤100 boxes, labels and scores; invert input resize','pool');roi.wire([(1390,441),(1390,500),(1680,500),(1680,550)],start='boxes',end='det');roi.wire([(2035,441),(2035,526),(2100,526),(2100,550)],start='softmax',end='det');roi.text(25,420,'R and N are dynamic. Their upper limits are fixed:',16);roi.text(25,465,'R ≤'+cap+' and N ≤100. All per-RoI dimensions are resolved.',16);roi.text(25,580,'Model outputs use 91-class IDs; the library maps valid COCO IDs to 80.',15)
  # Shared block definitions for the grouped RoI operations above.
  if v2:
   units=[('uc','Conv2d 3×3, p=1','256 channels; bias=False','conv2d'),('un','BatchNorm2d','eps=0.00001','norm'),('ua','ReLU','RoI conv unit','activation')]
  else:
   units=[('uc','Flatten spatial features','12544 values; skip for second FC','plain'),('un','Linear','1024 output features; bias=True','linear'),('ua','ReLU','Fully connected unit','activation')]
  for j,(id,label,detail,kind) in enumerate(units):box(roi,id,25+j*360,515,310,label,detail,kind)
  roi.connect('uc','un',from_port='right',to_port='left');roi.connect('un','ua',from_port='right',to_port='left')
  if masks:
   p=d.panel('mask','Mask branch after box selection',40,2650,2400,760,kind='plain')
   nodes=[('mroi','RoIAlign on selected detection boxes','N × 256 × 14 × 14; sampling_ratio=2','plain','RoIAlign'),('mconv','Conv3×3 + BatchNorm + ReLU ×4','N × 256 × 14 × 14','conv','RoIConv'),('mup','ConvTranspose2d 2×2, s=2 + ReLU','N × 256 × 28 × 28','conv2d',''),('mlog','Conv2d 1×1','N × 91 × 28 × 28 logits','conv2d','')]
   for j,(id,l,dt,k,bl) in enumerate(nodes):box(p,id,25+j*595,90,545,l,dt,k,bl)
   for a,z in zip(['mroi','mconv','mup'],['mconv','mup','mlog']):p.connect(a,z,from_port='right',to_port='left')
   box(p,'msig',1810,245,545,'Sigmoid and select each detection class','N × 1 × 28 × 28','activation');p.connect('mlog','msig');box(p,'mpad',1215,245,545,'Pad masks 1 pixel and expand boxes','30 × 30 masks; box scale 30/28','plain');p.connect('msig','mpad',from_port='left',to_port='right');box(p,'mresize',620,245,545,'Bilinear resize to each expanded box','align_corners=False; dynamic box width/height','norm');p.connect('mpad','mresize',from_port='left',to_port='right');box(p,'mpaste',25,245,545,'Clip and paste onto original canvas',f'N × 1 × {H} × {H}','plain');p.connect('mresize','mpaste',from_port='left',to_port='right');p.text(25,415,'The mask branch reuses FPN features and the final boxes from the box head. It does not rerun the backbone.',17);p.text(25,470,'91 mask channels are architectural slots; each detection selects its predicted class channel before pasting.',16)
   # Expand the shared RoIConv repeat unit down to primitives.
   box(p,'rc',25,595,620,'Conv2d 3×3, p=1, no bias','256 input/output channels','conv2d');box(p,'rbn',860,595,620,'BatchNorm2d','eps=0.00001','norm');box(p,'rrelu',1695,595,620,'ReLU','Per-RoI spatial size preserved','activation');p.connect('rc','rbn',from_port='right',to_port='left');p.connect('rbn','rrelu',from_port='right',to_port='left')
  if mobile:mobile_insets(d,insety,rows)
  else:
   resnet_insets(d,insety,norm,eps)
   for child in list(d.root):
    if child.get('id')=='resdef2':d.root.remove(child)
  roi_inset(d,insety,norm,eps,mobile)
  if symbolic:
   p=d.panel('variantvalues','N/S values for this shared MobileNet-FPN topology',40,insety+1160,2400,440)
   for j,(name,nv,sv) in enumerate([('Input H',320,800),('Maximum resized edge',640,1333),('F0/F1 square grid',10,25),('Pooled square grid',5,13),('Raw anchor rows',3375,21285),('Proposal limit K',150,1000)]):
    for xx,txt in zip([25,1050,1680],[name,str(nv),str(sv)]):p.text(xx,85+j*49,txt,17)
   p.text(1050,45,'N',18,weight=600);p.text(1680,45,'S',18,weight=600)
  else:d.text(50,HEIGHT-105,'Complete native inference path with random-weight CPU validation. Data-dependent selection counts are not fixed tensor dimensions.',15)
  path=out/f'{viewid}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{viewid}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));evidence.setdefault('routes',{})[viewid]=routes;views.append(dict(id=viewid,label=title,task='segment' if masks else 'detect',size='n/s' if symbolic else ('r50' if family=='mask_rcnn' else size),kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/{slug}/{viewid}.svg',html=f'/diagrams/models/{slug}/{viewid}.html',input='H from N/S table' if symbolic else f'1 × 3 × {H} × {H}',verification='cpu'));print(family,viewid,'routes',routes['total_findings'],flush=True)
 default='r50-segment' if family=='mask_rcnn' else 'm';(out/'manifest.json').write_text(json.dumps(dict(family=family,slug=slug,title='Mask R-CNN' if family=='mask_rcnn' else 'Faster R-CNN',source_revision=rev,default_view=default,views=views),indent=2)+'\n');evidence.update(verification='CPU each concrete path; random weights seeded 0; no downloads',dynamic_dimensions='R proposals and N detections depend on input values; bounds are printed',visual='Parent performs browser/PNG QA',reproduce=f'python scripts/model-diagrams/builders/{family}.py --source /path/to/libreyolo');ep.write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
