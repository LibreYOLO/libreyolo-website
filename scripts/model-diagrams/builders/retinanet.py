"""Native RetinaNet and FCOS diagrams, with reusable ResNet-50 schematics."""
from pathlib import Path
import argparse,os,sys,json,subprocess,importlib
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]

def resnet_backbone(d,x,y,w,h,norm='FrozenBatchNorm2d',eps='0.00001',H=800):
 p=d.panel('backbone','ResNet-50 backbone',x,y,w,h)
 stages=[('input','Input',f'3 × {H} × {H}','plain',''),('stem','Conv2d 7×7, s=2, p=3',f'64 × {H//2} × {H//2}; no bias','conv2d',''),('stemnorm',norm+' + ReLU',f'64 × {H//2} × {H//2}; eps={eps}','norm','NormReLU'),('pool','MaxPool2d 3×3, s=2, p=1',f'64 × {H//4} × {H//4}','pool','')]
 for i,(out,hidden,repeats) in enumerate([(256,64,3),(512,128,4),(1024,256,6),(2048,512,3)],2):stages.append((f'C{i}',f'C{i}: 1 projection + {repeats-1} identity blocks',f'{out} × {H//2**i} × {H//2**i}; hidden {hidden}','aggregate','ResNetBottleneck'))
 for i,(id,l,dt,k,bl) in enumerate(stages):box(p,id,35,75+i*108,w-70,l,dt,k,bl)
 seq(p,[s[0] for s in stages]);p.text(20,h-80,'Projection strides: C2=1, C3/C4/C5=2.',15);p.text(20,h-50,'The stride is in the bottleneck 3×3 convolution.',14)
 return p

def resnet_insets(d,y,norm,eps,W=2420):
 cols=[(40,'Projection bottleneck'),(850,'Identity bottleneck'),(1660,'Normalization and head unit')]
 for i,(x,title) in enumerate(cols):
  p=d.panel('resdef'+str(i),title,x,y,780,1070,kind='bottleneck' if i<2 else 'conv',dashed=True,block_type='ResNetBottleneck' if i<2 else 'NormReLU')
  if i<2:
   prefix='proj' if i==0 else 'identity';p.text(20,65,'Stage order: C2 / C3 / C4 / C5.',14)
   for id,yy,l,dt,k in [('in',90,'Input','64 / 256 / 512 / 1024 ch' if i==0 else '256 / 512 / 1024 / 2048 ch','plain'),('c1',175,'Conv2d 1×1','64 / 128 / 256 / 512 output channels','conv2d'),('n1',255,norm,'64 / 128 / 256 / 512 channels','norm'),('a1',330,'ReLU','','activation'),('c2',405,'Conv2d 3×3, p=1','s=1/2/2/2' if i==0 else 's=1','conv2d'),('n2',480,norm,'64 / 128 / 256 / 512 channels','norm'),('a2',555,'ReLU','','activation'),('c3',630,'Conv2d 1×1','256 / 512 / 1024 / 2048 output channels','conv2d'),('n3',705,norm,'No activation before addition','norm')]:box(p,prefix+id,330,yy,420,l,dt,k,h=43)
   seq(p,[prefix+x for x in ['in','c1','n1','a1','c2','n2','a2','c3','n3']]);p.sum(prefix+'sum',540,870);p.connect(prefix+'n3',prefix+'sum');box(p,prefix+'relu',330,945,420,'ReLU','Block output','activation');p.connect(prefix+'sum',prefix+'relu')
   if i==0:
    box(p,prefix+'skipconv',20,405,265,'Conv2d 1×1','256 / 512 / 1024 / 2048 ch','conv2d');box(p,prefix+'skipnorm',20,555,265,norm,'Stride 1 / 2 / 2 / 2','norm');p.connect(prefix+'in',prefix+'skipconv',from_port='left',via=[(152.5,111.5)]);p.connect(prefix+'skipconv',prefix+'skipnorm');p.connect(prefix+'skipnorm',prefix+'sum',via=[(152.5,870)],to_port='left')
   else:p.connect(prefix+'in',prefix+'sum',from_port='left',via=[(115,111.5),(115,870)],to_port='left');p.text(130,820,'identity',15)
  else:
   box(p,'normalization',115,100,550,norm,f'eps={eps}; learned scale and bias','norm');box(p,'nrelu',115,195,550,'ReLU','For stem and first two bottleneck convolutions','activation');p.connect('normalization','nrelu');p.text(25,315,'Frozen normalization uses stored mean/variance;',15);p.text(25,345,'BatchNorm in eval also uses its running statistics.',15);p.text(25,390,'Affine transform: (x - mean) / sqrt(variance + eps),',15);p.text(25,420,'then multiply scale and add bias.',15)
   box(p,'hc',115,530,550,'Head Conv2d 3×3, p=1','256 input/output channels','conv2d');box(p,'hgn',115,650,550,'Optional GroupNorm','32 groups; eps=0.00001','norm');box(p,'hr',115,770,550,'ReLU','This head unit repeats four times in each tower','activation');seq(p,['hc','hgn','hr']);p.text(25,930,'RetinaNet r50 has no head normalization.',15);p.text(25,962,'RetinaNet r50v2 and FCOS use GroupNorm.',15)

def main(family='retinanet'):
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 import torch
 torch.set_num_threads(4);isfcos=family=='fcos';sizes=['r50'] if isfcos else ['r50','r50v2'];source=importlib.import_module('libreyolo.models.'+family+'.nn');out=WEBSITE/f'public/diagrams/models/{family}';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/f'scripts/model-diagrams/evidence/{family}.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];evidence=json.loads(ep.read_text()) if args.diagram_only else dict(family=family,source_revision=rev,source_files=[f'libreyolo/models/{family}/nn.py',f'libreyolo/postprocess/{family}.py','torchvision.models.resnet.Bottleneck','torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork'],shapes={});views=[]
 for size in sizes:
  v2=size=='r50v2';norm='BatchNorm2d' if v2 else 'FrozenBatchNorm2d';eps='0.00001' if v2 or isfcos else '0';count=13343 if isfcos else 120087
  if not args.diagram_only:
   model=source.LibreFCOSModel(91).eval() if isfcos else source.LibreRetinaNetModel(size,91).eval();obs={}
   def hook(name):
    def record(mod,ins,output):
     def shape(v):
      if isinstance(v,torch.Tensor):return list(v.shape)
      if isinstance(v,(list,tuple)):return [shape(x) for x in v]
      if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
      return str(v)
     obs[name]={'input':shape(ins),'output':shape(output)}
    return record
   for name,mod in model.named_modules():
    if name and (name.count('.')<=3 or name.startswith('head.')):mod.register_forward_hook(hook(name))
   with torch.inference_mode():pred=model(torch.zeros(1,3,800,800))
   if isfcos:assert pred['cls_logits'].shape==(1,13343,91);assert pred['bbox_regression'].shape==(1,13343,4);assert pred['bbox_ctrness'].shape==(1,13343,1)
   else:assert pred.shape==(1,120087,84)
   evidence['shapes'][size]=obs
  title='FCOS ResNet-50' if isfcos else 'RetinaNet '+size.upper();d=Diagram(title,'Detection; 800 × 800 RGB canvas; batch 1; unfused eval. 91 class-head slots map to 80 COCO classes.',width=2480,height=3060,revision=rev,source_label=f'models/{family}/nn.py; postprocess/{family}.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/{family}/nn.py',logo=WEBSITE/'public/icon-128.png')
  resnet_backbone(d,40,230,450,1130,norm,eps);f=d.panel('fpn','Feature pyramid (FPN)',520,230,950,1130);h=d.panel('head','FCOS head' if isfcos else 'RetinaNet head',1500,230,940,1130)
  for lv,c,g,y in [(5,2048,25,110),(4,1024,50,365),(3,512,100,620)]:
   box(f,'lat'+str(lv),30,y,270,f'C{lv}: Conv2d 1×1',f'{c} input; 256 × {g} × {g}','conv2d');box(f,'P'+str(lv),625,y,290,f'P{lv}: Conv2d 3×3, p=1',f'256 × {g} × {g}','conv2d')
   if lv==5:f.connect('lat5','P5',from_port='right',to_port='left')
   else:
    f.sum('sum'+str(lv),450,y+23);f.connect('lat'+str(lv),'sum'+str(lv),from_port='right',to_port='left');f.connect('sum'+str(lv),'P'+str(lv),from_port='right',to_port='left')
  box(f,'up4',325,240,250,'Nearest resize','256 × 50 × 50','norm');box(f,'up3',325,495,250,'Nearest resize','256 × 100 × 100','norm');f.connect('lat5','up4',via=[(165,195),(450,195)]);f.connect('up4','sum4');f.connect('sum4','up3');f.connect('up3','sum3')
  box(f,'extra',30,820,270,'C5 continuation' if v2 else 'P5 continuation','2048 × 25 × 25' if v2 else '256 × 25 × 25');box(f,'P6',355,820,270,'Conv2d 3×3, s=2, p=1','P6: 256 × 13 × 13','conv2d');box(f,'erelu',675,820,240,'ReLU','P6 activation','activation');f.connect('extra','P6',from_port='right',to_port='left');f.connect('P6','erelu',from_port='right',to_port='left');box(f,'P7',675,930,240,'Conv2d 3×3, s=2','P7: 256 × 7 × 7; p=1','conv2d');f.connect('erelu','P7');f.text(20,1060,'Lateral sums feed upsampling before the output 3×3 convolutions.',15)
  h.text(25,65,'Apply the same tower weights to P3, P4, P5, P6 and P7.',15);box(h,'feature',315,110,310,'One pyramid feature','256 channels; grid 100 / 50 / 25 / 13 / 7')
  for side,x in [('class',25),('reg',525)]:
   box(h,side+'tower',x,230,365,'Head unit repeated 4 times','Conv3×3 '+('+ GroupNorm32 ' if v2 or isfcos else '')+'+ ReLU; 256 ch','conv','HeadUnit');h.connect('feature',side+'tower',via=[(470,190),(x+182.5,190)])
  box(h,'classpred',25,365,365,'Conv2d 3×3, p=1',('91' if isfcos else '819')+' logits/location; bias=True','conv2d');h.connect('classtower','classpred');box(h,'classes',25,485,365,'Reshape anchor/location rows',f'1 × {count:,} × 91','split');h.connect('classpred','classes')
  if isfcos:
   box(h,'regpred',485,365,205,'Conv2d 3×3, p=1','4 distances','conv2d');box(h,'ctrpred',715,365,205,'Conv2d 3×3, p=1','1 centerness logit','conv2d');h.wire([(707.5,276),(707.5,320),(587.5,320),(587.5,365)],start='regtower',end='regpred');h.wire([(707.5,276),(707.5,320),(817.5,320),(817.5,365)],start='regtower',end='ctrpred');box(h,'distance',485,485,205,'ReLU','1 × 13,343 × 4','activation');box(h,'centerness',715,485,205,'Reshape rows','1 × 13,343 × 1','split');h.connect('regpred','distance');h.connect('ctrpred','centerness')
  else:
   box(h,'regpred',525,365,365,'Conv2d 3×3, p=1','36 deltas/location; bias=True','conv2d');box(h,'distance',525,485,365,'Reshape anchor rows','1 × 120,087 × 4','split');seq(h,['regtower','regpred','distance'])
  h.text(25,620,'Class and regression towers have separate parameters.',15);h.text(25,655,'Intermediate conv bias: '+('enabled, including GroupNorm paths.' if isfcos else 'disabled with GroupNorm; enabled without norm.'),15)
  h.text(25,720,'Pyramid grid        predictions per location        rows',16,weight=600)
  for i,g in enumerate([100,50,25,13,7]):
   a=1 if isfcos else 9
   for xx,txt in zip([25,340,680],[f'{g} × {g}',str(a),f'{g*g*a:,}']):h.text(xx,762+i*46,txt,16)
  h.text(25,1050,'FCOS returns raw tensors plus anchor metadata.' if isfcos else 'RetinaNet decodes boxes and sigmoid scores inside forward().',15)
  resnet_insets(d,1400,norm,eps)
  p=d.panel('decode','Location/anchor decoding and selection',40,2510,2400,420)
  if isfcos:
   ops=[('a','One square per location','Anchor sizes 8,16,32,64,128; zero-based centers','plain'),('b','Scale ReLU distances by anchor size','Center minus left/top; plus right/bottom','linear'),('c','Sigmoid class and centerness logits','Score = sqrt(class probability × centerness)','activation'),('e','COCO slot mapping; threshold; top-K; NMS','Postprocessing occurs outside raw network forward','pool')]
  else:
   ops=[('a','Nine anchors per location','Three scales × ratios 0.5, 1, 2','plain'),('b','Decode center/size deltas','Weights all 1; exp sizes clamped at log(1000/16)','linear'),('c','Sigmoid classes; map COCO slots',f'Network output: 1 × {count:,} × 84','activation'),('e','Confidence filtering and class NMS','Postprocessing follows network forward','pool')]
  for j,(id,l,dt,k) in enumerate(ops):box(p,id,20+j*595,100,555,l,dt,k)
  p.connect('a','b',from_port='right',to_port='left');p.connect('b','e',from_port='bottom',to_port='bottom',via=[(892.5,205),(2082.5,205)]);p.connect('c','e',from_port='right',to_port='left')
  p.text(25,282,'Actual 800-pixel canvas grid spacing is floor(800/grid): 8,16,32,61,114. Anchor base sizes remain separately defined.',16)
  if not isfcos:p.text(25,325,'Base anchor sizes per level: (32,40,50), (64,80,101), (128,161,203), (256,322,406), (512,645,812).',15)
  p.text(25,367,'Random-weight CPU checks cover backbone, FPN, raw heads and model outputs. No pretrained weights were downloaded.',15)
  path=out/f'{size}.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));evidence.setdefault('routes',{})[size]=routes;views.append(dict(id=size,label=title,task='detect',size=size,kind='concrete',svg=f'/diagrams/models/{family}/{size}.svg',html=f'/diagrams/models/{family}/{size}.html',input='1 × 3 × 800 × 800',verification='cpu'));print(family,size,routes['total_findings'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(family=family,slug=family,title='FCOS' if isfcos else 'RetinaNet',source_revision=rev,default_view='r50',views=views),indent=2)+'\n');evidence.update(verification='CPU all variants, random weights, no downloads',family_view='RetinaNet v1/v2 differ in normalization and P6 source; separate concrete graphs.',visual='Parent performs browser/PNG QA',reproduce=f'python scripts/model-diagrams/builders/{family}.py --source /path/to/libreyolo');ep.write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
