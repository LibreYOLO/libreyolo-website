"""Source-verified FOMO s/m/l architecture posters; no external weights."""
from pathlib import Path
import argparse, os, sys, json, importlib.util, subprocess
WEBSITE=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args()
 src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 spec=importlib.util.spec_from_file_location('fomo_diagram_nn',src/'libreyolo/models/fomo/nn.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 import torch
 torch.set_num_threads(4)
 out=WEBSITE/'public/diagrams/models/fomo';out.mkdir(parents=True,exist_ok=True)
 evidence_path=WEBSITE/'scripts/model-diagrams/evidence/fomo.json'
 rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit']
 observed=json.loads(evidence_path.read_text())['shapes'] if args.diagram_only else {}
 cfgs=m.CONFIGS
 for size,cfg in cfgs.items():
  if args.diagram_only:continue
  model=m.LibreFOMOModel(size,nc=1).eval(); rec={}
  def hook(n):
   def record(mod,i,o):rec[n]={'input':list(i[0].shape),'output':list(o.shape)}
   return record
  for name,mod in model.named_modules():
   if name:mod.register_forward_hook(hook(name))
  with torch.inference_mode():y=model(torch.zeros(1,3,cfg['imgsz'],cfg['imgsz']))
  assert y.shape==(1,2,cfg['imgsz']//8,cfg['imgsz']//8)
  observed[size]=rec
 def values(size):
  cfg=cfgs[size];alpha=cfg['alpha'];return [m._make_divisible(v*alpha,8) for v in (32,16,24,32)]
 views=[]
 for size in [*cfgs,'family']:
  symbolic=size=='family';reference='m' if symbolic else size;H=cfgs[reference]['imgsz'];c0,c1,c2,c3=values(reference)
  cs=['C0','C1','C2','C3'] if symbolic else list(map(str,(c0,c1,c2,c3)))
  shape=lambda c,div: f'{c} × '+(f'H/{div} × H/{div}' if symbolic and div!=1 else ('H × H' if symbolic else f'{H//div} × {H//div}'))
  title='FOMO family' if symbolic else 'FOMO-'+size.upper()
  sub='Point detection; 1 foreground class + background; unfused PyTorch eval. Sizes exclude batch (batch = 1).'
  d=Diagram(title,sub,width=1850,height=2250,revision=rev,source_label='models/fomo/nn.py; models/fomo/utils.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/fomo/nn.py',logo=WEBSITE/'public/icon-128.png')
  p=d.panel('backbone','Truncated MobileNetV2',40,240,510,1000)
  def box(p,id,x,y,w,l,dt='',k='plain',bl='',h=49):return p.box(id,x,y,w,l,detail=dt,kind=k,block_type=bl,h=h,description=l+('. '+dt if dt else ''))
  def seq(p,ids):
   for a,b in zip(ids,ids[1:]):p.connect(a,b)
  steps=[('input','Input',shape('3',1),'plain',''),('stem','ConvBNReLU6 3×3, s=2',shape(cs[0],2),'conv','ConvBNReLU6'),('expanded','InvertedResidual (expansion 1)',shape(cs[1],2),'aggregate','ir0'),('b1','InvertedResidual (block 1, s=2)',shape(cs[2],4),'aggregate','ir1'),('b2','InvertedResidual (block 2)',shape(cs[2],4),'aggregate','ir2'),('b3','InvertedResidual (block 3, s=2)',shape(cs[3],8),'aggregate','ir3'),('b4','InvertedResidual (block 4)',shape(cs[3],8),'aggregate','ir45'),('b5','InvertedResidual (block 5)',shape(cs[3],8),'aggregate','ir45'),('b6','ConvBNReLU6 1×1',shape('6C3' if symbolic else str(6*c3),8),'conv','ConvBNReLU6')]
  for i,(id,l,dt,k,bl) in enumerate(steps):box(p,id,45,65+i*94,420,l,dt,k,bl)
  seq(p,[x[0] for x in steps]);p.text(30,958,'No box-regression tower or anchor decoding.',16)
  head=d.panel('head','Point head and postprocessing',580,240,1230,410)
  box(head,'headin',25,70,245,'Feature map',shape('6C3' if symbolic else str(6*c3),8),'plain')
  box(head,'logits',335,70,245,'Conv2d 1×1, bias=True',shape('2',8),'conv2d')
  box(head,'prob',650,70,245,'Softmax over 2 classes',shape('2',8),'activation')
  box(head,'fg',945,70,255,'Foreground probability',shape('1',8),'split')
  for a,b in [('headin','logits'),('logits','prob'),('prob','fg')]:head.connect(a,b,from_port='right',to_port='left')
  box(head,'threshold',945,180,255,'Threshold, sort scores','Descending foreground confidence','plain')
  box(head,'nms',650,180,245,'Grid-radius suppression','Default radius = 1 cell','pool')
  box(head,'point',335,180,245,'Grid-cell centers','x + 0.5; y + 0.5','plain')
  box(head,'result',25,180,245,'Rescale to original image','Rows: x, y, class, confidence','plain')
  head.connect('fg','threshold');
  for a,b in [('threshold','nms'),('nms','point'),('point','result')]:head.connect(a,b,from_port='left',to_port='right')
  head.text(25,290,'The neural network ends at logits. Softmax and point selection run in utils.postprocess().',16)
  if symbolic:
   
   for xx,txt in zip([25,110,255,380,505,630,765,975],['Size','Input H','C0','C1','C2','C3','Head input','Output grid']):head.text(xx,329,txt,15,weight=600)
   for j,sz in enumerate(cfgs):
    cc=values(sz);ii=cfgs[sz]['imgsz']
    for xx,txt in zip([25,110,255,380,505,630,765,975],[sz.upper(),ii,*cc,6*cc[3],f'{ii//8} × {ii//8}']):head.text(xx,354+22*j,str(txt),14)
  else:head.text(25,335,f'Random-weight CPU output: 1 × 2 × {H//8} × {H//8}. Input resolution is fixed when building the model.',16)
  # Labeled continuation keeps a long backbone tensor distinct from local head wires.
  p.text(300,934,'Feature F',14);p.wire([(255,866),(255,925),(290,925)],start='b6',end='headin');head.text(25,55,'Feature F from backbone',14)
  # Explicit residual units. Expansion, depthwise, projection, normalization, and activation are separate operations.
  specs=[('ir0','Initial block',c0,c0,c1,1,False,False,2),('ir1','Block 1',c1,c1*6,c2,2,False,True,4),('ir2','Block 2',c2,c2*6,c2,1,True,True,4),('ir3','Block 3',c2,c2*6,c3,2,False,True,8),('ir45','Blocks 4 and 5',c3,c3*6,c3,1,True,True,8)]
  for ix,(id,label,ci,ch,co,stride,resid,expand,div) in enumerate(specs):
   col=ix%3;row=ix//3;x=580+col*410;y=680+row*715
   pan=d.panel(id,label,x,y,390,685,kind='aggregate',dashed=True,block_type=id,description='MobileNetV2 inverted residual; exact operations for this stage.')
   if symbolic:
    ins=['C0','C1','C2','C2','C3'][ix];hidden=['C0','6C1','6C2','6C2','6C3'][ix];outs=['C1','C2','C2','C3','C3'][ix]
   else:ins,hidden,outs=map(str,(ci,ch,co))
   labels=[];yy=62
   for suffix,l,dt,k in [('i','Input',ins+' channels','plain')]+([('ex','Conv2d 1×1',hidden+' ch; bias=False','conv2d'),('ebn','BatchNorm2d + ReLU6','eps=0.001','norm')] if expand else [])+[('pad','Static zero padding',('TensorFlow SAME; input geometry fixed' if symbolic else 'L, R, T, B: '+str(m._same_pad_2d((H//(div//stride),)*2,(3,3),(stride,stride)))),'plain'),('dw','Depthwise Conv2d 3×3',f'{hidden} ch; groups={hidden}; s={stride}','conv2d'),('dbn','BatchNorm2d + ReLU6','eps=0.001','norm'),('proj','Conv2d 1×1',outs+' ch; bias=False','conv2d'),('pbn','BatchNorm2d','eps=0.001; no activation','norm')]:
    nid=id+'-'+suffix;box(pan,nid,90,yy,280,l,dt,k,h=44);labels.append(nid);yy+=65
   seq(pan,labels)
   if resid:
    pan.sum(id+'-add',230,yy+8);pan.connect(labels[-1],id+'-add');pan.connect(labels[0],id+'-add',from_port='left',via=[(30,84),(30,yy+8)],to_port='left');pan.text(35,yy-20,'identity',13)
   pan.text(20,658,('Each block has its own weights.' if ix==4 else ('Residual enabled.' if resid else 'No residual connection.')),14)
  # Primitive definitions are visible for composite stage labels.
  conv=d.panel('convdef','ConvBNReLU6',1400,1395,410,685,kind='conv',dashed=True,block_type='ConvBNReLU6')
  for i,(id,l,dt,k) in enumerate([('cpad','Static zero padding','SAME for the model input geometry','plain'),('cconv','Conv2d','k and stride given by stage','conv2d'),('cbn','BatchNorm2d','eps=0.001','norm'),('crelu','ReLU6','Clamp activations to [0, 6]','activation')]):box(conv,id,60,80+i*115,290,l,dt,k)
  seq(conv,['cpad','cconv','cbn','crelu']);conv.text(25,588,'Depthwise uses one group per input channel.',14);conv.text(25,617,'Projection Conv2d has no ReLU6.',14)
  d.text(50,2160,'Source-verified architecture. CPU shape evidence covers s, m, l with no downloaded weights. Point postprocessing is shown separately.',16)
  path=out/f'{size}.svg';d.save(path)
  subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/f'{size}.html')],check=True,stdout=subprocess.DEVNULL)
  check=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));assert check['total_findings']==0,check
  views.append(dict(id=size,label=title,task='point',size='s/m/l' if symbolic else size,kind='family' if symbolic else 'concrete',svg=f'/diagrams/models/fomo/{size}.svg',html=f'/diagrams/models/fomo/{size}.html',input='H × H RGB; H in table' if symbolic else f'1 × 3 × {H} × {H}',verification='cpu'))
 (out/'manifest.json').write_text(json.dumps(dict(family='fomo',slug='fomo',title='FOMO',source_revision=rev,default_view='m',views=views),indent=2)+'\n')
 evidence_path.write_text(json.dumps(dict(family='fomo',source_revision=rev,source_files=['libreyolo/models/fomo/nn.py','libreyolo/models/fomo/utils.py'],verification='cpu',weights='random; no downloads',shapes=observed,route_checker='zero findings in every view',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/fomo.py --source /path/to/libreyolo'),indent=2)+'\n')
 print('FOMO ready:',len(views),'views')
if __name__=='__main__':main()
