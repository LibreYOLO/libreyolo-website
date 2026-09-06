"""SSD300 VGG16, six scale-specific heads and default-box decoding."""
from pathlib import Path
import argparse,os,sys,json,subprocess
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.ssd.nn import LibreSSDModel
 import torch
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/ssd';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/ssd.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];evidence=json.loads(ep.read_text()) if args.diagram_only else dict(family='ssd',source_revision=rev,source_files=['libreyolo/models/ssd/nn.py','libreyolo/postprocess/ssd.py'],shapes={})
 if not args.diagram_only:
  model=LibreSSDModel(num_classes=91).eval();obs={}
  def hook(name):
   def record(mod,ins,output):
    def shape(v):
     if isinstance(v,torch.Tensor):return list(v.shape)
     if isinstance(v,(list,tuple)):return [shape(x) for x in v]
     if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
     return None
    obs[name]={'input':shape(ins),'output':shape(output)}
   return record
  for name,mod in model.named_modules():
   if name:mod.register_forward_hook(hook(name))
  with torch.inference_mode():pred=model(torch.zeros(1,3,300,300))
  assert pred['bbox_regression'].shape==(1,8732,4);assert pred['cls_logits'].shape==(1,8732,91);evidence['shapes']=obs
 d=Diagram('SSD300-VGG16','Detection; 300 × 300 RGB; batch 1. COCO: 91 head slots, including background and unused IDs, mapped to 80 classes.',width=2300,height=2790,revision=rev,source_label='models/ssd/nn.py; postprocess/ssd.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/ssd/nn.py',logo=WEBSITE/'public/icon-128.png')
 b=d.panel('backbone','VGG16 and SSD feature extractor',40,230,620,1700)
 stages=[('input','Input','3 × 300 × 300','plain',''),('conv1','Conv-ReLU 3×3 repeated 2 times','64 × 300 × 300','conv','ConvReLU'),('pool1','MaxPool2d 2×2, s=2','64 × 150 × 150','pool',''),('conv2','Conv-ReLU 3×3 repeated 2 times','128 × 150 × 150','conv','ConvReLU'),('pool2','MaxPool2d 2×2, s=2','128 × 75 × 75','pool',''),('conv3','Conv-ReLU 3×3 repeated 3 times','256 × 75 × 75','conv','ConvReLU'),('pool3','MaxPool2d 2×2, s=2, ceil=True','256 × 38 × 38','pool',''),('conv4','Conv-ReLU 3×3 repeated 3 times','512 × 38 × 38; unscaled continuation','conv','ConvReLU'),('pool4','MaxPool2d 2×2, s=2','512 × 19 × 19','pool',''),('conv5','Conv-ReLU 3×3 repeated 3 times','512 × 19 × 19','conv','ConvReLU'),('pool5','MaxPool2d 3×3, s=1, p=1','512 × 19 × 19','pool',''),('conv6','Conv-ReLU 3×3, dilation=6, p=6','1024 × 19 × 19','conv','ConvReLU'),('F1','Conv-ReLU 1×1','1024 × 19 × 19 (F1)','conv','ConvReLU'),('e2','Extra block','512 × 10 × 10 (F2)','aggregate','Extra'),('e3','Extra block','256 × 5 × 5 (F3)','aggregate','Extra'),('e4','Extra block','256 × 3 × 3 (F4)','aggregate','Extra'),('e5','Extra block','256 × 1 × 1 (F5)','aggregate','Extra')]
 for i,(id,l,dt,k,bl) in enumerate(stages):box(b,id,70,65+i*94,480,l,dt,k,bl)
 seq(b,[x[0] for x in stages]);b.text(20,1660,'All repeated convs use p=1 and s=1 unless marked.',15)
 scale=d.panel('l2','F0 feature normalization',690,230,1570,320,kind='plain',block_type='L2Normalize')
 box(scale,'f0in',25,90,360,'Unscaled conv4_3','512 × 38 × 38');box(scale,'norm',430,90,430,'L2 normalization across channels','Divide by max(L2 norm, 0.000000000001)','norm');box(scale,'scale',910,90,430,'Multiply learned channel scale','512 weights, initialized to 20; F0 = 512 × 38 × 38','linear');scale.connect('f0in','norm',from_port='right',to_port='left');scale.connect('norm','scale',from_port='right',to_port='left');scale.text(25,230,'Only F0 is normalized. The unscaled conv4_3 feature continues through pool4 to the later feature blocks.',16)
 h=d.panel('heads','Six independent box and class heads',690,580,1570,1100)
 features=[(512,38,4),(1024,19,6),(512,10,6),(256,5,6),(256,3,4),(256,1,4)]
 for i,(c,g,a) in enumerate(features):
  y=80+i*145;prefix='h'+str(i);box(h,prefix+'in',20,y,270,f'F{i}',f'{c} × {g} × {g}')
  box(h,prefix+'box',355,y,330,'Conv2d 3×3, p=1',f'{a*4} output channels; bias=True','conv2d');box(h,prefix+'class',355,y+67,330,'Conv2d 3×3, p=1',f'{a*91} output channels; bias=True','conv2d')
  h.connect(prefix+'in',prefix+'box',from_port='right',to_port='left');h.connect(prefix+'in',prefix+'class',from_port='right',to_port='left',via=[(320,y+23),(320,y+90)])
  box(h,prefix+'bflat',740,y,280,'Reshape and permute',f'1 × {a*g*g:,} × 4','split');box(h,prefix+'cflat',740,y+67,280,'Reshape and permute',f'1 × {a*g*g:,} × 91','split')
  h.connect(prefix+'box',prefix+'bflat',from_port='right',to_port='left');h.connect(prefix+'class',prefix+'cflat',from_port='right',to_port='left');h.wire([(1020,y+23),(1048,y+23)],start=prefix+'bflat',end='B'+str(i));h.text(1055,y+28,'B'+str(i),13,weight=600);h.wire([(1020,y+90),(1048,y+90)],start=prefix+'cflat',end='C'+str(i));h.text(1055,y+95,'C'+str(i),13,weight=600)
 box(h,'boxcat',1190,100,340,'Concat box levels','1 × 8,732 × 4','concat',h=220)
 box(h,'classcat',1190,540,340,'Concat class levels','1 × 8,732 × 91','concat',h=220)
 for i in range(6):
  for pre,end,yy in [('B','boxcat',155+24*i),('C','classcat',595+24*i)]:
   h.text(1120,yy+5,pre+str(i),13,weight=600);h.wire([(1150,yy),(1190,yy)],start='h'+str(i)+('bflat' if pre=='B' else 'cflat'),end=end)
 h.text(25,990,'Concat box rows across F0...F5: 1 × 8,732 × 4. Concat class rows: 1 × 8,732 × 91.',18,weight=600);h.text(25,1040,'Anchor-row order is spatial location, then anchor; every level has its own prediction convolutions.',16)
 conv=d.panel('conv','Conv-ReLU',690,1710,620,220,kind='conv',dashed=True,block_type='ConvReLU');box(conv,'primitiveconv',20,85,260,'Conv2d','Bias=True; k/s/p from stage','conv2d');box(conv,'relu',335,85,260,'ReLU','No BatchNorm','activation');conv.connect('primitiveconv','relu',from_port='right',to_port='left')
 extra=d.panel('extra','Extra block (fully resolved)',1340,1710,920,220,kind='aggregate',dashed=True,block_type='Extra')
 extra.text(20,63,'Each line is Conv2d 1×1 + ReLU, then Conv2d 3×3 + ReLU.',15)
 for j,line in enumerate(['F2: 1024 input; 256 hidden; 512 output; second conv s=2, p=1.','F3: 512 input; 128 hidden; 256 output; second conv s=2, p=1.','F4: 256 input; 128 hidden; 256 output; second conv s=1, p=0.','F5: 256 input; 128 hidden; 256 output; second conv s=1, p=0.']):extra.text(20,95+j*29,line,15)
 # Separate explicit extra-block circuit, followed by decoding branches.
 dec=d.panel('decode','Extra-block internals and SSD decoding',40,1970,2220,690,kind='plain')
 for i,(id,l,dt,k) in enumerate([('ext1','Conv2d 1×1','Hidden widths listed above','conv2d'),('extrelu1','ReLU','','activation'),('ext3','Conv2d 3×3','Output/stride/padding listed above','conv2d'),('extrelu2','ReLU','','activation')]):box(dec,id,25+i*330,70,280,l,dt,k)
 for a,z in zip(['ext1','extrelu1','ext3'],['extrelu1','ext3','extrelu2']):dec.connect(a,z,from_port='right',to_port='left')
 box(dec,'offset',30,230,540,'Box offsets: 8,732 × 4','Divide center deltas by 10; size deltas by 5','split');box(dec,'anchors',665,230,800,'Generate 8,732 default boxes','Steps: 8, 16, 32, 64, 100, 300; centers have +0.5 offset','plain');box(dec,'logits',1580,230,570,'Class logits: 8,732 × 91','Softmax over 91 slots','activation')
 box(dec,'decodebox',270,385,880,'Apply deltas to default-box centers and sizes','Size exponent is clamped at log(1000/16); exp sizes; convert xyxy','linear');dec.wire([(300,276),(300,345),(450,345),(450,385)],start='offset',end='decodebox');dec.wire([(1065,276),(1065,365),(950,365),(950,385)],start='anchors',end='decodebox');box(dec,'select',1580,385,570,'Drop background; map sparse COCO IDs','Threshold; top-400 per class','split');dec.connect('logits','select');box(dec,'nms',1200,505,950,'Class-wise NMS; retain at most 200; invert image resize','Final boxes, scores and 80-class IDs','pool');dec.connect('select','nms',via=[(1865,475),(1675,475)]);dec.connect('decodebox','nms',from_port='right',to_port='left',via=[(1175,408),(1175,528)])
 dec.text(25,610,'Default-box scales: 0.07, 0.15, 0.33, 0.51, 0.69, 0.87; next scale 1.05. Aspect ratios per level: [2], [2,3], [2,3], [2,3], [2], [2].',15)
 dec.text(25,644,'Each level includes its scale square and geometric-mean square, plus reciprocal aspect-ratio pairs. Sizes are clipped to [0,1] before pixel scaling.',15)
 path=out/'300.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/'300.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));evidence.update(routes=routes,verification='CPU random-weight forward; all feature/head modules hooked',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/ssd.py --source /path/to/libreyolo');ep.write_text(json.dumps(evidence,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(family='ssd',slug='ssd',title='SSD300',source_revision=rev,default_view='300',views=[dict(id='300',label='SSD300 VGG16',task='detect',size='300',kind='concrete',svg='/diagrams/models/ssd/300.svg',html='/diagrams/models/ssd/300.html',input='1 × 3 × 300 × 300',verification='cpu')]),indent=2)+'\n');print('SSD ready; route findings',routes['total_findings'])
if __name__=='__main__':main()
