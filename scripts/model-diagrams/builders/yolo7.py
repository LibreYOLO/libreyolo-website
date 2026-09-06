"""YOLO7-B from the in-tree MIT MMT YAML, including sequential SPP pools."""
from pathlib import Path
import argparse,os,sys,json,subprocess,math
from yolo9 import box,seq
WEBSITE=Path(__file__).resolve().parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));ap.add_argument('--diagram-only',action='store_true');args=ap.parse_args();src=args.source.resolve();scripts=src/'skills/libreyolo-make-diagram/scripts';sys.path.insert(0,str(src));sys.path.insert(0,str(scripts));from svg_diagram import Diagram
 from libreyolo.models.yolo7.net import YOLOv7Model
 import torch,yaml
 torch.set_num_threads(4);out=WEBSITE/'public/diagrams/models/yolov7';out.mkdir(parents=True,exist_ok=True);ep=WEBSITE/'scripts/model-diagrams/evidence/yolo7.json';rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit'];ev=json.loads(ep.read_text()) if args.diagram_only else dict(family='yolo7',source_revision=rev,source_files=['libreyolo/models/yolo7/v7.yaml','libreyolo/models/yolo7/net.py','libreyolo/models/yolo7/blocks.py','libreyolo/postprocess/yolo7.py']);model=YOLOv7Model(num_classes=80).eval()
 if not args.diagram_only:
  obs={}
  def hook(name):
   def record(mod,ins,out):
    def shape(v):
     if isinstance(v,torch.Tensor):return list(v.shape)
     if isinstance(v,(list,tuple)):return [shape(x) for x in v]
     return None
    obs[name]={'input':shape(ins),'output':shape(out)}
   return record
  for name,mod in model.named_modules():
   if name:mod.register_forward_hook(hook(name))
  with torch.inference_mode():pred=model(torch.zeros(1,3,640,640))
  assert [list(x.shape) for x in pred]==[[1,255,g,g] for g in [80,40,20]];ev['shapes']=obs
 obs=ev['shapes'];cfg=yaml.safe_load((src/'libreyolo/models/yolo7/v7.yaml').read_text());entries=[]
 for section in ['backbone','head']:
  for item in cfg['model'][section]:entries.extend(item.items())
 shapes={0:[1,3,640,640]};sources={};labels={};kinds={};heights={};coords={};tags={}
 for i,(typ,info) in enumerate(entries,1):
  origin=model._sources[i-1];origin=origin if isinstance(origin,list) else[origin];sources[i]=[i-1 if x==-1 else x for x in origin];meta=obs.get('layers.'+str(i-1));shapes[i]=meta['output'] if meta else [1,sum(shapes[x][1] for x in sources[i]),*shapes[sources[i][0]][2:]]
  if typ=='Concat':assert all(shapes[x][2:]==shapes[sources[i][0]][2:] for x in sources[i])
  ar=info.get('args',{});label=typ
  if typ=='Conv':label=f'Conv {ar["kernel_size"]}×{ar["kernel_size"]}, s={ar.get("stride",1)}'
  elif typ=='Pool':label='MaxPool 2×2, s=2, p=0'
  elif typ=='UpSample':label='Nearest upsample ×2'
  if info.get('tags'):label+=' ('+info['tags']+')';tags[info['tags']]=i
  labels[i]=label;kinds[i]='conv' if typ=='Conv' else 'pool' if typ=='Pool' else 'norm' if typ=='UpSample' else 'concat' if typ=='Concat' else 'spp' if typ=='SPPCSPConv' else 'bottleneck' if typ=='RepConv' else 'activation';heights[i]=max(68,40+20*(len(sources[i])-1))
 num=len(entries);cols=math.ceil(num/28);rows=math.ceil(num/cols);colh=[]
 for col in range(cols):
  yy=80
  for i in range(col*rows+1,min(num,(col+1)*rows)+1):coords[i]=(col,yy);yy+=heights[i]+34
  colh.append(yy+35)
 mainh=max(colh);W=cols*770+80;defy=240+mainh+45;HEIGHT=defy+1470
 d=Diagram('YOLO7-B','Detection; 640 × 640 RGB; 80 classes; batch 1; unfused native eval. Layer numbers preserve the MIT v7.yaml graph.',width=W,height=HEIGHT,revision=rev,source_label='models/yolo7/v7.yaml; models/yolo7/blocks.py',source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{rev}/libreyolo/models/yolo7/v7.yaml',logo=WEBSITE/'public/icon-128.png');d.text(50,216,'Matching F labels continue a tensor across columns. All YAML branches are explicit; block internals follow the in-tree implementation.',15)
 ps=[];cross=set()
 for col in range(cols):
  lo=col*rows+1;hi=min(num,(col+1)*rows);p=d.panel('layers'+str(col),f'Layers {lo} to {hi}',40+col*770,240,746,mainh);ps.append(p)
  for i in range(lo,hi+1):
   sh=shapes[i];detail=' / '.join(' ×'.join(map(str,v[1:])) for v in sh) if sh and isinstance(sh[0],list) else ' ×'.join(map(str,sh[1:]));box(p,'L'+str(i),140,coords[i][1],420,'L'+str(i)+' '+labels[i],detail,kinds[i],entries[i-1][0],h=heights[i])
  if col==0:p.text(140,60,'Input: 3 × 640 × 640',15);p.wire([(350,65),(350,80)],start='input',end='L1')
 for col,p in enumerate(ps):
  intervals=[]
  for dest,srcs in sources.items():
   if coords[dest][0]!=col:continue
   for j,origin in enumerate(srcs):
    if origin and coords[origin][0]==col and origin!=dest-1:intervals.append((origin,dest,j))
  lanes=[];assigned={}
  for origin,dest,j in sorted(intervals):
   lane=next((k for k,last in enumerate(lanes) if last<origin),len(lanes))
   if lane==len(lanes):lanes.append(dest)
   else:lanes[lane]=dest
   assigned[origin,dest,j]=lane
  assert len(lanes)<=7,(col,len(lanes))
  for dest,srcs in sources.items():
   if coords[dest][0]!=col:continue
   yy=coords[dest][1];port_index=0
   for j,origin in enumerate(srcs):
    if origin==0:continue
    if coords[origin][0]!=col:
     off=16+20*port_index;port_index+=1;p.text(16,yy+off+5,'F'+str(origin),13,weight=600);p.wire([(85,yy+off),(140,yy+off)],start='L'+str(origin),end='L'+str(dest));cross.add(origin)
    elif origin==dest-1:p.connect('L'+str(origin),'L'+str(dest))
    else:
     sy=coords[origin][1]+heights[origin]/2;off=16+20*port_index;port_index+=1;lx=584+24*assigned[origin,dest,j];p.wire([(560,sy),(lx,sy),(lx,yy+off),(560,yy+off)],start='L'+str(origin),end='L'+str(dest))
 for origin in cross:
  col,yy=coords[origin];p=ps[col];ym=yy+heights[origin]/2;p.wire([(140,ym),(105,ym)],start='L'+str(origin),end='continuation'+str(origin));p.text(18,ym+5,'F'+str(origin),13,weight=600)
 # Primitive blocks and exact SPPCSP forward (sequential 5/9/13 pools).
 pw=(W-120)/3
 p=d.panel('convdefs','Conv and RepConv',40,defy,pw,1310,kind='conv',dashed=True,block_type='Conv')
 for j,(id,l,dt,k) in enumerate([('conv','Conv2d','No bias; p=(k-1)/2; groups 1','conv2d'),('bn','BatchNorm2d','eps=.001; momentum=.03','norm'),('silu','SiLU','Disabled inside the two RepConv branches','activation')]):box(p,id,100,85+j*130,pw-200,l,dt,k)
 seq(p,['conv','bn','silu']);box(p,'repin',240,565,pw-480,'RepConv input','128 /256 /512 channels')
 for prefix,x in [('a',25),('b',pw/2+20)]:
  box(p,'rep'+prefix,x,725,pw/2-60,'Conv2d '+('3×3' if prefix=='a' else '1×1'),'256 /512 /1024 output channels','conv2d');box(p,'rep'+prefix+'bn',x,855,pw/2-60,'BatchNorm2d','eps=.001; no branch activation','norm');cx=x+(pw/2-60)/2;p.connect('repin','rep'+prefix,via=[(pw/2,665),(cx,665)]);p.connect('rep'+prefix,'rep'+prefix+'bn')
 p.sum('repsum',pw/2,1065);p.connect('repabn','repsum',via=[(25+(pw/2-60)/2,1065)],to_port='left');p.connect('repbbn','repsum',via=[(pw/2+20+(pw/2-60)/2,1065)],to_port='right');box(p,'repact',240,1170,pw-480,'SiLU','No identity BN branch','activation');p.connect('repsum','repact')
 p=d.panel('spp','SPPCSPConv',60+pw,defy,pw,1310,kind='pool',dashed=True,block_type='SPPCSPConv')
 cx=pw/2;xx=230;ww=pw-460
 for id,yy,l,dt,k in [('si',55,'Input','1024 ×20 ×20','plain'),('pre1',145,'Conv 1×1','512 channels','conv'),('pre3',225,'Conv 3×3','512 channels','conv'),('prelast',305,'Conv 1×1','512 channels','conv'),('p5',400,'MaxPool 5×5','s=1, p=2;512 channels','pool'),('p9',495,'MaxPool 9×9','s=1, p=4;512 channels','pool'),('p13',590,'MaxPool 13×13','s=1, p=6;512 channels','pool'),('pcat',780,'Concat four sequential taps','2048 channels','concat'),('post1',880,'Conv 1×1','512 channels','conv'),('post3',970,'Conv 3×3','512 channels','conv'),('merge',1110,'Concat main and short branches','1024 channels','concat'),('out',1205,'Conv 1×1','512 ×20 ×20','conv')]:box(p,id,xx,yy,ww,l,dt,k,'Conv' if k=='conv' else '',h=43)
 seq(p,['si','pre1','pre3','prelast','p5','p9','p13'])
 for j,id in enumerate(['prelast','p5','p9','p13']):
  sx,sy=p.port(id,'bottom');lane=25+j*38;bottom=755-j*24;port=xx+35+j*(ww-70)/3;p.wire([(sx,sy),(sx,sy+14),(lane,sy+14),(lane,bottom),(port,bottom),(port,780)],start=id,end='pcat')
 seq(p,['pcat','post1','post3']);box(p,'short',pw-205,880,185,'Conv 1×1','1024 input;512 output','conv','Conv');p.connect('si','short',from_port='right',via=[(pw-112.5,76.5)]);p.wire([(cx,1013),(cx,1060),(xx+95,1060),(xx+95,1110)],start='post3',end='merge');p.wire([(pw-112.5,926),(pw-112.5,1084),(xx+ww-95,1084),(xx+ww-95,1110)],start='short',end='merge');p.connect('merge','out');p.text(20,1285,'The in-tree pools are sequential, with kernels 5 then 9 then 13.',14)
 p=d.panel('headdef','Implicit detection head and decoding',80+2*pw,defy,pw,1310,kind='plain',block_type='MultiheadDetection')
 for j,(id,l,dt,k) in enumerate([('hi','Each scale separately','256×80² /512×40² /1024×20²','plain'),('add','Add learned per-channel ImplicitA','256 /512 /1024 bias parameters','linear'),('hc','Conv2d1×1','255 output channels; bias=True','conv2d'),('multiply','Multiply learned ImplicitM','255 scale parameters per head','linear'),('raw','Raw predictions','255 ×80²;255 ×40²;255 ×20²','plain')]):box(p,id,105,70+j*120,pw-210,l,dt,k)
 seq(p,['hi','add','hc','multiply','raw']);box(p,'sigmoid',105,705,pw-210,'Reshape 3 anchors ×85 values; sigmoid','Applied to all five box/objectness fields and 80 classes','activation');p.connect('raw','sigmoid')
 for j,(id,l,dt) in enumerate([('xy','Center coordinates','(2×sigmoid(xy)-.5+grid) ×stride'),('wh','Width and height','(2×sigmoid(wh))² ×anchor size'),('score','Class confidence','sigmoid(objectness) ×sigmoid(class)')]):box(p,id,30+j*(pw-60)/3,855,(pw-90)/3,l,dt,'plain');sx=220+j*(pw-440)/2;tx=30+j*(pw-60)/3+(pw-90)/6;p.wire([(sx,751),(sx,790+j*18),(tx,790+j*18),(tx,855)],start='sigmoid',end=id)
 p.text(25,995,'Strides 8/16/32. Per-scale anchor (w,h) pairs:',15)
 for j,anchors in enumerate(model.anchors):p.text(25,1035+j*42,str(list(zip(anchors[0::2],anchors[1::2]))),15)
 p.text(25,1190,'Flatten 25,200 anchor rows; convert to xyxy; threshold and class NMS.',14);p.text(25,1235,'No pretrained weights were used for the CPU shape check.',14)
 path=out/'b.svg';d.save(path);subprocess.run([sys.executable,str(scripts/'wrap_svg.py'),str(path),'--output',str(out/'b.html')],check=True,stdout=subprocess.DEVNULL);routes=json.loads(subprocess.check_output([sys.executable,str(scripts/'check_routes.py'),str(path)],text=True));ev.update(all_layer_shapes=shapes,sources=sources,tags=tags,routes=routes,verification='CPU 640-square random-weight forward; all actual layer outputs checked',scope='MIT MultimediaTechLab graph; SPPCSPConv uses sequential 5/9/13 pooling exactly as implemented',visual='Parent performs browser/PNG QA',reproduce='python scripts/model-diagrams/builders/yolo7.py --source /path/to/libreyolo');ep.write_text(json.dumps(ev,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(family='yolo7',slug='yolov7',title='YOLOv7',source_revision=rev,default_view='b',views=[dict(id='b',label='YOLO7-B',task='detect',size='b',kind='concrete',svg='/diagrams/models/yolov7/b.svg',html='/diagrams/models/yolov7/b.html',input='1 × 3 × 640 × 640',verification='cpu')]),indent=2)+'\n');print('YOLO7 ready',num,'layers; route findings',routes['total_findings'])
if __name__=='__main__':main()
