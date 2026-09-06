from alexnet import *
def main():
 a=setup();b=Book(a,'facerec','Face recognition',sourcefile='model.py');pre=load(a.source,'facerec','preprocess.py');align=load(a.source,'facerec','align.py');import numpy as np
 crop=np.zeros((112,112,3),dtype=np.uint8);blob=pre.preprocess_aligned(crop,pre.PreprocCfg.arcface());matrix=align.estimate_norm(align.ARCFACE_DST_112);normalized=pre.l2_normalize(np.ones((2,512),dtype=np.float32))
 b.evidence={'preprocess_cpu':{'input':list(crop.shape),'output':list(blob.shape),'dtype':str(blob.dtype),'range':[float(blob.min()),float(blob.max())]},'alignment_cpu':matrix.tolist(),'normalization_cpu':{'input':[2,512],'output':list(normalized.shape),'l2_norms':np.linalg.norm(normalized,axis=1).tolist()},'boundary':'Recognition is an externally supplied ONNX artifact. Its internal operators were unavailable under the no-download constraint; no iResNet100 internals are inferred. Named l artifact declares 512-dimensional output in weights.py.'}
 d=b.diagram('Face recognition l','Embedding pipeline, aligned input 112 × 112, 512-dimensional output per face. Actual recognition and detector graphs have dedicated views.',1800,1880)
 p=d.panel('faces','Face regions',30,220,550,1210)
 p.box('image',100,65,340,'RGB image',detail='H × W × 3, uint8')
 p.box('detector',25,205,225,'Face detector',detail='External ONNX or caller model',font_size=14)
 p.box('boxes',305,205,220,'Supplied face boxes',detail='Detector bypass',font_size=14)
 p.connect('image','detector',via=[(270,155),(137.5,155)]);p.connect('image','boxes',via=[(270,175),(415,175)])
 p.box('regions',100,355,340,'Choose face source',detail='Boxes and optional 5-point landmarks');p.wire([(137.5,254),(137.5,310),(200,310),(200,355)],start='detector',end='regions');p.wire([(415,254),(415,330),(340,330),(340,355)],start='boxes',end='regions')
 p.box('warp',25,525,225,'Similarity transform',detail='5 landmarks to canonical template',font_size=14)
 p.box('crop',305,525,220,'Center-square crop',detail='When landmarks are unavailable',font_size=14)
 p.connect('regions','warp',via=[(270,460),(137.5,460)]);p.connect('regions','crop',via=[(270,485),(415,485)])
 p.box('aligned',100,705,340,'Choose aligned crop',detail='112 × 112 × 3 RGB');p.wire([(137.5,574),(137.5,650),(200,650),(200,705)],start='warp',end='aligned');p.wire([(415,574),(415,670),(340,670),(340,705)],start='crop',end='aligned')
 p.text(25,820,'Both routes run once per detected face.',17);p.text(25,860,'Detection graph is supplied externally.',17)
 p=d.panel('recognition','Recognition',620,220,570,1210)
 ids=chain(p,'rec',[('Aligned face crop','112 × 112 × 3 RGB, uint8','plain'),('Cast to float32','112 × 112 × 3','plain'),('Subtract mean','127.5 from each RGB channel','plain'),('Multiply scale','1 / 127.5','plain'),('Transpose and add batch','1 × 3 × 112 × 112','plain'),('ONNX Runtime recognition','iResNet100 ONNX; output 1 × 512','plain'),('Collect all face outputs','F × 512; F is the number of faces','concat'),('L2 normalize rows','F × 512; denominator clipped at 1e-10','plain')],w=470,gap=115)
 d.connect('aligned',ids[0],from_port='right',via=[(600,949.5),(600,265),(885,265)])
 p=d.panel('usage','Embedding output',1230,220,540,1210,kind='pool')
 ids=chain(p,'out',[('Results embeddings','F × 512 normalized identity vectors','plain'),('Optional verification','Select highest-confidence face per image','plain'),('Dot product','Cosine similarity between two unit vectors','attention'),('Compare threshold','same_person = similarity >= threshold','plain')],w=440,gap=120)
 p.text(25,650,'Gallery identification is optional.',17);p.text(25,690,'Output dimension for custom ONNX is D,',17);p.text(25,725,'read from that artifact or its runtime output.',17)
 p=d.panel('boundary','External graph boundary',30,1480,1740,230)
 p.text(25,70,'This drawing covers the in-tree orchestration, alignment, preprocessing and output contract.',18)
 p.text(25,112,'The l-core-embed view shows all recognition blocks; detector-640 shows the face-detection backbone, FPN and 12 output heads.',17)
 p.text(25,154,'Those views are derived from the actual licensed ONNX operators and inferred tensor shapes; their SHA-256 hashes are recorded.',17)
 b.save(d,'l-embed','l',task='embed',verification='source',input='Per face: 1 × 3 × 112 × 112')
 graphs=json.loads((b.path/'onnx-graphs.json').read_text());recognition_graph(b,graphs);detector_graph(b,graphs)
 b.views=[b.views[1],b.views[2],b.views[0]]
 b.evidence['onnx_artifacts']=json.loads(Path('/tmp/libreyolo-diagram-facerec/metadata.json').read_text()) if Path('/tmp/libreyolo-diagram-facerec/metadata.json').exists() else {'recognition_sha256':'a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60','detector_sha256':'8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'}
 b.evidence['boundary']='Both actual ONNX graphs were downloaded under their documented permissive licenses and inspected with ONNX shape inference at batch1. Full recognition and detector views replace the earlier opaque-core boundary.'
 b.finish()

def recognition_graph(b,graphs):
 import re
 g=graphs['librefacerec-l'];nodes=g['nodes'];groups={}
 for n in nodes:
  match=re.match(r'/layer(\d+)/layer\d+\.(\d+)/',n['name'])
  if match:groups.setdefault((int(match[1]),int(match[2])),[]).append(n)
 counts=[len({k[1] for k in groups if k[0]==s}) for s in range(1,5)];assert counts==[3,13,30,3]
 d=b.diagram('Face recognition l: ONNX iResNet100','Aligned RGB input 112 × 112, batch 1, 512-dimensional embedding. Actual exported ONNX operators, with fused convolution parameters.',2320,2180)
 p=d.panel('network','Recognition network',25,220,450,1630)
 items=[('Aligned RGB face','3 × 112 × 112','plain'),('Normalize (x - 127.5) / 127.5','3 × 112 × 112','plain'),('Conv 3×3 / 1','3 to 64, p=1; fused bias','conv2d'),('PReLU','64 × 112 × 112','activation')]+[(f'Layer {i+1}',f'{[64,128,256,512][i]} × {56//2**i} × {56//2**i}, n={counts[i]}','bottleneck') for i in range(4)]+[('BatchNormalization','512 × 7 × 7','norm'),('Flatten','25,088','plain'),('Gemm','25,088 to 512, bias=True','linear'),('BatchNormalization','512','norm'),('L2 normalize','512-dimensional identity vector','plain')]
 chain(p,'recnet',items,w=370,gap=105)
 for i,c in enumerate([64,128,256,512]):
  p=d.panel('stage'+str(i),f'Layer {i+1}',510+i*450,220,425,1630,kind='bottleneck',dashed=True)
  for part,y,blocknum,n in [('entry',65,0,1),('repeat',855,1,counts[i]-1)]:
   group=groups[(i+1,blocknum)];first=group[0];shape=g['shapes'][first['inputs'][0]][1:];p.text(20,y,('Downsample block' if part=='entry' else 'Identity block')+f', n={n}',17,weight=700);p.text(20,y+35,'Input '+dim(shape),14);start=y+65;p.dot(140,start)
   main=[node for node in group if node['op']!='Add' and '/downsample/' not in node['name']];ids=[]
   for j,node in enumerate(main):
    label=node['op'];detail=dim(node['shape'][1:]);kind='norm' if label=='BatchNormalization' else 'activation' if label=='PRelu' else 'conv2d'
    if label=='Conv':
     a=node['attributes'];label=f'Conv {a["kernel_shape"][0]}×{a["kernel_shape"][1]} / {a["strides"][0]}';detail+=f', p={a["pads"][0]}'
    id=f'{i}-{part}-{j}';p.box(id,20,start+35+j*105,240,label,detail=detail,kind=kind,font_size=14);ids.append(id)
    if j:p.connect(ids[-2],ids[-1])
   p.wire([(140,start),(140,start+35)],end=ids[0]);addy=start+530;p.sum(f'sum{i}{part}',140,addy);p.connect(ids[-1],f'sum{i}{part}')
   skip=[node for node in group if '/downsample/' in node['name']]
   if skip:
    node=skip[0];a=node['attributes'];p.box(f'skip{i}',280,start+160,125,'Conv 1×1 / 2',detail=dim(node['shape'][1:]),kind='conv2d',font_size=12);p.wire([(140,start),(342.5,start),(342.5,start+160)],end=f'skip{i}');p.connect(f'skip{i}',f'sum{i}{part}',via=[(342.5,addy)],to_port='right')
   else:p.wire([(140,start),(342.5,start),(342.5,addy),(153,addy)],end=f'sum{i}{part}')
   p.text(20,addy+55,'Output '+dim(group[-1]['shape'][1:]),14)
 p=d.panel('source','Artifact verification',25,1910,2265,155)
 p.text(25,65,'librefacerec-l.onnx: 103 Conv, 51 BatchNormalization, 50 PRelu, 49 Add, one Flatten and one Gemm.',17)
 p.text(25,108,'SHA-256 a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60. Shape inference pinned batch 1.',14)
 b.save(d,'l-core-embed','l',task='embed',verification='source',input='1 × 3 × 112 × 112')

def detector_graph(b,graphs):
 g=graphs['librefacerec-det'];assert len(g['nodes'])==106
 d=b.diagram('Face recognition default detector: ONNX graph','Face boxes and five landmarks, 640 × 640 artifact input. Three output strides: 8, 16 and 32.',2920,3590)
 p=d.panel('backbone','Backbone',25,220,560,1580)
 items=[('Input','3 × 640 × 640','plain'),('Conv 3×3 / 2 + ReLU','3 to 16, 320 × 320','conv'),('Separable unit','16 to 16, 320 × 320','conv'),('MaxPool 2×2 / 2','16 × 160 × 160','pool'),('Separable unit','16 to 16, 160 × 160','conv'),('Separable unit','16 to 32, 160 × 160','conv'),('Separable unit','32 to 32, 160 × 160','conv'),('Separable unit','32 to 64, 160 × 160','conv'),('MaxPool 2×2 / 2','64 × 80 × 80','pool'),('Two separable units','64 to 64, 80 × 80','conv'),('MaxPool 2×2 / 2','64 × 40 × 40','pool'),('Two separable units','64 to 64, 40 × 40','conv'),('MaxPool 2×2 / 2','64 × 20 × 20','pool'),('Two separable units','64 to 64, 20 × 20','conv')];chain(p,'back',items,w=465,gap=101)
 p=d.panel('neck','Feature pyramid',620,220,1180,1580,kind='aggregate')
 ids=chain(p,'fp',[('Deep backbone output','64 × 20 × 20','plain'),('Separable unit','64 to 64, 20 × 20; stride-32 head feature','conv'),('Nearest resize ×2','64 × 40 × 40','pool')],w=480,gap=140)
 p.box('tap40',680,345,450,'Backbone 40×40 tap',detail='64 × 40 × 40');p.sum('add40',270,510);p.connect(ids[-1],'add40');p.connect('tap40','add40',via=[(905,510)],to_port='right')
 p.box('neck40',30,630,480,'Separable unit',detail='64 × 40 × 40; stride-16 head feature',kind='conv');p.connect('add40','neck40');p.box('resize80',30,790,480,'Nearest resize ×2',detail='64 × 80 × 80',kind='pool');p.connect('neck40','resize80')
 p.box('tap80',680,790,450,'Backbone 80×80 tap',detail='64 × 80 × 80');p.sum('add80',270,970);p.connect('resize80','add80');p.connect('tap80','add80',via=[(905,970)],to_port='right')
 p.box('neck80',30,1090,480,'Separable unit',detail='64 × 80 × 80; stride-8 head feature',kind='conv');p.connect('add80','neck80')
 p.text(30,1260,'Each feature feeds four independent prediction towers.',17)
 p=d.panel('unit','Separable unit',1835,220,1060,750,kind='conv',dashed=True)
 chain(p,'unit',[('Conv 1×1','Input to output channels, bias=True','conv2d'),('Depthwise Conv 3×3 / 1','Output channels, groups=output channels, p=1','conv2d'),('ReLU','Same tensor shape','activation')],w=910,gap=145)
 p.text(30,590,'Resolved channel pairs in this graph: 16/16, 16/32, 32/32, 32/64, 64/64.',17)
 p=d.panel('stem','Stem primitive',1835,1010,1060,790,kind='conv',dashed=True)
 chain(p,'stem',[('Conv 3×3 / 2','3 to 16, padding 1, bias=True','conv2d'),('ReLU','16 × 320 × 320','activation')],w=910,gap=145)
 p.text(30,490,'ONNX BatchNorm parameters are folded into the exported Conv weights/biases.',17)
 p.text(30,555,'Head activation is sigmoid for cls and obj; bbox and kps outputs remain linear.',17)
 for si,(stride,h,locs) in enumerate([(8,80,6400),(16,40,1600),(32,20,400)]):
  p=d.panel('head'+str(stride),f'Stride {stride} output heads',25+si*970,1860,940,1450,kind='pool')
  for j,(name,channels,sigmoid) in enumerate([('cls',1,True),('obj',1,True),('bbox',4,False),('kps',10,False)]):
   xx=20+j*230;ops=[('Feature input',f'64 × {h} × {h}','plain'),('Conv 1×1',f'64 to {channels}','conv2d'),('Conv 3×3 / 1',f'g={channels}, p=1','conv2d'),('Transpose NHWC',f'{h} × {h} × {channels}','plain'),('Reshape',f'{locs} × {channels}','plain')]
   if sigmoid:ops.append(('Sigmoid',f'{locs} × {channels}','activation'))
   ids=chain(p,name+str(stride),ops,x=xx,w=205,y=90,gap=135);x,y=p.port(ids[-1]);p.text(x,y+65,f'{name}_{stride}',17,weight=700,anchor='middle')
  p.text(20,1080,'kps stores 5 coordinate pairs per location.',17)
  p.text(20,1150,'Each branch has separate learned Conv weights.',17)
 d.text(35,3390,'Actual librefacerec-det.onnx: 106 operators, 53 Conv, 15 Relu, 4 MaxPool, 2 Resize, 2 Add, 12 Transpose, 12 Reshape, 6 Sigmoid.',16)
 d.text(35,3440,'SHA-256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4. Postprocessing/5-point alignment is shown in the pipeline view.',14)
 b.save(d,'detector-640','det',task='detect',verification='source',input='1 × 3 × 640 × 640')

if __name__=='__main__':main()
