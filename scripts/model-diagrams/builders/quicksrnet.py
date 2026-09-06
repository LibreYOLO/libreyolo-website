"""QuickSRNet architecture and reusable drawing I/O for the dense-vision batch.

Architecture labels are independently authored from the in-tree sources.
The shared functions below handle packaging only, never infer model topology.
"""
from pathlib import Path
import argparse, importlib.util, json, os, subprocess, sys
WEBSITE=Path(__file__).resolve().parents[3]
DEFAULT_SOURCE=Path('/Users/xuban.ceccon/Documents/personal/libreyolo')

def environment(description):
 p=argparse.ArgumentParser(description=description)
 p.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',DEFAULT_SOURCE)))
 p.add_argument('--verify',action='store_true',help='Run random-weight CPU shape checks')
 a=p.parse_args()
 a.source=a.source.resolve()
 sys.path.insert(0,str(a.source/'skills/libreyolo-make-diagram/scripts'))
 a.revision=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
 return a

def nn_module(a,family):
 name='diagram_'+family
 spec=importlib.util.spec_from_file_location(name,a.source/f'libreyolo/models/{family}/nn.py')
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except ImportError as exc:
  if 'relative import' not in str(exc):raise
  from importlib import import_module
  sys.path.insert(0,str(a.source))
  m=import_module('libreyolo.models.'+family+'.nn')
 return m

def tensor_shapes(o):
 if hasattr(o,'shape'):return list(o.shape)
 if isinstance(o,dict):return {str(k):tensor_shapes(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [tensor_shapes(v) for v in o]
 return str(type(o))

def cpu_probe(model,shape,names=(),input_range=None,input_tensor=None):
 import torch
 torch.manual_seed(0);torch.set_num_threads(4)
 model.eval();results={}
 for name,module in model.named_modules():
  if name in names:module.register_forward_hook(lambda m,i,o,n=name:results.__setitem__(n,{'input':tensor_shapes(i),'output':tensor_shapes(o)}))
 with torch.inference_mode():
  value=input_tensor if input_tensor is not None else torch.randn(*shape) if input_range is None else torch.rand(*shape)*(input_range[1]-input_range[0])+input_range[0]
  out=model(value)
 return {'device':'cpu','torch':torch.__version__,'input':list(shape),'output':tensor_shapes(out),'hooks':results,'parameters':sum(p.numel() for p in model.parameters()),'pretrained':False}

def evidence_path(family):return WEBSITE/'scripts/model-diagrams/evidence'/f'{family}.json'
def write_evidence(a,family,records,sources,notes=()):
 p=evidence_path(family);p.parent.mkdir(parents=True,exist_ok=True)
 data={'family':family,'source_revision':a.revision,'sources':sources,'records':records,'notes':list(notes),'visual_qa':'Parent agent; not executed by this builder.'}
 p.write_text(json.dumps(data,indent=2)+'\n');return data

def read_evidence(family):
 p=evidence_path(family)
 return json.loads(p.read_text()) if p.exists() else {'records':{}}

def diagram(a,title,subtitle,family,width,height):
 from svg_diagram import Diagram
 return Diagram(title,subtitle,width=width,height=height,source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{a.revision}/libreyolo/models/{family}/nn.py',source_label=f'libreyolo/models/{family}/nn.py and model.py',revision=a.revision,logo=WEBSITE/'public/icon-128.png')

def op(p,id,x,y,w,label,detail='',kind='plain',h=49,block=''):
 return p.box(id,x,y,w,label,h=h,detail=detail,kind=kind,block_type=block,description=detail or label,font_size=15)
def chain(p,ids):
 for first,second in zip(ids,ids[1:]):p.connect(first,second)
def finish_view(a,d,slug,id,label,task,size,kind,input,verification):
 from check_routes import check
 out=WEBSITE/'public/diagrams/models'/slug;out.mkdir(parents=True,exist_ok=True)
 for e in d.root.iter():
  if e.get('class')=='wire':e.set('stroke-width','2')
 for e in d.global_wires.iter():
  if e.get('class')=='wire':e.set('stroke-width','2')
 path=d.save(out/f'{id}.svg')
 routes=check(d.root)
 (out/f'{id}.routes.json').write_text(json.dumps(routes,indent=2)+'\n')
 subprocess.run([sys.executable,str(a.source/'skills/libreyolo-make-diagram/scripts/wrap_svg.py'),str(path),'--output',str(out/f'{id}.html')],check=True,stdout=subprocess.DEVNULL)
 if routes['findings']:print(f'{slug}/{id}: {len(routes["findings"])} route findings',file=sys.stderr)
 return {'id':id,'label':label,'task':task,'size':size,'kind':kind,'svg':f'/diagrams/models/{slug}/{id}.svg','html':f'/diagrams/models/{slug}/{id}.html','input':input,'verification':verification}
def manifest(a,family,slug,title,views):
 data={'family':family,'slug':slug,'title':title,'source_revision':a.revision,'default_view':next(v['id'] for v in views if v['kind']=='concrete'),'views':views}
 p=WEBSITE/'public/diagrams/models'/slug/'manifest.json';p.write_text(json.dumps(data,indent=2)+'\n');print(f'{family}: {len(views)} views, {p}')

def main():
 a=environment('Build QuickSRNet Medium 2x diagram')
 if a.verify:
  m=nn_module(a,'quicksrnet').QuickSRNet(scale=2,num_channels=32,num_intermediate_layers=5)
  write_evidence(a,'quicksrnet',{'m2':cpu_probe(m,(1,3,64,64),['cnn','conv_last','clip_output','depth_to_space'])},['libreyolo/models/quicksrnet/model.py:QUICKSRNET_SIZE_CONFIGS','libreyolo/models/quicksrnet/nn.py:QuickSRNet'],['Only registered size is m2. Native images may vary; 64×64 is the configured representative input.'])
 ev=read_evidence('quicksrnet');verified='cpu' if 'm2' in ev['records'] else 'source'
 d=diagram(a,'QuickSRNet Medium 2x','RGB super-resolution, native eval. Input 3 × 64 × 64. All shapes exclude batch.','quicksrnet',1300,1230)
 p=d.panel('network','Super-resolution network',40,230,560,870)
 nodes=[('input','Input RGB','3 × 64 × 64','plain'),('stem','Conv2d 3×3, stride 1','3 to 32 channels; padding 1','conv2d'),('clip','Hardtanh [0,1]','32 × 64 × 64','activation'),('trunk','Feature block, n=5','32 × 64 × 64','conv'),('last','Conv2d 3×3, stride 1','32 to 12 channels; padding 1','conv2d'),('clipo','Hardtanh [0,1]','12 × 64 × 64','activation'),('shuffle','PixelShuffle ×2','Rearrange 12 channels to 3 RGB channels','aggregate'),('output','Restored RGB','3 × 128 × 128','plain')]
 for i,(id,label,detail,kind) in enumerate(nodes):op(p,id,90,65+i*96,380,label,detail,kind,block='feature' if id=='trunk' else '')
 chain(p,[n[0] for n in nodes])
 q=d.panel('featuredef','Repeated feature block',640,230,620,440,kind='conv',dashed=True,block_type='feature')
 for i,(id,label,detail,kind) in enumerate([('fi','Feature input','32 × 64 × 64','plain'),('fc','Conv2d 3×3','32 to 32; stride 1, padding 1, bias=True','conv2d'),('fa','Hardtanh [0,1]','Elementwise clamp to [0,1]','activation'),('fo','Feature output','32 × 64 × 64','plain')]):op(q,id,95,65+i*88,430,label,detail,kind)
 chain(q,['fi','fc','fa','fo'])
 q=d.panel('rearrange','PixelShuffle indexing',640,710,620,390,kind='aggregate',dashed=True)
 op(q,'ps-in',95,65,430,'12 channels at 64 × 64','Each RGB channel has four subpixel positions','split')
 op(q,'ps-r',95,170,430,'Reshape and permute','3 × 2 × 2 × 64 × 64','aggregate')
 op(q,'ps-out',95,275,430,'RGB at 128 × 128','Channel positions become spatial offsets','plain');chain(q,['ps-in','ps-r','ps-out'])
 view=finish_view(a,d,'quicksrnet','m2-restore','Medium 2x','restore','m2','concrete','3×64×64',verified)
 manifest(a,'quicksrnet','quicksrnet','QuickSRNet',[view])
if __name__=='__main__':main()

def meta_probe(model,shape,names=()):
 import torch
 model.eval();results={}
 for name,module in model.named_modules():
  if name in names:module.register_forward_hook(lambda m,i,o,n=name:results.__setitem__(n,{'input':tensor_shapes(i),'output':tensor_shapes(o)}))
 with torch.inference_mode():out=model(torch.empty(*shape,device='meta'))
 return {'device':'meta','torch':torch.__version__,'input':list(shape),'output':tensor_shapes(out),'hooks':results,'parameters':sum(p.numel() for p in model.parameters()),'pretrained':False,'limitation':'Meta forward validates shape propagation only; numerical inference was not executed.'}

def construct_meta(factory):
 """Materialize constructor-only linspace schedules on CPU, parameters on meta.

Several audited backbones call .item() on their small stochastic-depth schedule.
Making that constant CPU-valued preserves its exact numbers without allocating
model weights. The override ends before any forward execution.
 """
 import torch
 from unittest.mock import patch
 original=torch.linspace
 def cpu_schedule(*args,**kwargs):
  kwargs['device']='cpu';return original(*args,**kwargs)
 with patch.object(torch,'linspace',cpu_schedule),torch.device('meta'):
  return factory()
