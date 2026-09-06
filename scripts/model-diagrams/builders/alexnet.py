"""AlexNet diagram and small authoring utilities for the classifier builders."""
import argparse,importlib.util,json,os,sys,subprocess
from pathlib import Path
WEBSITE=Path(__file__).resolve().parents[3]
def setup():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(os.environ.get('LIBREYOLO_SOURCE',WEBSITE.parent/'libreyolo')));p.add_argument('--verify',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.source/'skills/libreyolo-make-diagram/scripts'));return a

def load(source,family,file='nn.py'):
 spec=importlib.util.spec_from_file_location('diagram_'+family,source/'libreyolo/models'/family/file);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;
 try:spec.loader.exec_module(m)
 except ImportError as e:
  if 'relative import' not in str(e):raise
  from importlib import import_module
  sys.path.insert(0,str(source));m=import_module('libreyolo.models.'+family+'.'+Path(file).stem)
 return m

def shapes(model,shape,device='meta'):
 import torch
 seen={};handles=[]
 for name,m in model.named_modules():
  if name:handles.append(m.register_forward_hook(lambda m,inp,out,name=name:seen.__setitem__(name,{'input':list(inp[0].shape),'output':list(out.shape)}) if isinstance(out,torch.Tensor) else None))
 with torch.inference_mode():out=model.eval()(torch.zeros(shape,device=device))
 for h in handles:h.remove()
 return {'device':device,'input':list(shape),'output':list(out.shape),'modules':seen}

def dim(s):return ' × '.join(map(str,s))
def op(m,shape):
 import torch.nn as nn
 if isinstance(m,nn.Conv2d):return (f'Conv2d {m.kernel_size[0]}×{m.kernel_size[1]} / {m.stride[0]}',f'{dim(shape)}; p={m.padding[0]}, g={m.groups}','conv2d')
 if isinstance(m,nn.Linear):return ('Linear',f'{m.in_features} to {m.out_features}','linear')
 if isinstance(m,(nn.MaxPool2d,nn.AdaptiveAvgPool2d,nn.AvgPool2d)):return (type(m).__name__,dim(shape),'pool')
 if isinstance(m,nn.Dropout):return ('Dropout (eval identity)',f'p={m.p}; {dim(shape)}','plain')
 return(type(m).__name__,dim(shape),'norm' if 'Norm' in type(m).__name__ else 'activation')
class Book:
 def __init__(self,args,family,title,sourcefile='nn.py'):
  self.a=args;self.family=family;self.title=title;inventory=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text());self.slug=next((f['slug'] for f in inventory['families'] if f['family']==family),family);self.path=WEBSITE/'public/diagrams/models'/self.slug;self.path.mkdir(parents=True,exist_ok=True);self.views=[];self.evidence={};self.sourcefile=sourcefile if sourcefile.startswith('libreyolo/') else f'libreyolo/models/{family}/{sourcefile}';self.rev=json.loads((WEBSITE/'scripts/model-diagrams/coverage.json').read_text())['source_commit']
 def diagram(self,title,subtitle,width,height):
  from svg_diagram import Diagram
  return Diagram(title,subtitle,width=width,height=height,revision=self.rev,source_label=self.sourcefile,source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{self.rev}/{self.sourcefile}',logo=WEBSITE/'public/icon-128.png')
 def save(self,d,id,size,kind='concrete',verification='meta',task='classify',input='1 × 3 × 224 × 224'):
  from wrap_svg import wrap
  d.save(self.path/f'{id}.svg');wrap(self.path/f'{id}.svg',self.path/f'{id}.html')
  self.views.append(dict(id=id,label=d.root.find('{http://www.w3.org/2000/svg}title').text,task=task,size=size,kind=kind,svg=f'/diagrams/models/{self.slug}/{id}.svg',html=f'/diagrams/models/{self.slug}/{id}.html',input=input,verification=verification))
 def finish(self):
  (self.path/'manifest.json').write_text(json.dumps(dict(family=self.family,slug=self.slug,title=self.title,source_revision=self.rev,default_view=self.views[0]['id'],views=self.views),indent=2)+'\n')
  evidence=WEBSITE/'scripts/model-diagrams/evidence'/f'{self.family}.json';evidence.write_text(json.dumps(dict(source_revision=self.rev,source=self.sourcefile,shapes=self.evidence,verification_limits='Meta checks validate shape propagation, not numerical behavior. CPU checks use random weights. Source inspection establishes functional branches. Browser and PNG QA belongs to integration review.'),indent=2)+'\n')
  results=[]
  for v in self.views:
   p=subprocess.run([sys.executable,str(self.a.source/'skills/libreyolo-make-diagram/scripts/check_routes.py'),str(self.path/f"{v['id']}.svg")],capture_output=True,text=True);results.append(v['id']+': '+p.stdout)
  (WEBSITE/'scripts/model-diagrams/evidence'/f'{self.family}.md').write_text('# '+self.title+'\n\nRebuild with `python scripts/model-diagrams/builders/'+self.family+'.py --source /path/to/libreyolo`. All variants come from the in-tree source.\n\nRoute checks:\n\n```\n'+''.join(results)+'```\n')

def chain(p,prefix,items,x=30,y=65,w=280,gap=72):
 ids=[]
 for j,item in enumerate(items):
  label,detail,kind=item[:3];id=f'{prefix}{j}';p.box(id,x,y+j*gap,w,label,detail=detail,kind=kind,font_size=15);ids.append(id)
  if j:p.connect(ids[j-1],id)
 return ids

def main():
 a=setup();b=Book(a,'alexnet','AlexNet');m=load(a.source,'alexnet');import torch
 torch.set_num_threads(2);model=m.AlexNet().eval();ev=shapes(model,[1,3,224,224],'cpu');b.evidence['b']=ev
 d=b.diagram('AlexNet b','Classification, 224 × 224 input, 1,000 classes. Tensor sizes exclude batch.',1420,1480)
 groups=[list(model.features.named_children())[:6],list(model.features.named_children())[6:]]
 ids=[]
 for i,g in enumerate(groups):
  p=d.panel('p'+str(i),'Features '+str(i+1),40+i*460,220,420,850)
  items=[op(layer,ev['modules']['features.'+name]['output'][1:]) for name,layer in g]
  if i==0:items.insert(0,('Input','3 × 224 × 224','plain'))
  ids.append(chain(p,'f'+str(i),items,w=330))
 p=d.panel('head','Classifier',960,220,420,1130,kind='pool');items=[op(model.avgpool,[256,6,6]),('Flatten','9,216','plain')]+[op(layer,ev['modules']['classifier.'+name]['output'][1:]) for name,layer in model.classifier.named_children()];ids.append(chain(p,'h',items,w=330))
 for i in range(2):
  start=ids[i][-1];end=ids[i+1][0];sx,sy=d.port(start,'right');ex,ey=d.port(end,'top');lane=470+i*460;d.connect(start,end,from_port='right',via=[(lane,sy),(lane,265),(ex,265)])
 d.text(50,1115,'Native single-tower graph: no local response normalization or grouped convolution.',17)
 d.text(50,1150,'Convolutions and Linear layers include bias. MaxPool kernels are 3×3 with stride 2.',17)
 d.text(50,1185,'Output: 1,000 logits. Dropout is inactive in the displayed eval graph.',17)
 b.save(d,'b-classify','b',verification='cpu');b.finish()
if __name__=='__main__':main()
