"""DA3MONO-LARGE: mono DINOv2/DPT core, explicit sky branch and inverse depth."""
from quicksrnet import *
from depth_anything import vit_definition,dpt_topology,fusion_definitions

def main():
 a=environment('Build Depth Anything 3 monocular large')
 if a.verify:
  import torch
  torch.set_num_threads(4);nn=nn_module(a,'depth_anything3')
  m=construct_meta(nn.LibreDepthAnything3Net)
  class Core(torch.nn.Module):
   def __init__(self,model):super().__init__();self.model=model
   def forward(self,x):return self.model.forward_network(x)
  record=meta_probe(Core(m),(1,3,504,504),['model.backbone','model.head','model.head.scratch.refinenet1','model.head.scratch.output_conv1'])
  depth=torch.linspace(1,10,504*504).reshape(1,1,504,504);sky=torch.zeros_like(depth);sky[:,:,:,252:]=1
  with torch.inference_mode():finish=m.finish_depth(depth,sky)
  record['finish_depth_cpu']={'input':tensor_shapes(depth),'output':tensor_shapes(finish),'finite':bool(torch.isfinite(finish).all()),'note':'Synthetic positive depth plus half-sky mask; network remains meta.'}
  write_evidence(a,'depth_anything3',{'l':record},['libreyolo/models/depth_anything3/nn.py:LibreDepthAnything3Net','libreyolo/models/depth_anything3/_vendor/dinov2/vision_transformer.py:mono get_intermediate_layers','libreyolo/models/depth_anything3/_vendor/dpt.py:DPT and FeatureFusionBlock'],['Actual neural core meta forward with504 canvas; numerical network forward not run.','Sky heuristic and inverse-depth conversion separately executed on CPU synthetic504 depth/sky maps.','DA3 LayerNorm epsilon defaults1e-5, unlike V2 vendor1e-6. No alternate-view attention, QK normalization or RoPE in mono preset.','Meta constructor materializes its small linspace schedule on CPU; weights stay meta.'])
 ev=read_evidence('depth_anything3')
 d=diagram(a,'Depth Anything 3 Large','DA3MONO-LARGE, inverse-depth output, input 3 × 504 × 504, native eval. Shapes exclude batch.','depth_anything3',1900,4780)
 p=d.panel('backbone','Single-view DINOv2-Large',40,230,570,1760)
 rows=[('image','ImageNet normalization + singleton view','B × 1 × 3 × 504 × 504','norm'),('patch','Conv2d patch14, stride14','1024 × 36 × 36;1296 patch tokens','conv2d'),('pos','CLS + interpolated learned positions','1297 × 1024; position grid37×37 resized36×36','aggregate')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,65,65+i*135,440,label,detail,kind)
 chain(p,[r[0] for r in rows]);prev='pos'
 for i,(end,n) in enumerate([(4,5),(11,7),(17,6),(23,6)]):
  yy=525+i*245;op(p,f'block{i}',65,yy,440,f'Transformer blocks, n={n}',f'Ends at zero-based index{end};16 heads','attention',block='vit')
  if i==0:p.connect(prev,f'block{i}')
  else:p.connect(prev,f'block{i}',from_port='right',to_port='right',via=[(540,yy-220.5),(540,yy+24.5)])
  op(p,f'backbone-tap{i}',65,yy+110,440,'LayerNorm + remove CLS',f'T{i+1}: 1024 × 36 × 36','norm');p.connect(f'block{i}',f'backbone-tap{i}');prev=f'block{i}'
 p.text(25,1595,'alt_start=qknorm_start=rope_start=-1.',14)
 p.text(25,1635,'No alternating view attention, QK norm, RoPE or camera token.',13)
 p.text(25,1675,'cat_token=False; tap width stays1024.',14)
 dpt_topology(d,1024,256,[256,512,1024,1024],36,504)
 vit_definition(d,1024,16,1297,y=2070,eps='1e-5')
 h=d.panel('heads','Shared output feature and two heads',1300,2070,560,1420)
 op(h,'conv1',70,65,420,'Conv2d 3×3','256 to128;288 × 288','conv2d')
 op(h,'resize',70,165,420,'Bilinear resize','128 × 504 × 504; align_corners=True','pool');h.connect('conv1','resize')
 for side,x,activation in [('depth',15,'Exponential'),('sky',300,'ReLU')]:
  for j,(id,label,detail,kind) in enumerate([('conv','Conv2d 3×3','128 to32; s1,p1','conv2d'),('relu','ReLU','','activation'),('class','Conv2d 1×1','32 to1','conv2d'),('act',activation,'1 × 504 × 504','activation')]):op(h,side+id,x,330+j*155,245,label,detail,kind)
  h.wire([(170 if side=='depth' else 390,214),(170 if side=='depth' else 390,285),(x+122.5,285),(x+122.5,330)],start='resize',end=side+'conv');chain(h,[side+s for s in ['conv','relu','class','act']])
 h.text(20,1125,'Depth output is positive relative depth.',14)
 h.text(20,1170,'Sky is ReLU output, not a probability sigmoid.',14)
 h.text(20,1250,'No confidence head when output_dim=1.',14)
 h.text(20,1315,'Sky handling and inversion execute after this shared core.',13)
 fusion_definitions(d,256,y=3600)
 q=d.panel('finish','Native depth finishing',1450,3600,410,1050)
 rows=[('mask','Non-sky mask','sky <0.3, per image','plain'),('guard','Count both mask regions','Apply only if each has more than10 pixels','plain'),('sample','Non-sky depth values','Sample100000 if there are more','pool'),('quantile','99th percentile','Far-depth value from non-sky depth','pool'),('fill','Replace sky pixels','Use far depth where sky>=0.3','aggregate'),('clamp','Clamp depth minimum1e-6','Positive denominator','activation'),('recip','Reciprocal','Inverse relative depth','linear'),('out','Remove singleton view','B × 1 × 504 × 504','plain')]
 for i,(id,label,detail,kind) in enumerate(rows):op(q,id,30,65+i*110,350,label,detail,kind)
 chain(q,[r[0] for r in rows]);q.connect('guard','clamp',from_port='left',to_port='left',via=[(12,199.5),(12,639.5)]);q.text(20,995,'If counts fail, keep original depth before inversion.',12)
 d.text(50,4430,'Raw adapters:256×144²,512×72²,1024×36²,1024×18². Fusion outputs36²,72²,144²,288² before the final resize.',15)
 d.text(50,4470,'Sky statistics are computed independently per batch image; the wrapper never mixes images into one scene.',15)
 rec=ev['records'].get('l',{});view=finish_view(a,d,'depth-anything-3','l-depth','Large monocular','depth','l','concrete','3×504×504',rec.get('device','source'))
 manifest(a,'depth_anything3','depth-anything-3','Depth Anything 3',[view])
if __name__=='__main__':main()
