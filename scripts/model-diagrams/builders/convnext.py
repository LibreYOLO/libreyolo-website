from alexnet import *
def main():
 a=setup();b=Book(a,'convnext','ConvNeXt');m=load(a.source,'convnext');import torch
 torch.set_num_threads(2)
 for size in m.ARCH_DEFS:
  dev='cpu' if size=='t' else 'meta'
  with torch.device(dev):model=m.ConvNeXt(size)
  b.evidence[size]=shapes(model,[1,3,224,224],dev)
 for size in list(m.ARCH_DEFS)+['family']:
  sym=size=='family';spec=m.ARCH_DEFS['t' if sym else size];C=[f'C{i+1}' for i in range(4)] if sym else spec['dims'];N=[f'n{i+1}' for i in range(4)] if sym else spec['depths'];H=[56,28,14,7]
  d=b.diagram('ConvNeXt '+size,'Classification, 224 × 224 input, 1,000 classes. Tensor sizes exclude batch.',2260,1870)
  p=d.panel('net','Network',30,220,415,1320)
  items=[('Input','3 × 224 × 224','plain'),('Conv2d 4×4 / 4',f'{C[0]} × 56 × 56','conv2d'),('LayerNorm over channels',f'{C[0]} × 56 × 56','norm')]+[(f'Stage {i+1}',f'{C[i]} × {H[i]} × {H[i]}, n={N[i]}','aggregate') for i in range(4)]+[('AdaptiveAvgPool2d',f'{C[3]} × 1 × 1','pool'),('LayerNorm over channels',f'{C[3]} × 1 × 1','norm'),('Flatten',str(C[3]),'plain'),('Linear classifier',f'{C[3]} to 1,000 logits','linear')]
  chain(p,'main',items,w=340,gap=95)
  for i in range(4):
   p=d.panel('stage'+str(i),f'Stage {i+1}',470+i*445,220,420,1320,kind='bottleneck',dashed=True)
   c=C[i];hi=H[i];prev=C[i-1] if i else C[0];items=[]
   if i:items=[('LayerNorm over channels',f'{prev} × {hi*2} × {hi*2}','norm'),('Conv2d 2×2 / 2',f'{prev} to {c}, p=0','conv2d')]
   else:items=[('Identity downsample',f'{c} × {hi} × {hi}','plain')]
   ids=chain(p,'down'+str(i),items,w=280,y=65)
   p.text(25,275,f'ConvNeXtBlock, repeat {N[i]} times',16,weight=700)
   p.dot(170,320);p.connect(ids[-1],ids[-1]) if False else None
   sx,sy=p.port(ids[-1],'right');p.wire([(sx,sy),(380,sy),(380,300),(170,300),(170,320)],start=ids[-1],arrow=False)
   hidden=f'4{c}' if sym else c*4
   ops=[('Depthwise Conv2d 7×7',f'{c} channels, g={c}, s=1, p=3','conv2d'),('Permute NCHW to NHWC',f'{hi} × {hi} × {c}','plain'),('LayerNorm',f'{c} channels, eps=1e-6','norm'),('Linear',f'{c} to {hidden}','linear'),('GELU',f'{hi} × {hi} × {hidden}','activation'),('Linear',f'{hidden} to {c}','linear'),('Permute NHWC to NCHW',f'{c} × {hi} × {hi}','plain'),('Multiply layer scale',f'Learned gamma: {c} channels','plain')]
   ids=chain(p,'block'+str(i),ops,w=280,y=355,gap=90);p.wire([(170,320),(170,355)],end=ids[0]);p.sum('sum'+str(i),170,1120);p.connect(ids[-1],'sum'+str(i));p.wire([(170,320),(380,320),(380,1120),(183,1120)],end='sum'+str(i));p.text(30,1180,f'Output {c} × {hi} × {hi}',16)
  p=d.panel('table','Variant values',30,1580,2200,175)
  for j,(label,x) in enumerate([('Size',25),('C1, C2, C3, C4: stage channels',200),('n1, n2, n3, n4: block counts',880)]):p.text(x,62,label,17,weight=700)
  for j,(key,s) in enumerate(m.ARCH_DEFS.items()):
   p.text(25,95+j*29,key,17);p.text(200,95+j*29,', '.join(map(str,s['dims'])),17);p.text(880,95+j*29,', '.join(map(str,s['depths'])),17)
  b.save(d,size+'-classify',size,kind='family' if sym else 'concrete',verification='source' if sym else b.evidence[size]['device'])
 b.finish()
if __name__=='__main__':main()
