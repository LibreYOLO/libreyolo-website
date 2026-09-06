from clip import *
def main():
 a=setup();b=Book(a,'swin','Swin',sourcefile='classifier.py');m=load(a.source,'swin','classifier.py');conf=load(a.source,'swin','config.py').SWIN_CONFIGS;import torch
 torch.set_num_threads(2)
 for size in conf:
  dev='cpu' if size=='t' else 'meta'
  with torch.device(dev):model=m.SwinClassifier(size)
  b.evidence[size]=shapes(model,[1,3,224,224],dev)
 for size in list(conf)+['family']:
  sym=size=='family';s=conf['t' if sym else size];C=[f'C{i+1}' for i in range(4)] if sym else [s['embed_dim']*2**i for i in range(4)];H=[56,28,14,7];heads=[f'h{i+1}' for i in range(4)] if sym else s['num_heads'];N=[f'n{i+1}' for i in range(4)] if sym else s['depths']
  d=b.diagram('Swin '+size,'Classification, 224 × 224 input, 1,000 classes. Stage tensors use H × W × C, excluding batch.',2740,4210)
  p=d.panel('network','Network',25,220,440,2150)
  items=[('Input NCHW','3 × 224 × 224','plain'),('Conv2d 4×4 / 4',f'{C[0]} × 56 × 56','conv2d'),('Permute to NHWC',f'56 × 56 × {C[0]}','plain'),('LayerNorm',f'56 × 56 × {C[0]}','norm')]+[(f'Stage {i+1}',f'{H[i]} × {H[i]} × {C[i]}, n={N[i]}','attention') for i in range(4)]+[('LayerNorm',f'7 × 7 × {C[3]}','norm'),('Mean over H and W',str(C[3]),'pool'),('Linear classifier',f'{C[3]} to 1,000 logits','linear')]
  chain(p,'net',items,w=350,y=65,gap=95)
  p.text(25,1300,'Window size: 7 × 7 throughout.',17)
  p.text(25,1340,'Final stage: shift is disabled.',17)
  p.text(25,1380,'No CLS token or absolute position.',17)
  p.text(25,1420,'Dropout and stochastic depth are absent.',16)
  for i in range(4):
   c=C[i];h=H[i];hd='C/h' if sym else c//heads[i];mid=f'4{c}' if sym else 4*c
   p=d.panel('stage'+str(i),f'Stage {i+1}',495+i*555,220,530,2150,kind='attention',dashed=True)
   if i:
    prev=C[i-1];four=f'4{prev}' if sym else prev*4
    items=[('Input from previous stage',f'{h*2} × {h*2} × {prev}','plain'),('Reshape 2×2 neighborhoods',f'{h} × {h} × {four}','plain'),('LayerNorm',f'{four} channels','norm'),('Linear reduction (no bias)',f'{four} to {c}','linear')]
   else:items=[('Input from patch embedding',f'56 × 56 × {c}','plain'),('Identity downsample',f'56 × 56 × {c}','plain')]
   ids=chain(p,'merge'+str(i),items,w=330,y=65,gap=85)
   p.text(25,480,f'Window block, total repeats {N[i]}',18,weight=700)
   p.text(25,515,'All blocks use shift 0.' if i==3 else 'Odd block shift 0; even block shift 3.',16)
   p.dot(195,560);sx,sy=p.port(ids[-1],'right');p.wire([(sx,sy),(490,sy),(490,540),(195,540),(195,560)],start=ids[-1],arrow=False)
   count=(h//7)**2;items=[('LayerNorm',f'{h} × {h} × {c}','norm'),('Cyclic roll H and W','0 pixels' if i==3 else 'Odd: 0; even: -3 pixels','plain'),('Partition 7×7 windows',f'{count} windows, 49 × {c} each','plain'),('Window self-attention',f'{heads[i]} heads, width {hd}','attention'),('Reverse window partition',f'{h} × {h} × {c}','plain'),('Reverse cyclic roll','0 pixels' if i==3 else 'Odd: 0; even: +3 pixels','plain')]
   ids=chain(p,'window'+str(i),items,w=330,y=600,gap=85);p.wire([(195,560),(195,600)],end=ids[0]);p.sum('sum1'+str(i),195,1155);p.connect(ids[-1],'sum1'+str(i));p.wire([(195,560),(490,560),(490,1155),(208,1155)],end='sum1'+str(i))
   p.dot(195,1210);p.wire([(195,1168),(195,1210)],start='sum1'+str(i),arrow=False)
   ids=chain(p,'mlp'+str(i),[('Flatten spatial tokens',f'{h*h} × {c}','plain'),('LayerNorm',f'{h*h} × {c}','norm'),('Linear',f'{c} to {mid}','linear'),('GELU',f'{h*h} × {mid}','activation'),('Linear',f'{mid} to {c}','linear')],w=330,y=1250,gap=100)
   p.wire([(195,1210),(195,1250)],end=ids[0]);p.sum('sum2'+str(i),195,1790);p.connect(ids[-1],'sum2'+str(i));p.wire([(195,1210),(490,1210),(490,1790),(208,1790)],end='sum2'+str(i));p.box('reshape'+str(i),30,1860,330,'Reshape to NHWC',detail=f'{h} × {h} × {c}');p.connect('sum2'+str(i),'reshape'+str(i));p.text(25,1990,'No spatial padding needed at this input.',16)
   att=attention(d,'attn'+str(i),f'Stage {i+1} window attention',25+i*680,2430,c,heads[i],49,hd,relative=True,height=1270,shift_mask='0 only' if i==3 else '0 / -100, even blocks')
  p=d.panel('variants','Variant values',25,3740,2690,270)
  for x,label in [(25,'Size'),(260,'C1, C2, C3, C4'),(900,'n1, n2, n3, n4'),(1500,'h1, h2, h3, h4')]:p.text(x,65,label,18,weight=700)
  for j,(key,spec) in enumerate(conf.items()):
   for x,val in [(25,key),(260,', '.join(str(spec['embed_dim']*2**i) for i in range(4))),(900,', '.join(map(str,spec['depths']))),(1500,', '.join(map(str,spec['num_heads'])))]:p.text(x,105+j*35,val,17)
  b.save(d,size+'-classify',size,kind='family' if sym else 'concrete',verification='source' if sym else b.evidence[size]['device'])
 b.finish()
if __name__=='__main__':main()
