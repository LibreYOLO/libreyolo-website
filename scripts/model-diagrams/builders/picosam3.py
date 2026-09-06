from alexnet import *
def main():
 a=setup();b=Book(a,'picosam3','PicoSAM3');m=load(a.source,'picosam3');import torch
 torch.set_num_threads(2);model=m.PicoSAM3Network().eval();ev=shapes(model,[1,3,96,96],'cpu');b.evidence['pico']=ev
 d=b.diagram('PicoSAM3 pico','ROI mask segmentation, 96 × 96 input. One mask logit map. Tensor sizes exclude batch.',2320,3500)
 p=d.panel('network','Encoder and decoder',25,220,2270,1270)
 p.box('input',35,70,340,'ROI image',detail='3 × 96 × 96')
 for i,(c,h) in enumerate(zip([48,96,160,256],[96,48,24,12])):
  yy=190+i*235;p.box('e'+str(i),35,yy,340,f'Encoder stage {i+1}',detail=f'{c} × {h} × {h}',kind='conv');p.connect('input' if i==0 else 'down'+str(i-1),'e'+str(i))
  p.box('down'+str(i),35,yy+95,340,'Conv2d 3×3 / 2',detail=f'{c} × {h//2} × {h//2}, p=1',kind='conv2d');p.connect('e'+str(i),'down'+str(i))
  destc=[40,80,128,192][i];p.box('skip'+str(i),620,yy,330,'Skip Conv2d 1×1',detail=f'{c} to {destc}, bias=True',kind='conv2d');p.connect('e'+str(i),'skip'+str(i),from_port='right',to_port='left')
  p.box('up'+str(i),1130,yy+90,345,f'Upsample block {4-i}',detail=f'{destc} × {h} × {h}',kind='conv');p.sum('add'+str(i),1550,yy+24.5)
  p.connect('skip'+str(i),'add'+str(i),from_port='right',to_port='left')
  p.connect('up'+str(i),'add'+str(i),from_port='right',to_port='bottom',via=[(1550,yy+114.5)])
  if i<3:p.connect('add'+str(i+1),'up'+str(i)) if False else None
 p.box('bottle',570,1100,420,'Bottleneck',detail='320 × 6 × 6',kind='bottleneck');p.connect('down3','bottle',from_port='right',to_port='left',via=[(470,1014.5),(470,1124.5)])
 p.connect('bottle','up3',from_port='right',to_port='bottom',via=[(1060,1124.5),(1060,1065),(1302.5,1065)])
 for i in range(3):p.connect('add'+str(i+1),'up'+str(i),from_port='right',to_port='bottom',via=[(1680,190+(i+1)*235+24.5),(1680,190+i*235+175),(1302.5,190+i*235+175)])
 ids=chain(p,'out',[('Channel attention (ECA)','40 × 96 × 96','attention'),('Refine mask logits','1 × 96 × 96','conv'),('Sigmoid and ROI resize','Mask pasted into original image','plain')],x=1820,w=400,y=190,gap=150);p.connect('add0',ids[0],from_port='right',to_port='left')
 # Every separable block is expanded with its own actual input/output widths.
 specs=[('Encoder 1','encoder_stage1'),('Encoder 2','encoder_stage2'),('Encoder 3','encoder_stage3'),('Encoder 4','encoder_stage4'),('Bottleneck','bottleneck'),('Upsample 1','up1'),('Upsample 2','up2'),('Upsample 3','up3'),('Upsample 4','up4'),('Refine','refine')]
 for j,(title,name) in enumerate(specs):
  mm=getattr(model,name);items=[]
  for sub,layer in mm.named_modules():
   if sub and not list(layer.children()):
    sh=ev['modules'][name+'.'+sub]['output'][1:]
    if isinstance(layer,torch.nn.Upsample):items.append(('Nearest upsample ×2',dim(sh),'pool'))
    else:
     label,detail,kind=op(layer,sh)
     if isinstance(layer,torch.nn.Conv2d) and layer.dilation[0]!=1:detail+=f', d={layer.dilation[0]}'
     items.append((label,detail,kind))
  p=d.panel('definition'+str(j),title,25+j%5*460,1540+j//5*880,440,840,kind='conv',dashed=True)
  chain(p,'def'+str(j),items,w=360,gap=57)
 p=d.panel('eca','ECA channel attention',25,3340,2270,0) if False else None
 # ECA is a complete miniature directed graph along the bottom row.
 p=d.panel('eca','Channel attention (ECA)',25,3340,2270,360,kind='attention',dashed=True)
 for j,(label,detail,kind) in enumerate([('Input','40 × 96 × 96','plain'),('Global average pool','40 × 1 × 1','pool'),('Conv2d 1×1','40 to 40, no bias','conv2d'),('Sigmoid','40 × 1 × 1','activation'),('Multiply gate and input','40 × 96 × 96','plain')]):
  p.box('gate'+str(j),30+j*440,90,375,label,detail=detail,kind=kind)
  if j:p.connect('gate'+str(j-1),'gate'+str(j),from_port='right',to_port='left')
 p.connect('gate0','gate4',via=[(217.5,230),(1977.5,230)],to_port='bottom')
 # Expand the viewBox to include the final ECA panel and provenance.
 d.height=3850;d.root.set('height','3850');d.root.set('viewBox','0 0 2320 3850')
 d.text(40,3740,'Box/point prompts select an ROI before this network. The decoder has additive projected skips; it does not concatenate encoder maps.',16)
 b.save(d,'pico-segment','pico',task='segment',verification='cpu',input='1 × 3 × 96 × 96');b.finish()
if __name__=='__main__':main()
