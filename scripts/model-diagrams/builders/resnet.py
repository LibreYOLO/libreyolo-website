#!/usr/bin/env python3
"""Reproduce ResNet diagrams and CPU shape evidence using the pinned local source."""
import argparse, importlib.util, json, sys
from pathlib import Path
from alexnet import setup,Book,shapes
args=setup();book=Book(args,'resnet','ResNet');OUT=book.path
from svg_diagram import Diagram
from wrap_svg import wrap
REV=book.rev
SOURCE='libreyolo/models/resnet/nn.py'
spec=importlib.util.spec_from_file_location('diagram_resnet_source',args.source/SOURCE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
VARIANTS={s:tuple(v['layers']) for s,v in mod.ARCH_DEFS.items()}
HW=[56,28,14,7]

def build(size):
    symbolic=size.startswith('family'); bottleneck=size in ['50','101','family-bottleneck']; expansion=4 if bottleneck else 1; W=[v*expansion for v in [64,128,256,512]];ns=[f'n{i}' for i in range(1,5)] if symbolic else VARIANTS[size]
    title=('ResNet-50 / 101 family' if bottleneck else 'ResNet-18 / 34 family') if symbolic else f'ResNet-{size}'
    d=Diagram(title,'Classification, 224 × 224 input, 1,000 classes. Tensor sizes exclude batch.',width=1860,height=2800 if bottleneck else 2210,logo=book.path.parents[2]/'icon-128.png',revision=REV,source_label=SOURCE,source_url=f'https://github.com/LibreYOLO/libreyolo/blob/{REV}/{SOURCE}')
    p=d.panel('network','Network',40,220,1780,450)
    stem=[('input','Input','3 × 224 × 224','plain'),('stem-conv','Conv2d 7×7 / 2','64 × 112 × 112, p=3','conv2d'),('stem-bn','BatchNorm2d','64 × 112 × 112','norm'),('stem-relu','ReLU','64 × 112 × 112','activation'),('stem-pool','MaxPool2d 3×3 / 2','64 × 56 × 56, p=1','pool')]
    for j,(id,label,detail,kind) in enumerate(stem):
        p.box(id,25,60+j*70,245,label,detail=detail,kind=kind)
        if j:p.connect(stem[j-1][0],id)
    for i in range(4):
        x=340+i*295
        p.box(f'stage{i+1}',x,225,250,f'Layer {i+1}: '+('Bottleneck' if bottleneck else 'BasicBlock'),detail=f'{W[i]} × {HW[i]} × {HW[i]}, repeats={ns[i]}',h=56,kind='bottleneck',block_type=f'layer{i+1}',description=f'Layer {i+1} uses '+('one stride-1 channel projection followed by identity blocks.' if i==0 and bottleneck else ('identity shortcuts throughout.' if i==0 else 'one stride-2 projection block followed by identity blocks.')))
        if i:p.connect(f'stage{i}',f'stage{i+1}',from_port='right',to_port='left')
        p.text(x,311,'Projects 64 to 256 channels' if i==0 and bottleneck else ('Identity blocks only' if i==0 else 'First block downsamples'),15)
    p.connect('stem-pool','stage1',via=[(300,364.5),(300,253)],from_port='right',to_port='left')
    if bottleneck:
        for x,label in [(340,'Size'),(530,'n1'),(650,'n2'),(770,'n3'),(890,'n4')]:p.text(x,75,label,17,weight=700)
        for yy,sz in [(115,'50'),(160,'101')]:
            p.text(340,yy,'ResNet-'+sz,17)
            for xx,value in zip([530,650,770,890],VARIANTS[sz]):p.text(xx,yy,str(value),17)
    # Classifier remains fully expanded in a horizontal row.
    for id,x,label,detail,kind in [('avg',720,'AdaptiveAvgPool2d',f'{W[-1]} × 1 × 1','pool'),('flat',1040,'Flatten (from dim 1)',str(W[-1]),'plain'),('fc',1360,'Linear + bias',f'{W[-1]} inputs, 1,000 logits','linear')]:
        p.box(id,x,365,270,label,detail=detail,kind=kind)
    p.connect('stage4','avg',from_port='right',via=[(1710,253),(1710,337),(855,337)])
    p.connect('avg','flat',from_port='right',to_port='left');p.connect('flat','fc',from_port='right',to_port='left')
    d.text(50,710,'Bottleneck definitions' if bottleneck else 'BasicBlock definitions',26,weight=700)
    d.text(380,710,'Conv2d bias=False. k=kernel, s=stride, p=padding. ReLU follows each addition.',17)
    def block(panel,prefix,y,cin,cout,hi,ho,project,count):
        panel.text(18,y,('Projection block' if project else 'Identity block')+f' (n={count})',18,weight=700)
        panel.text(18,y+26,f'Input {cin} × {hi} × {hi}',14)
        mid=155; start=y+46;stride=2 if hi!=ho else 1
        panel.dot(mid,start)
        ops=[('c1','Conv2d 3×3',f'{cin} to {cout}, s={stride}, p=1','conv2d'),('bn1','BatchNorm2d',f'{cout} × {ho} × {ho}','norm'),('r1','ReLU',f'{cout} × {ho} × {ho}','activation'),('c2','Conv2d 3×3',f'{cout} to {cout}, s=1, p=1','conv2d'),('bn2','BatchNorm2d',f'{cout} × {ho} × {ho}','norm')]
        if bottleneck:
            inner=cout//4
            ops=[('c1','Conv2d 1×1',f'{cin} to {inner}, s=1, p=0','conv2d'),('bn1','BatchNorm2d',f'{inner} × {hi} × {hi}','norm'),('r1','ReLU',f'{inner} × {hi} × {hi}','activation'),('c2','Conv2d 3×3',f'{inner} to {inner}, s={stride}, p=1','conv2d'),('bn2','BatchNorm2d',f'{inner} × {ho} × {ho}','norm'),('r2','ReLU',f'{inner} × {ho} × {ho}','activation'),('c3','Conv2d 1×1',f'{inner} to {cout}, s=1, p=0','conv2d'),('bn3','BatchNorm2d',f'{cout} × {ho} × {ho}','norm')]
        for j,(id,label,detail,kind) in enumerate(ops):
            panel.box(prefix+id,50,start+25+j*66,210,label,detail=detail,kind=kind,font_size=15)
            if j:panel.connect(prefix+ops[j-1][0],prefix+id)
        panel.wire([(mid,start),(mid,start+25)],end=prefix+'c1')
        addy=start+(583 if bottleneck else 385);panel.sum(prefix+'sum',mid,addy)
        panel.connect(prefix+('bn3' if bottleneck else 'bn2'),prefix+'sum')
        if project:
            panel.box(prefix+'sc',280,start+91,125,f'Conv2d 1×1 / {stride}',detail=f'{cin} to {cout}, p=0',kind='conv2d',font_size=13)
            panel.box(prefix+'sb',280,start+223,125,'BatchNorm2d',detail=f'{cout} × {ho} × {ho}',kind='norm',font_size=13)
            panel.wire([(mid,start),(342.5,start),(342.5,start+91)],end=prefix+'sc')
            panel.connect(prefix+'sc',prefix+'sb')
            panel.connect(prefix+'sb',prefix+'sum',via=[(342.5,addy)],to_port='right')
        else:
            panel.wire([(mid,start),(342.5,start),(342.5,addy),(168,addy)],end=prefix+'sum')
            panel.text(285,start+185,'identity',14)
        panel.box(prefix+'out',50,addy+38,210,'ReLU',detail=f'{cout} × {ho} × {ho}',kind='activation')
        panel.connect(prefix+'sum',prefix+'out')
        panel.text(50,addy+109,f'Output {cout} × {ho} × {ho}',14)
    for i in range(4):
        panel=d.panel(f'def{i+1}',f'Layer {i+1}',40+450*i,745,430,1875 if bottleneck else 1325,kind='bottleneck',dashed=True,block_type=f'layer{i+1}',description=f'All operations and shortcuts used in layer {i+1}.')
        n=ns[i]; block(panel,f'l{i+1}a',65,W[i-1] if i else 64,W[i],HW[i-1] if i else HW[i],HW[i],i>0 or bottleneck,1 if i or bottleneck else n)
        if i or bottleneck:
            block(panel,f'l{i+1}b',970 if bottleneck else 710,W[i],W[i],HW[i],HW[i],False,f'{n}-1' if symbolic else n-1)
            panel.text(18,940 if bottleneck else 677,'Then repeat the identity block below.',15)
        else:
            panel.text(20,730,'Stage repeats',22,weight=700)
            columns=[('Variant',20),('n1',198),('n2',258),('n3',318),('n4',378)]
            for label,x in columns:
                panel.text(x,770,label,17,weight=700,anchor='start' if x==20 else 'middle')
            for y,size_key in [(812,'18'),(854,'34')]:
                panel.text(20,y,f'ResNet-{size_key}',17)
                for x,value in zip([198,258,318,378],VARIANTS[size_key]):
                    panel.text(x,y,str(value),17,anchor='middle')
            from svg_diagram import element
            for y in [782,826,868]:
                element(panel.ops,'line',x1=20,y1=y,x2=405,y2=y,stroke='#cbbeb4',stroke_width=0.7)
            panel.text(20,910,'n1..n4: total BasicBlocks per layer.',16)
            panel.text(20,949,'Widths are fixed at 64, 128, 256, 512.',16)
            panel.text(20,988,'Only repeat counts vary in this pair.',16)
            panel.text(20,1050,'ResNet-50/101 use Bottleneck blocks',16)
            panel.text(20,1081,'and are outside this shared topology.',16)
            panel.text(20,1143,'Outputs are logits from model.forward.',16)
            panel.text(20,1174,'Softmax is outside this network.',16)
    if bottleneck:
        d.text(50,2655,'n1..n4: block counts. ResNet-50: 3, 4, 6, 3. ResNet-101: 3, 4, 23, 3. Bottleneck expansion is 4.',17)
    d.text(50,2690 if bottleneck else 2110,'Original diagram from LibreYOLO MIT source. Unfused inference graph.',15)
    book.save(d,size+'-classify',size,kind='family' if symbolic else 'concrete',verification='source' if symbolic else book.evidence[size]['device'])
import torch
torch.set_num_threads(2)
for size in VARIANTS:
    device='cpu' if size=='18' else 'meta'
    with torch.device(device):model=mod.ResNet(size=size,num_classes=1000)
    book.evidence[size]=shapes(model,[1,3,224,224],device)
    build(size)
for size in ['family-basic','family-bottleneck']:build(size)
book.finish()
