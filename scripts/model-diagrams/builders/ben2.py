"""BEN2 Base: five image fields, shared Swin-B, multi-field cross-attention/refinement."""
from quicksrnet import *

def main():
 a=environment('Build BEN2 Base1024 multi-field matting architecture')
 if a.verify:
  nn=nn_module(a,'ben2');m=construct_meta(nn.LibreBEN2Model)
  rec=meta_probe(m,(1,3,1024,1024),['backbone','output5','multifieldcrossatt','dec_blk4','dec_blk3','dec_blk2','dec_blk1','insmask_head','upsample1','upsample2','output'])
  write_evidence(a,'ben2',{'b':rec},['libreyolo/models/ben2/nn.py:LibreBEN2Model,BEN2Backbone,MultiFieldCrossAttention,MultiFieldRefinement','libreyolo/models/birefnet/nn.py:shared Swin-v1 primitives','libreyolo/models/ben2/utils.py:ImageNet normalization'],['Actual1024meta forward,notnumericalCPU inference. No pretrainedweights.','Eachimage becomes4local512quadrants plusoneglobal512image. Backbone sees5fields inbatch.','Fivebackboneoutputs includepatchembedding andstage1 atsame128grid.','Finaloutput islogits; sharedmatte postprocess appliessigmoidonce. Optionalforegroundcolour/video helpers arenotpart ofthenativegraph.'])
 ev=read_evidence('ben2');d=diagram(a,'BEN2 Base','Alpha-matte logits,normalized RGB3 × 1024 × 1024,native eval. Five-field dimensions include thefield batch.','ben2',1900,6380)
 p=d.panel('fields','Five fields and shared backbone',40,230,570,1650)
 rows=[('input','Normalized RGB image','3 × 1024 × 1024','plain'),('fields','Four512² quadrants + global512² resize','5 × 3 × 512 × 512;global bilinearresize','aggregate'),('bb','Shared Swin-v1 backbone','Basewidth128;depths[2,2,18,2];window12','attention'),('outputs','Five feature outputs','Patch128×128²;stages128×128²,256×64²,512×32²,1024×16²','split')]
 for j,(id,label,detail,kind) in enumerate(rows):op(p,id,55,65+j*250,460,label,detail,kind,block='swin' if id=='bb' else '')
 chain(p,[r[0] for r in rows]);op(p,'proj',55,1100,460,'Independent Conv3×3 + InstanceNorm + GELU','Channels128,128,256,512,1024 each to128','conv',block='cig');p.connect('outputs','proj')
 p.text(25,1350,'Each level keeps5fields. Projected names:E1,E2,E3,E4,E5.',14)
 p.text(25,1410,'E1/E2 grids128²;E3 grid64²;E4 grid32²;E5 grid16².',14)
 p.text(25,1490,'Independent shallowConv3×3 makes128×1024² original-image features.',13)
 q=d.panel('pyramid','Multi-field decoder',650,230,590,1650)
 rows=[('mcfa','MultiFieldCrossAttention onE5','4local +1global;5×128×16²','attention'),('ref4','E4 + bilinear resize(E5);Refinement4','5×128×32²','aggregate'),('conv4','Conv3×3 +InstanceNorm +GELU','128to128;32²','conv'),('ref3','E3 + bilinear resize(previous);Refinement3','5×128×64²','aggregate'),('conv3','Conv3×3 +InstanceNorm +GELU','128to128;64²','conv'),('ref2','E2 + bilinear resize(previous);Refinement2','5×128×128²','aggregate'),('conv2','Conv3×3 +InstanceNorm +GELU','128to128;128²','conv'),('ref1','E1 + bilinear resize(previous);Refinement1','5×128×128²','aggregate'),('conv1','Conv3×3 +InstanceNorm +GELU','128to128;128²','conv')]
 for j,(id,label,detail,kind) in enumerate(rows):op(q,id,55,65+j*160,480,label,detail,kind,block='mcfa' if id=='mcfa' else 'refinement' if id.startswith('ref') else 'cig')
 chain(q,[r[0] for r in rows]);q.text(25,1560,'Every refinement updateslocal fields andadds them back intoglobal context.',13)
 r=d.panel('output','Field merge and full-resolution output',1280,230,580,1650)
 rows=[('merge','Reassemble4localfields; addresizedglobal','128 × 256 × 256','aggregate'),('maskhead','ThreeConv3×3 mask-head operations','128to384to384to128;IN+GELU afterfirsttwo','conv'),('shallow1','Addbilinear-resized shallowfeature','128 × 256 × 256','aggregate'),('up1','Nearest×2;Conv3×3+IN+GELU','128 × 512 × 512','conv'),('shallow2','Addbilinear-resized shallowfeature','128 × 512 × 512','aggregate'),('up2','Nearest×2;Conv3×3+IN+GELU','128 × 1024 × 1024','conv'),('final','Conv3×3,padding1','128to1;1 × 1024 × 1024 logits','conv2d')]
 for j,(id,label,detail,kind) in enumerate(rows):op(r,id,55,65+j*210,470,label,detail,kind)
 chain(r,[r[0] for r in rows]);r.text(25,1570,'Training sideout1...5 arestored butdo notexecute ininference.',13)
 b=d.panel('swin','Swin-v1 backbone internals',40,1940,590,1650,kind='attention',dashed=True,block_type='swin')
 rows=[('patch','Convpatch4,stride4 +LayerNorm','3to128;128²;retainthispatchfeature','conv'),('s1','Swinblocks,n=2','128channels,4heads;stage1output128²','attention'),('m1','PatchMerging','Concat2×2:512;LN;Linear512to256;64²','aggregate'),('s2','Swinblocks,n=2','256channels,8heads;stage2output64²','attention'),('m2','PatchMerging','Concat2×2:1024;LN;Linear1024to512;32²','aggregate'),('s3','Swinblocks,n=18','512channels,16heads;stage3output32²','attention'),('m3','PatchMerging','Concat2×2:2048;LN;Linear2048to1024;16²','aggregate'),('s4','Swinblocks,n=2','1024channels,32heads;stage4output16²','attention')]
 for j,(id,label,detail,kind) in enumerate(rows):op(b,'b'+id,65,65+j*175,460,label,detail,kind)
 chain(b,['b'+r[0] for r in rows]);b.text(25,1530,'Eachstage output getsits ownLayerNorm beforedecoder projection.',13)
 c=d.panel('mcfa','MultiFieldCrossAttention at16²',670,1940,590,1650,kind='attention',dashed=True,block_type='mcfa')
 rows=[('local','Reassemblelocal4×16² into32²','128channels','aggregate'),('pool','Adaptivepools to16²,4²,2²;concat tokens','276memorytokens,128channels','pool'),('global','Globalquery cross-attention','256queries;K276+2D sinepositions,V276;1head128','attention'),('gn','Residual+LN;FFN128to256to128;residual+LN','Globalstate16²','norm'),('split','Splitupdatedglobal into4quadrants','Eachglobalquadrant8²=64memorytokens','split'),('localca','Fourindependent localcross-attentions','Eachlocal256queries toitsglobal64keys;1head128','attention'),('ln','Residual+LN;FFN128to256to128;residual+LN','Local4×128×16²','norm'),('out','Concatlocal andglobal alongfield batch','5×128×16²','concat')]
 for j,(id,label,detail,kind) in enumerate(rows):op(c,'mf'+id,55,65+j*175,480,label,detail,kind)
 chain(c,['mf'+r[0] for r in rows]);c.text(25,1510,'Sinepositions only onglobal Q/K. FFNs useGELU;dropout inactive ineval.',13)
 f=d.panel('refinement','MultiFieldRefinement at32²,64²,128²',1300,1940,560,1650,kind='attention',dashed=True,block_type='refinement')
 rows=[('split','Split4local and1global field','Each128channels','split'),('gate','GlobalConv1×1(128to1)+sigmoid','Nearestresizeto2H×2H,split4tiles,gate localfeatures','attention'),('pool','Splitglobal into4quadrants;adaptivepools','TargetH/2,H/4,H/8;concatper-fieldmemory','pool'),('attn','Fourindependent cross-attentions','Q=H²;KV=336/1344/5376 forH32/64/128','attention'),('res','Localresidual+LN','Gatedlocalfeaturesprovide residual','norm'),('ffn','Linear128to256;GELU;Linear256to128','Residual+LayerNorm','linear'),('gadd','Reassembleupdatedlocals;resizeandaddtoglobal','Nearestresizeof2H×2H toH×H','aggregate'),('out','Concatlocal/global','5×128×H×H;returntokenattention alongside','concat')]
 for j,(id,label,detail,kind) in enumerate(rows):op(f,'rf'+id,45,65+j*175,470,label,detail,kind)
 chain(f,['rf'+r[0] for r in rows]);f.text(20,1520,'Thepublic decoder ignores auxiliarytokenattention outputs.',13)
 z=d.panel('primitives','Shared primitives',40,3650,1820,970,kind='attention',dashed=True)
 z.text(25,65,'Swin block: LayerNorm,windowattention,residual,LayerNorm,Linear4C,GELU,LinearC,residual.',15)
 z.text(25,110,'C=[128,256,512,1024];MLP=[512,1024,2048,4096];headwidth32;window12,alternatingshift0/6.',14)
 rows=[('qkv','Q/K/V linear projections','128to128 infieldattention;SwinQKV widths384,768,1536,3072','linear'),('qk','ScaledQK transpose','Fieldattention headwidth128;Swinheadwidth32','attention'),('pos','Addpositions orrelativewindowbias whereused','Fieldglobal:sinepositions;Swin:529-entryrelativebiastable +shiftmask','attention'),('soft','Softmax overkeys','','attention'),('av','Weights×V;headconcat;outputprojection','Outputhasquery token countandoriginalchannelwidth','attention')]
 for j,(id,label,detail,kind) in enumerate(rows):op(z,id,25,200+j*130,850,label,detail,kind)
 chain(z,[r[0] for r in rows])
 for j,(id,label,detail,kind) in enumerate([('conv','Conv3×3,padding1','DeclaredCin/Cout;biasTrue','conv2d'),('norm','InstanceNorm2d','Perimagechannel spatialnormalization','norm'),('gelu','GELU','','activation')]):op(z,id+'prim',990,245+j*190,770,label,detail,kind)
 chain(z,['convprim','normprim','geluprim']);z.text(990,890,'Optional foreground-colour refinement is outsideBEN2 Base.',14)
 sw=d.panel('swinblockdetail','Swin block',40,4680,590,1500,kind='attention',dashed=True)
 swrows=[('x','Stage feature','Widths128,256,512,1024','plain'),('n1','LayerNorm','epsilon1e-5','norm'),('part','Pad,shift0/6,partition12×12windows','144tokens/window;headwidth32','split'),('att','Window attention','QKV,relativebias,shiftmask,softmax,V,projection','attention'),('rev','Reverse windows,undo shift,crop','Restore originalgrid','aggregate')]
 for j,(nid,lab,det,kind) in enumerate(swrows):op(sw,'sw'+nid,75,65+j*145,440,lab,det,kind)
 chain(sw,['sw'+r[0] for r in swrows]);sw.sum('swsum1',295,840);sw.connect('swrev','swsum1');sw.connect('swx','swsum1',from_port='left',to_port='left',via=[(25,89.5),(25,840)])
 for j,(nid,lab,det,kind) in enumerate([('n2','LayerNorm','epsilon1e-5','norm'),('f1','Linear4× expansion','128/256/512/1024to512/1024/2048/4096','linear'),('g','GELU','','activation'),('f2','Linearback tostagewidth','512/1024/2048/4096to128/256/512/1024','linear')]):op(sw,'sw'+nid,75,950+j*105,440,lab,det,kind)
 chain(sw,['swsum1','swn2','swf1','swg','swf2']);sw.sum('swsum2',295,1390);sw.connect('swf2','swsum2');sw.dot(295,900);sw.wire([(295,900),(25,900),(25,1390),(282,1390)],start='swsum1',end='swsum2')
 up=d.panel('field-update','Field update after cross-attention',670,4680,590,1500,kind='attention',dashed=True)
 op(up,'ux',75,65,440,'Input X','128channel featuretokens');op(up,'ua',75,215,440,'Cross-attention update','Query X;context as specified inMCFA/refinement','attention');up.connect('ux','ua');up.sum('usum1',295,420);up.connect('ua','usum1');up.connect('ux','usum1',from_port='left',to_port='left',via=[(25,89.5),(25,420)])
 for j,(nid,lab,det,kind) in enumerate([('n1','LayerNorm128','','norm'),('f1','Linear128to256','','linear'),('g','GELU','','activation'),('f2','Linear256to128','','linear')]):op(up,'u'+nid,75,535+j*170,440,lab,det,kind)
 chain(up,['usum1','un1','uf1','ug','uf2']);up.sum('usum2',295,1250);up.connect('uf2','usum2');up.connect('un1','usum2',from_port='left',to_port='left',via=[(25,559.5),(25,1250)]);op(up,'un2',75,1360,440,'LayerNorm128','Updated featuretokens','norm');up.connect('usum2','un2')
 mh=d.panel('mask-head-detail','Instance mask head',1300,4680,560,1500,kind='conv',dashed=True)
 mhrows=[('c1','Conv3×3','128to384;padding1','conv2d'),('n1','InstanceNorm384','','norm'),('g1','GELU','','activation'),('c2','Conv3×3','384to384;padding1','conv2d'),('n2','InstanceNorm384','','norm'),('g2','GELU','','activation'),('c3','Conv3×3','384to128;padding1','conv2d')]
 for j,(nid,lab,det,kind) in enumerate(mhrows):op(mh,'mh'+nid,65,65+j*180,430,lab,det,kind)
 chain(mh,['mh'+r[0] for r in mhrows]);mh.text(20,1430,'Input128×256²;output128×256²before shallowfeature addition.',13)
 rec=ev['records'].get('b',{});view=finish_view(a,d,'ben2','b-matte','Base','matte','b','concrete','3×1024×1024 normalizedRGB',rec.get('device','source'));manifest(a,'ben2','ben2','BEN2',[view])
if __name__=='__main__':main()
