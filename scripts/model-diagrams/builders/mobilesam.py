from sam import *
def main():
 a=setup();b=Book(a,'mobilesam','MobileSAM',sourcefile='model.py');m=load(a.source,'mobilesam','model.py');import torch
 torch.set_num_threads(2)
 model=m.MobileSAMNetwork().to('meta').eval()
 with torch.inference_mode():out=model(pixel_values=torch.zeros(1,3,1024,1024,device='meta'),input_boxes=torch.zeros(1,1,4,device='meta'),multimask_output=True)
 b.evidence['tiny']={'device':'meta','input':[1,3,1024,1024],'image_embeddings':list(out.image_embeddings.shape),'pred_masks':list(out.pred_masks.shape),'iou_scores':list(out.iou_scores.shape),'encoder_config':{'widths':[64,128,160,320],'depths':[2,2,6,2],'heads':[2,4,5,10],'windows':[7,7,14,7]}}
 d=b.diagram('MobileSAM tiny','Promptable segmentation, 1024 × 1024 input, one box, multimask output. TinyViT encoder and native SAM decoder.',2740,7550)
 p=d.panel('image','TinyViT image encoder',25,220,605,1720)
 items=[('RGB input after normalize/pad','3 × 1,024 × 1,024','plain'),('Conv2d 3×3 / 2','3 to 32, 512 × 512','conv2d'),('BatchNorm2d','32 × 512 × 512','norm'),('GELU','32 × 512 × 512','activation'),('Conv2d 3×3 / 2','32 to 64, 256 × 256','conv2d'),('BatchNorm2d','64 × 256 × 256','norm'),('MBConv, repeats=2','64 × 256 × 256','bottleneck'),('Patch merge 1','128 × 128 × 128','conv'),('TinyViT block, repeats=2','16,384 tokens × 128','attention'),('Patch merge 2','160 × 64 × 64','conv'),('TinyViT block, repeats=6','4,096 tokens × 160','attention'),('Patch merge 3 (stride 1)','320 × 64 × 64','conv'),('TinyViT block, repeats=2','4,096 tokens × 320','attention'),('Conv2d 1×1 + channel LN','320 to 256, 64 × 64','conv'),('Conv2d 3×3 + channel LN','256 to 256, p=1, 64 × 64','conv')]
 chain(p,'im',items,w=490,gap=100)
 common_heads(d)
 p=d.panel('mb','MBConv',25,5200,650,2010,kind='bottleneck',dashed=True)
 p.text(30,70,'Input: 64 × 256 × 256',17);p.dot(260,110)
 ids=chain(p,'mbops',[('Conv2d 1×1','64 to 256, no bias','conv2d'),('BatchNorm2d','256 × 256 × 256','norm'),('GELU','256 × 256 × 256','activation'),('Depthwise Conv2d 3×3','256 to 256, g=256, p=1','conv2d'),('BatchNorm2d','256 × 256 × 256','norm'),('GELU','256 × 256 × 256','activation'),('Conv2d 1×1','256 to 64, no bias','conv2d'),('BatchNorm2d','64 × 256 × 256','norm')],w=460,y=155,gap=110);p.wire([(260,110),(260,155)],end=ids[0]);p.sum('mbsum',260,1110);p.connect(ids[-1],'mbsum');p.wire([(260,110),(600,110),(600,1110),(273,1110)],end='mbsum');p.box('mbgelu',30,1200,460,'GELU',detail='64 × 256 × 256',kind='activation');p.connect('mbsum','mbgelu')
 p=d.panel('tinyblock','TinyViT block',705,5200,650,2010,kind='attention',dashed=True)
 p.text(25,70,'Stage order in numeric lists: 128, 160, 320.',16);p.dot(240,110)
 items=[('Pad and partition windows','128 pads to 133; 64 pads to 70','plain'),('LayerNorm','128 / 160 / 320 channels','norm'),('Window self-attention','4 / 5 / 10 heads, 32 channels per head','attention'),('Reverse windows and crop','128×128 / 64×64 / 64×64','plain')]
 ids=chain(p,'tiny1',items,w=420,y=155,gap=115);p.wire([(240,110),(240,155)],end=ids[0]);p.sum('tinys1',240,680);p.connect(ids[-1],'tinys1');p.wire([(240,110),(610,110),(610,680),(253,680)],end='tinys1')
 ids=chain(p,'local',[('Reshape NCHW','128 / 160 / 320 channels','plain'),('Depthwise Conv2d 3×3','groups=128 / 160 / 320, p=1','conv2d'),('BatchNorm2d','128 / 160 / 320 channels','norm'),('Flatten spatial tokens','16,384 / 4,096 / 4,096 tokens','plain')],w=420,y=760,gap=95);p.connect('tinys1',ids[0]);p.dot(240,1170);p.wire([p.port(ids[-1]),(240,1170)],start=ids[-1],arrow=False)
 ids=chain(p,'tinymlp',[('LayerNorm','128 / 160 / 320 channels','norm'),('Linear','128 to 512 / 160 to 640 / 320 to 1,280','linear'),('GELU','512 / 640 / 1,280 channels','activation'),('Linear','512 to 128 / 640 to 160 / 1,280 to 320','linear')],w=420,y=1210,gap=115);p.wire([(240,1170),(240,1210)],end=ids[0]);p.sum('tinys2',240,1750);p.connect(ids[-1],'tinys2');p.wire([(240,1170),(610,1170),(610,1750),(253,1750)],end='tinys2')
 p=d.panel('merge','Patch merging',1380,5200,650,2010,kind='conv',dashed=True)
 items=[('Reshape input to NCHW','64×256×256 / 128×128×128 / 160×64×64','plain'),('Conv2d 1×1','64 to 128 / 128 to 160 / 160 to 320','conv2d'),('BatchNorm2d','128 / 160 / 320 channels','norm'),('GELU','128 / 160 / 320 channels','activation'),('Depthwise Conv2d 3×3','Strides 2 / 2 / 1, p=1','conv2d'),('BatchNorm2d','128 / 160 / 320 channels','norm'),('GELU','128 / 160 / 320 channels','activation'),('Conv2d 1×1','128 to 128 / 160 to 160 / 320 to 320','conv2d'),('BatchNorm2d','128 / 160 / 320 channels','norm'),('Flatten and transpose','16,384×128 / 4,096×160 / 4,096×320','plain')]
 chain(p,'mergeops',items,w=545,gap=105)
 p.text(25,1220,'No residual connection around patch merging.',16)
 p.text(25,1300,'Encoder neck primitives:',18,weight=700)
 chain(p,'neck',[('Conv2d 1×1','320 to 256, bias=False','conv2d'),('Channel LayerNorm','256 × 64 × 64','norm'),('Conv2d 3×3 / 1','256 to 256, p=1, bias=False','conv2d'),('Channel LayerNorm','256 × 64 × 64','norm')],w=545,y=1350,gap=105)
 attention(d,'tinyattn','Window self-attention',2055,5200,'128 / 160 / 320','4 / 5 / 10','49 / 196 / 49',32,relative=True,height=2010)
 d.text(2075,6500,'Attention bias uses absolute relative offsets.',16)
 d.text(2075,6550,'Stage 1: 361 windows of 7×7.',16);d.text(2075,6590,'Stage 2: 25 windows of 14×14.',16);d.text(2075,6630,'Stage 3: 100 windows of 7×7.',16)
 d.text(35,7310,'All numeric lists are aligned by the three transformer stages. Their actual dimensions are written in the stage graph and definitions.',16)
 b.save(d,'tiny-box-segment','tiny',task='segment',verification='meta',input='Image: 1 × 3 × 1024 × 1024; box: 1 × 1 × 4');b.finish()
if __name__=='__main__':main()
