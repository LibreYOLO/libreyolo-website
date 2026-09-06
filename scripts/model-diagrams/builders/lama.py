"""LaMa architecture extracted from the exact Apache-2.0 OpenCV Zoo ONNX.

Fetch outside repo with hf download opencv/inpainting_lama
inpainting_lama_2025jan.onnx --revision aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f
--local-dir /tmp/libreyolo-diagram-artifacts/lama. --verify needs onnx installed;
normal rebuild uses recorded artifact metadata and the source-checked graph.
"""
from quicksrnet import *
import hashlib,collections,re
HASH='7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2'
REV='aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f'
ARTIFACT=Path(os.environ.get('LIBREYOLO_LAMA_ONNX','/tmp/libreyolo-diagram-artifacts/lama/inpainting_lama_2025jan.onnx'))

def main():
 a=environment('Build the pinned LaMa ONNX architecture with expanded Fourier operations')
 if a.verify:
  import onnx
  raw=ARTIFACT.read_bytes();assert len(raw)==92591623 and hashlib.sha256(raw).hexdigest()==HASH
  m=onnx.shape_inference.infer_shapes(onnx.load(str(ARTIFACT)),data_prop=True)
  shapes={v.name:[z.dim_value if z.HasField('dim_value') else z.dim_param for z in v.type.tensor_type.shape.dim] for v in [*m.graph.input,*m.graph.value_info,*m.graph.output]};shapes.update({v.name:list(v.dims) for v in m.graph.initializer})
  convs=[{'name':n.name,'op':n.op_type,'weights':shapes.get(n.input[1]),'attrs':{at.name:onnx.helper.get_attribute_value(at) for at in n.attribute}} for n in m.graph.node if n.op_type in ['Conv','ConvTranspose']]
  record={'device':'source','artifact':{'path_outside_repository':str(ARTIFACT),'repo':'opencv/inpainting_lama','revision':REV,'sha256':HASH,'bytes':len(raw),'opset':[(x.domain,x.version) for x in m.opset_import],'nodes':len(m.graph.node),'operators':dict(collections.Counter(n.op_type for n in m.graph.node)),'convolutions':convs},'input':[[1,3,512,512],[1,1,512,512]],'output':[1,3,512,512],'numerical_inference':False,'shape_note':'ONNX inference resolves weight dimensions but many activation dimensions remain symbolic due shape/Pad subgraphs. Spatial sizes are also checked algebraically from fixed512 input and exact kernel/stride/padding attributes.'}
  write_evidence(a,'lama',{'b':record},['libreyolo/models/lama/nn.py:OpaqueLaMaONNX and pinned checksum','libreyolo/models/lama/NOTICE:Apache-2.0 graph provenance','libreyolo/models/lama/utils.py:preprocess_image_and_mask','libreyolo/postprocess/lama.py:postprocess',f'https://huggingface.co/opencv/inpainting_lama/resolve/{REV}/inpainting_lama_2025jan.onnx'],['92.6MB artifact downloaded read-only outside repository; exact checksum verified. No remote Python or model inference executed.','18 FFC residual blocks, model.5...model.22, eachwithtwoFFC units. Local128/global384 channels at64×64.','INT8 parameter storage feeds299 DequantizeLinear nodes (block_size8); executable convolutions operate on dequantized tensors.','Fourier transforms are lowered into Sin/Cos, MatMul/Einsum, shape operations and normalization; no opaque FFT custom operator.','PNG/browser QA belongs to parent.'])
 ev=read_evidence('lama')
 d=diagram(a,'LaMa Base','Pinned OpenCV Zoo ONNX,512 × 512 BGR image + binary fill mask. Native wrapper accepts4-channel guided input.','lama',1900,4900)
 p=d.panel('main','Encoder and decoder',40,230,610,1890)
 rows=[('masked','Image × (1-mask), concat mask','4 × 512 × 512','concat'),('stem','ReflectPad3; Conv7×7; ReLU','4 to64;512×512; BatchNorm fused','conv'),('down1','ReflectPad1; Conv3×3,s2; ReLU','64 to128;256×256; BatchNorm fused','conv'),('down2','ReflectPad1; Conv3×3,s2; ReLU','128 to256;128×128; BatchNorm fused','conv'),('split','Parallel Conv3×3,s2 from256 channels','Local128 + global384, each64×64; BN fused/ReLU','split'),('blocks','FFC residual block,n=18','Local128/global384 at64×64','bottleneck'),('concat','Concat final local and global','512 × 64 × 64','concat'),('up1','ConvTranspose3×3,s2,p1,op1; BN; ReLU','512 to256;128×128','conv'),('up2','ConvTranspose3×3,s2,p1,op1; BN; ReLU','256 to128;256×256','conv'),('up3','ConvTranspose3×3,s2,p1,op1; BN; ReLU','128 to64;512×512','conv'),('last','ReflectPad3; Conv7×7','64 to3;512×512','conv2d'),('sigmoid','Sigmoid','3-channel generated BGR in[0,1]','activation')]
 for i,(id,label,detail,kind) in enumerate(rows):op(p,id,70,65+i*140,470,label,detail,kind,block='ffc-res' if id=='blocks' else '')
 chain(p,[r[0] for r in rows]);p.text(25,1815,'op:output_padding. Actual artifact uses18 residual blocks.',14)
 r=d.panel('resdef','FFC residual block',690,230,560,1890,kind='bottleneck',dashed=True,block_type='ffc-res')
 op(r,'ri',65,65,430,'Input pair L,G','128 and384 channels;64×64')
 op(r,'ffc1',65,260,430,'FFC + separate BN/ReLU','Output pair L1,G1','conv',block='ffc');r.connect('ri','ffc1')
 op(r,'ffc2',65,475,430,'FFC + separate BN/ReLU','Output pair L2,G2','conv',block='ffc');r.connect('ffc1','ffc2')
 op(r,'l2',15,710,235,'L2','128 × 64 × 64');op(r,'g2',310,710,235,'G2','384 × 64 × 64')
 r.wire([(195,524),(195,645),(132.5,645),(132.5,710)],start='ffc2',end='l2');r.wire([(380,524),(380,645),(427.5,645),(427.5,710)],start='ffc2',end='g2')
 r.sum('ladd',132.5,1010);r.sum('gadd',427.5,1010);r.connect('l2','ladd');r.connect('g2','gadd')
 op(r,'L',15,890,55,'L',h=26);op(r,'G',490,890,55,'G',h=26);r.connect('L','ladd',via=[(42.5,1010)],to_port='left');r.connect('G','gadd',via=[(517.5,1010)],to_port='right')
 r.text(25,1165,'L/G identity residuals remain separate; there is no cross-add.',14)
 r.text(25,1230,'Each residual block executes two identical-shape FFC units.',14)
 r.text(25,1295,'Blocks5...22 have the same graph and independent parameters.',14)
 r.text(25,1375,'The global spectral branch is expanded below.',14)
 h=d.panel('composite','Mask composite and wrapper output',1290,230,570,1890)
 for name,x,label in [('pred',20,'Generated BGR × mask'),('orig',315,'Input BGR × (1-mask)')]:op(h,name,x,65,235,label,'3 × 512 × 512','linear')
 h.sum('blend',285,340);h.wire([(137.5,114),(137.5,340),(272,340)],start='pred',end='blend');h.wire([(432.5,114),(432.5,340),(298,340)],start='orig',end='blend')
 rows=[('255','Multiply255','BGR byte-domain values','linear'),('clip','Clip[0,255]','ONNX output3 × 512 × 512','activation'),('resize','Wrapper: resize to original H×W','Bilinear; convert BGR toRGB','pool'),('cast','Clip and cast uint8','H × W × 3','plain'),('preserve','Copy original RGB outside fill mask','Exact original pixels on unmasked region','aggregate')]
 for i,(id,label,detail,kind) in enumerate(rows):op(h,id,65,510+i*225,440,label,detail,kind)
 chain(h,['blend']+[r[0] for r in rows]);h.text(25,1740,'The wrapper retains the original image/mask before512 resize.',13)
 f=d.panel('ffc','Fast Fourier Convolution unit',40,2180,1120,1190,kind='conv',dashed=True,block_type='ffc')
 branches=[('ll',25,'Local L','ReflectPad1; Conv3×3','128 to128'),('gl',305,'Global G','ReflectPad1; Conv3×3','384 to128'),('lg',585,'Local L','ReflectPad1; Conv3×3','128 to384'),('gg',865,'Global G','Spectral transform','384 to384')]
 for key,x,label,operation,detail in branches:
  op(f,key+'in',x,65,230,label,'64 × 64 grid');op(f,key+'op',x,225,230,operation,detail,'conv',block='spectral' if key=='gg' else '');f.connect(key+'in',key+'op')
 f.sum('local-add',275,455);f.sum('global-add',845,455)
 for key,cx,dest,side in [('ll',140,'local-add','left'),('gl',420,'local-add','right'),('lg',700,'global-add','left'),('gg',980,'global-add','right')]:f.connect(key+'op',dest,via=[(cx,455)],to_port=side)
 for key,x,ch in [('local',65,128),('global',635,384)]:
  op(f,key+'bn',x,605,420,'BatchNormalization',f'{ch} channels','norm');op(f,key+'relu',x,780,420,'ReLU',f'{ch} × 64 × 64','activation');f.connect(key+'-add',key+'bn');f.connect(key+'bn',key+'relu')
 f.text(25,1035,'Local output=Conv(L)+Conv(G). Global output=Conv(L)+Spectral(G).',15)
 f.text(25,1080,'The3×3 path convolutions are unbiased; normalization follows each sum.',14)
 s=d.panel('spectral','Global spectral transform',1200,2180,660,1190,kind='attention',dashed=True,block_type='spectral')
 steps=[('reduce','Conv1×1 (BN fused); ReLU','384 to192;64×64','conv'),('fft','Real2D Fourier transform','192 complex channels;64×33 spectrum','attention'),('pack','Pack real and imaginary as channels','384 × 64 × 33','concat'),('spectralconv','Conv1×1 (BN fused); ReLU','384 to384;spectral mixing','conv'),('unpack','Unpack to192 complex channels','64×33 spectrum','split'),('ifft','Inverse real2D Fourier transform','192 × 64 × 64','attention')]
 for i,(id,label,detail,kind) in enumerate(steps):op(s,id,105,65+i*135,480,label,detail,kind,block='fourier' if id in ['fft','ifft'] else '')
 chain(s,[r[0] for r in steps]);s.sum('sadd',345,930);s.connect('ifft','sadd');s.connect('reduce','sadd',from_port='left',to_port='left',via=[(35,89.5),(35,930)])
 op(s,'expand',105,1030,480,'Conv1×1, no bias','192 to384','conv2d');s.connect('sadd','expand')
 z=d.panel('fourier','Fourier lowering and parameter storage',40,3430,1820,1190,kind='attention',dashed=True,block_type='fourier')
 rows=[('basis','Coordinate/frequency ranges and phase','Sin/Cos bases for separable64-point transforms','plain'),('matmul','Real/imaginary MatMul and Einsum','Apply cosine and sine bases along width and height','attention'),('norm','Normalize by square-root transform lengths','Orthonormal transform scaling','linear'),('half','Keep nonnegative width frequencies','33 complex coefficients along width','split')]
 for i,(id,label,detail,kind) in enumerate(rows):op(z,id,65,65+i*180,700,label,detail,kind)
 chain(z,[r[0] for r in rows]);z.text(25,880,'Inverse reconstructs conjugate-symmetric frequencies and uses inverse phase before returning real values.',15)
 op(z,'quant',1030,65,700,'Stored INT8 parameter blocks','Scale and zero-point tensors; block_size8','plain')
 op(z,'dq',1030,270,700,'DequantizeLinear','(integer-zero_point) × scale','linear');z.connect('quant','dq')
 op(z,'wr',1030,475,700,'Reshape weights to convolution tensors','Float Conv/ConvTranspose operands','plain');z.connect('dq','wr')
 z.text(1030,725,'Artifact:18,001 primitive nodes.',15)
 z.text(1030,770,'222 Conv,3 ConvTranspose,299 DequantizeLinear.',15)
 z.text(1030,815,'Fourier paths are lowered; no opaque FFT custom operator.',15)
 z.text(25,970,'Source-level grouping preserves the exported arithmetic; shape bookkeeping and repeated trig operations are summarized by the DFT definition.',14)
 z.text(25,1030,'Architecture extracted from the exact licensed artifact. ONNX shape inference plus kernel arithmetic; numerical inference was not executed.',14)
 d.text(50,4710,'Artifact revision '+REV+'; SHA-256 '+HASH,12)
 view=finish_view(a,d,'lama','b-inpaint','Base pinned ONNX architecture','restore','b','concrete','image3×512×512 + mask1×512×512','source')
 manifest(a,'lama','lama','LaMa',[view])
if __name__=='__main__':main()
