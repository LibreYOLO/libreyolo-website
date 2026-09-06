from sam2 import *
def main():
 a=setup();b=Book(a,'edgetam','EdgeTAM',sourcefile='libreyolo/models/sam/edgetam.py');import torch,transformers
 from transformers.models.edgetam.configuration_edgetam import EdgeTamMaskDecoderConfig
 from transformers.models.edgetam.modeling_edgetam import EdgeTamMaskDecoder
 cfg=EdgeTamMaskDecoderConfig();cfg._attn_implementation='eager'
 with torch.device('meta'):decoder=EdgeTamMaskDecoder(cfg).eval()
 with torch.inference_mode():out=decoder(image_embeddings=torch.zeros(1,256,64,64,device='meta'),image_positional_embeddings=torch.zeros(1,256,64,64,device='meta'),sparse_prompt_embeddings=torch.zeros(1,1,2,256,device='meta'),dense_prompt_embeddings=torch.zeros(1,256,64,64,device='meta'),multimask_output=True,high_resolution_features=[torch.zeros(1,32,256,256,device='meta'),torch.zeros(1,64,128,128,device='meta')])
 b.evidence={'configuration_source':{'repo':'LibreYOLO/LibreEdgeTAM','revision':'c81728930ecc22d5d88d9ab9f3db3fcd99440143','file':'config.json'},'backend':{'package':'transformers','version':transformers.__version__,'license':'Apache-2.0'},'decoder_meta_outputs':[list(v.shape) for v in out],'boundary':'RepViT-m1 is selected through TimmWrapperConfig in the downloaded configuration. timm source/runtime is not installed in this environment. Feature widths and downstream image-mode decoder are verified, but RepViT internal operators are not reconstructed from its name.'}
 d=b.diagram('EdgeTAM edge','Image segmentation, 1024 × 1024 input, one box, multimask output. RepViT backend boundary; verified prompt and decoder internals.',3040,6550)
 p=d.panel('image','Image encoder interface',25,220,605,1720)
 items=[('RGB input','3 × 1,024 × 1,024','plain'),('Bilinear square resize','1,024 × 1,024, ImageNet normalization','plain'),('RepViT-m1 backend','External timm feature extractor','plain'),('Four backbone feature maps','48, 96, 192, 384 channels','aggregate'),('Image feature pyramid','Four 1×1 lateral projections to 256','aggregate'),('High-res s0 projection','32 × 256 × 256','conv2d'),('High-res s1 projection','64 × 128 × 128','conv2d'),('Main image embedding','256 × 64 × 64, add no-memory vector','plain')]
 # Feature outputs are parallel; the linear network stops at the pyramid.
 ids=chain(p,'image',items[:4],w=490,gap=130);p.box('pyramid-outputs',30,585,490,'Image feature pyramid',h=100,detail='Three independent image-head feature outputs',kind='aggregate')
 p.connect(ids[-1],'pyramid-outputs')
 for j,(label,detail,kind) in enumerate(items[5:]):
  p.box('outfeature'+str(j),30,910+j*170,490,label,detail=detail,kind=kind)
  p.wire([(520,610+j*30),(535+j*25,610+j*30),(535+j*25,934.5+j*170),(520,934.5+j*170)],start='pyramid-outputs',end='outfeature'+str(j))
 p.text(25,1520,'RepViT internal operators are not present',17);p.text(25,1555,'in this source checkout or installed runtime.',17)
 heads(d)
 fpn(d,25,5200,[48,96,192,384])
 p=d.panel('boundary','Backend verification boundary',1560,5200,1450,1010)
 p.text(30,80,'The library selects repvit_m1 through an embedded TimmWrapperConfig.',19)
 p.text(30,140,'Its configured backbone widths are 48, 96, 192 and 384.',18)
 p.text(30,200,'The RepViT architecture is not inferred from those four numbers.',18)
 p.text(30,275,'Prompt encoder, two-way decoder, output MLPs and high-resolution additions are fully drawn.',17)
 p.text(30,330,'The installed Apache-2.0 EdgeTamMaskDecoder passed a no-weight meta forward.',17)
 p.text(30,405,'Image mode does not run the video memory encoder, memory attention or perceiver resampler.',17)
 p.text(30,490,'Configuration source: LibreYOLO/LibreEdgeTAM, revision c81728930ecc.',17)
 b.save(d,'edge-box-segment','edge',task='segment',verification='source',input='Image: 1 × 3 × 1024 × 1024; box: 1 × 1 × 4');b.finish()
if __name__=='__main__':main()
