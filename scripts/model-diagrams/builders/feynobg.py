"""FeyNobg reuses BiRefNet's exact graph with Swin stage3 depth24."""
from quicksrnet import *
from birefnet import render

def main():
 a=environment('Build FeyNobg Large architecture')
 if a.verify:
  nn=nn_module(a,'feynobg');m=construct_meta(lambda:nn.LibreFeyNobgModel('l'))
  rec=meta_probe(m,(1,3,1024,1024),['bb','squeeze_module','decoder.decoder_block4','decoder.decoder_block1','decoder.conv_out1'])
  write_evidence(a,'feynobg',{'l':rec},['libreyolo/models/feynobg/nn.py:FEYNOBG_DIMS and LibreFeyNobgModel','libreyolo/models/birefnet/nn.py:shared architecture'],['Actual default1024 meta forward; no numerical inference.','Only architectural delta from BiRefNet-L is Swin depth[2,2,24,2] instead of[2,2,18,2].','fp8/fp16 checkpoint variants change parameter precision, not the topology or declared size. This diagram shows the native float architecture.'])
 ev=read_evidence('feynobg');view=render(a,'feynobg','l',192,[2,2,24,2],12,ev);manifest(a,'feynobg','feynobg','FeyNobg',[view])
if __name__=='__main__':main()
