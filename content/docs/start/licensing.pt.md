---
title: Licenciamento
seo_title: 'Licenciamento do LibreYOLO: código e pesos'
description: >-
  O código próprio do LibreYOLO é MIT. O código upstream incorporado e os
  checkpoints publicados carregam suas próprias licenças, e várias delas são não
  comerciais.
lead: >-
  O LibreYOLO reúne três coisas licenciadas separadamente: o código dele, o
  código upstream incorporado a uma família de modelos e os checkpoints
  pré-treinados. Muitas vezes não é a mesma licença.
keywords:
  - licença libreyolo
  - biblioteca de visão computacional mit
  - pesos de modelo não comercial
  - licença de checkpoint de modelo
  - detecção de objetos apache-2.0
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## O código próprio do LibreYOLO

A biblioteca é MIT. Isso cobre a API Python, a CLI, os trainers, os validadores
e os exportadores, os carregadores de dataset e os scripts de conversão em
`weights/`. Use em um produto comercial ou de código fechado, mantenha a linha
de copyright e o texto da licença junto de qualquer cópia que você redistribuir,
e a obrigação termina aí.

A concessão para no código. O arquivo
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)
diz isso sem rodeios:

> Essas licenças variam e nem todas são permissivas: alguns pesos publicados são
> não comerciais ou restritos de outra forma, e esta Licença MIT não se estende
> a eles. Escolher um modelo é escolher a licença dele.

## Código upstream, por família

A maioria das famílias são ports de pesquisa publicada, e várias incorporam
código-fonte upstream diretamente. Um arquivo incorporado mantém o cabeçalho de
copyright original e a licença original. A MIT não sobrescreve isso, e o
LibreYOLO não relicencia o trabalho de ninguém. Apache-2.0 e BSD-3-Clause são as
duas que mais aparecem.

A Apache-2.0 cobre a linha DETR e boa parte do trabalho com transformers: DETR
da Meta AI (FAIR), Deformable DETR da SenseTime, LW-DETR da Baidu, OV-DEIM de
Leilei Wang e coautores, a implementação do SegFormer que o LibreYOLO porta do
Hugging Face Transformers, PP-OCRv5 dos PaddlePaddle Authors, SwinIR do Computer
Vision Lab da ETH Zurich e Depth Anything 3 do ByteDance Seed. Também cobre os
classificadores derivados do timm, de Ross Wightman e dos contribuidores do
timm, entre eles ResNet, DeiT, EfficientNetV2, MobileNetV4 e Swin, cujos nomes
de módulo espelham os do timm para que os tensores da ImageNet carreguem sem
alteração.

A BSD-3-Clause cobre tudo que deriva do torchvision: Faster R-CNN, Mask R-CNN,
FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN e DeepLabv3.

A MIT cobre um grupo menor, incluindo NAFNet da Megvii, CenterNet de Xingyi
Zhou e YOLOv7 tal como relançado pelos próprios autores, Kin-Yiu Wong e Hao-Tang
Tsui, no MultimediaTechLab. As famílias YOLOv1 a YOLOv4 reproduzem arquiteturas
do projeto Darknet, de Joseph Redmon e, no caso do YOLOv4, de Alexey
Bochkovskiy. O Darknet é de domínio público, então essas não carregam obrigação
nenhuma.

Uma subárvore incluída não tem licença de código aberto. A família DEIMv2 traz o
código do backbone DINOv3 da Meta Platforms sob o DINOv3 License Agreement, uma
licença customizada fora da OSI. Redistribuir esse código significa distribuir
uma cópia do acordo junto, e o acordo proíbe o uso para atividades sujeitas ao
ITAR, para fins militares ou de guerra, para indústrias nucleares, espionagem e
desenvolvimento de armas. Esses termos valem só para aquela subárvore.

Dois arquivos do repositório têm o quadro completo.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) lista cada
subárvore de terceiros incluída, com o caminho, o arquivo de licença e a origem
upstream.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
lista os projetos upstream dos quais o LibreYOLO deriva e reproduz o texto de
cada licença na íntegra.

## Pesos, por checkpoint

Nenhum arquivo de pesos pré-treinados vai dentro do pacote. Os checkpoints
publicados ficam no Hugging Face sob a [organização
LibreYOLO](https://huggingface.co/LibreYOLO), e cada repositório carrega o
próprio `LICENSE` e a própria atribuição, refletindo o projeto de onde os pesos
vieram.

Esse repositório é a fonte autoritativa dos termos. Não esta página, não a
página do modelo e não o resumo na árvore de código. Veja
[checkpoints e pesos](/docs/weights) para saber como os arquivos são nomeados e
de onde são baixados.

As licenças diferem entre famílias, e diferem entre arquivos dentro de uma mesma
família. Dois exemplos do segundo caso:

- Os checkpoints COCO do YOLO9 são MIT. `LibreYOLO9P2s-visdrone.pt`, treinado no
  VisDrone2019-DET, é CC BY-NC-SA 3.0, que é não comercial.
- Os checkpoints de detecção do RF-DETR são Apache-2.0. Os checkpoints de caixa
  orientada são CC BY 4.0, porque passaram por fine-tuning em um dataset do
  Roboflow Universe publicado sob CC BY 4.0 e os pesos carregam adiante a
  exigência de atribuição desse dataset.

Entre famílias a variação é maior, e vários checkpoints publicados não podem ser
usados em um produto comercial:

- O SegFormer é a separação mais clara entre as duas camadas. A implementação é
  um port Apache-2.0 do código do Hugging Face Transformers. Os checkpoints
  ADE20K publicados são convertidos do release da NVIDIA sob a NVIDIA Source
  Code License, que permite a redistribuição mas limita o uso a pesquisa ou
  avaliação não comerciais, e leva esse limite adiante para os trabalhos
  derivados. Esses checkpoints não são cobertos pelos termos permissivos do
  LibreYOLO.
- Os checkpoints do OV-DEIM são CC BY-NC 4.0, confirmado pelo autor upstream.
  Toda predição também carrega a torre de texto MobileCLIP-B(LT) da Apple, cuja
  licença restringe o uso a pesquisa, um termo mais estrito que o do próprio
  checkpoint.
- O código do SenseNova-Vision é Apache-2.0 e os pesos dele são CC BY-NC 4.0. O
  loader imprime o aviso de uso não comercial antes de todo download automático.

Algumas famílias não têm checkpoint nenhum hospedado pelo LibreYOLO, e as
páginas delas dizem isso na linha Weights. O SAM 3 tem acesso restrito no
Hugging Face sob a SAM License própria da Meta e é baixado direto da Meta. Os
assets de release do MiDaS são buscados nas URLs oficiais e verificados por hash
em vez de rehospedados. O Dome-DETR é linkado ao upstream porque o model card
dele não declara licença nenhuma nos metadados enquanto o texto afirma
Apache-2.0 e ao mesmo tempo restringe o uso a pesquisa acadêmica, e essas duas
coisas não batem. As arquiteturas TEED e DexiNed são MIT, mas os checkpoints
publicados pelos autores foram treinados no BIPED, cujos termos de dataset são
não comerciais, então o LibreYOLO não os inclui nem os baixa automaticamente.

Vários checkpoints do torchvision não têm arquivo de licença próprio. O
LibreYOLO os espelha na licença que o projeto que os publica usa, declara em
cada model card que a base é implícita e não uma concessão por checkpoint, e
repete o próprio aviso do torchvision de que os termos de um modelo pré-treinado
podem derivar dos dados de treinamento.

## Encontrar os termos de um modelo

A página do modelo traz uma linha **Licenses** no cabeçalho, na forma
`Code X, weights Y`, que linka para baixo, até a seção Licensing da página. Essa
seção lista o trabalho original e seus autores, a licença upstream, a origem
upstream, a licença do código do LibreYOLO, os pesos e uma interpretação do que
os termos permitem. A tabela Checkpoints da mesma página tem uma coluna
**Weights license**, com uma linha por arquivo publicado, então uma família com
termos mistos os mostra arquivo por arquivo.

Tudo isso é renderizado a partir dos mesmos dados contra os quais a biblioteca é
verificada, e é por isso que esta página não repete a informação em forma de
tabela. Uma matriz de licenças escrita à mão fica errada dentro de uma mesma
versão, e errar aqui sai caro.

Na árvore de código, os equivalentes são `NOTICE` para o código incluído,
`THIRD_PARTY_NOTICES.txt` para os projetos upstream e seus textos de licença, e
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
para um resumo por família dos checkpoints publicados.

Depois confira o repositório do Hugging Face do arquivo exato que você está
prestes a baixar. Ele é autoritativo, e pode mudar sem que uma página de
documentação mude junto.

## Uso comercial

O código raramente é o problema. MIT, Apache-2.0 e BSD-3-Clause permitem todas o
uso comercial e de código fechado. Cada uma pede que você mantenha o texto da
licença e os avisos de atribuição nas cópias que redistribuir, a Apache-2.0
também concede uma licença de patentes, e nenhuma delas impõe condições ao
código da sua própria aplicação.

É nos checkpoints que os produtos empacam. Um checkpoint não comercial continua
não comercial por mais permissivo que seja o código em volta, e converter o
arquivo não muda os termos aplicáveis a ele, que é o que
`weights/LICENSE_NOTICE.txt` diz diretamente. Um artefato ONNX ou TensorRT
construído a partir de um checkpoint restrito herda a restrição.

Quando uma licença leva a restrição adiante para os trabalhos derivados, como a
NVIDIA Source Code License faz, o fine-tuning também não escapa dela. Treinar a
mesma arquitetura do zero com dados que você tem o direito de usar escapa: o
código é permissivo, então um modelo que você mesmo treina é seu, e os termos do
checkpoint pré-treinado nunca entram nele. A página do SegFormer detalha isso
para os pesos dela; leia a linha Interpretation na página da família que você
pretende colocar em produção.

Decida a questão da licença quando escolher o modelo, e não na hora de publicar,
e leia os termos do arquivo que você de fato baixou, porque uma família com um
checkpoint permissivo pode ter um restrito do lado.

## Isto não é aconselhamento jurídico

Esta página descreve as licenças envolvidas. É uma descrição, não aconselhamento
jurídico, e não cria garantia nenhuma. Se a resposta importa comercialmente,
leia as licenças você mesmo e busque o seu próprio parecer jurídico.
