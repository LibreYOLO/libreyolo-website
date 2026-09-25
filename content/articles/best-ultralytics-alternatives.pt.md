---
title: "Melhores alternativas ao Ultralytics em 2026: opções YOLO"
description: "Um guia prático das melhores alternativas de código aberto ao Ultralytics YOLO em 2026: licenças, cobertura de tarefas e limites de deploy, além de onde o LibreYOLO, a biblioteca YOLO com licença MIT mais completa, se encaixa."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "O Ultralytics YOLO é gratuito para uso comercial?"
    a: "Somente sob AGPL-3.0, que exige a liberação do código-fonte do aplicativo que você criar com ele, incluindo serviços em rede. O uso comercial de código fechado exige a licença comercial paga do Ultralytics. Alternativas permissivas (MIT, Apache-2.0) evitam isso."
  - q: "Qual é a alternativa ao Ultralytics com menos restrições de licença?"
    a: "Para uma stack permissiva que você pode distribuir em qualquer lugar: LibreYOLO (código MIT), RF-DETR (Apache-2.0) e YOLOX (Apache-2.0). Todas evitam o copyleft da AGPL."
  - q: "O que está substituindo o YOLO em 2026?"
    a: "Cada vez mais, detectores transformer em tempo real: RT-DETR, RF-DETR e a linha D-FINE e DEIM. Eles igualam ou superam a acurácia do YOLO com latência semelhante em hardware capaz. CNNs no estilo YOLO ainda vencem em dispositivos de borda pequenos, onde esses transformers rodam mal e não têm um caminho maduro para NCNN ou CPU."
  - q: "Eles rodam em um Raspberry Pi ou em uma NPU?"
    a: "Depende da exportação. Detectores CNN como YOLOX e RTMDet exportam para NCNN e rodam em um Pi, e alguns (YOLOX) também funcionam na NPU Hailo. Detectores transformer como RF-DETR não rodam assim: precisam de uma GPU ou de uma CPU potente."
---

Aviso: o LibreYOLO é nosso, e está em primeiro lugar nesta lista. Usamos todas as outras ferramentas aqui em trabalho real e indicamos onde elas superam o LibreYOLO.

A maioria das equipes deixa o Ultralytics por um de três motivos:

- **Licença.** Os modelos YOLO do Ultralytics usam AGPL-3.0 (YOLOv5 desde 2023 e v8 em diante). O padrão é conhecido: você cria um produto, distribui, e então uma revisão jurídica ou um cliente aponta a licença do modelo. A escolha passa a ser abrir o código-fonte de todo o seu aplicativo ou comprar uma licença comercial. O copyleft da AGPL alcança seu código assim que você o distribui ou oferece como serviço. Esse é um motivo comum para procurar outra opção.
- **Abertura.** Um único fornecedor controla os modelos, o código de treinamento e os termos. Algumas equipes querem pesos e código sob licenças permissivas, sobre os quais possam construir sem pedir autorização.
- **O modelo.** YOLO nem sempre é a melhor arquitetura para a tarefa. Detectores transformer em tempo real como RT-DETR, RF-DETR e D-FINE podem ser mais precisos com a mesma latência. O problema é que eles ficam espalhados por repositórios de pesquisa e, como você verá, o LibreYOLO roda os três com uma única API. Portanto, esse é um motivo para trocar de biblioteca, não para sair dela.

Avaliamos cada ferramenta abaixo por três critérios: uma licença que permita distribuir seu produto, se ela instala e roda com uma versão atual do PyTorch e se exporta para o hardware em que você realmente faz deploy.

Um padrão se repete ao longo do texto: cada alternativa é boa, mas usá-la individualmente dá trabalho. O YOLOX mal instala, o MMDetection é um purgatório de versões fixas de wheels, a família RT-DETR é código de pesquisa sem canal de suporte e o Detectron2 está congelado desde 2021. O LibreYOLO está em primeiro lugar porque já integra a maioria delas, com manutenção ativa, licença MIT e uma stack atual, tudo por trás de uma API familiar. Ele é menos um item desta lista e mais a camada que fica acima dela.

## Resumo rápido

| Ferramenta | Licença | Tarefas cobertas | Ideal para | Exportação para dispositivos de borda |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (código) | Detecção, segmentação, pose, classificação, profundidade, estimativa de olhar, tracking | O hub MIT para mais de 20 famílias de modelos com uma única API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Detecção, segmentação, pose (preview) | Detecção e segmentação em produção | ONNX, TFLite, TensorRT; sem NCNN ou Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Pré-treinamento, destilação | Dados sem rótulos, destilação de DINOv2/v3 | n/a (não é um detector) |
| **YOLOX** | Apache-2.0 | Detecção | Detecção em tempo real sem âncoras | O projeto upstream é difícil de instalar, roda pelo LibreYOLO |
| **MMDetection / fork OneDL** | Apache-2.0 | Tudo (pesquisa) | Variedade de arquiteturas, reprodução de artigos | Via exportação |
| **Detectron2** | Apache-2.0 | Detecção, segmentação, keypoints | Mask R-CNN e a família R-CNN | Manual |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Detecção | DETRs em tempo real de última geração | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Detecção, segmentação, pose | ViTs compactos para dispositivos de borda destilados de DINOv3 | Nível de pesquisa |
| **RT-DETR (v1-v4)** | Apache-2.0 | Detecção | Detecção em tempo real de ponta | ONNX, TensorRT |

## 1. LibreYOLO

Licença: MIT (código). Ideal para quem busca a biblioteca YOLO com licença MIT mais completa e uma alternativa direta ao Ultralytics.

**O LibreYOLO é a biblioteca YOLO com licença MIT que roda todos os modelos com uma única API.** É o nosso projeto e, depois de conhecer o restante desta lista, seu papel deve ficar claro: ele é o hub que roda as alternativas. O LibreYOLO já integra quase todos os repositórios trabalhosos acima, YOLOX, RTMDet, RF-DETR, D-FINE, DEIM e a linha RT-DETR, em uma API familiar e mantida. Assim, você aproveita o modelo sem ter de investigar instalações nem lidar com a licença.

Há também um motivo que vai além da conveniência. YOLO começou como pesquisa aberta: o Darknet de Joseph Redmon, das versões v1 a v3, era gratuito para qualquer pessoa usar e desenvolver. A era AGPL fechou esse acesso. O LibreYOLO existe para tornar esse trabalho acessível de novo, como seus criadores pretendiam: com licença permissiva, mantido pela comunidade e sem se conectar a serviços externos. São duas histórias.

**Primeiro: é a alternativa ao Ultralytics YOLO com licença MIT.** Ele mantém a API que você já conhece, então seus datasets no formato YOLO e seus scripts funcionam com pequenas alterações. Mas a licença é MIT, em vez de AGPL: o copyleft não alcança seu código, você não precisa comprar uma licença comercial e não há telemetria se conectando a serviços externos, como ocorre por padrão no Ultralytics. Se você chegou aqui por causa da licença, esse é o ponto principal. YOLO9 e RF-DETR são os padrões mais testados e prontos para produção. O restante do catálogo é mais recente e está claramente marcado como experimental. Preferimos dizer isso a fingir que tudo já foi testado em condições reais.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Segundo: ele cobre muito mais do que o Ultralytics.** Uma única API abrange mais de 20 famílias de modelos, CNNs YOLO, DETRs transformer, backbones ViT, SAM e outros, em vez de apenas a linha de um fornecedor. E não são só detectores:

- **Tarefas:** segmentação de instâncias e semântica, pose, classificação e caixas orientadas, além de duas tarefas que o Ultralytics não oferece: estimativa de profundidade monocular (Depth Anything V2) e estimativa de olhar (L2CS).
- **A camada prática:** tracking de múltiplos objetos com IDs persistentes (ByteTrack e OC-SORT), inferência de vídeo com subamostragem de quadros e visualização ao vivo, inferência por blocos para imagens de drones e satélites, uma interface web com arrastar e soltar (`libreyolo ui`) e um comando `doctor` que confere seu dataset antes de você gastar tempo em um treinamento.
- **Treinamento avançado:** acumulação de gradiente, congelamento de camadas, retomada do treinamento, data augmentation no teste, fine-tuning com LoRA/DoRA, múltiplas GPUs e registro com TensorBoard/MLflow/Weights & Biases.
- **Exportação:** sete formatos (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite), com quantização INT8/FP16, NMS integrado e metadados do modelo integrados.
- **Roda em qualquer lugar:** aceita caminhos, URLs, PIL, NumPy, tensores ou bytes brutos e roda em CUDA, Apple Silicon (MPS) ou apenas CPU, sem alterar o código.

Quando o próprio repositório de um detector forte é abandonado ou difícil de instalar, o LibreYOLO roda os pesos com manutenção ativa e em uma stack atual.

## 2. RF-DETR

Licença: Apache-2.0 (Nano, Small, Medium, Large). Ideal para detecção e segmentação em produção.

O RF-DETR, da Roboflow e publicado na ICLR 2026, é o modelo mais pronto para produção desta lista. Foi bem projetado, vem de uma equipe que entrega produtos e é uma ótima primeira escolha quando você precisa de detecção ou segmentação confiável para um sistema real. O pacote `rfdetr` e os pesos de Nano a Large usam Apache-2.0. Os pesos maiores, XL e 2XL, usam a licença PML da Roboflow.

## 3. Lightly

Ideal para dados sem rótulos, pré-treinamento auto-supervisionado e destilação de um backbone DINO.

Lightly não é um detector. É uma forma de aproveitar dados que você ainda não rotulou e vem em duas partes com licenças diferentes.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Um framework de componentes de aprendizado auto-supervisionado: losses, cabeças, data augmentation, bancos de memória e implementações de referência de mais de vinte métodos (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE e outros). Você escreve o loop de treinamento; ele oferece tudo de que precisa para pré-treinar uma representação do zero com imagens sem rótulos.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (com opções comerciais e gratuitas para a comunidade).** A opção pronta para usar, e o fluxo de trabalho a que a maioria das pessoas se refere. Aponte para um modelo fundacional como DINOv2 ou DINOv3 e para seus dados sem rótulos, e ele destila esse backbone em um modelo menor que você pode colocar em produção (YOLO, RT-DETR, ViT ou uma rede personalizada) com poucas linhas e sem rótulos. Observe a licença: AGPL-3.0 tem o mesmo copyleft que leva as pessoas a procurar alternativas ao Ultralytics. Leia os termos se pretende distribuir software de código fechado ou opte pela licença comercial.

Use Lightly quando o gargalo for ter um bom backbone, mas nenhum rótulo. Ele complementa os detectores desta lista, em vez de competir com eles. Os mais recentes também mostram isso: RT-DETRv4 incorpora a destilação de modelos fundacionais de visão ao treinamento, e DEIMv2 integra diretamente backbones DINOv3 ao detector.

## 4. YOLOX

Licença: Apache-2.0. Ideal para detecção em tempo real sem âncoras, depois de conseguir instalar.

O repositório upstream está congelado desde seu último release, v0.3.0, em abril de 2022, tem mais de 700 issues abertas e é difícil de instalar em uma stack de 2026. Não há `pyproject.toml`, e `requirements.txt` fixa `onnx-simplifier==0.4.10`, que não tem wheels pré-compiladas para versões posteriores ao Python 3.10. Em um interpretador moderno, o pacote é compilado a partir do código-fonte e exige cmake e uma toolchain C++. O NumPy também não tem versão fixada, o que facilita conflitos de ABI do NumPy 2.0 com versões antigas de pycocotools e torch. A detecção principal funciona depois que você resolve a cadeia de dependências, mas chegar até lá tem seu custo.

O modelo ainda é bom: um detector em tempo real sem âncoras da Megvii, com Apache-2.0 tanto para o código quanto para os pesos, e uma baseline razoável, embora detectores mais recentes já o tenham superado. Reviver um repositório abandonado, mas sólido, é o tipo de tarefa que um agente de programação resolve em uma tarde hoje, então o estado da instalação é menos impeditivo do que parece. O LibreYOLO também roda os pesos do YOLOX diretamente em uma stack atual. De qualquer forma, vale a pena usar essa arquitetura. Escrevemos um guia mais completo [aqui](/articles/yolox-with-libreyolo).

## 5. O universo MMDetection

Licença: em sua maioria Apache-2.0. Ideal para variedade de arquiteturas e reprodução de artigos.

Durante anos, o MMDetection manteve a implementação oficial de quase todos os artigos sobre detecção: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN e centenas de configurações. Em 2026, o OpenMMLab reduziu o desenvolvimento da série mm. O último release do [MMDetection](https://github.com/open-mmlab/mmdetection) foi o v3.3.0, em janeiro de 2024, as issues ficam sem resposta e a stack está presa à versão fixada de `mmcv`, cujas wheels pré-compiladas ficam atrás das versões atuais do PyTorch e CUDA. Por isso, uma instalação nova costuma falhar até você rebaixar todas as versões. Falamos sobre esse problema [aqui](/articles/rtmdet-without-mmdetection).

A empresa holandesa VBTI mantém o projeto vivo. O [fork OneDL](https://github.com/VBTI-development/onedl-mmdetection) republica toda a stack com o prefixo `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` e outros), recompilada para PyTorch 2.x, CUDA e Python 3.10 ou superior, o que resolve o conflito de versões. Usa Apache-2.0 e tem manutenção ativa, com a versão v3.5.1 lançada em maio de 2026. Para ter a variedade do MMDetection sem o trabalho de instalação, use esse fork. É uma equipe pequena, então contribua de volta se depender dele.

Duas ferramentas relacionadas:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) está em modo de manutenção e não tem release marcado desde v0.6, em 2021, mas é confiável e ainda é a fonte mais direta de Mask R-CNN e da família R-CNN para segmentação de instâncias e panóptica.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, com manutenção ativa) inclui Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN e Keypoint R-CNN. Não tem YOLO ou DETR modernos, e você escreve seu próprio loop de treinamento, mas é a baseline sem dependências adicionais.

Um alerta para uso comercial: **MMYOLO**, o hub de reimplementações YOLO do OpenMMLab, usa GPL-3.0 em vez de Apache-2.0 e está congelado desde agosto de 2023.

## 6. A família RT-DETR

Licença: Apache-2.0 em todos. Ideal para detecção em tempo real de última geração com código em nível de pesquisa.

A fronteira atual da detecção em tempo real é formada por um grupo de detectores transformer relacionados, em vez de um YOLO. A maioria descende do RT-DETR, compartilha boa parte do código de backbone e encoder e quase todos usam Apache-2.0. Por isso, é fácil trocar entre eles. A maioria faz apenas detecção, com uma exceção (EdgeCrafter), que estende as mesmas ideias de destilação à segmentação e à pose. O custo comum é serem repositórios de pesquisa: treinamento por arquivos de configuração, ergonomia diferente do Ultralytics e nenhum canal de suporte. Os modelos são fortes e a exportação ONNX e TensorRT costuma ser de primeira classe.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Reformula a regressão de caixas como refinamento de distribuição com auto-destilação. Tem boa acurácia por latência (D-FINE-X chega a cerca de 55.8 AP em aproximadamente 13 ms em uma T4), tamanhos de N a X e integração com Hugging Face Transformers, o que facilita carregá-lo e fazer fine-tuning fora do próprio repositório.
- **[DEIM e DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) é uma receita de treinamento (correspondência Dense O2O) que pode ser adicionada ao D-FINE ou RT-DETRv2 e reduz o tempo de treinamento aproximadamente pela metade. DEIMv2 adiciona recursos DINOv3 e chega a um nível Atto com menos de 1M de parâmetros para dispositivos móveis; DEIMv2-S é o primeiro modelo com menos de 10M de parâmetros a ultrapassar 50 AP. Tem manutenção ativa e recebeu atualizações até 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). O trabalho mais recente do mesmo laboratório e o único desta lista que vai além da detecção: cobre detecção (ECDet), segmentação de instâncias (ECSeg) e pose (ECPose), cada uma nos tamanhos S/M/L/X, todos sob Apache-2.0. Adapta um ViT grande pré-treinado com DINOv3 como professor especializado por tarefa e o destila em backbones compactos de aluno, projetados para dispositivos de borda. Os números são fortes para o tamanho: ECDet-S chega a 51.7 AP com menos de 10M de parâmetros apenas no COCO, ECPose-X atinge 74.8 AP (acima dos 71.6 do YOLO26Pose-X) e sua segmentação de instâncias é competitiva com a do RF-DETR. É muito recente, então trate como um lançamento de pesquisa, mas é raro ver um modelo compacto que cobre as três tarefas.
- **[RT-DETR e RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). O trabalho original de que “DETRs superam YOLOs em detecção em tempo real” (CVPR 2024) e a origem da família: sem NMS, ponta a ponta e fácil de exportar. v2 é uma atualização de 2024 com um conjunto de melhorias gratuitas, um upgrade direto com a mesma latência. O original em Paddle e a versão canônica para PyTorch compartilham esse repositório, que também está no HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) e **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Universidade de Pequim e Tsinghua, final de 2025). v3 funciona somente com PaddlePaddle. v4 é o melhor da linha atualmente (X perto de 57 AP), usa PyTorch e destila um modelo fundacional de visão em um detector leve sem custo adicional na inferência. A base de código também reproduz D-FINE, DEIM e RT-DETRv2 por meio de alterações na configuração.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). Um DETR com ViT padrão, apresentado como substituto transformer do YOLO, em tamanhos de tiny a xlarge, com exportação ONNX e TensorRT. Também serve de base ao OVLW-DETR de vocabulário aberto. A atividade recente é menor; trate-o como um lançamento de pesquisa estável.

O LibreYOLO já integra muitos desses modelos (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter) em uma única API, tendo YOLO9 e RF-DETR como os modelos mais testados. Para implementações de referência e os checkpoints de pesquisa mais recentes, os repositórios originais acima são a fonte oficial.

## Perguntas frequentes

**O Ultralytics YOLO é gratuito para uso comercial?**
Somente sob AGPL-3.0, que exige a liberação do código-fonte do aplicativo que você criar com ele, incluindo serviços em rede. O uso comercial de código fechado exige a licença comercial paga do Ultralytics. Alternativas permissivas (MIT, Apache-2.0) evitam isso.

**Qual é a alternativa ao Ultralytics com menos restrições de licença?**
Para uma stack permissiva que você pode distribuir em qualquer lugar: LibreYOLO (código MIT), RF-DETR (Apache-2.0) e YOLOX (Apache-2.0). Todas evitam o copyleft da AGPL.

**O que está substituindo o YOLO em 2026?**
Cada vez mais, detectores transformer em tempo real: RT-DETR, RF-DETR e a linha D-FINE e DEIM. Eles igualam ou superam a acurácia do YOLO com latência semelhante em hardware capaz. CNNs no estilo YOLO ainda vencem em dispositivos de borda pequenos, onde esses transformers rodam mal e não têm um caminho maduro para NCNN ou CPU.

**Eles rodam em um Raspberry Pi ou em uma NPU?**
Depende da exportação. Detectores CNN como YOLOX e RTMDet exportam para NCNN e rodam em um Pi, e alguns (YOLOX) também funcionam na NPU Hailo. Detectores transformer como RF-DETR não rodam assim: precisam de uma GPU ou de uma CPU potente.

## Experimente

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

O LibreYOLO usa licença MIT, roda em Linux, Mac e Windows e funciona em GPU, Apple Silicon ou apenas CPU sem alterar o código. Uma única API abrange RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter e outros, para detecção, segmentação, pose, classificação, profundidade, estimativa de olhar e tracking.

Dê uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
