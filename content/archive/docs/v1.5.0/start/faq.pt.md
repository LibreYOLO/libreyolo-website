---
title: FAQ
seo_title: Perguntas frequentes sobre o LibreYOLO
description: >-
  Respostas curtas para as perguntas que atravessam todos os modelos do
  LibreYOLO: hardware, licenciamento, pesos, dispositivos, treinamento,
  cobertura de exportação e a CLI.
lead: >-
  Respostas para perguntas que não são específicas de uma família de modelos.
  Tudo que é específico de uma família fica na página daquela família.
keywords:
  - libreyolo faq
  - libreyolo perguntas frequentes
  - libreyolo precisa de gpu
  - licença libreyolo uso comercial
  - onde ficam os pesos do libreyolo
  - libreyolo cli
  - libreyolo sem internet
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Por qual modelo devo começar?

YOLOv9 para um detector CNN e RF-DETR para um baseado em transformer. Os dois
estão no nível flagship, o que significa que os recursos são projetados e
validados em GPU neles antes dos demais. Veja
[YOLOv9](/docs/models/yolov9) e [RF-DETR](/docs/models/rf-detr), ou
[todos os modelos](/docs/models) para o resto.

## Preciso de uma GPU?

Não. Todo modelo roda em CPU, e tudo no [início rápido](/docs/quickstart) foi
escrito para rodar nela. Uma GPU muda quanto tempo o treinamento e a inferência
em vídeo levam, não se funcionam.

## Como o LibreYOLO escolhe o dispositivo?

O padrão é `device="auto"`, que usa CUDA quando o PyTorch informa que está
disponível, depois Metal Performance Shaders quando estão disponíveis, e CPU
caso contrário. Para fixar, passe `device` para o modelo ou para `predict`,
`train`, `val` e `export`. Ele aceita `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, um
inteiro simples como `0`, ou uma string de dígitos; os dois últimos se expandem
para `cuda:<n>`.

`libreyolo checks` imprime o build do Torch, suas versões de CUDA e cuDNN, e
todas as GPUs que ele consegue enxergar. Se esse comando não mostrar CUDA, o
wheel do PyTorch é um build de CPU; a [instalação](/docs/install) explica como
substituí-lo.

## Para onde vão os pesos baixados?

Para `weights/`, relativo ao diretório de trabalho. Uma referência de modelo sem
componente de diretório é resolvida ali e baixada no primeiro uso; uma
referência que inclui um diretório é usada exatamente como foi escrita e nunca é
baixada. Veja [checkpoints e pesos](/docs/weights).

## Dá para rodar sem acesso à rede?

Dá. Baixe os checkpoints uma vez em uma máquina conectada, copie o diretório
`weights/` para a outra, e nada mais vai acessar a rede. Um caminho
compartilhado somente leitura também funciona, já que uma referência que contém
um diretório é interpretada literalmente. Os datasets são resolvidos em
`~/datasets`, ou em `LIBREYOLO_DATASETS_DIR`.

## Posso usar o LibreYOLO comercialmente?

O código é licenciado sob MIT. Os pesos pré-treinados são um caso à parte: eles
podem herdar termos do projeto ou do dataset de onde vieram, e esses termos não
são uniformes nem dentro de uma mesma família. A licença no repositório
específico do Hugging Face é a que vale, e toda página de modelo traz uma seção
de licenciamento que a reproduz. Quando os pesos são restritos, o LibreYOLO
imprime a restrição antes de o download começar.

## Posso carregar um checkpoint de outro projeto?

Geralmente sim, passando o caminho dele para `LibreYOLO()`. Layouts upstream
reconhecidos são convertidos no carregamento, mantendo a contagem e os nomes das
classes, e um checkpoint do LibreYOLO é gravado ao lado do original.
[Importar pesos existentes](/docs/migrate) cobre o que é reconhecido e o que
precisa de um script de conversão.

## Por que o train lança NotImplementedError?

Porque essa família vem só com inferência, e a exceção diz o motivo. Predição,
validação e, onde há suporte, exportação funcionam; não existe loop de
treinamento para essa arquitetura no LibreYOLO. O nível de suporte no
cabeçalho da página do modelo avisa antes de você tentar. Veja
[conceitos básicos](/docs/concepts).

## O que o val retorna?

Um dicionário simples, não um objeto. As chaves de detecção incluem
`metrics/precision`, `metrics/recall`, `metrics/mAP50` e `metrics/mAP50-95`.
Outras tarefas retornam as chaves que fazem sentido para elas, como
`metrics/accuracy_top1` para classificação ou `metrics/PQ`, `metrics/SQ` e
`metrics/RQ` para segmentação panóptica.

## Como rodo em uma pasta, um vídeo ou uma webcam?

Passe como fonte. Um caminho de arquivo é uma imagem, um diretório é todas as
imagens dentro dele, um caminho de vídeo é um vídeo, um inteiro é o índice de
uma webcam, e uma URL RTSP, RTMP, TCP, UDP ou HLS é um stream ao vivo. Um
arquivo `.streams` lista várias fontes de uma vez. Fontes ao vivo exigem
`stream=True`, que entrega um `Results` por quadro em vez de montar uma lista;
vale usar a mesma flag para vídeos longos e diretórios grandes. Só as URLs de
páginas do YouTube precisam de um extra, `libreyolo[stream]`.

## Como mantenho só algumas classes?

Passe `classes` para `predict` com os índices das classes que você quer, por
exemplo `classes=[0, 2]`. `conf` define o limiar de confiança, padrão `0.25`, e
`max_det` limita as detecções por imagem, padrão `300`.

## A CLI usa flags ou pares chave=valor?

Chave e valor unidos por um sinal de igual, em todo comando:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` aceita um caminho ou um nome curto no formato `family-size`,
opcionalmente com um sufixo de tarefa, e `libreyolo models` lista todos os
válidos. Comandos de diagnóstico e de inventário também aceitam `--json`, que
imprime os mesmos dados como um objeto legível por máquina no stdout.

## Todo modelo exporta para todo formato?

Não. A cobertura é por família e por tarefa, não é uniforme, e cada formato tem
seu próprio extra para instalar. Cada página de modelo traz a matriz de
exportação da sua família; a [seção de exportação](/docs/export) cobre os
formatos em si.

## Qual é a diferença entre segment, semantic e panoptic?

São três tarefas distintas. `segment` produz uma máscara por objeto detectado.
`semantic` rotula cada pixel com uma classe e não separa nada em instâncias.
`panoptic` dá a cada pixel exatamente um rótulo, juntando coisas contáveis com
regiões amorfas. Elas têm ground truth diferente, campos de resultado diferentes
e métricas diferentes, e uma família suporta as que aparecem na sua lista de
tarefas.

## Como treino com as minhas próprias classes?

Escreva um YAML de dataset com `train`, `val` e `names`. As labels ficam ao lado
das imagens em uma árvore `labels/` paralela, um `.txt` por imagem, com
coordenadas normalizadas. `nc` é opcional e precisa bater com `names` quando
está presente. Rode `libreyolo doctor <data.yaml>` antes: ele procura problemas
no dataset e sai com código diferente de zero quando encontra erros, o que
permite usá-lo como gate de CI.

## Por que o carregamento imprime um aviso de metadados?

Porque o checkpoint não traz os metadados v1.0 completos. O carregamento
continua por um caminho de compatibilidade, e o aviso diz exatamente quais
chaves estão faltando. Rode `libreyolo metadata path=<file>` para ver o que
existe, e veja [checkpoints e pesos](/docs/weights) para saber o que o schema
exige.

## Um import parou de funcionar depois de uma atualização. O que mudou?

Dois nomes de classe foram renomeados por consistência: `LibreYOLORTDETR` virou
`LibreRTDETR` e `LibreYOLORFDETR` virou `LibreRFDETR`. Os nomes antigos
continuam funcionando e emitem um `DeprecationWarning` apontando para o novo,
então o código existente continua rodando enquanto você o atualiza.
