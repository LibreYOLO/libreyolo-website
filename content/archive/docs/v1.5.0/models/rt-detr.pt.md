---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 e RT-DETRv4 no LibreYOLO'
description: >-
  Use RT-DETR, RT-DETRv2 e RT-DETRv4 no LibreYOLO para detecção de objetos, além
  de caixas orientadas no RT-DETRv2. Instale, rode predições, treine, valide e
  exporte, com pesos sob Apache-2.0.
lead: >-
  Um transformer de detecção feito para inferência em tempo real: ele decodifica
  um conjunto fixo de queries em vez de uma grade densa, então não roda NMS. O
  LibreYOLO traz três versões dele, diferenciadas pelo checkpoint que você
  carrega, e a versão 2 também serve caixas orientadas.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - transformer de detecção em tempo real
  - DETR
  - detecção de objetos
  - detecção de caixas orientadas
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vídeo
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A versão faz parte do nome do arquivo, e a factory decide pelo
        # checkpoint, então as três carregam do mesmo jeito.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Qualquer fonte que a biblioteca aceita: arquivo, pasta, URL, índice
        # de webcam, stream RTSP ou uma lista .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Caixas orientadas
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Só na versão 2. O sufixo -obb seleciona a tarefa, e o checkpoint é

        # reconhecido como orientado por seus tensores, então não precisa de

        # argumento task. Estes pesos são DOTA v1.0, 15 classes aéreas a 1024
        px.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radianos

        print(obb.xyxyxyxy)  # as mesmas linhas como quatro pontos de canto

        print(result.boxes.xyxy)  # caixas envolventes alinhadas aos eixos
    - label: 'Caixas orientadas, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRTDETRr18.pt")


        # O coco128.yaml baixa uma amostra de 128 imagens no primeiro uso.
        Aponte

        # `data` para o YAML do seu próprio dataset em uma execução de verdade.

        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Precisa do extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() retorna um dict simples, não um objeto
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Contra o COCO
      language: bash
      code: >
        # O coco-val-only.yaml busca as 5000 imagens de val2017 e pula o
        conjunto

        # de treinamento. Ele traz um script de download embutido, então precisa

        # de permissão explícita a menos que o dataset já esteja local.

        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Caixas orientadas
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A validação orientada faz o matching com IoU rotacionado, então uma
        # predição no lugar certo com o ângulo errado conta como erro.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Precisa do extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Caixas orientadas
      language: bash
      code: >
        # ONNX e TorchScript são os alvos validados para a tarefa orientada, em

        # FP32, batch 1, num canvas fixo de 1024 por 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Instalação

O RT-DETR não precisa de nenhum extra opcional. Tudo que ele importa já está na
instalação base, e o extra `rtdetr` é um nome estável que não acrescenta nada.

```bash
pip install libreyolo
```

O fine-tuning com adaptadores via `lora=True` é a exceção, e precisa do extra
`lora`.

```bash
pip install "libreyolo[lora]"
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det` filtram
uma decodificação top-k sobre queries e classes; não há etapa de NMS para
ajustar, e `iou` é aceito mas não usado. Um checkpoint orientado preenche
`result.obb` nativamente e também preenche `result.boxes` com os retângulos
envolventes alinhados aos eixos. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Três versões, duas tarefas entre elas, e os códigos de tamanho não seguem uma
série única. A versão 1 nomeia seus tamanhos pelo backbone, ResNet ou HGNetv2. A
versão 2 reaproveita só os nomes de ResNet: a versão 1 já traz os dois tamanhos
HGNetv2, e os resultados da versão 2 ali ficaram próximos o bastante para que o
LibreYOLO não publique pesos duplicados para eles. A versão 4 usa uma série de
letras simples, que colide com os nomes HGNetv2 da versão 1, então um código de
tamanho sozinho não identifica um modelo. A versão está escrita no nome do
arquivo do checkpoint.

<benchmark-table task="detect" />

<va-embed />

A versão 2 mantém a arquitetura e o layout do state dict da versão 1 e muda como
a atenção deformável amostra, e é por isso que as duas são diferenciadas pelos
metadados do checkpoint em vez de pelo formato. A versão 4 é outra linhagem: ela
reaproveita a arquitetura e o trainer do D-FINE, e seus pesos vêm da destilação
de um modelo de fundação de visão DINOv3 como professor em um aluno HGNetv2. No
LibreYOLO, `LibreRTDETRv4` é uma subclasse de `LibreDFINE` com a cabeça de
máscaras fixada em desligado, então ele fica só em detecção.

### Caixas orientadas na versão 2

A versão 2 é a única versão que carrega uma segunda tarefa. Suas tarefas
suportadas são `detect` e `obb`, e as duas não compartilham grafo nem série de
tamanhos. A detecção usa os tamanhos ResNet a 640 px; a detecção orientada usa
uma série HGNetv2, n, s, m, l e x, a 1024 px, e o tamanho de entrada é resolvido
por tarefa, não por família. Um checkpoint é reconhecido como orientado pelos
próprios tensores, pelas cabeças de caixa de cinco coordenadas e pelos parâmetros
de amostragem da versão 2, então pesos `-obb` carregam no grafo orientado sem um
argumento `task`, e uma incompatibilidade entre os dois é um erro duro em vez de
uma reinterpretação silenciosa.

Os arquivos publicados vão de `LibreRTDETRv2n-obb.pt` a
`LibreRTDETRv2x-obb.pt`. São os checkpoints oficiais de escala única do DOTA
v1.0 convertidos para o formato do LibreYOLO, 15 classes aéreas de avião e navio
a porto e helicóptero, e os nomes das classes estão gravados no checkpoint.
Diferente do lado da detecção, a tarefa orientada é só de inferência: predição,
validação e exportação funcionam, e `train()` em um modelo orientado levanta
erro. Tracking e test-time augmentation também não suportam caixas orientadas.
[Detecção orientada](/docs/tasks/oriented-detection) cobre a tarefa, o formato
das labels e as métricas.

## Treinamento

O treinamento parte de um checkpoint publicado. `pretrained` é aceito e depois
descartado nas três versões, então `pretrained=False` não te dá um modelo
inicializado aleatoriamente. Tudo nesta seção é sobre detecção: a tarefa
orientada da versão 2 é só de inferência, e não há caminho de transferência dos
pesos de detecção para ela, porque as duas usam backbones diferentes.

<code-tabs name="train" />

O learning rate é o argumento que precisa estar certo, e cada versão traz seu
próprio padrão em vez do padrão geral da biblioteca. A assinatura de `train()` no
Python lê esse valor da config de treinamento daquela versão, e o CLI resolve o
mesmo valor quando `lr0` não é passado. As versões 1 e 2 também aceitam
`lr_backbone` e o deixam por padrão em um vigésimo de `lr0`, seguindo a receita
original; a versão 4 roda pelo trainer do D-FINE, que em vez disso escala o grupo
de parâmetros do backbone com `backbone_lr_mult`.

Deixe `imgsz` no tamanho nativo do checkpoint a menos que você tenha um motivo
para mudar. Validação e predição em outros tamanhos funcionam, com um resíduo: um
tamanho retangular cuja contagem de tokens bate com a do tamanho nativo ainda
reaproveita um embedding construído para a proporção errada.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

As linhas da tabela de benchmark acima vêm do harness de benchmark do LibreYOLO;
a nota abaixo daquela tabela registra qual dataset as produziu e traz os links
dos registros de execução.

A validação orientada passa pela mesma chamada e reporta as mesmas chaves, mais
quatro repetidas sob um sufixo `(OBB)`. O matching usa IoU rotacionado em vez do
IoU dos retângulos envolventes, então um erro de ângulo é uma falha.
`augment=True` é rejeitado nesta tarefa.

## Exportação

<export-matrix />

A matriz cobre a linhagem como uma página só: onde as três versões discordam
sobre um formato, a célula mostra a mais fraca das três, então nada aqui é
prometido a mais para a versão que você carregar. A linha orientada pertence só à
versão 2. ONNX e TorchScript são validados ali, em FP32, batch 1 e um canvas fixo
de 1024 por 1024; OpenVINO, TensorRT e ExecuTorch convertem e recarregam, mas não
atingiram paridade de saída bruta em todo o conjunto de queries, então as caixas
do topo batem a uma fração de pixel enquanto a cauda desvia.

Um artefato exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

O nome do arquivo carrega a versão, depois o tamanho, depois a tarefa. Os pesos
de detecção são `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` e
`LibreRTDETRv4<size>.pt`, todos a 640 px. Pesos orientados existem só para a
versão 2 e acrescentam o sufixo da tarefa, de `LibreRTDETRv2n-obb.pt` a
`LibreRTDETRv2x-obb.pt`, todos a 1024 px e treinados no DOTA v1.0 em vez do COCO.

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />

O bloco acima é o que os autores publicam para a detecção das versões 1 e 2. Os
pesos orientados da versão 2 têm um terceiro upstream, o repositório RiO-DETR sob
Apache-2.0 em
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), que é
de onde vêm os checkpoints do DOTA; cite esse projeto se você usou um deles. A
versão 4 é
um artigo separado de um grupo diferente e tem seu próprio bloco de citação em
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
cite esse se você usou um checkpoint da versão 4.
