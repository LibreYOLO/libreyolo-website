---
title: YOLOv9
families: [yolo9]
seo_title: "YOLOv9: preveja, treine e exporte sob a MIT"
description: "Rode o YOLOv9 no LibreYOLO, incluindo a cabeça end-to-end sem NMS e a cabeça de stride 4 para objetos pequenos. Instale, faça predições, treine, valide e exporte."
lead: "Um detector convolucional de estágio único: uma passada pontua uma grade densa de caixas e o NMS descarta as duplicadas. O LibreYOLO traz três variantes dele, uma delas sem etapa de NMS."
keywords: [YOLOv9, YOLO9, detecção de objetos, detecção sem NMS, detecção end-to-end, detecção de objetos pequenos, yolov9 python, treinar yolov9, programmable gradient information, GELAN]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Sem NMS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Mesma chamada, checkpoint diferente. A cabeça end-to-end devolve suas
        # próprias predições de maior pontuação, então nenhum NMS roda e iou é ignorado.
        model = LibreYOLO("LibreYOLO9E2Es.pt")
        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)

        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Objetos pequenos
      language: python
      code: |
        from libreyolo import LibreYOLO9P2

        # A variante de stride 4 não tem checkpoint COCO próprio, então indique
        # um de detecção base: o backbone e o neck dele carregam sem alterações
        # e a torre da cabeça de stride 4 parte de inicialização aleatória.
        model = LibreYOLO9P2(None, size="s")
        model.train(data="my-dataset.yaml", epochs=100, pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Contra o COCO
      language: bash
      code: |
        # O yaml de COCO que vem junto carrega um script de download embutido,
        # então ele precisa de permissão explícita a menos que o dataset já esteja local.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Com NMS no grafo
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreYOLO9s.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Instalação

O YOLOv9 não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. Nos modelos base e de
stride 4, `conf` define o limiar de confiança e `iou` o limiar do NMS. O modelo
end-to-end não roda NMS e ignora `iou`, então `conf` e `max_det` são o que dá
forma à saída dele. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Variantes

Três variantes compartilham um backbone. As três só detectam, e aceitam os
mesmos argumentos.

O modelo base prediz em três escalas de características e elimina as caixas
duplicadas com NMS.

O modelo end-to-end mantém essa cabeça e acrescenta ao lado dela um ramo de
correspondência um-para-um. A inferência lê só o ramo um-para-um e pega as
predições de maior pontuação dele, então nenhum NMS roda. Escolha esse quando o
runtime em que você faz deploy não tiver operador de NMS.

O modelo de stride 4 puxa mais um nível acima no backbone, estende o neck até
ele e prediz em quatro escalas em vez de três. A escala extra é para objetos que
cobrem poucos pixels; o único checkpoint publicado para ele é treinado com
imagens aéreas. Checkpoints de detecção base se transferem para ele: o backbone
e o neck carregam sem alterações, as três torres de cabeça pré-treinadas sobem
uma posição e a torre de stride 4 parte de inicialização aleatória.

<benchmark-table task="detect" />

<va-embed />

## Treinamento

<code-tabs name="train" />

`pretrained` decide de onde a execução parte. Passe `True` para carregar o
checkpoint publicado do mesmo modelo e tamanho, ou um nome ou caminho para
qualquer outra coisa. Tensores cuja forma não bate são pulados em vez de
recusados, e a execução registra quantos foram carregados, então um checkpoint
treinado com um número diferente de classes ainda é um ponto de partida
utilizável.

O modelo de stride 4 não tem checkpoint COCO publicado próprio, então `True`
resolve ali para um arquivo que não existe e o download falha. Indique um
checkpoint de detecção base no lugar.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Uma marca vale para as três variantes: onde elas diferem, a matriz traz a mais
fraca das três.

Um artefato exportado é recarregado pelo `LibreYOLO()` a partir do sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta.

Para o modelo de detecção base, a metade de pós-processamento disso pode ir para
dentro do grafo. `nms=True` em uma exportação para ONNX coloca a supressão
dentro do modelo, e a primeira saída passa a ser um tensor fixo
`(1, max_det, 6)` cujas linhas são `x1, y1, x2, y2, score, class`, preenchidas
com zeros além da contagem de detecções. Esse grafo é de batch 1 e não carrega
eixos dinâmicos. Os modelos end-to-end e de stride 4 não aceitam a flag.

Cada formato instala um extra diferente e aceita alguns argumentos próprios. As
duas coisas estão na página daquele formato.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

Um checkpoint aqui não é MIT. O modelo de stride 4 treinado no VisDrone2019-DET
herda os termos CC BY-NC-SA 3.0 desse dataset: só uso não comercial, share-alike
em tudo que derivar dele, e fora da licença permissiva sob a qual o resto desta
família é distribuído. Ele prediz as classes aéreas do VisDrone em vez das do
COCO. A biblioteca imprime tudo isso antes de baixar o arquivo.

</provenance-box>

## Citação

<citation-block />
