---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: faça predições e exporte sob Apache-2.0'
description: >-
  Use o DINO-DETR no LibreYOLO para detecção de objetos. Instale, faça
  predições, valide e exporte três tamanhos com anchors de denoising, todos
  licenciados sob Apache-2.0.
lead: >-
  O DINO-DETR, publicado pela IDEA Research com o nome DINO, combina treinamento
  com denoising contrastivo e seleção mista de queries sobre a atenção esparsa
  do Deformable DETR. O LibreYOLO inclui três tamanhos para detecção, somente
  inferência.
keywords:
  - DINO-DETR
  - DINO
  - transformer de detecção
  - detecção de objetos python
  - denoising contrastivo
  - anchor boxes com denoising
  - seleção mista de queries
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() devolve um dict simples, não um objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Instalação

O DINO-DETR não precisa de nenhum extra opcional. Tudo o que ele importa está
na instalação base, usando o mesmo núcleo de atenção deformável multiescala em
PyTorch puro da família Deformable DETR do LibreYOLO.

```bash
pip install libreyolo
```

Instalar `libreyolo[hub-kernels]` é opcional. Uma vez presente o pacote
`kernels`, o LibreYOLO busca um kernel compilado de atenção deformável
multiescala no Hugging Face Hub em runtime e o usa no lugar do núcleo em
PyTorch puro; `LIBREYOLO_HUB_KERNELS=0` desliga isso de novo.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` devolvido é o mesmo que toda família devolve, então trocar
por outro detector é uma mudança de uma linha. `conf` e `max_det` filtram a
seleção de queries; `iou` é aceito por paridade de API mas não tem efeito,
porque o decoder é um preditor de conjuntos sem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

O DINO-DETR é somente inferência no LibreYOLO. O upstream treina com denoising
contrastivo e matching húngaro; essa receita não está implementada aqui, então
`train()` levanta `NotImplementedError`.

## Variantes

Três checkpoints, todos na mesma resolução de entrada. `r50` e `r50s5`
compartilham um backbone ResNet-50 e diferem em quantas escalas de mapas de
características alimentam o decoder, quatro contra cinco. `swinl` troca o
backbone por Swin-L e também amostra cinco escalas.

## Validação

`val()` devolve um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidas contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que
todo formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

Os três checkpoints oficiais vêm da pasta de publicação no Google Drive dos
autores, não de uma model card do Hugging Face. O repositório upstream declara
Apache-2.0 no nível do repositório, mas não anexa um arquivo de licença nem
metadados de licença aos checkpoints em si, então a base para a redistribuição
é essa declaração no nível do repositório e não uma concessão específica dos
checkpoints. Todos os mirrors do LibreYOLO incluem o texto literal da licença
Apache-2.0 do upstream junto com um aviso explicando isso.

</provenance-box>

## Citação

<citation-block />
