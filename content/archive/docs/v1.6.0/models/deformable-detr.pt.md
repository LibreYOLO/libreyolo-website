---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: predição e exportação, Apache-2.0'
description: >-
  Rode o Deformable DETR no LibreYOLO para detecção de objetos. Instale, faça
  predição, validação e exportação de cinco tamanhos com atenção esparsa, todos
  sob licença Apache-2.0.
lead: >-
  O Deformable DETR troca a cross-attention densa do DETR por uma amostragem
  esparsa e multiescala ao redor de cada ponto de referência, o que foi o que
  tornou os detectores transformer viáveis de treinar. O LibreYOLO traz cinco
  tamanhos para detecção, apenas inferência.
keywords:
  - Deformable DETR
  - detection transformer
  - atenção esparsa
  - atenção multiescala
  - detecção de objetos python
  - transformer para detecção de objetos
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() retorna um dict simples, não um objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide o caminho pelo sufixo do arquivo, então um artefato
        # exportado carrega como qualquer checkpoint e retorna o mesmo Results.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Instalação

O Deformable DETR não precisa de nenhum extra opcional. Tudo o que ele importa
está na instalação base, com um núcleo de atenção deformável multiescala em
PyTorch puro.

```bash
pip install libreyolo
```

Instalar `libreyolo[hub-kernels]` é opcional. Com o pacote `kernels` presente,
o LibreYOLO baixa em tempo de execução um kernel compilado de atenção
deformável multiescala do Hugging Face Hub e o usa no lugar do núcleo em
PyTorch puro; `LIBREYOLO_HUB_KERNELS=0` desativa isso de novo.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det`
filtram a seleção de queries; `iou` é aceito por paridade de API, mas não tem
efeito, porque o decoder é um preditor de conjuntos sem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

No LibreYOLO, o Deformable DETR é apenas para inferência. O upstream treina com
matching húngaro e uma focal loss de classificação; essa receita não está
implementada aqui, então `train()` levanta `NotImplementedError`.

## Variantes

Cinco checkpoints cobrem as configurações publicadas, todas na mesma resolução
de entrada. O `r50ss` restringe a atenção a uma única escala de características;
o `r50ssdc5` acrescenta a isso um estágio C5 dilatado no backbone. O `r50` é a
configuração multiescala padrão, que amostra em quatro níveis de mapas de
características. O `r50refine` adiciona refinamento iterativo dos bounding boxes
ao longo das camadas do decoder, e o `r50twostage` gera suas propostas de região
iniciais a partir da saída do encoder em vez de usar queries aprendidas.

## Validação

`val()` retorna um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é recarregado por `LibreYOLO()` pelo sufixo do arquivo,
então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e retorna
o mesmo `Results`. [Exportação](/docs/export) lista os argumentos que todo
formato aceita.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
