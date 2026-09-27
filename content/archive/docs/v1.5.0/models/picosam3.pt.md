---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: segmentação no edge guiada por caixas no LibreYOLO'
description: >-
  Use o PicoSAM3 no LibreYOLO para segmentação de regiões guiada por caixas em
  sensores edge. Instale, faça predições e exporte o checkpoint pico sob a
  Apache-2.0.
lead: >-
  O PicoSAM3 é uma CNN compacta destilada do SAM 2.1 e do SAM 3, feita para
  segmentação de regiões de interesse guiada por caixas em sensores como o Sony
  IMX500. O LibreYOLO dá suporte a ele por uma factory LibreSAM dedicada,
  separada da factory de detectores LibreYOLO(), e só com prompts de caixa.
keywords:
  - PicoSAM3
  - Segment Anything
  - segmentação no edge
  - segmentar imagem por caixa
  - região de interesse
  - inferência no sensor
  - IMX500
  - destilação de conhecimento
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt de caixa
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # O PicoSAM3 tem um único tamanho, "pico", então não há outro alias.
        model = LibreSAM("picosam3")

        # bboxes= é o único prompt suportado: [x1, y1, x2, y2] ou uma lista de
        # caixas, uma máscara por caixa. Cada caixa é expandida em 10%, virada
        # quadrada, recortada à imagem e redimensionada para 96x96 antes da CNN.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # um polígono por máscara
        print(result.boxes.xyxy)    # caixa justa derivada da máscara
    - label: 'Codifique uma vez, envie vários prompts'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() guarda a imagem de origem em cache; o PicoSAM3 roda um

        # forward completo da CNN por caixa, então isso economiza a carga e o

        # decode da imagem, não um passo de encoder como nas outras famílias
        SAM.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (13 por padrão) e dynamic (True por padrão, só o eixo de batch)
        # são os únicos argumentos de exportação que esta família aceita.
    - label: Usar o arquivo exportado
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # O PicoSAM3 exporta sua CNN de ROI 96x96 crua: roi_image ->
        mask_logits.

        # Aqui não há pré/pós-processamento do lado do LibreYOLO para
        reaproveitar,

        # porque export() não é roteado de volta por LibreYOLO() como acontece

        # com o checkpoint de um detector.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Instalação

O PicoSAM3 precisa do extra `sam`: o download de pesos do próprio LibreYOLO
ainda passa pelo ferramental do Hugging Face que vem com `transformers`, mesmo
que a inferência rode sobre uma CNN nativa, sem `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Predição

`LibreSAM(...)` (ou o `LibrePicoSAM3(...)` específico da família) é um ponto de
entrada separado de `LibreYOLO(...)`: ele devolve um segmentador guiado por
prompts em vez de um detector, porque aqui um forward pass não significa nada
sem um prompt. Não existe comando de CLI `libreyolo predict` para esta família;
use a API Python.

<code-tabs name="predict" />

O PicoSAM3 aceita apenas `bboxes=`; passar `points=`, `labels=`, `masks=`,
`text=`, `multimask=True` ou omitir a caixa para segmentar tudo levanta um
`ValueError` claro em todos os casos, já que nenhum desses modos existe no
modelo original. `conf` filtra pela qualidade de máscara predita (IoU), não por
uma confiança de detecção, e precisa estar entre `0.0` e `1.0`. Toda máscara
carrega o id de classe `0`, com nome `"object"`. `train()`, `val()` e `track()`
levantam `NotImplementedError`; use o LibreSAM2 ou o LibreSAM3 para prompts de
ponto, texto, máscara ou de segmentar tudo. Veja [predição](/docs/predict) para
os tipos de fonte.

## Variantes

Um único tamanho, pico, com entrada de ROI fixa de 96 px: o PicoSAM3 roda um
forward completo da CNN por caixa em vez de codificar a imagem inteira uma vez
só.

## Exportação

<export-matrix />

O PicoSAM3 é a única família do nível SAM que exporta: ele leva sua CNN de ROI
96x96 crua para ONNX, `roi_image -> mask_logits`, sem NMS nem pós-processamento
de máscara embutidos. As outras famílias SAM levantam `NotImplementedError` no
`export()`, já que a separação entre encoder e decoder delas ainda não tem um
contrato de exportação definido para runtime. Um grafo PicoSAM3 exportado não
carrega de volta por `LibreYOLO()`; rode ele direto com um runtime como o
`onnxruntime`, aplicando o mesmo pré-processamento de ROI quadrada com 10% de
folga mostrado acima.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O PicoSAM3 é destilado do SAM 2.1 e do SAM 3, usados como modelos professores.
O LibreYOLO não incorpora nem redistribui o código ou os pesos de nenhum dos
dois professores nesta família; só a CNN estudante compacta e seu checkpoint
convertido são distribuídos.

</provenance-box>

## Citação

<citation-block />
