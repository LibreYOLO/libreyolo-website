---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: detecção e reconhecimento de texto no LibreYOLO'
description: >-
  Use o PP-OCRv5 no LibreYOLO para OCR multilíngue de texto em cena. Instale,
  faça predições e valide os checkpoints t e l, com licença Apache-2.0.
lead: >-
  O PP-OCRv5 é o pipeline de detecção e reconhecimento de texto do PaddleOCR: um
  detector de binarização diferenciável localiza os quadriláteros de texto e um
  reconhecedor SVTR/CTC os lê. O LibreYOLO o porta para PyTorch em dois níveis.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - detecção de texto em imagens
  - reconhecimento de texto
  - extrair texto de imagem python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Quadriláteros
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # polígonos (N, 4, 2) em ordem de leitura: superior-esquerdo,
        # superior-direito, inferior-direito, inferior-esquerdo. Os
        # quadriláteros de detecção são polígonos de verdade (texto
        # rotacionado), então preenchem result.ocr, não result.boxes.
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # métrica principal
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Instalação

O PP-OCRv5 não precisa de nenhum extra além do pacote base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

Cada checkpoint reúne as duas etapas, detecção e reconhecimento, em um único
arquivo `.pt`, com o charset de reconhecimento e os padrões do pipeline
guardados nos metadados do checkpoint. O reconhecedor lê chinês simplificado e
tradicional, inglês, japonês e pinyin com um único dicionário. `result.ocr` é
um payload `OCRRegions`: `.data` guarda os polígonos de quatro pontos,
`.texts` as transcrições, `.conf` a pontuação de reconhecimento de cada região
e `.det_conf` a pontuação de detecção. Fontes com várias imagens rodam
sequencialmente: o pipeline de duas etapas não agrupa em batch entre imagens.
Veja [predição](/docs/predict) para fontes, streaming e tratamento dos
resultados.

## Variantes

Dois níveis: `t`, construído sobre os backbones mais leves
PP-LCNetV3/PP-OCRv5_mobile para uso em CPU, e `l`, construído sobre os
backbones de servidor PP-HGNetV2 para maior acurácia. Os dois níveis rodam a
detecção com um limite fixo do lado maior e reconhecem os recortes em batches;
`rec_batch` controla quantos recortes passam pelo reconhecedor a cada forward
pass.

## Validação

`val()` mede o pipeline contra um diretório de imagens mais um arquivo
`labels/<split>.jsonl`, ou o YAML de dataset equivalente, em que cada label
lista os polígonos das regiões de texto de cada imagem e suas transcrições.
Ele reporta o hmean de detecção (precisão/recall/F1 com correspondência por
IoU), o F1 end-to-end (o hmean mais uma correspondência exata da transcrição
depois da normalização, a métrica de fitness do checkpoint) e o 1-NED, a
distância de edição normalizada média sobre os pares correspondentes.

<code-tabs name="val" />

## Exportação

<export-matrix />

O PP-OCRv5 é um pipeline de duas redes, detecção e reconhecimento andando
juntos, não um único grafo rastreável, e a exportação não está implementada
para ele: nenhum formato é suportado ainda. Faça fine-tuning direto no código
de treinamento upstream sob Apache-2.0 e converta o resultado com
`weights/convert_ppocr_weights.py` se você precisar de um checkpoint fora
desse formato.

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
