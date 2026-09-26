---
title: OCR
seo_title: 'OCR: detecção e reconhecimento de texto no LibreYOLO'
description: >-
  Encontre e leia texto em imagens com o LibreYOLO. Faça predições de
  quadriláteros e transcrições, rotule um dataset JSONL e valide com hmean, F1
  end-to-end e 1-NED.
lead: >-
  O OCR localiza o texto de uma imagem e o lê. O LibreYOLO expõe isso como a
  tarefa ocr, que devolve um polígono de quatro pontos e uma transcrição por
  região de texto, em ordem de leitura.
keywords:
  - ocr python
  - extrair texto de imagem python
  - reconhecimento de texto em cena
  - detecção de texto em imagens
  - PP-OCRv5 python
  - ler texto de imagem com python
last_verified: 1.5.0
snippets:
  predict:
    - label: Ler o texto de uma imagem
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # O nível t é o mais leve dos dois, feito para CPU. O SAMPLE_IMAGE
        # mantém isso executável; aponte para uma imagem sua que tenha texto.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Ler os quadriláteros
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        regions = result.ocr

        print(regions.data.shape)   # polígonos (N, 4, 2), TL TR BR BL

        print(regions.xyxy)         # envoltórias alinhadas aos eixos desses
        polígonos

        print(regions.det_conf)     # pontuação de detecção, separada de .conf
    - label: Filtrar pela confiança do reconhecimento
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Indexe por posições, não por máscara booleana: o fatiamento leva

        # junto as transcrições e os dois arrays de pontuação, além da
        geometria.

        regions = result.ocr.numpy()

        keep = regions[np.flatnonzero(regions.conf >= 0.9)]

        print(keep.texts)
  val:
    - label: Validar e ler as chaves das métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Definição

A tarefa `ocr` faz duas coisas em uma só chamada: localiza cada região de texto
de uma imagem e a transcreve. As regiões voltam como polígonos de quatro pontos
em vez de boxes alinhados aos eixos, porque texto em cena costuma estar
rotacionado, e em ordem de leitura, de cima para baixo e depois da esquerda para
a direita.

Uma predição preenche `result.ocr`, um payload `OCRRegions`. `.data` é um array
float `(N, 4, 2)` de polígonos em pixels da imagem original, ordenados como canto
superior esquerdo, superior direito, inferior direito e inferior esquerdo;
`.texts` é a lista das N transcrições; `.conf` é a pontuação de reconhecimento de
cada região e `.det_conf` a pontuação de detecção; `.xyxy` dá a envoltória
alinhada aos eixos de cada polígono. Como os quadriláteros são polígonos de
verdade, eles não preenchem `result.boxes`. Fatiar um `OCRRegions` leva junto as
transcrições e os dois arrays de pontuação, além da geometria.

## Modelos

Duas famílias atendem `ocr`.

O [PP-OCRv5](/docs/models/pp-ocrv5) é o pipeline dedicado: um detector de
binarização diferenciável encontra os quadriláteros de texto e um reconhecedor
SVTR/CTC os lê, com as duas etapas reunidas em um único arquivo `.pt` junto com o
charset de reconhecimento. Ele vem em dois níveis, um mais leve para CPU e um de
servidor para maior acurácia, e um único dicionário cobre chinês simplificado e
tradicional, inglês, japonês e pinyin.

O [SenseNova-Vision](/docs/models/sensenova-vision) chega ao OCR gerando as
palavras como texto com tags a partir do mesmo checkpoint de 7B que atende suas
outras seis tarefas, carregado com `LibreVLM("sensenova-vision", task="ocr")`.
Ele precisa do extra `sensenova`, e seus pesos são restritos a uso não comercial;
a licença está na página dele.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

O PP-OCRv5 roda a detecção com um limite fixo do lado maior e depois reconhece as
regiões recortadas em batches, com `rec_batch` controlando quantos recortes
passam pelo reconhecedor a cada forward pass. Fontes com várias imagens rodam
sequencialmente, porque um pipeline de duas etapas não agrupa em batch entre
imagens. Veja [predição](/docs/predict) para fontes, streaming e tratamento dos
resultados.

## Formato do dataset

Os rótulos de OCR são um arquivo JSONL por split, um objeto JSON por imagem, ao
lado das próprias imagens.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Cada linha nomeia uma imagem e lista suas regiões:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` é um quadrilátero de quatro pontos em coordenadas absolutas de pixel,
ordenado como canto superior esquerdo, superior direito, inferior direito e
inferior esquerdo. Uma região cujo texto não pode ser lido é rotulada com
`"text": "###"`, a convenção don't-care do ICDAR: ela fica de fora da pontuação
de reconhecimento, e uma predição que a sobrepõe é ignorada em vez de contar como
falso positivo.

Passar o diretório raiz como `data=` já basta. Um YAML de dataset é a
alternativa, com `path` mais os nomes opcionais dos diretórios `images` e
`labels`, e `nc: 1` com `names: {0: text}` como placeholders de esquema, já que um
modelo de OCR devolve `Results.ocr` em vez de detecções. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

Nenhuma das duas famílias de OCR tem implementação de treinamento: `train()`
levanta `NotImplementedError` nas duas, e o suporte a OCR cobre apenas predição e
validação. A página do PP-OCRv5 indica o código de treinamento upstream sob
Apache-2.0 e o script de conversão que traz um checkpoint com fine-tuning de
volta para o LibreYOLO.

## Validação

`val()` pontua o pipeline inteiro, detecção e reconhecimento juntos, casando
polígonos previstos com polígonos do ground truth um a um com IoU acima de 0.5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` e `metrics/det_hmean` pontuam só a
localização: para haver correspondência basta a sobreposição dos polígonos, seja
qual for a transcrição. `metrics/e2e_precision`, `metrics/e2e_recall` e
`metrics/e2e_f1` acrescentam a leitura: a correspondência exige a mesma
sobreposição de polígonos e uma transcrição idêntica depois da normalização NFKC
e da remoção de espaços em branco, e a comparação continua sensível a maiúsculas
e minúsculas. `metrics/e2e_f1` também é o `fitness`, o número que a seleção do
melhor checkpoint lê.

`metrics/rec_1-NED` avalia o reconhecedor sozinho, sobre os pares que a detecção
já casou: um menos a distância de edição normalizada, de modo que uma transcrição
que erra um caractere pontua perto de 1, enquanto o F1 end-to-end dá 0 a ela.

## Exportação

Nenhum formato de exportação está disponível para esta tarefa. O PP-OCRv5 são
duas redes trabalhando juntas em vez de um único grafo rastreável, e `export()`
levanta erro para todo formato nas duas famílias. Para fazer deploy fora do
LibreYOLO, faça fine-tuning upstream e use o caminho de deploy upstream.
