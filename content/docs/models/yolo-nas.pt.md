---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: faça predições, treine e exporte no LibreYOLO'
description: >-
  Use o YOLO-NAS no LibreYOLO para detecção e pose. Os pesos da Deci.AI são
  proprietários e de uso não comercial, e o LibreYOLO não publica nenhum deles.
lead: >-
  Um detector convolucional cujo backbone e neck saíram da busca de arquiteturas
  da Deci.AI, construído com blocos RepVGG preparados para quantização. Seus
  pesos são da Deci.AI, licenciados apenas para uso não comercial, e o LibreYOLO
  não publica nenhum deles.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - detecção de objetos
  - estimativa de pose
  - detector quantizável
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Um nome que ainda não esteja em disco é baixado da CDN da Deci. O
        # download imprime antes os termos de licença da Deci; ficar com o
        # arquivo significa aceitá-los.
        model = LibreYOLO("LibreYOLONASs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pose
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O sufixo -pose seleciona a cabeça de pose e seu próprio conjunto de
        pesos.

        model = LibreYOLO("LibreYOLONASs-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Do zero
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Nenhum checkpoint da Deci é tocado: o modelo parte de pesos
        aleatórios,

        # então o que sai da execução deriva apenas dos seus dados.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Contra o COCO
      language: bash
      code: >
        # O yaml do COCO que vem junto carrega um script de download embutido,

        # então precisa de permissão explícita a menos que o dataset já esteja
        local.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A fábrica resolve pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Instalação

O YOLO-NAS não precisa de nada além do pacote base.

```bash
pip install libreyolo
```

## Predição

Um nome de checkpoint que ainda não esteja em disco é baixado da CDN pública da
Deci, não da organização do LibreYOLO, que não hospeda nenhum desses pesos.
Antes de a transferência começar, a biblioteca imprime os termos de licença da
Deci uma vez por processo, e antes de o arquivo baixado ser aberto seu SHA-256 é
conferido contra um valor fixado. O que esses termos permitem está em
[licenciamento](#licensing).

<code-tabs name="predict" />

O objeto `Results` devolvido é o mesmo que toda família devolve, então trocar
por outro detector é uma mudança de uma linha. `conf` define o limiar de
confiança e `iou` o limiar do NMS. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Variantes

Detecção e pose são a mesma arquitetura sob cabeças diferentes, e aceitam os
mesmos argumentos. Os tamanhos da tabela abaixo são os de detecção; pose é
publicado nesses e em um tamanho menor. A cabeça de pose prevê o conjunto de
keypoints do COCO.

<benchmark-table task="detect" />

<va-embed />

## Treinamento

<code-tabs name="train" />

`epochs`, `lr0` e `amp` são resolvidos por tarefa quando você os omite, então
uma execução de pose parte de padrões diferentes dos de uma execução de
detecção. O otimizador é o AdamW por padrão. O número de classes vem do YAML do
dataset e a cabeça é reconstruída para ele antes da primeira época; na cabeça de
pose o número de keypoints é tratado do mesmo jeito, então um checkpoint de pose
do COCO pode receber fine-tuning sobre um esqueleto de outro tamanho.

O fine-tuning parte dos pesos da Deci, que é o que a licença da Deci cobre.
Treinar a partir de um modelo inicializado aleatoriamente não envolve nenhum
checkpoint da Deci, e é isso que o terceiro snippet acima faz.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` devolve um dicionário de chaves `metrics/` cobrindo precisão, recall,
mAP 50 e mAP 50-95, medidos contra qualquer dataset no formato em que você
treinou.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta pelo `LibreYOLO()` conforme o sufixo
do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint
e devolve o mesmo `Results`. Rodar o grafo em um runtime puro, sem o LibreYOLO
instalado, também é suportado, mas aí o pré-processamento e o pós-processamento
ficam por sua conta. Cada formato instala um extra diferente e aceita alguns
argumentos próprios. Os dois estão na página daquele formato.

Uma exportação é outra cópia dos mesmos pesos em um contêiner diferente.
Exportar um checkpoint da Deci não muda nem de onde os pesos vieram nem a
licença que os cobre.

<code-tabs name="export" />

## Checkpoints

Não há nenhum para listar. A licença da Deci proíbe a redistribuição, então a
organização do LibreYOLO não publica pesos do YOLO-NAS e o download é resolvido
em outro lugar: um nome no formato `LibreYOLONAS<size>.pt`, ou
`LibreYOLONAS<size>-pose.pt` para pose, corresponde ao objeto equivalente na CDN
pública da Deci.

Só dá para baixar assim os checkpoints cujo SHA-256 a biblioteca fixa. Qualquer
outra coisa falha de forma fechada em vez de abrir um pickle de terceiros não
verificado, e precisa ser baixada à mão e passada como caminho. Um arquivo que
já esteja em disco carrega a partir do seu caminho, sem download e sem barreira
de checksum. Isso inclui um `.pth` da Deci com o nome original, que o carregador
reconhece.

## Licenciamento

<provenance-box>

O LibreYOLO não hospeda nem espelha esses pesos: não existe nada desta família
na organização do LibreYOLO no Hugging Face. Em vez disso, todo download
automático vai para a CDN pública da Deci, imprime os termos da Deci uma vez por
processo antes de começar, e é conferido contra um SHA-256 fixado antes de o
arquivo ser aberto.

Treinar a partir de um modelo inicializado aleatoriamente é a alternativa. A
arquitetura é Apache-2.0 no upstream e MIT aqui, então um modelo treinado desse
jeito com os seus próprios dados não deriva de nenhum checkpoint da Deci.

</provenance-box>

## Citação

O YOLO-NAS foi lançado sem paper. A entrada abaixo é a que os autores pedem, e
cobre o SuperGradients, a biblioteca em que ele foi distribuído.

<citation-block />
