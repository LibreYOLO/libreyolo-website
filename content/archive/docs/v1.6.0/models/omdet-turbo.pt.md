---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo no LibreYOLO: detecção zero-shot em tempo real'
description: >-
  Use o OMDet-Turbo no LibreYOLO para detecção de vocabulário aberto em tempo
  real. Instale o extra openvocab e faça predições com um vocabulário de texto
  livre.
lead: >-
  O OMDet-Turbo é um detector de objetos de vocabulário aberto em tempo real,
  desenvolvido pelo Om AI Lab, que desacopla os embeddings de classe do prompt
  de tarefa de linguagem. O LibreYOLO o encapsula como uma família somente de
  predição no seu tier de detectores de vocabulário aberto.
keywords:
  - OMDet-Turbo
  - OmDet
  - detecção de vocabulário aberto
  - detecção de objetos em tempo real
  - detecção zero-shot
  - detectar objetos por texto
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Limiar de NMS personalizado
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # O OMDet-Turbo é a única família deste tier que respeita iou=: o
        # próprio pós-processamento dele recebe o limiar de supressão como
        # argumento, e assume 0.5 quando iou= não é definido.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Instalação

O OMDet-Turbo é carregado pelo tier de detectores de vocabulário aberto do
LibreYOLO, que precisa do extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Esse extra traz junto `transformers` e `timm`, as bibliotecas da Hugging Face
que este tier chama; o backbone Swin do OMDet-Turbo é carregado pelo wrapper
`TimmBackbone` do `transformers`.

## Predição

O OMDet-Turbo não é um checkpoint que o LibreYOLO carrega por meio de
`LibreYOLO()`. Ele é carregado pela factory irmã `LibreOpenVocab`, que baixa um
snapshot da Hugging Face no primeiro uso e o guarda em cache em `weights/`.

<code-tabs name="predict" />

`set_classes()` define um vocabulário de texto persistente: chame de novo para
substituir a lista inteira, ou não chame para manter os rótulos COCO-80 padrão,
e um resultado vazio é um desfecho válido, não um erro. Diferente do Grounding
DINO, o OMDet-Turbo desacopla os embeddings de classe do prompt de tarefa de
linguagem, então o pós-processamento do `transformers` retorna rótulos que
mapeiam direto de volta para a lista de classes consultada, sem nenhuma etapa de
desambiguação de frases.

O OMDet-Turbo não tem limiar de token de texto: só `conf` filtra as detecções, e
passar `text_threshold` levanta erro. É a única família deste tier que roda a
própria non-maximum suppression dentro de
`post_process_grounded_object_detection`, então `iou` é respeitado aqui em vez
de gerar um aviso. `imgsz` e `augment=True` são rejeitados de saída: o
processador do `transformers` é quem cuida do redimensionamento, e o data
augmentation em tempo de teste está fora do escopo deste tier. `predict()` em
uma única imagem retorna um `Results`, não uma lista; passe um diretório, uma
lista de imagens ou `stream=True` com uma fonte de vídeo para obter vários. Não
há caminho de CLI para esta família, `libreyolo predict` só carrega checkpoints
`.pt` por meio de `LibreYOLO()`, então as famílias de `LibreOpenVocab` rodam a
partir do Python. Veja [predição](/docs/predict) para tipos de fonte e
streaming.

## Variantes

Um checkpoint, `t`, o único tamanho deste tier. Ele espelha
`omlab/omdet-turbo-swin-tiny-hf` em uma revisão upstream fixada, por meio do
`OmDetTurboForObjectDetection` do `transformers`; o arquivo de pesos espelhado é
byte a byte idêntico a esse snapshot upstream. Ainda não há números publicados
de acurácia ou de latência para esta família.

Treinamento, validação de dataset e exportação estão todos fora do escopo deste
tier: `train()`, `val()` e `export()` levantam `NotImplementedError` de forma
incondicional. Este é um wrapper somente de predição em volta de um checkpoint
publicado.

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
