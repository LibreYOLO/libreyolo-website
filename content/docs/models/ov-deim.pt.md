---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM no LibreYOLO: detecção de vocabulário aberto'
description: >-
  Use OV-DEIM no LibreYOLO para detecção de vocabulário aberto em tempo real, no
  estilo DETR. Instale o extra openvocab e faça predições com um vocabulário de
  texto livre.
lead: >-
  OV-DEIM é um detector de objetos de vocabulário aberto no estilo DETR que casa
  as queries do decoder com embeddings de texto de uma torre de texto MobileCLIP
  embutida. O LibreYOLO o porta nativamente como uma família apenas de predição
  dentro do seu tier de detectores de vocabulário aberto.
keywords:
  - OV-DEIM
  - DEIMv2
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

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Substituir o vocabulário
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # Uma segunda chamada a set_classes() substitui o vocabulário por
        # completo e gera os embeddings dele de novo pela torre de texto; um
        # resultado vazio é um desfecho válido, não um erro.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Instalação

O OV-DEIM carrega pelo tier de detectores de vocabulário aberto do LibreYOLO,
que precisa do extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Ao contrário do resto deste tier, o OV-DEIM é um port nativo do LibreYOLO em vez
de um wrapper de `transformers`, não existe uma classe de modelo de
`transformers` para ele, mas o mesmo extra cobre os pacotes `huggingface_hub`,
`safetensors`, `regex` e `ftfy` de que ele precisa no momento da predição.

## Predição

O OV-DEIM não é um checkpoint que o LibreYOLO carregue por `LibreYOLO()`. Ele
carrega pela fábrica irmã `LibreOpenVocab`, que baixa um snapshot do Hugging
Face no primeiro uso e o mantém em cache em `weights/`.

<code-tabs name="predict" />

`set_classes()` define um vocabulário de texto persistente: chame de novo para
substituir a lista por completo, ou pule a chamada para manter os rótulos
COCO-80 padrão, e um resultado vazio é um desfecho válido, não um erro. Cada
query do decoder é pontuada por similaridade de cosseno contra os embeddings de
texto de uma torre de texto MobileCLIP-B(LT) embutida, calculados online para
qualquer vocabulário que esteja definido e mantidos em cache até que ele mude,
então prompts arbitrários funcionam sem nenhum arquivo de embeddings
pré-computados.

O OV-DEIM não tem limiar de token de texto: só `conf` filtra as detecções, e
passar `text_threshold` lança um erro. A correspondência é uma seleção top-K
um-para-um, então nada aqui roda non-maximum suppression, e `iou` é aceito por
compatibilidade de API, mas avisa e não faz nada. `imgsz` e `augment=True` são
rejeitados de saída: o modelo tem uma entrada com letterbox fixo própria, e o
data augmentation em tempo de teste está fora do escopo deste tier. `predict()`
em uma única imagem retorna um `Results`, não uma lista; passe um diretório, uma
lista de imagens ou `stream=True` para uma fonte de vídeo para obter vários. Não
há caminho de CLI para esta família, `libreyolo predict` só carrega checkpoints
`.pt` por `LibreYOLO()`, então as famílias do `LibreOpenVocab` rodam a partir do
Python. Veja [predição](/docs/predict) para tipos de fonte e streaming.

Toda chamada a `predict()` também roda a torre de texto MobileCLIP-B(LT)
embutida para gerar os embeddings do vocabulário atual; veja Licenciamento para
o que isso acrescenta aos termos.

## Variantes

Três checkpoints, `s`, `m` e `l`. `s` é o tamanho padrão deste tier quando
nenhum é informado. Ao contrário do resto deste tier, o OV-DEIM é um port nativo
em vez de um wrapper de `transformers`: o LibreYOLO incorpora os módulos do
detector sob a mesma licença Apache-2.0 do código upstream e reutiliza o
adaptador de backbone DINOv3 já construído para a família DEIMv2. O backbone do
checkpoint `l` é um fine-tune do DINOv3-S, licenciado à parte sob a DINOv3
License da Meta. Ainda não há números publicados de acurácia ou de latência para
esta família.

Treinamento, validação de dataset e exportação estão todos fora do escopo deste
tier: `train()`, `val()` e `export()` lançam `NotImplementedError` de forma
incondicional. Isto é um wrapper apenas de predição em volta de um checkpoint
publicado.

## Checkpoints

Todo arquivo de pesos publicado desta família.

<checkpoint-table />

## Licenciamento

<provenance-box>

O OV-DEIM sobrepõe três licenças upstream a cada chamada de predição: os pesos
do detector sob a CC BY-NC 4.0 do próprio OV-DEIM, a torre de texto online sob a
Machine Learning Research Model license da Apple (uso apenas para pesquisa) e,
no caso do checkpoint `l`, um fine-tune do backbone DINOv3-S sob a DINOv3
License da Meta. Os textos das três licenças acompanham o repositório de pesos
do LibreYOLO.

</provenance-box>

## Citação

<citation-block />
