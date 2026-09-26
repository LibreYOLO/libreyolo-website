---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO no LibreYOLO: detecção de conjunto aberto'
description: >-
  Use o Grounding DINO no LibreYOLO para detectar qualquer objeto descrito por
  texto. Instale o extra openvocab e faça predições com um vocabulário de texto
  livre.
lead: >-
  O Grounding DINO é um detector de objetos de conjunto aberto, desenvolvido
  pela IDEA Research, que pontua uma imagem contra um prompt de texto livre em
  vez de uma lista fixa de classes. O LibreYOLO o encapsula como uma família
  somente de predição no seu tier de detectores de vocabulário aberto.
keywords:
  - Grounding DINO
  - detecção de vocabulário aberto
  - detectar objetos por texto
  - detecção zero-shot
  - detectar objetos sem treinar
  - open-set detection
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Limiar de texto
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtra pelo score da caixa, e text_threshold pelo score de token
        # da frase decodificada. Ambos assumem 0.25 quando não são definidos.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Instalação

O Grounding DINO é carregado pelo tier de detectores de vocabulário aberto do
LibreYOLO, que precisa do extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Esse extra traz junto `transformers` e `timm`, as bibliotecas da Hugging Face
que este tier chama.

## Predição

O Grounding DINO não é um checkpoint que o LibreYOLO carrega por meio de
`LibreYOLO()`. Ele é carregado pela factory irmã `LibreOpenVocab`, que baixa um
snapshot da Hugging Face no primeiro uso e o guarda em cache em `weights/`.

<code-tabs name="predict" />

`set_classes()` define um vocabulário de texto persistente: chame de novo para
substituir a lista, ou não chame para manter os rótulos COCO-80 padrão. O
Grounding DINO decodifica frases livres a partir da própria saída de texto e as
mapeia de volta para esse vocabulário por conta própria, uma correspondência
normalizada exata vence, uma correspondência de token inteiro é aceita, e uma
frase ambígua ou sem correspondência é descartada em vez de adivinhada, então
`school bus` nunca acaba mapeado para `bus` ou `school` sozinhos. Um
vocabulário longo o bastante para ultrapassar o limite de tokens do codificador
de texto é dividido em vários prompts, executado como passes forward separados
e reunido de volta em um único conjunto de detecções limitado por `max_det`.

`iou` é aceito por compatibilidade de API, mas emite um aviso e não faz nada, já
que nada aqui roda non-maximum suppression. `imgsz` e `augment=True` são
rejeitados de saída: o processador do `transformers` é quem cuida do
redimensionamento, e o data augmentation em tempo de teste está fora do escopo
deste tier. `predict()` em uma única imagem retorna um `Results`, não uma lista;
passe um diretório, uma lista de imagens ou `stream=True` com uma fonte de vídeo
para obter vários. Não há caminho de CLI para esta família, `libreyolo predict`
só carrega checkpoints `.pt` por meio de `LibreYOLO()`, então as famílias de
`LibreOpenVocab` rodam a partir do Python. Veja [predição](/docs/predict) para
tipos de fonte e streaming.

## Variantes

Dois checkpoints, `t` e `b`. `t` é o tamanho padrão deste tier quando nenhum é
informado. Ambos espelham o release oficial da IDEA Research por meio do
`GroundingDinoForObjectDetection` do `transformers`, baixado uma vez para um
snapshot da Hugging Face hospedado pelo LibreYOLO que preserva os arquivos
originais. Ainda não há números publicados de acurácia ou de latência para esta
família.

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
