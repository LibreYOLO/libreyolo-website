---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 no LibreYOLO: detecção de objetos zero-shot'
description: >-
  Use o OWLv2 no LibreYOLO para detectar qualquer objeto descrito por texto.
  Instale o extra openvocab e faça predições com um vocabulário de texto livre.
lead: >-
  O OWLv2 é um detector de objetos de vocabulário aberto, desenvolvido pela
  Google Research, que pontua regiões da imagem contra embeddings de texto
  vindos de um codificador no estilo CLIP. O LibreYOLO o encapsula como uma
  família somente de predição no seu tier de detectores de vocabulário aberto.
keywords:
  - OWLv2
  - OWL-ViT
  - detecção de vocabulário aberto
  - detecção zero-shot
  - detectar objetos por texto
  - detectar objetos sem treinar
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vocabulário padrão
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # Não chamar set_classes() mantém o vocabulário COCO-80 padrão do tier.
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Instalação

O OWLv2 é carregado pelo tier de detectores de vocabulário aberto do LibreYOLO,
que precisa do extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Esse extra traz junto `transformers` e `timm`, as bibliotecas da Hugging Face
que este tier chama.

## Predição

O OWLv2 não é um checkpoint que o LibreYOLO carrega por meio de `LibreYOLO()`.
Ele é carregado pela factory irmã `LibreOpenVocab`, que baixa um snapshot da
Hugging Face no primeiro uso e o guarda em cache em `weights/`.

<code-tabs name="predict" />

`set_classes()` define um vocabulário de texto persistente: chame de novo para
substituir a lista, ou não chame para manter os rótulos COCO-80 padrão. Cada
label é envolvida em um template de prompt fixo antes de chegar à torre de
texto, do mesmo jeito que o `Owlv2ForObjectDetection` do `transformers` foi
treinado.

O OWLv2 não tem limiar de token de texto: só `conf` filtra as detecções, e
passar `text_threshold` levanta erro. `iou` é aceito por compatibilidade de API,
mas emite um aviso e não faz nada, já que nada aqui roda non-maximum
suppression. `imgsz` e `augment=True` são rejeitados de saída: o processador do
`transformers` é quem cuida do redimensionamento, e o data augmentation em tempo
de teste está fora do escopo deste tier. `predict()` em uma única imagem retorna
um `Results`, não uma lista; passe um diretório, uma lista de imagens ou
`stream=True` com uma fonte de vídeo para obter vários. Não há caminho de CLI
para esta família, `libreyolo predict` só carrega checkpoints `.pt` por meio de
`LibreYOLO()`, então as famílias de `LibreOpenVocab` rodam a partir do Python.
Veja [predição](/docs/predict) para tipos de fonte e streaming.

## Variantes

Dois checkpoints, `b16` (base, patch size 16) e `l14` (large, patch size 14).
`b16` é o tamanho padrão deste tier quando nenhum é informado. Ambos espelham o
release oficial da Google Research por meio do `Owlv2ForObjectDetection` do
`transformers`, baixado uma vez para um snapshot da Hugging Face hospedado pelo
LibreYOLO que preserva os arquivos originais. Ainda não há números publicados de
acurácia ou de latência para esta família.

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
