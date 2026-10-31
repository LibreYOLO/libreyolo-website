---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision no LibreYOLO: 7 tarefas, um checkpoint'
description: >-
  Use o SenseNova-Vision no LibreYOLO para detecção, segmentação, panóptica,
  pose, pontos, profundidade e OCR a partir de um único checkpoint generativo
  guiado por prompt.
lead: >-
  O SenseNova-Vision é um modelo multimodal unificado que trata as tarefas de
  visão como geração guiada por prompt sobre um decoder compartilhado: boxes,
  pontos, keypoints e palavras de OCR saem como texto etiquetado, e mapas de
  profundidade, de máscara e panópticos saem como imagens que um decoder
  renderiza. O LibreYOLO o carrega pelo LibreVLM e dá suporte a sete tarefas a
  partir do único checkpoint de 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - modelo multimodal unificado
  - Bagel
  - detecção por prompt
  - percepção densa
  - segmentação por referência
  - referring segmentation
  - segmentação panóptica
  - estimativa de profundidade python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() troca de tarefa no mesmo modelo já carregado.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Segmentação por referência e panóptica
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # A segmentação é por referência: precisa de uma frase alvo, não de uma
        lista de classes.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Sem vocabulário personalizado, a panóptica cai nas categorias

        # panópticas do COCO em que o checkpoint foi ajustado.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Pontos, pose e OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Sem vocabulário definido, a pose cai em "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Instalação

O SenseNova-Vision precisa do próprio extra, que traz junto o `accelerate` para o dispatch de modelo grande de que este checkpoint precisa e, em plataformas que não sejam macOS, o `bitsandbytes` para o carregamento em 4 bits.

```bash
pip install "libreyolo[sensenova]"
```

O checkpoint está espelhado no Hugging Face sob a org do próprio LibreYOLO e baixa automaticamente no primeiro uso; ele é CC BY-NC 4.0, apenas para uso não comercial, e o loader imprime esse aviso antes de todo download automático. Veja Licenciamento abaixo.

## Predição

<code-tabs name="predict" />

Toda predição é uma decodificação por difusão sobre o backbone Bagel-MoT compartilhado, então ele é um modelo de capacidade e não de tempo real: espere uma latência por imagem bem mais alta que a de um detector ou segmentador feito sob medida. `dtype="auto"` (o padrão) carrega bf16 em uma GPU com memória suficiente e cai para a quantização NF4 de 4 bits nos demais casos, o que exige o `bitsandbytes`; passe `dtype="bf16"` para forçar precisão total em uma GPU grande o bastante. `noise_seed=42` na construção fixa a semente do amostrador de difusão para saídas densas reproduzíveis; passe `noise_seed=None` para desativar a semente.

As sete tarefas compartilham um único checkpoint carregado: `set_task()` alterna entre elas sem recarregar. `set_classes()` define o vocabulário ativo; detecção, pontos, pose e panóptica aceitam uma lista de classes, enquanto a segmentação é por referência e precisa exatamente da frase a isolar. Cada tarefa retorna o objeto `Results` padrão com um payload diferente preenchido: `boxes` para detect, `points` para point, `boxes` e `keypoints` para pose, `ocr` para OCR, `depth_map` para depth, `masks` para segment e `panoptic` (com `segments_info`) para panoptic. Veja [predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Checkpoints

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
