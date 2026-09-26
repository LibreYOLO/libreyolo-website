---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS no LibreYOLO: análise de imagem any-to-any'
description: >-
  Use o LibreMODUS no LibreYOLO para profundidade, normais, bordas e detecção, e
  para compô-las com any2any(). Somente inferência; os pesos são carregados do
  EPFL-VILAB.
lead: >-
  O LibreMODUS é uma integração somente de inferência do checkpoint MODUS
  14B-A7B, um modelo any-to-any que transforma uma entrada derivada de imagem em
  outra: RGB na entrada, profundidade na saída; profundidade na entrada, normais
  na saída; qualquer uma delas mais uma frase, caixas na saída. O LibreYOLO
  suporta quatro tarefas pela API padrão de predição e um conjunto mais amplo
  por meio de any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - estimativa de profundidade
  - normais de superfície
  - detecção de bordas
  - detecção por frase de texto
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Sem vocabulário personalizado, detect decodifica os tokens de label
        # COCO do checkpoint em ids de classe COCO-80 contíguos.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding por frase
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() muda a detecção para grounding por frase: cada frase
        # roda de forma independente e retorna pelo mesmo contrato Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # De uma a três entradas derivadas de imagem (rgb, depth, normal,
        # canny/edge), mais texto auxiliar opcional, compostas para um alvo.
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # O grounding via any2any() precisa de uma entrada de texto com a frase.
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Instalação

O LibreMODUS precisa do próprio extra, que traz o `accelerate` para o dispatch de modelo grande de que esse checkpoint precisa.

```bash
pip install "libreyolo[modus]"
```

O LibreYOLO não redistribui nem espelha os pesos do MODUS. Por padrão, carregar um modelo `LibreMODUS` baixa os arquivos necessários diretamente de `EPFL-VILAB/MODUS` em uma revisão fixada do Hugging Face, e um download novo sempre exige a conta autenticada do próprio usuário no Hugging Face, mesmo que o gate de hospedagem do upstream esteja temporariamente aberto. Revise e aceite os termos do upstream e depois faça a autenticação:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Para evitar qualquer requisição de rede, aponte para um snapshot que você já tenha:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Esse diretório precisa conter `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` e `merges.txt`. Veja Licenciamento abaixo para saber o que os termos do checkpoint permitem.

## Predição

<code-tabs name="predict" />

A API padrão de tarefas cobre quatro tarefas, cada uma mapeada para um alvo do MODUS: `depth` para profundidade relativa (`result.depth_map`), `normal` para normais de superfície (`result.normal_map`), `edge` para bordas no estilo Canny (`result.edges`) e `detect` para caixas COCO-80 (`result.boxes`), a menos que `set_classes()` a mude para grounding por frase. `set_task()` alterna entre elas no mesmo modelo já carregado. A receita publicada usa dez passos de amostragem de fluxo com guidance de texto 4.0 e guidance de imagem 2.0; sobrescreva esses valores com `inference_steps=`, `inference_cfg=` e `inference_image_cfg=` na construção.

`any2any()` alcança a superfície pública de análise mais ampla: de uma a três entradas derivadas de imagem (`rgb`, `depth`, `normal`, `canny`/`edge`), mais texto auxiliar opcional, compostas em direção a qualquer um destes alvos: profundidade, normais, bordas, bordas derivadas do SAM, detecção COCO ou grounding por frase. Todas as entradas derivadas de imagem precisam descrever o mesmo canvas alinhado; o LibreMODUS rejeita larguras e alturas incompatíveis em vez de redimensioná-las de forma independente. `chain=(...)` gera alvos intermediários e os realimenta no mesmo contexto, dentro do orçamento de três condições com que o checkpoint foi treinado. `verify=N` (N >= 2) gera N candidatos e mantém o que obtém a maior pontuação em uma verificação restrita de autoconsistência, exposta como `result.verification_score`.

`dtype="bf16"` (o padrão) corresponde à precisão do checkpoint publicado; `dtype="fp8"` armazena os pesos lineares elegíveis do tronco do decoder em E4M3 com uma escala por canal de saída, converte uma única vez para um cache local em `~/.cache/libreyolo/modus/fp8` e desquantiza para o dtype de entrada a cada multiplicação de matrizes, ou seja, ele troca memória em vez de trocar acurácia no nível das ativações.

`train()`, `val()` e `export()` todos lançam exceção: o LibreMODUS é somente inferência, não há validação em dataset e não existe caminho de exportação para ONNX, TensorRT ou TFLite. `predict()` em batch e o data augmentation em tempo de teste também não são suportados; cada chamada trata uma imagem.

## Licenciamento

<provenance-box>

O LibreYOLO não hospeda nem espelha o checkpoint do MODUS em lugar nenhum, incluindo a própria org no Hugging Face: carregá-lo sempre puxa a revisão fixada diretamente de EPFL-VILAB/MODUS, ou lê um snapshot já em disco no `checkpoint_path`.

</provenance-box>

## Citação

<citation-block />
