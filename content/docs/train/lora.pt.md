---
title: Fine-tuning com LoRA
seo_title: Fine-tuning com LoRA no LibreYOLO
description: >-
  Faça fine-tuning de um detector transformer com pouca VRAM usando lora=True.
  Quais nove famílias têm suporte, a receita de adaptadores de cada uma e como
  os checkpoints se comportam.
lead: >-
  O LoRA congela as partes pesadas pré-treinadas de um modelo e treina pequenos
  adaptadores de baixo rank ao lado delas, mais as camadas que precisam
  continuar densas. No LibreYOLO toda a interface pública é um booleano.
keywords:
  - fine tuning lora
  - fine tuning eficiente em parâmetros
  - peft
  - dora
  - treinar com pouca vram
  - rf-detr lora
  - d-fine lora
  - fundir adaptadores lora
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: A exportação funde os adaptadores
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Fundir em memória
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Instalação

O LoRA se apoia na dependência opcional `peft`.

<code-tabs name="install" />

Sem ela, `lora=True` levanta um `ImportError` citando esse comando, em vez de
treinar um fine-tuning completo por acidente.

## Como usar

<code-tabs name="train" />

`lora=True` é toda a interface. Rank, alpha, dropout e módulos alvo são fixos por
família para bater com cada referência upstream, e não são botões expostos ao
usuário.

Uma família que não tem suporte a LoRA levanta erro na configuração, em vez de
ignorar a flag:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

A CLI rejeita antes disso, antes de o modelo ser construído, usando a própria
allowlist das mesmas nove famílias.

## Quais famílias

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 e v4, EC e ConvNeXt. A trava é o
atributo `supports_lora` na classe de treinador de cada família, e a CLI carrega
uma allowlist correspondente.

A cobertura por tarefa é mais estreita que a cobertura por família. D-FINE e EC
só têm suporte a detecção, e seus caminhos de segmentação e de pose levantam
erro. O caminho semântico do RF-DETR levanta erro. ConvNeXt é classificação.

Todo o resto levanta erro. Não existe modo parcial nem silencioso.

## O que cada receita faz

As receitas diferem porque as arquiteturas diferem, e uma receita que funciona em
um backbone ViT não tem onde se prender em um backbone convolucional.

O RF-DETR usa DoRA, o LoRA decomposto em pesos, com rank 16 e alpha 16 nas
projeções de atenção `query`, `key` e `value` do backbone DINOv2, seguindo a
referência do RF-DETR. O backbone ViT congela; o projetor, o decodificador e a
cabeça de detecção continuam treinando normalmente.

D-FINE, DEIM e RT-DETR v1, v2 e v4 combinam um backbone convolucional com um
encoder híbrido transformer e um decodificador deformável, então a divisão muda
de lugar. O backbone convolucional congela por inteiro, o que também pula o
backward pass dele. Os blocos transformer congelam seus pesos base e treinam
adaptadores LoRA simples, com os mesmos rank 16 e alpha 16, nas suas camadas
lineares: o feed-forward `linear1` e `linear2`, o gate, e as projeções de atenção
deformável. Todo o resto, a fusão de convoluções do encoder, as projeções de
entrada, as cabeças de predição e os embeddings de query, continua treinando de
forma densa.

Dois detalhes dessa receita são deliberados. A self-attention do decodificador
fica congelada e sem adaptadores, porque o `nn.MultiheadAttention` do PyTorch lê
`out_proj.weight` diretamente e passaria por cima de um adaptador injetado sem
avisar. E é LoRA simples em vez de DoRA, porque várias camadas lineares do
decodificador são inicializadas com zero por design e a normalização de magnitude
do DoRA divide pela norma dos pesos.

O DEIMv2 usa a mesma receita, com suas camadas feed-forward SwiGLU `w12` e `w3`
como alvos. Seus tamanhos S, M, L e X também trazem um backbone ViT DINOv3, onde
a base ViT congela e suas camadas de atenção fundida `qkv` recebem adaptadores,
enquanto a pirâmide de convoluções do Spatial Tuning Adapter continua treinando
como análoga ao projetor. Esses adaptadores de `qkv` entram mesmo quando a config
já vinha com o ViT congelado, já que adaptar um backbone congelado é justamente o
objetivo. Os tamanhos abaixo de S usam um backbone convolucional e ficam com a
receita simples.

O EC é um DETR cujo backbone é um ViT cercado por uma pirâmide de projetores
convolucionais treinável. A base ViT congela e suas camadas `qkv` recebem
adaptadores, os blocos transformer ficam com a receita compartilhada, e o
projetor e as cabeças continuam densos.

Os blocos do ConvNeXt carregam MLPs lineares em channels-last, `fc1` e `fc2`, e
esses recebem adaptadores simples. As convoluções depthwise, as normalizações e
os parâmetros de layer-scale congelam. A cabeça de classificação continua densa
para que contagens personalizadas de classes continuem funcionando.

As cabeças de detecção e de classificação continuam sempre treináveis em todas as
receitas, porque uma contagem personalizada de classes precisa de uma cabeça
treinada do zero.

## Checkpoints e exportação

`best.pt` e `last.pt` guardam os tensores dos adaptadores, então uma execução com
LoRA é retomada ou inspecionada como qualquer outra. Carregar um desses
checkpoints exige o extra `lora` instalado, porque o carregador repete a injeção
dos adaptadores para que as chaves batam.

`export()` funde os adaptadores em pesos densos, então um artefato exportado não
carrega nenhuma dependência de `peft`. A mesma fusão está disponível diretamente
para um modelo em memória.

<code-tabs name="merge" />

Depois de uma fusão a árvore de módulos fica totalmente densa e uma segunda fusão
não faz nada.

## O que economiza, e o que não

O LoRA corta memória de otimizador e de gradientes, e nas famílias que congelam o
backbone por completo ele também pula o backward pass desse backbone.

A memória de ativações não muda. As ativações do forward ainda precisam ser
guardadas para tudo o que continua treinável, e em geral é isso que define o
pico. Para o orçamento de VRAM mais apertado, reduza também `batch` ou `imgsz`.

## Relacionado

- [Congelamento de camadas](/docs/train/layer-freezing) para a outra forma de
  treinar um subconjunto dos pesos, que funciona em todas as famílias e não
  precisa de dependência extra. `freeze` e `lora=True` se combinam: os parâmetros
  dos adaptadores continuam treináveis mesmo quando o grupo de backbone que os
  contém está congelado.
- [Hiperparâmetros](/docs/train/hyperparameters) para `batch`, `imgsz` e o resto
  de `train()`.
