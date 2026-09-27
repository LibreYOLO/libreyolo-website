---
title: Atualizando para a 1.5.0
seo_title: Atualizar o LibreYOLO da 1.4.0 para a 1.5.0
description: >-
  As quatro mudanças de código que a 1.5.0 exige, as três mudanças que alteram
  métricas e os ajustes menores de comportamento que vale conhecer antes de
  comparar execuções.
lead: >-
  Nada foi removido da API pública de modelos: toda classe e função que
  funcionava na 1.4.0 continua importando. Quatro argumentos mudaram de forma, e
  três padrões alteram números com os quais você talvez esteja comparando.
keywords:
  - atualizar libreyolo
  - migração libreyolo 1.5.0
  - allow_experimental removido
  - mudanças incompatíveis libreyolo
  - yolox bn eps
  - faster-coco-eval padrão
last_verified: 1.5.0
meta:
  - label: Aplica-se a
    value: 1.4.0 para 1.5.0
  - label: Mudanças de código necessárias
    value: 'Quatro, todas pontuais'
  - label: Resultados que mudam
    value: 'Backend COCO, eps do BN do YOLOX, multi-escala do D-FINE'
  - label: Remoções da API pública
    value: Nenhuma
source_hash: ab38d8ef7b53f596
---

Esta página é sobre atualizar o próprio LibreYOLO. Se o que você procura é como
carregar um checkpoint de um projeto upstream, isso é
[importar pesos existentes](/docs/migrate), um assunto diferente.

A entrada completa da versão está no [changelog](/docs/changelog). O que vem a
seguir é só a parte que exige algo de você.

## Mudanças de código que você precisa fazer

### `allow_experimental=True` não existe mais

A trava de confirmação sumiu, junto com o mecanismo
`ddp_aware(experimental_key=...)` por trás dela. O treinamento e a exportação de
EC, RTMDet, PicoDet e FOMO antes exigiam o argumento, então qualquer script que
treine uma dessas famílias é afetado.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: exclua o argumento
model.train(data="data.yaml", epochs=100)
```

Não há camada de compatibilidade. Uma chamada que ainda passa o argumento levanta
`TypeError`. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` foi removido junto. O hook
`get_download_notice()` sobrevive, e continua sendo sobrescrito por MiDaS,
SegFormer e YOLO9-P2.

Os níveis de suporte continuam publicados, só não são mais um argumento: veja
[níveis de estabilidade](/docs/reference/stability-tiers).

### O tier de exportação `"experimental"` não existe mais

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Código que ramifica pela string do tier deve ler `"available"` onde lia
`"experimental"`. O `BaseExporter` não emite mais um `RuntimeWarning` para esses
formatos. O estado de cada formato está listado na
[matriz de exportação](/docs/reference/export-matrix).

### `pretrained=False` com `resume` agora é rejeitado

Antes a combinação seguia adiante de forma incoerente. Agora ela levanta:

```
ValueError: pretrained=False cannot be combined with resume.
```

Escolha um. `pretrained=False` parte de uma inicialização nova com seed, que na
1.5.0 funciona para todas as famílias treináveis em vez de três delas, e `resume`
continua uma execução interrompida a partir do checkpoint dela. Os dois estão
documentados em [treinamento](/docs/train).

### O `--imgsz` da CLI é uma string, não um int

Mais restrito do que parece. Os dois casos abaixo não são afetados:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # continua ok
```

```python
model.predict("img.jpg", imgsz=640)   # continua ok
```

Só precisa mudar o código que chama as funções de comando da [CLI](/docs/cli)
diretamente do Python, porque `predict`, `train` e `val` ampliaram `--imgsz` de
`int` para `str` para que ele aceite tamanhos retangulares:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, e "480x640" agora também funciona
```

O padrão de `train` agora é a string `"640"`. `export --imgsz` já era uma string,
e `profile` não mudou.

## Números que mudam

Três mudanças alteram métricas nas configurações padrão. Se você acompanha
resultados entre versões, leia isto antes de comparar uma execução da 1.5.0 com
uma da 1.4.0.

### faster-coco-eval é o backend padrão de métricas COCO

`val()` e a validação de treinamento a cada época agora calculam as métricas COCO
com o backend C++ faster-coco-eval em vez de pycocotools.

A troca foi decidida com base em paridade medida em todos os 100 splits de teste
do RF100-VL: 1381 de 1400 valores de métrica idênticos bit a bit, desvio máximo
de 2.22e-16, deltas das métricas principais exatamente 0, com 15.6x mais
velocidade no geral e 56x em datasets com muitas detecções. Seus números não
deveriam mudar. Ainda assim, eles são produzidos por uma implementação diferente,
e é por isso que isto está na lista.

pycocotools continua sendo o fallback automático quando faster-coco-eval não está
instalado. Para forçá-lo:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` faz o mesmo globalmente. O backend efetivamente
usado é registrado em nível INFO, exposto como `model.last_eval_backend` depois
de `val()`, e incluído como `eval_backend` no payload JSON da
[CLI](/docs/cli/val). Instale o caminho rápido com
`pip install libreyolo[fast-eval]`.

### Checkpoints do YOLOX treinados antes da 1.5.0 precisam de um override de eps

Esta é a armadilha da versão. Leia se você já fez fine-tuning do
[YOLOX](/docs/models/yolox).

O YOLOX especifica BatchNorm com `eps=1e-3` e `momentum=0.03`. Até a 1.5.0 esses
valores eram aplicados como um conserto posterior que não sobrevivia à
reconstrução por contagem de classes que o `train()` faz quando o `nc` do seu
dataset difere do `nc` do checkpoint. Um fine-tuning assim treinava e reportava a
validação durante o treinamento com o `eps=1e-5` padrão do torch, e depois era
recarregado para inferência com `1e-3`: os mesmos tensores sob normalizações
diferentes.

Os tamanhos com convolução regular quase não mudam. O `n` depthwise muda muito,
porque o `running_var` por canal dele é pequeno o bastante para o eps dominar. No
`ball` do RF100-VL, o mesmo checkpoint nano marca **0.566** de mAP50-95 avaliado
com o eps de treinamento e **0.151** depois de um recarregamento padrão.

Um checkpoint treinado antes da 1.5.0 carrega a semântica de eps=1e-5. Para
reportar números fiéis a ele, ou você avalia com o eps do BN sobrescrito para
1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

ou incorpora `sqrt((var + 1e-3) / (var + 1e-5))` aos pesos do BN uma vez e salva
o resultado. Checkpoints treinados na 1.5.0 ou depois não precisam de nenhum dos
dois.

### O treinamento multi-escala do D-FINE usa a receita por tamanho do upstream

`base_size_repeat` era fixado em 3 para todos os tamanhos. Agora ele é resolvido
por tamanho como o upstream especifica: **n** treina em tamanho fixo com
multi-escala desligado, **s** 20, **m** 6, **l** 4, **x** 3. Antes só o x batia,
então n, s, m e l veem uma distribuição de escalas diferente e convergem para
métricas diferentes.

Para restaurar o comportamento antigo, defina o valor explicitamente:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

O DEIM continua usando o 3 fixo. Os detalhes da família estão em
[D-FINE](/docs/models/d-fine).

## Vale saber, sem ação necessária

- **Os resultados com `imgsz` retangular mudaram porque antes estavam errados.**
  As coordenadas das caixas, o redimensionamento de máscaras do RTMDet, o
  reescalonamento do YOLO-NAS e o escalonamento do ground truth no validador
  agora usam altura e largura por eixo em vez de um único escalar. Com `imgsz`
  quadrado nada muda, bit a bit. Inferência ou validação retangular rodada na
  1.4.0 saía com escala errada. O YOLO-NAS agora rejeita `imgsz` retangular de
  cara, em vez de produzir silenciosamente uma saída errada.
- **Os dicionários de métricas ganharam chaves.** `max_det`, `ar_max_det` e
  `AR_max_det` do avaliador COCO, e `metrics/loss` mais `metrics/loss/ce` do
  FOMO. Os valores nos padrões não mudaram, mas qualquer coisa que itere sobre as
  chaves de métricas, incluindo [loggers](/docs/train/loggers) personalizados e
  cabeçalhos de CSV, vê colunas novas.
- **Execuções do YOLO9 com seed que disparam uma reconstrução da cabeça** partem
  de uma inicialização diferente, porque a seed agora é aplicada antes da
  reconstrução, e não depois. Um fine-tuning com seed feito na 1.4.0 para uma
  contagem de classes diferente não é reproduzível bit a bit na 1.5.0.
- **`libreyolo[hub-kernels]` em CUDA agora realmente ativa o kernel nativo de
  MS-deform-attn.** A 1.4.0 o escondia atrás de uma condição que o RF-DETR nunca
  satisfazia, então o kernel nunca rodava. As predições podem variar dentro da
  tolerância de ponto flutuante no RF-DETR e nas outras famílias com atenção
  deformável. Instalações padrão não são afetadas, e `LIBREYOLO_HUB_KERNELS=0`
  desativa o kernel.
- **`libreyolo predict` descarta opções não suportadas em vez de levantar erro.**
  A CLI filtra os kwargs pela assinatura de `__call__` do modelo, então uma opção
  que a família não aceita é ignorada em vez de levantar `TypeError`. Um erro de
  digitação no nome de uma flag agora passa despercebido.
- **Fontes ao vivo mudam o formato da saída JSON.** Webcams, streams RTSP e
  captura de tela habilitam o streaming implicitamente, o que emite um registro
  por quadro em vez de um por chamada. Essas
  [fontes](/docs/predict/sources) são novas na 1.5.0, então nenhum script da
  1.4.0 é afetado.
- **Reexportar `rfdetr-pose` ou `yolonas-pose` para ONNX gera nomes de saída
  diferentes.** A 1.4.0 interpretava as cabeças de pose multi-tensor deles como
  segmentação por causa de uma heurística de contagem de saídas. Arquivos `.onnx`
  que já estão no disco não são tocados.
- **Em uma instalação sem torch**, os resultados guardam arrays numpy em vez de
  `torch.Tensor`, então `.boxes.data` retorna um tipo diferente e o desempate do
  NMS pode divergir do torchvision. Com o torch instalado, o comportamento é
  idêntico byte a byte. Veja
  [instalação leve](/docs/lightweight-install).
- **Os objetos de configuração validam mais na construção.** `TrainConfig` ganhou
  um `__post_init__` onde não tinha nenhum, então uma configuração que já era
  inválida agora levanta erro na hora em vez de falhar no meio de uma execução. A
  serialização de `ValidationConfig` ganhou uma chave `edge_thresholds`, o que
  quebra um round-trip estrito `ValidationConfig(**dump)` a partir de um dump da
  1.4.0.
- **Os nomes de arquivo de pesos das famílias com sufixo de tarefa são resolvidos
  de outro jeito.** `segformer-b0` agora resolve para `LibreSegformerb0-sem.pt`.
  Isso corrige os 404 do download automático e quebra qualquer script que tenha
  fixado o nome antigo sem sufixo.
- **O marcador do pytest `experimental_backend` agora é `extended_backend`.** Só
  importa se você roda a suíte de testes com `-m`.

## Checkpoints e datasets

Checkpoints escritos pela 1.4.0 carregam sem mudança. O
[schema](/docs/reference/checkpoint-schema) ganhou `imgsz_h` e `imgsz_w` para
modelos retangulares, e continua escrevendo o escalar `imgsz = max(h, w)` para
leitores mais antigos. As exportações [ExecuTorch](/docs/export/executorch) e
[MNN](/docs/export/mnn) agora exigem um arquivo auxiliar, `<program>.pte.json` e
`<model>.mnn.json` respectivamente, e as exportações de HRNet carregam
`pose_input: "person_crop"`. Os formatos de dataset não mudaram.
