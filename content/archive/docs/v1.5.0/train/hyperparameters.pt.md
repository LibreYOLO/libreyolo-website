---
title: Hiperparâmetros
seo_title: Hiperparâmetros de treinamento no LibreYOLO
description: >-
  Os argumentos de train() que importam: epochs, batch, lr0, optimizer, EMA,
  autobatch, acumulação de gradiente e resume, além do porquê de os padrões
  mudarem conforme a família.
lead: >-
  Todo argumento de treinamento é um campo de uma dataclass TrainConfig. A
  classe base define o campo e seu valor padrão; cada família de modelos faz
  subclasse dela e sobrescreve os padrões que sua receita publicada altera.
keywords:
  - argumentos de treinamento yolo
  - learning rate yolo
  - tamanho de batch treinamento
  - autobatch yolo
  - media movel exponencial ema
  - acumulacao de gradiente pytorch
  - retomar treinamento yolo
  - early stopping patience
  - amp bfloat16
  - train config yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Ler os padrões resolvidos de uma família
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # Imprime os padrões de train, val e predict, incluindo as sobrescritas
        de família.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 sonda a memória da GPU e resolve para uma potência de dois
        concreta.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 micro-batches de 16 por passo do otimizador, batch efetivo de 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Carrega o checkpoint da execução interrompida e então pede para
        retomar.

        model = LibreYOLO("runs/train/exp/weights/last.pt")

        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # As chaves do yaml são nomes de campos de TrainConfig. Kwargs
        explícitos vencem.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Definindo argumentos

`train()` recebe argumentos nomeados e a CLI aceita os mesmos nomes na forma
`key=value`.

<code-tabs name="train" />

Os dois caminhos terminam no mesmo lugar. Os kwargs são entregues a
`TrainConfig.from_kwargs()`, que constrói a dataclass de config da família.

## Um erro de digitação não levanta exceção

`from_kwargs()` descarta qualquer chave que não seja um campo da config e emite um
`UserWarning` que a nomeia. O treinamento então começa com o padrão no lugar:

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Nada falha, a execução termina, e o learning rate nunca foi o que quem chamou
pediu. Leia os avisos na primeira época de uma receita nova. A CLI é mais rígida,
porque valida os nomes das flags antes de a config ser construída, então uma flag
de CLI escrita errada é rejeitada de imediato.

## Os padrões são por família

`TrainConfig` define o campo e um padrão base. Cada família faz subclasse dela e
sobrescreve o que sua receita publicada altera, então não existe uma única resposta
correta para "qual é o learning rate padrão".

Os padrões base são `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` e `amp=True`. Três exemplos do quanto uma família se afasta disso:

| Campo | Base | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE e DEIM vêm com `amp=False` porque o decoder do D-FINE limita as ativações
a 65504, o maior valor finito de float16. YOLO-NAS e FOMO também deixam isso
desligado por padrão. A flag `--amp` da CLI tem `True` como padrão para todas as
famílias, então ela conta como fornecida pelo usuário e sobrescreve o padrão da
família; não mexa nela a menos que você queira mesmo mudá-la.

Para ler os padrões reais de uma família em vez de adivinhar:

<code-tabs name="defaults" />

## Tamanho de batch

`batch` é o batch global. Em treinamento multi-GPU cada rank carrega
`batch // world_size`, então o número que você passa é o número de imagens por
passo do otimizador, independentemente de quantas GPUs estejam envolvidas. Veja
[Treinamento multi-GPU](/docs/train/multi-gpu).

`batch=-1` liga o autobatch. O trainer sonda o modelo em modo de treinamento com
um backward pass real em potências de dois, ajusta uma reta à curva de memória e
escolhe a maior potência de dois estritamente abaixo do valor extrapolado que cabe
em 60 por cento da VRAM total.

<code-tabs name="autobatch" />

Sondar em modo de treinamento com um backward pass é justamente o ponto: uma sonda
em modo de inferência não enxerga as ativações retidas nem os tensores de
gradiente, que em uma CNN profunda consomem várias vezes mais memória do que a
inferência. O
RF-DETR baixa a fração alvo para 45 por cento, porque o backward sintético da sonda
ainda subestima o custo do seu critério e das camadas auxiliares do decoder.

O autobatch é um recurso de CUDA. Em CPU ou MPS ele registra uma linha no log e
mantém o batch padrão.

## Acumulação de gradiente

`nbs` define o tamanho de batch nominal, ou efetivo. O trainer acumula
`round(nbs / batch)` micro-batches por passo do otimizador.

<code-tabs name="accumulate" />

Deixado como `None`, o padrão, a acumulação fica desligada e o treinamento não
muda.

## Learning rate e schedule

`lr0` é o learning rate inicial e `optimizer` aceita `sgd`, `adam` e `adamw`.
`momentum` é o momentum do SGD ou o beta1 do Adam, `weight_decay` é o termo L2, e
`nesterov` se aplica ao SGD.

O schedule é moldado por `scheduler`, `warmup_epochs`, `warmup_lr_start` e
`min_lr_ratio`. `no_aug_epochs` define quantas épocas finais rodam sem augmentation
forte, e vários schedules também o usam para moldar sua cauda, então ele não é
puramente um parâmetro de augmentation. O que cada família faz com a metade de
augmentation dele está em [Aumento de dados](/docs/train/augmentations).

Algumas famílias acrescentam seus próprios parâmetros de learning rate.
`backbone_lr_mult` escala o grupo do backbone em relação à cabeça, `clip_max_norm`
define o clipping de gradiente, e o SegFormer usa `head_lr_mult` para rodar sua
cabeça de decodificação a dez vezes a taxa do backbone. Eles ficam na subclasse de
config da família, não na base.

## EMA

`ema=True` mantém uma média móvel exponencial dos pesos ao lado dos pesos
treinados. Vem ligado por padrão em todo lugar, menos no FOMO.

`ema_decay` é o decaimento alvo. O decaimento entra em rampa em vez de começar já
no alvo: o valor efetivo na atualização `n` é `ema_decay * (1 - exp(-n / tau))` com
`tau` valendo 2000 por padrão, então as atualizações iniciais acompanham o modelo
mais de perto e as tardias o suavizam. Os padrões por família vão de `0.997` no
YOLO-NAS pose, passando por `0.9998` no YOLOX, até `0.9999` no YOLOv9 e na linha
DETR.

Os pesos do EMA são o que passa pela validação e o que `best.pt` e `last.pt`
carregam. Os pesos treinados brutos também ficam guardados, sob a chave
`train_model`, para que um resume continue a partir da trajetória treinada e não da
média.

## Precisão

`amp=True` roda o forward pass sob o autocast do CUDA. `amp_dtype` seleciona
`float16` (o padrão) ou `bfloat16`; `fp16` e `bf16` são grafias aceitas.

O float16 precisa de escalonamento dinâmico da loss e recebe um `GradScaler` ativo.
A faixa de expoente mais larga do bfloat16 não precisa, então o scaler dele é
construído mas fica desabilitado, o que mantém o caminho do otimizador idêntico.
Pedir bfloat16 em um dispositivo CUDA sem suporte a bfloat16 levanta exceção na
configuração em vez de degradar em silêncio.

## Saída, checkpoints e parada

As execuções são escritas em `project/name`. `project` tem `runs/train` como padrão
em todo lugar, mas `name` é uma das sobrescritas por família: o padrão base é `exp`,
enquanto o YOLOv9 usa `yolo9_exp` e o D-FINE usa `dfine_exp`. Com
`exist_ok=False`, o padrão, um diretório já existente ganha um sufixo incrementado
em vez de ser sobrescrito.

`save_period` escreve um `weights/epoch_<N>.pt` extra a cada N épocas, além de
`weights/last.pt` depois de cada época e `weights/best.pt` sempre que a métrica
acompanhada melhora. `eval_interval` define com que frequência a validação roda, e
`patience` para a execução após esse número de épocas sem melhora, com `0`
desativando o early stopping.

`cache` acelera as épocas repetidas mantendo as imagens decodificadas na RAM
(`True` ou `"ram"`) ou como arquivos `.npy` ao lado das fontes (`"disk"`). As
leituras em cache são byte a byte idênticas às leituras novas. Com workers no
dataloader, `"disk"` é a mais segura das duas.

## Resume

`resume=True` continua uma execução interrompida. O checkpoint precisa ser
carregado antes, porque o resume o lê a partir do modelo, e não de um argumento
separado.

<code-tabs name="resume" />

O resume restaura os pesos treinados, o estado do otimizador, os pesos do EMA e a
contagem de atualizações, o acompanhamento da melhor métrica, a escala do
`GradScaler`, e os estados aleatórios do PyTorch, do CUDA e do NumPy. Ele começa na
época do checkpoint mais um e avança o schedule até essa posição.

Duas coisas que ele não faz. `resume=True` não pode ser combinado com
`pretrained`, o que levanta exceção. E quando a chave de melhor métrica do
checkpoint difere da da execução atual, o acompanhamento da melhor métrica é zerado
com um aviso, em vez de comparar valores que não significam a mesma coisa.

## Receitas em um arquivo

`cfg=` carrega um mapeamento YAML de nomes de campos de `TrainConfig` e o mescla
abaixo dos argumentos nomeados explícitos, então um kwarg sempre vence o arquivo.

<code-tabs name="cfg" />

`size` e `num_classes` são removidos do arquivo, porque a instância do modelo já os
possui. Não existe flag `--cfg` na CLI; o caminho do arquivo é um argumento de
Python.

## Relacionados

- [Datasets](/docs/train/datasets) para o que `data=` aceita.
- [Aumento de dados](/docs/train/augmentations) para os parâmetros de augmentation
  e quais famílias os respeitam.
- [Congelamento de camadas](/docs/train/layer-freezing) e [LoRA](/docs/train/lora)
  para treinar um subconjunto dos pesos.
- [Validação e métricas](/docs/train/validation) para o que a execução reporta.
