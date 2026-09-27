---
title: Treinamento multi-GPU
seo_title: Treinamento multi-GPU no LibreYOLO
description: >-
  Treine em várias GPUs com device="0,1". Como a biblioteca lança os workers de
  DDP, por que batch é o batch global, quando definir sync_bn e o caminho do
  torchrun.
lead: >-
  O treinamento multi-GPU no LibreYOLO é o DistributedDataParallel do PyTorch:
  um processo por GPU, cada um com uma réplica completa do modelo e uma fatia de
  cada batch, e a média dos gradientes é feita entre os ranks a cada passo.
keywords:
  - treinamento ddp pytorch
  - treinamento multi gpu
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - tamanho de batch global
  - backend nccl gloo
  - multi gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # O guard __main__ é obrigatório: cada worker lançado reimporta este
        # módulo e, sem o guard, ele relançaria o treinamento recursivamente.
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # batch global: 16 imagens por GPU em duas GPUs
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Execução
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Sondado uma vez na GPU 0 e escalado para um múltiplo do world size.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Rodar em duas GPUs

Passe uma lista de dispositivos. Nada mais muda.

<code-tabs name="train" />

Quando recebe mais de um dispositivo e não há ambiente de torchrun, o `train()` do
modelo salva os pesos em um arquivo temporário, resolve o autobatch se ele foi
pedido e lança um processo worker por GPU com `torch.multiprocessing.spawn`. Cada
worker reimporta a classe do modelo, reconstrói o modelo a partir dos pesos
salvos e roda o caminho normal de dispositivo único, porque de dentro de um
worker lançado as variáveis de ambiente do torchrun estão definidas. Quando a
execução termina, o melhor checkpoint do rank 0 é carregado de volta na instância
do modelo de quem fez a chamada.

`device` aceita `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` e
`"auto"`. Só uma lista com mais de um índice CUDA dispara o spawn.

## O guard `__main__` é obrigatório

Os workers lançados reimportam o módulo de onde vieram. Sem um guard
`if __name__ == "__main__":`, essa importação executa de novo a chamada de
treinamento e cada worker lança os próprios workers. A biblioteca detecta o caso
e levanta um erro em vez de deixar a recursão acontecer:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Tudo o que cruza para dentro de um worker passa por pickle, então `callbacks=`
precisa ser picklable. Uma classe em nível de módulo funciona; um closure ou uma
lambda não, e o erro diz isso e aponta os loggers embutidos como alternativa.

## batch é o batch global

`batch` é o número de imagens por passo do otimizador somando todas as GPUs. O
dataloader de cada rank é construído com `batch // world_size` e um
`DistributedSampler`, então `batch=32` em duas GPUs significa 16 imagens por GPU,
não 32.

Um batch que não é divisível pelo world size levanta um erro em vez de treinar
silenciosamente com um tamanho diferente:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

A média dos gradientes é feita pelo próprio DDP, então a loss é repassada sem
escala. Multiplicá-la pelo world size além disso inflaria o learning rate efetivo
por um fator próximo ao número de GPUs.

## Autobatch sob DDP

`batch=-1` funciona e devolve um batch global divisível pelo world size.

<code-tabs name="autobatch" />

No caminho do spawn a sondagem roda no processo pai, no primeiro dispositivo,
antes de qualquer worker existir, então cada worker recebe um inteiro concreto e
nenhuma coordenação entre processos é necessária. Sob torchrun, o rank 0 sonda e
transmite o resultado como um único tensor long.

A sondagem mede a capacidade de uma GPU e multiplica pelo world size. Quando
`nbs` está definido, o batch global é limitado a `nbs` e arredondado para baixo
até um múltiplo do world size, de modo que adicionar GPUs reduz o número de
passos de acumulação em vez de encolher o batch por GPU. A mecânica da sondagem
em si está em [Hiperparâmetros](/docs/train/hyperparameters).

## SyncBatchNorm

Sob DDP as camadas de BatchNorm de cada rank enxergam apenas a própria fatia. Com
`batch // world_size` essa fatia pode ficar pequena o bastante para as
estatísticas acumuladas degradarem o modelo convergido em relação a uma execução
em uma GPU só.

`sync_bn=True` converte cada BatchNorm em SyncBatchNorm para que as estatísticas
sejam calculadas sobre o batch global. A conversão só acontece quando o modo
distribuído está ativo, então uma execução em uma GPU só não é afetada pela flag
de um jeito nem de outro.

Ela já vem ligada por padrão nas famílias convolucionais carregadas de
BatchNorm: YOLOX, YOLOv7, YOLOv9 e suas variantes, YOLO-NAS, PicoDet, RTMDet e
FOMO. Todas as outras famílias deixam a flag desligada por padrão. Quando um
modelo contém BatchNorm, `sync_bn` está desligado e o batch por rank fica abaixo
de 16, o trainer emite um aviso.

<code-tabs name="syncbn" />

Não existe flag de CLI para `sync_bn`. É um argumento de Python.

## Lançando com torchrun

O torchrun também funciona, e é a escolha certa quando um escalonador de cluster
já cuida do lançamento dos processos. Escreva o script para um único dispositivo
e deixe o torchrun definir o ambiente de rank.

<code-tabs name="torchrun" />

Não combine os dois. Com o ambiente do torchrun presente, `device="0,1"` não faz
spawn; o trainer assume `cuda:LOCAL_RANK` e o torchrun cuida da contagem de
processos.

## Comportamento dos ranks

O rank 0 é dono de todo efeito colateral. Ele resolve o diretório da execução e
transmite o nome resolvido para que todos os ranks concordem, escreve checkpoints
e artefatos e dispara os callbacks e loggers do usuário. Os outros ranks treinam
e contribuem com gradientes.

Cada rank semeia o RNG do dataloader e do data augmentation de forma diferente, a
partir do `seed` configurado, para que os ranks não sorteiem as mesmas
augmentações.

## Plataforma e backend

O backend é escolhido automaticamente: NCCL quando CUDA e NCCL estão ambos
disponíveis, Gloo caso contrário. O NCCL não é compilado no Windows, então
execuções no Windows usam Gloo sem nenhuma configuração. O grupo de processos é
inicializado com um timeout de três horas.

## O que não roda sob DDP

- Captura de CUDA graph. `cuda_graph=True` registra uma linha e treina em modo
  eager. Veja [Desempenho de treinamento](/docs/train/performance).
- O profiler de treinamento. `profile=True` é ignorado com um aviso.

Nem toda família suporta o spawn automático. Vinte e quatro suportam, cobrindo as
famílias de detecção, classificação, semântica e restauração que treinam. Uma
família sem esse suporte, ao receber um dispositivo multi-GPU, levanta um erro que
nomeia a API do modelo e o comando do torchrun em vez de treinar silenciosamente
em uma GPU só.

## Relacionados

- [Hiperparâmetros](/docs/train/hyperparameters) para `batch`, `nbs` e retomada.
- [Loggers de experimentos](/docs/train/loggers) para a restrição de
  picklabilidade dos callbacks.
- [GPUs na nuvem](/docs/train/cloud-gpus) para alugar uma máquina multi-GPU.
