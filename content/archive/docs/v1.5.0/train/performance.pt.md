---
title: Desempenho do treinamento
seo_title: 'Treinar mais rápido: grafos CUDA, AMP, profiler'
description: >-
  Deixe um treinamento mais rápido: capture o passo em grafos CUDA, escolha um
  dtype para o AMP e use o profiler integrado para achar onde o tempo realmente
  vai parar.
lead: >-
  Três alavancas mudam a velocidade de um passo de treinamento: a precisão
  mista, a captura do forward e do backward da rede em grafos CUDA, e o que o
  profiler disser que está de fato segurando o passo.
keywords:
  - grafos cuda treinamento
  - acelerar treinamento yolo
  - treinamento com precisão mista
  - treinar com bfloat16
  - profiler pytorch
  - treinamento limitado pelo dataloader
  - overhead de lançamento de kernels
  - uso da gpu no treinamento
last_verified: 1.5.0
snippets:
  profile:
    - label: Perfilar e continuar treinando
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Perfila uma janela curta de passos reais, imprime um veredicto e
        # depois continua a execução com os hooks removidos.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: Só medir e parar
      language: bash
      code: |
        # Define no_aug_epochs=0 e roda só as épocas necessárias para encher a janela.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Detalhar o resultado
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Meça antes de mudar qualquer coisa

As três alavancas abaixo resolvem problemas diferentes, e aplicar a errada não
muda nada. O profiler diz qual problema você tem.

<code-tabs name="profile" />

`profile=True` mede uma janela de passos reais de treinamento, cinco descartados
e depois vinte medidos por padrão, imprime um relatório, grava seus artefatos e
então segue treinando com os hooks removidos. Não custa nada quando está
desligado, e é ignorado em treinamento distribuído.

O relatório termina em um de quatro veredictos:

| Veredicto | Significado | Alavancas |
|---|---|---|
| `dataloader` | a GPU espera pelos dados de entrada | mais `workers`, `cache="ram"` ou `"disk"`, augmentation mais leve, batch maior |
| `host / launch` | a GPU é alimentada devagar demais, muitos kernels minúsculos | batch maior, grafos CUDA, menos sincronizações com o host por passo |
| `compute` | a GPU está saturada | AMP ou bfloat16, ou aceitar |
| `memory-pressure` | thrash do alocador, VRAM no limite | reduza o batch; os números de utilização aqui não são confiáveis |

O número de utilização é o tempo em que os kernels ficam ocupados dividido pelo
tempo do passo sem sincronização. A janela é dividida de propósito: a primeira metade roda sem
sincronização extra, para que o veredicto reflita a sobreposição real, e só a
segunda metade cerca cada fase com um sync para atribuir o tempo de GPU.
Sincronizar cada fase dá folga aos workers do dataloader e esconde a starvation,
então os números de composição nunca são usados para escolher o veredicto.

Quatro arquivos aparecem no diretório da execução: `timeline.html`, que abre
sozinho no navegador, `profile_trace.json` para o Perfetto ou o Nsight,
`profile_summary.json` e `profile.json`, o autocontido, feito para copiar por aí
e devolver aos subcomandos `libreyolo profile`.

Vale saber duas coisas sobre o `profile run`. Ele define `no_aug_epochs=0`,
porque o profiler mede a época 0 e uma execução curta com o `no_aug_epochs`
padrão perfilaria o dataloader mais leve, sem augmentation, em vez do que o
treinamento de fato usa. E `--repeat N` reporta média e desvio padrão, o que
importa porque um passo limitado por lançamento é ruidoso o bastante para que
uma única execução engane; ele grava diretórios por tentativa, `prof_1`,
`prof_2` e assim por diante, além de um `profile_repeat.json` agregado.

## Precisão mista

`amp=True` é o padrão para a maioria das famílias e roda o forward sob o
autocast do CUDA. `amp_dtype` escolhe entre `float16` e `bfloat16`.

<code-tabs name="amp" />

O float16 precisa de escalonamento dinâmico da loss e recebe um gradient scaler
ativo; a faixa de expoente mais larga do bfloat16 não precisa, então o scaler
dele fica desativado. Quatro famílias vêm com `amp=False`, D-FINE, DEIM, YOLO-NAS
e FOMO, e a configuração do DEIM se propaga para o RT-DETRv4 por herança. O
D-FINE explica o motivo: seu decoder limita as ativações em 65504, o maior valor
finito de float16.

A semântica dos argumentos, incluindo o que um pedido de bfloat16 faz em
hardware sem suporte a bfloat16, está em
[Hiperparâmetros](/docs/train/hyperparameters).

## Grafos CUDA

`cuda_graph=True` captura o forward e o backward de treinamento da rede em um
grafo CUDA, eliminando o overhead de lançamento de kernels a cada passo.

<code-tabs name="graph" />

A flag é sempre segura de passar. Uma família, tarefa ou configuração que não
pode ser capturada registra uma linha de log e treina em eager, sem mudanças.

Só a rede é capturada. A loss fica em eager por design, porque as losses de
detecção selecionam com máscaras booleanas, rodam matching húngaro e ramificam
conforme o resultado da atribuição, e um grafo não consegue gravar nada disso. O
passo do otimizador, o clipping de gradiente, a atualização do EMA e o schedule
de learning rate também ficam em eager.

Isso limita o ganho à fatia do passo que é rede, e essa fatia varia muito.
Medido em uma RTX 5070 Ti a 640 px, batch 8: 84 por cento de um passo do
YOLOv9-t é rede, 44 por cento de um passo do YOLOv7-b, 31 por cento de um passo
do YOLOX-t e 26 por cento de um passo do RTMDet-t. Os dois últimos passam a
maior parte de um passo dentro de seus label assigners, então capturar a rede é
o que menos os ajuda.

### Quanto isso vale

Condições de todos os números abaixo: RTX 5070 Ti, Windows, AMP, um processo por
braço a partir de um estado salvo compartilhado, reproduzindo um batch real para
tirar o dataloader do caminho, o passo mais rápido de 24 depois do warmup.
Detecção a 640 px, classificação a 224 px. O tamanho de batch é por linha.

| Família | Tamanho | Batch | Eager | Com grafo | Speedup |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Esses números isolam o passo de GPU. Um fine-tuning completo também paga pelo
dataloader e pela validação. YOLOv9-t em um conjunto de detecção de 406 imagens,
20 épocas, batch 8, 640 px, 4 workers de dataloader, na mesma máquina: 428.4 s
de wall clock em eager contra 367.7 s com grafo, um ganho de 1.16x, com mAP50-95
de 0.6394 nos dois braços.

Três coisas mexem nesses números. Batches pequenos são limitados por lançamento
e os grandes são limitados por computação, então o RT-DETR-r18 ganha 1.19x com
batch 2 e 1.04x com batch 8. O overhead de lançamento é maior no Windows, e os
ganhos no Linux ficam entre um terço e a metade dos valores da tabela. E uma
execução limitada pelo dataloader não vê mudança nenhuma no wall clock, e é por
isso que o profiler vem primeiro.

A captura entra em ação do mesmo jeito com `amp=False`, mas os kernels fp32
rodam por mais tempo, então o passo fica menos limitado por lançamento e a
maioria das famílias ganha menos. No mesmo hardware, o MobileNetV4-s com batch
16 vai de 2.74x sob AMP para 3.61x em fp32, enquanto o YOLOv9-t com batch 8 vai
de 1.99x para 1.69x e o RT-DETR-r18 com batch 4 vai de 1.12x para 0.99x.

### Onde a captura se aplica

| Tarefa | Famílias |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Todo o resto cai para eager com uma linha de log: outras tarefas nessas
famílias, famílias que não estão na lista, execuções distribuídas e execuções de
destilação. Uma falha de captura em runtime também derruba o resto da execução
para eager, em vez de quebrar.

Nos detectores encoder-decoder, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 e v4, e EC,
só o backbone e o encoder são capturados. O decoder deles lê o ground truth para
montar as queries de contrastive denoising, e o número dessas queries acompanha a
maior contagem de ground truth do batch, então sua contagem de tokens muda de
batch para batch.

### Shapes

Um grafo é válido exatamente para o shape de entrada com que foi capturado. O
trainer conta os shapes de batch e captura assim que um shape se repete três
vezes. Batches em qualquer outro shape rodam em eager: os batches multi-escala e
o último batch parcial de uma época.

Essa é a armadilha das famílias DETR, que redimensionam todo batch por padrão.
Com `multi_scale=True`, uma execução curta pode nunca ver um shape com
frequência suficiente para chegar a capturar. Passe `multi_scale=False` quando o
objetivo for o speedup.

O YOLOX muda o que a região capturada calcula no meio de uma execução, ligando
seu ramo de regressão L1 quando o mosaico se encerra em `no_aug_epochs`. O
trainer invalida a captura nesse ponto e recaptura assim que o novo shape se
estabiliza.

### Numérica e memória

A maioria das famílias reproduz bit a bit a trajetória da loss em eager sob AMP.
FOMO e LingBot-Vision diferem no último bit do float32 por causa de uma ordem de
soma diferente. Os detectores de deformable attention, D-FINE, DEIM, DEIMv2,
RT-DETR, RF-DETR e EC, também não reproduzem suas próprias execuções em eager,
porque esse backward acumula com atômicos e as convoluções TF32 escolhem uma
ordem de redução a cada lançamento; a execução com grafo fica dentro dessa
dispersão. O RTMDet difere em cerca de 3e-4 relativo em dois de 139 gradientes,
porque compartilha convoluções da cabeça entre os níveis da pirâmide e os dois
caminhos de backward somam três contribuições em ordem diferente. O SegFormer
tem stochastic depth dentro da região capturada, então um grafo reproduzido puxa
seu próprio fluxo aleatório e é estatisticamente equivalente ao eager, não
idêntico; o gerenciador registra isso uma vez no momento da captura.

Com `amp=False`, resultado bit a bit idêntico não está disponível para nada
neste hardware, com ou sem captura. Duas execuções eager idênticas do YOLOv9-t
com a mesma semente divergem 36 por cento relativo ao longo de 20 passos, e o
YOLOX-t, 2.6 por cento, porque o cuDNN escolhe um algoritmo não determinístico
de gradiente de pesos para alguns shapes de convolução fp32.

Um grafo capturado fixa buffers estáticos de entrada, saída e workspace, então o
pico de VRAM sobe mais ou menos um conjunto extra de ativações. Nas famílias
acima, a alocação de pico variou entre -5 e +19 por cento. O custo relativo é
maior nos modelos pequenos de classificação, cujas ativações já são pequenas de
início: o ResNet-18 a 224 px, batch 16, foi de 0.48 GB em eager para 0.57 GB com
grafo. Se isso empurrar uma execução além do limite, reduza o batch ou deixe a
flag desligada.

## Relacionados

- [Hiperparâmetros](/docs/train/hyperparameters) para `batch`, `nbs`, `cache` e
  `workers`.
- [Treinamento multi-GPU](/docs/train/multi-gpu), onde nem os grafos CUDA nem o
  profiler estão disponíveis.
- [Grafos CUDA](/docs/reference/cuda-graphs) para a matriz de suporte combinada
  de inferência e treinamento, os pontos de divisão da captura e o contrato
  numérico.
