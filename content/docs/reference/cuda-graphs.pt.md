---
title: Grafos CUDA
seo_title: Matriz de suporte a grafos CUDA no LibreYOLO
description: >-
  Quais famílias capturam o forward na predição e o forward e o backward no
  treinamento, o que é garantido sobre os números, onde uma captura é dividida e
  por que uma família não suportada levanta erro.
lead: >-
  Um grafo CUDA registra uma execução de uma sequência fixa de kernels e a
  reproduz como um único lançamento. LibreYOLO captura a inferência em 39
  famílias verificadas e o treinamento em 24, sempre por família, sempre depois
  de uma checagem de paridade bit a bit e nunca como fallback silencioso.
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - grafos cuda pytorch
  - acelerar inferência yolo cuda graph
  - treinar yolo com cuda graph
  - capture_error_mode thread_local
last_verified: 1.5.0
verification: >-
  Lista de famílias de inferência derivada da matriz CAPTURABLE em
  tests/e2e/test_cuda_graph_families.py na v1.5.0. Lista de famílias de
  treinamento, classes de paridade e tempos vindos de
  docs/training_cuda_graphs.md. A API e o NotImplementedError, de
  BaseModel._require_cuda_graph_support, cuda_graph_scope e capture_graph em
  libreyolo/models/base/model.py, com a variável de classe SUPPORTS_CUDA_GRAPH.
  As divisões por costura, lidas dos overrides de _get_graph_runner nas famílias
  depth_anything3, birefnet, ppocr, sam e sensenova e em
  libreyolo/models/base/detr_cuda_graph.py. capture_error_mode, de
  libreyolo/models/base/cuda_graph.py e libreyolo/training/cuda_graph.py. O
  fallback de treinamento, de libreyolo/training/trainer.py e a flag
  --cuda-graph, de libreyolo/cli/commands/train.py.
meta:
  - label: Famílias de inferência
    value: '39'
  - label: Famílias de treinamento
    value: '24'
  - label: Flag de inferência
    value: predict(cuda_graph=True)
    mono: true
  - label: Flag de treinamento
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Predição
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # True captura no primeiro uso de cada shape de entrada.
        # "auto" espera uma shape se repetir antes de pagar o custo da captura.
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Treinamento
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Treinar pela CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## O que é capturado

Um grafo registra uma sequência fixa de kernels e os endereços de memória que
eles leem e escrevem. Ele não registra valores, shapes nem fluxo de controle. A
reprodução é um único lançamento em vez de centenas, e é por isso que o ganho é
maior em redes pequenas com batches pequenos, onde uma etapa é dominada pelo
overhead de lançamento e não pela aritmética.

Os dois pontos de entrada capturam quantidades diferentes de trabalho.

| | Dentro do grafo | Eager |
|---|---|---|
| Inferência | O forward da rede, `model._forward(x)` | Pré-processamento, NMS, todo o pós-processamento |
| Treinamento | O forward e o backward da rede | Loss, passo do otimizador, clipping de gradiente, EMA, agendamento do learning rate |

Nem o NMS nem a loss de detecção são candidatos. Os dois selecionam com máscaras
booleanas, rodam matching húngaro ou um assigner e ramificam com base no
resultado, que é exatamente o que um grafo não consegue registrar. Mantê-los de
fora é o que torna a captura segura, em vez de uma limitação a contornar.

<code-tabs name="usage" />

`cuda_graph` aceita três valores na predição. `False` é o padrão. `True` captura
na primeira vez que cada shape de entrada aparece. `"auto"` espera uma shape se
repetir, então trabalhos pontuais e com shapes variáveis nunca pagam por uma
captura que não vão reutilizar. `capture_graph(imgsz=None, batch=1, dtype=None)`
tira esse custo da primeira requisição, `graph_info()` informa os grafos
capturados e as contagens de replay, e `release_graphs()` os libera.

No treinamento a flag é um booleano simples, `--cuda-graph` na CLI. Veja
[desempenho de predição](/docs/predict/performance) e
[desempenho de treinamento](/docs/train/performance) para os controles ao redor.

## Suporte na inferência

O suporte é por família, declarado pela variável de classe
`SUPPORTS_CUDA_GRAPH`, e uma família só é marcada depois de capturar e
reproduzir de forma bit a bit idêntica contra duas entradas de teste vindas de
distribuições diferentes. Essa matriz de paridade compartilhada cobre 39
famílias em nove tarefas.

| Tarefa | Famílias |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Várias famílias aparecem em mais de uma tarefa, então a matriz tem mais linhas
do que famílias distintas. Outras três famílias capturam por caminhos de código
específicos, com testes dedicados próprios, em vez de pela matriz
compartilhada, e não fazem parte das 39: PP-OCR, SAM e SenseNova.

A verificação é bit a bit, não aproximada. Uma versão anterior do protocolo
julgava a paridade por magnitude relativa e rebaixou por engano três famílias
saudáveis, YOLOX, EfficientNetV2 e YOLOv7, cuja diferença entre eager e grafo
fica em torno de 1e-7 e ainda assim é bit a bit idêntica na entrada de teste que
importa.

## Suporte no treinamento

A captura no treinamento passou de duas famílias para 24 nesta versão, em cinco
tarefas.

| Tarefa | Famílias |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Todo o resto treina em modo eager: outras tarefas nessas mesmas famílias,
famílias fora da lista, execuções distribuídas e execuções de destilação. A
captura também é pulada enquanto uma shape ainda é nova, já que o caminho de
treinamento espera uma shape de entrada se repetir três vezes antes de capturar,
o que significa que `multi_scale=True` pode nunca capturar.

## Duas respostas diferentes para uma família não suportada

O caminho de inferência levanta erro. `predict(cuda_graph=True)` em uma família
que não optou por entrar levanta `NotImplementedError` nomeando a família, em
vez de rodar em modo eager e deixar você acreditar que ganhou uma aceleração que
não ganhou. A razão é que uma captura ruim não falha alto: a reprodução de um
forward que faz algo não capturável devolve números errados em silêncio, então o
suporte precisa ser uma afirmação explícita por família, e não uma tentativa com
fallback.

O caminho de treinamento registra em log. `train(cuda_graph=True)` é sempre
seguro de passar, e uma família, tarefa ou configuração que não pode ser
capturada escreve uma linha e treina em modo eager, sem mudanças. Uma captura
que falha no meio de uma execução também derruba o resto da execução para eager,
em vez de abortá-la. A assimetria é deliberada: a predição é uma chamada que
você conserta no ponto da chamada, enquanto uma execução de treinamento não
deveria morrer na sexta hora por causa de uma otimização opcional.

## Divisão por costura

Algumas famílias não podem ser capturadas inteiras porque um estágio realmente
faz algo que um grafo não consegue registrar. Em vez de descartar a família, a
captura é dividida em uma costura verificada: a parte capturável é reproduzida,
o resto roda em modo eager, e a saída combinada é a mesma de rodar tudo em
eager.

| Família | Capturado | Eager, e por quê |
|---|---|---|
| Depth Anything 3 | A rede | A etapa de céu, que é trabalho visível ao host depois do forward |
| BiRefNet | O encoder, `forward_enc` | O decoder, cujo `deform_conv2d` é reproduzido com um resultado diferente sob captura |
| PP-OCR | O estágio de detecção, `forward_det` | O reconhecimento, porque a largura dos recortes varia por linha |
| SAM | O encoder de imagem | O caminho de prompt, que roda muitas vezes por encode |
| SenseNova | A torre de visão | A geração autorregressiva, com um cache KV que cresce a cada passo |
| Detectores encoder-decoder | Backbone e encoder | Decoder e critério húngaro |

A divisão do BiRefNet vale ler duas vezes: o `deform_conv2d` se comportando mal
sob captura se reproduz em uma chamada isolada, fora de qualquer modelo.
Substituí-lo por um equivalente em PyTorch puro foi rejeitado porque isso teria
deslocado também as predições em eager, e os números do eager são o contrato.

O caso encoder-decoder cobre D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4 e EC. O decoder deles constrói queries de contrastive denoising a
partir do ground truth, e o número dessas queries vem da maior contagem de
ground truth do batch, então a contagem de tokens do decoder muda de batch para
batch. Essa é a única coisa que um grafo não tolera. Backbone mais encoder é
mais ou menos de um quinto a um quarto de uma etapa nessas famílias, e é por
isso que elas ficam no fim da tabela de aceleração.

O PP-OCR captura um grafo por shape de entrada da detecção, limitado pelo teto
de cache do runner, e devolve o resultado eager quando nenhum escopo de captura
está ativo.

## Números

A maioria das famílias é bit a bit idêntica, e onde não é, a razão é nomeada em
vez de deixada no ar. No passo zero do treinamento a loss é bit a bit idêntica
nas 24 famílias e nenhum buffer de BatchNorm difere; a comparação de gradientes
é o que separa as categorias.

| Classe | Famílias | Significado |
|---|---|---|
| Exata | A maioria das 24 | Todo gradiente bit a bit idêntico |
| 1 ULP | fomo, lingbotvision | O último bit do float32, cerca de 1e-7 relativo, por uma ordem de soma diferente |
| Ruído do eager | A linhagem DETR | O grafo difere do eager não mais do que duas execuções eager diferem entre si |
| Arredondamento de ponto flutuante | rtmdet | 137 de 139 gradientes bit a bit idênticos, dois diferem em cerca de 3e-4 |
| Fluxo de RNG próprio | segformer | O stochastic depth fica dentro da região capturada |

A classe do ruído do eager é a que mais importa ler direito. Nessas famílias,
duas execuções eager com a mesma seed já discordam, então bit a bit idêntico não
é uma barra que a execução em grafo não alcançou; é uma barra que ninguém
alcança. Isso vale de forma mais ampla com `amp=False`, onde um não determinismo
relativo medido de 3.2e-7 em um gradiente de peso fp32 se acumula: duas
execuções eager de YOLOv9-t com a mesma seed divergem em 36 por cento ao longo
de 20 passos, e desligar o TF32 não resolve.

## Pin memory

A captura roda com `capture_error_mode="thread_local"`. No modo `"global"`
padrão do PyTorch, uma thread de pin memory do DataLoader preparando o próximo
batch chama `cudaHostAlloc`, que ao mesmo tempo invalida a captura em andamento
e é envenenada por ela, então a execução morre na próxima busca de batch com um
erro levantado de dentro da thread de pin memory. Essa combinação foi observada
duas vezes em uma campanha de treinamento real antes de ser diagnosticada.

O modo thread-local restringe apenas a thread que captura. A thread de pin nunca
toca no stream de captura, então nada do que ela faz pertence ao grafo em
primeiro lugar. O treinamento vai além e substitui temporariamente uma subclasse
de `torch.cuda.CUDAGraph` que força o modo, porque `make_graphed_callables` não
expõe nenhum argumento para isso, sob um lock para que duas capturas
concorrentes não deixem a substituição instalada.

## O que isso vale

Medido em uma RTX 5070 Ti com AMP, um processo por braço, reproduzindo um batch
real para tirar o dataloader do caminho, o mais rápido de 24 passos depois do
warmup. Detecção a 640 px, classificação a 224 px.

| Família | Batch | Aceleração |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Todo o resto | varia | 1.04x a 1.26x |

Uma execução inteira ganha menos, porque um grafo não acelera o dataloader nem a
validação. Um fine-tuning de YOLOv9-t de 20 épocas em 406 imagens foi de 428.4 s
para 367.7 s, um ganho ponta a ponta de 1.16x, com um mAP50-95 idêntico de
0.6394 nos dois braços e losses por época idênticas.

O teto é definido por quanto de uma etapa é rede. No mesmo hardware, a 640 px e
batch 8, isso é 84 por cento no YOLOv9-t, mas só 26 por cento no RTMDet-t, que
passa a maior parte de uma etapa no seu assigner de labels. O overhead de
lançamento é maior no Windows, então os ganhos no Linux ficam em cerca de um
terço a metade desta tabela, e uma execução limitada pelo dataloader não vê
mudança nenhuma no tempo de relógio. O pico de memória varia entre 5 por cento
menor e 19 por cento maior.

## Ressalvas

Um grafo registra endereços, não valores, então qualquer coisa que realoque
parâmetros o descarta. Trocar de dispositivo por `predict(device=...)`,
quantizar e desquantizar invalidam os grafos capturados.

O tamanho de batch importa mais do que a família: o RT-DETR-r18 ganha 1.19x com
batch 2 e 1.04x com batch 8, porque um batch grande é limitado por computação e
tem menos overhead de lançamento a remover.

A suíte de paridade de inferência rodou sem o pacote opcional `kernels`
instalado, então a segurança da captura com os kernels compilados do Hub ativos
não é coberta por ela. Defina `LIBREYOLO_HUB_KERNELS=0` para tirá-los do caminho
enquanto isola um problema de captura. Veja [kernels](/docs/reference/kernels).
