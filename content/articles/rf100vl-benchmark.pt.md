---
title: "Benchmark RF100-VL: como os modelos LibreYOLO generalizam?"
description: Resultados do RF100-VL que medem como YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter e RF-DETR generalizam em 100 datasets do mundo real após o fine-tuning.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

O RF100-VL avalia o quanto um detector se adapta além do COCO. Cada modelo passa por fine-tuning separadamente em 100 datasets muito diferentes, incluindo imagens infravermelhas, raios X, microscopia, esportes, imagens do espectro de rádio, objetos minúsculos e videogames. Avaliamos 17 configurações do LibreYOLO: **1700 fine-tunings no total.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Resultados

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

O RF-DETR-L liderou esta comparação com **61.76**. O M veio em seguida, com 61.13, depois o S, com 60.41. Os quatro resultados do RF-DETR são próximos dos [números publicados pela Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), o que é uma validação útil do treinamento do RF-DETR no LibreYOLO.

O tamanho não previu todos os resultados. YOLO-NAS-S e M ficaram praticamente empatados, com 58.00 e 57.99. O EdgeCrafter-M marcou 57.93, acima do S, com 55.99, e do L, com 56.11. Pretendemos investigar essas regressões. Esta varredura não é um experimento controlado de escala: a resolução, os pesos pré-treinados, a precisão e as receitas variam.

O RF-DETR-S com LoRA no backbone marcou 57.19, em comparação com 60.41 no fine-tuning completo. O fine-tuning completo vence em 95 datasets. A mediana registrada é apenas sete minutos mais lenta, enquanto o tempo total somado dos jobs quase não muda.

As linhas do YOLOv9 são anteriores a correções importantes no treinamento. O resultado anômalo do M ajudou a descobrir a ausência do PGI, uma convenção diferente de letterbox e um limite de 100 rótulos no treinamento. Os números arquivados descrevem o código que foi executado. Eles não são um veredito sobre a arquitetura do YOLOv9.

Escolha um modelo abaixo para inspecionar suas pontuações nos 100 datasets.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Metodologia

Para cada dataset, treinamos em `train`, selecionamos o melhor checkpoint de AP50:95 na validação e o avaliamos uma vez em `test`. A pontuação principal é a média não ponderada desses 100 resultados de teste.

| Configuração | Regra da campanha |
|---|---|
| Treinamento | 100 épocas; sem early stopping |
| Batch efetivo | 16 |
| Seed | 0; uma execução concluída por dataset |
| Checkpoint | Melhor AP50:95 de validação usando pesos EMA |
| Pontuação no teste | pycocotools com `maxDets=500` |
| Ajuste | Uma receita fixa por família; sem ajuste por dataset |

Cada família manteve sua própria receita de treinamento:

| Família | Resolução | Precisão | Descrição da data augmentation |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip; augmentations fortes desativadas nas 15 épocas finais |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip; cauda de 15 épocas |
| YOLOX S/M | 640 | FP32 | Mesma receita da família, com learning rate específico por tamanho |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV e flip; mosaic desativado |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; padrões da família LibreYOLO |
| RF-DETR N/S/M/L e S LoRA | 384/512/576/704; LoRA em 512 | BF16 | Multi-scale, crop/resize e flip |

Estes resultados usam uma única seed. Diferenças pequenas não são melhorias comprovadas. O RF-DETR M e L usaram acumulação de gradientes para caber na memória disponível, e alguns datasets densos precisaram de batches físicos menores. As receitas públicas de RF-DETR também começam a partir de checkpoints do COCO, enquanto a Roboflow usou checkpoints privados de Objects365 para sua tabela histórica.

Os tempos de treinamento são registros da campanha, não benchmarks controlados de velocidade. Às vezes, os jobs compartilharam GPUs, foram repetidos ou retomados. Não executamos o protocolo opcional de latência em um único artefato T4. O YOLO-NAS usa os [pesos pré-treinados da Deci com licença separada para uso não comercial](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).

## O que melhorou com essa carga de trabalho

Testes pequenos podem provar que o treinamento começa. Eles não reproduzem semanas de fine-tuning contínuo em muitas arquiteturas e datasets. Nessa escala, caminhos lentos, pressão de memória e problemas de correção ficaram difíceis de ignorar.

- **Carregamento de imagens e validação.** Armazenamos em cache o redimensionamento determinístico antes da augmentação, reutilizamos workers e buffers do validador e, quando havia suporte, geramos gráficos dos forwards de validação. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA graphs no treinamento.** O suporte à captura passou do YOLOv9 e RF-DETR para 24 famílias. Também corrigimos uma condição de corrida no pin-memory do DataLoader e a invalidação do grafo nas transições do treinamento. Um teste do YOLOv9-T caiu de 428.4 para 367.7 segundos com o mesmo AP reportado. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Pontuação COCO.** A avaliação em datasets densos às vezes demorava mais que o treinamento. A integração faster-coco-eval pontuou as predições salvas de todos os 100 splits de teste em 8.4 segundos, em vez de 131.4. Dos 1,400 valores de métricas, 1,381 eram bit a bit idênticos; a maior diferença foi 2.22e-16 e o AP principal não mudou. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR e caminhos de treinamento compartilhados.** Um matching L1 mais rápido, menos sincronizações com o dispositivo, AdamW fundido e memória limitada do matcher reduziram um passo do RF-DETR-S de 266 para 234 ms no teste registrado. A mesma investigação melhorou outros cinco matchers, o logging do YOLOv9 e a reutilização de tensores do EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Atenção.** Direcionamos a atenção compatível para o SDPA do PyTorch, integramos uma implementação CUDA com licença Apache-2.0 e corrigimos a execução com precisão mista. Também escrevemos um kernel Triton de deformable attention dentro do repositório para inferência. Nos testes documentados, ele foi cerca de quatro vezes mais rápido isoladamente e cerca de 7% mais rápido de ponta a ponta para RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Correção do treinamento.** Corrigimos a reconstrução do BatchNorm no YOLOX e restauramos o PGI, os metadados de letterbox, a capacidade de rótulos e o warmup de momentum no YOLOv9. O benchmark forneceu os checkpoints e os padrões de falha que revelaram os dois problemas. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Esses números vêm de testes separados em cargas de trabalho diferentes. Não devem ser multiplicados para formar um único ganho de velocidade. O LibreYOLO agora é mais rápido e confiável para cargas de trabalho reais de treinamento do que era antes desta campanha.

## Harness e artefatos

O [harness público de treinamento](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) distribui datasets entre GPUs, retoma jobs interrompidos, lida com recuperação de OOM e registra as receitas, versões dos datasets, logs, checkpoints e predições.

A [skill run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) fornece a agentes de programação com IA o protocolo e as instruções de operação do Vast.ai. O [arquivo de resultados](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) contém todas as execuções publicadas.

## Agradecimentos à Roboflow

Joseph Nelson ofereceu suporte de GPU depois que escrevi que queria fazer um benchmark além do COCO, mas não tinha como pagar pelas execuções. Matvei Popov compartilhou as configurações de referência, explicou as definições de avaliação e acompanhou a chegada dos resultados.

Esse apoio nos permitiu validar o treinamento do LibreYOLO em 17 configurações, publicar evidências que ajudam as pessoas a escolher um modelo, disponibilizar o harness e levar a biblioteca ao limite até que seus pontos fracos ficassem claros.

Obrigado, Joseph, Matvei e equipe da Roboflow, pelo poder computacional, pelo tempo e pela orientação.
