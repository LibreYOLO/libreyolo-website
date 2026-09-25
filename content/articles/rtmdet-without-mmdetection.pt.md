---
title: "Como usar RTMDet sem mmdetection: detecção de objetos em tempo real"
description: RTMDet é um dos detectores mais rápidos e com melhor acurácia disponíveis. A instalação oficial não funciona com nenhuma versão do PyTorch lançada nos últimos dois anos. Veja a alternativa.
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Por que o mmcv falha ao instalar com versões recentes do PyTorch?"
    a: "A versão mais recente do wheel do mmcv, a 2.2.0, de abril de 2024, só oferece binários pré-compilados até torch 2.4 / CUDA 12.1. Em qualquer versão mais recente do PyTorch, o mmcv precisa compilar suas operações em C++/CUDA a partir do código-fonte. Isso leva de 10 a 30 minutos e exige um toolkit CUDA correspondente, nvcc e um compilador C++ compatível. É daí que surgem erros como a ausência do módulo mmcv._ext."
  - q: "Posso usar RTMDet sem instalar mmdetection?"
    a: "Sim. O LibreYOLO carrega pesos do RTMDet convertidos dos checkpoints originais do OpenMMLab, e a inferência é bit a bit equivalente à do mmdetection com o mesmo checkpoint. Carregue LibreRTMDets.pt pelo nome e o download é feito automaticamente. Os cinco tamanhos, de Tiny a X, rodam a 640 px."
  - q: "RTMDet é gratuito para uso comercial?"
    a: "Sim. MMDetection e RTMDet usam Apache 2.0, os pesos convertidos mantêm os mesmos termos Apache 2.0, e o código do LibreYOLO usa MIT. Não há restrições comerciais."
  - q: "Quando mmdetection ainda é a escolha certa?"
    a: "Se você precisa treinar ou fazer fine-tuning do RTMDet com todo o pipeline de data augmentation do MMDetection, ou já está profundamente integrado ao ecossistema OpenMMLab e o mmcv está funcionando. Para inferência e deploy, o LibreYOLO é o melhor caminho."
---

RTMDet é um detector em tempo real do OpenMMLab que alcança alta acurácia no COCO e continua rápido o suficiente para deploy. A arquitetura é limpa. A instalação, não.

Para executar RTMDet a partir da fonte oficial, é preciso montar uma pilha OpenMMLab de quatro camadas:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

As versões compatíveis têm uma faixa estreita. O mmdet 3.3.0 exige `mmcv < 2.2.0`, mas `mim install mmcv>=2.0.0` instala sem problemas a versão 2.2.0, que depois falha em uma asserção durante a execução. Você precisa fixar a versão manualmente.

Depois vem o problema maior. O wheel mais recente do mmcv é a versão 2.2.0, lançada em abril de 2024 e nunca atualizada desde então. Ela só oferece binários pré-compilados até **torch 2.4 / CUDA 12.1**. Hoje, um `pip install torch` em uma instalação nova instala PyTorch 2.12. Essa diferença significa que o mmcv precisa compilar suas operações em C++/CUDA a partir do código-fonte: de 10 a 30 minutos, um toolkit CUDA correspondente, `nvcc` e um compilador C++ compatível. É daí que vem esta lista de falhas documentadas:

* `ModuleNotFoundError: No module named 'mmcv._ext'`: as operações compiladas não foram construídas ou não são compatíveis,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` em qualquer placa da série RTX 30,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`: a pilha obriga você a voltar ao Python 3.8,

* falha de segmentação ao importar por incompatibilidade da versão do GCC.

O próprio FAQ do OpenMMLab recomenda instalar o mmcv com pip em vez de mim para evitar a incompatibilidade de versões. A página Get Started recomenda usar mim. Os dois documentos são oficiais. Eles se contradizem.

Mesmo depois de colocar a pilha para funcionar, a inferência ainda exige escolher um arquivo de configuração cujo nome codifica a receita de treinamento, além de baixar separadamente um checkpoint com nome contendo um hash e conectá-los por meio de um registry que você precisa aprender primeiro.

O LibreYOLO substitui tudo isso:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

Sem `mim`, sem matriz de versões com quatro pacotes, sem compilar a partir do código-fonte, sem arquivos de configuração, sem procurar o hash do checkpoint e sem fixar Python 3.8. Os pesos são convertidos dos checkpoints originais do OpenMMLab, e a inferência é bit a bit equivalente à do mmdetection com o mesmo checkpoint.

Há cinco tamanhos disponíveis: Tiny, Small, Medium, Large e X. Todos rodam a 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Quando a implementação original ainda é a escolha certa

Se você precisa treinar ou fazer fine-tuning do RTMDet com todo o pipeline de data augmentation do MMDetection, ou se já está profundamente integrado ao ecossistema OpenMMLab e o mmcv está funcionando, continue usando essa opção. O LibreYOLO é o melhor caminho para inferência e deploy.

## Uma observação sobre a licença

MMDetection e RTMDet usam Apache 2.0. O código do LibreYOLO usa MIT. Os pesos mantêm os mesmos termos Apache 2.0 da fonte original. Não há restrições comerciais.

## Experimente

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

O LibreYOLO usa a licença MIT, roda em Linux, Mac e Windows e funciona em GPU, Apple Silicon e CPU comum sem nenhuma alteração no código. Uma API reúne RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, segmentação, pose, profundidade e muito mais.

Dê uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
