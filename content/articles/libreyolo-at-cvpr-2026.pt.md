---
title: "LibreYOLO no CVPR 2026: detecção de objetos com Edge AI"
description: "No CVPR 2026, em Denver, uma equipe da Jabra e da IT University of Copenhagen usou o LibreYOLOXs como modelo de exemplo no tutorial \"Edge AI in Action\", executando-o em um Hailo-8L e em Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "Qual é a velocidade do LibreYOLOXs em hardware de borda?"
    a: "No tutorial do CVPR 2026, o LibreYOLOXs rodou a 19.0 ms por frame (52.6 FPS) em um Raspberry Pi 5 com Hailo-8L e a 6.69 ms no HTP/DSP de um Qualcomm QCS6490, após a quantização INT8."
  - q: "Como faço deploy de um modelo LibreYOLO em um acelerador de borda?"
    a: "Exporte para ONNX com a chamada de exportação do LibreYOLO e, em seguida, compile para o destino: o Hailo Dataflow Compiler gera um HEF para chips Hailo, e a stack SNPE / QAIRT / AI Hub atende ao Qualcomm Snapdragon. Esse é o pipeline completo que o tutorial do CVPR percorreu."
---

![CVPR 2026, Denver, Colorado, 3 a 7 de junho](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

O LibreYOLO apareceu no CVPR 2026.

O CVPR é amplamente reconhecido como a conferência de visão computacional mais prestigiada do mundo. Neste ano, o evento aconteceu em Denver, e uma equipe da Jabra e da IT University of Copenhagen apresentou ali um tutorial chamado "Edge AI in Action: Mastering On-Device Inference". Eles escolheram o LibreYOLOXs como modelo de exemplo para levar a detecção de objetos a chips de borda.

## O que eles criaram

O tutorial percorreu todo o pipeline de borda, de ponta a ponta. Comece com um modelo LibreYOLOXs. Exporte-o para ONNX. Em seguida, compile esse ONNX para dois hardwares bem diferentes: um acelerador Hailo-8L e o Qualcomm Snapdragon.

## Tempo real em um Hailo-8L

Eles compilaram o ONNX para HEF com o Hailo Dataflow Compiler e, em seguida, executaram o modelo em um Raspberry Pi 5 com o AI HAT Hailo-8L.

![LibreYOLOXs rodando em um Raspberry Pi 5 com Hailo-8L, detectando pessoas e bolsas em uma rua movimentada](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Ele rodou a 19.0 ms por frame, 52.6 FPS, em um Raspberry Pi.

## Tempo real em Snapdragon

No lado da Qualcomm, eles quantizaram o LibreYOLOXs para INT8 e o executaram com a stack SNPE, QAIRT e AI Hub, com benchmark em um QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs detectando ao vivo uma TV, um laptop, um teclado, um mouse e um celular no app EdgeVision AI em um telefone Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

O resultado foi 6.69 ms no HTP/DSP.

## Agradecimentos

Muito obrigado a Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera e Shan Ahmed Shaffi por apresentarem o projeto.

- Tutorial e slides: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Modelos LibreYOLOXs quantizados para Qualcomm: [no Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Experimente

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # depois compile para o seu dispositivo de borda
```

O LibreYOLO usa a licença MIT, roda em Linux, Mac e Windows e funciona em GPU, Apple Silicon e CPU comum sem nenhuma alteração no código. Uma API cobre YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentação, pose, profundidade e muito mais.

Dê uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
