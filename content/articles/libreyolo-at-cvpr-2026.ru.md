---
title: "LibreYOLO на CVPR 2026: запуск детекции объектов на edge-устройствах"
description: "На CVPR 2026 в Денвере команда Jabra и IT University of Copenhagen использовала LibreYOLOXs как пример модели на занятии «Edge AI in Action», запустив её на Hailo-8L и Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "Как быстро LibreYOLOXs работает на edge-оборудовании?"
    a: "На учебном занятии CVPR 2026 LibreYOLOXs обрабатывала кадр за 19.0 ms (52.6 FPS) на Raspberry Pi 5 с Hailo-8L и за 6.69 ms на HTP/DSP Qualcomm QCS6490 после квантизации INT8."
  - q: "Как развернуть модель LibreYOLO на edge-ускорителе?"
    a: "Экспортируйте модель в ONNX с помощью функции экспорта LibreYOLO, затем скомпилируйте её для целевого оборудования: Hailo Dataflow Compiler создаёт файл HEF для чипов Hailo, а стек SNPE / QAIRT / AI Hub предназначен для Qualcomm Snapdragon. Именно этот пайплайн показали на занятии CVPR от начала до конца."
---

![CVPR 2026, Денвер, Колорадо, 3–7 июня](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO побывала на CVPR 2026.

CVPR широко признана самой престижной конференцией по компьютерному зрению в мире. В этом году она проходила в Денвере, где команда Jabra и IT University of Copenhagen провела учебное занятие «Edge AI in Action: Mastering On-Device Inference». В качестве примера модели для переноса детекции объектов на edge-чипы команда выбрала LibreYOLOXs.

## Что они собрали

На занятии показали полный edge-пайплайн от начала до конца. Сначала берут модель LibreYOLOXs. Экспортируют её в ONNX. Затем компилируют ONNX для двух совершенно разных типов оборудования: ускорителя Hailo-8L и Qualcomm Snapdragon.

## Работа в реальном времени на Hailo-8L

Они скомпилировали ONNX в HEF с помощью Hailo Dataflow Compiler, а затем запустили модель на Raspberry Pi 5 с Hailo-8L AI HAT.

![LibreYOLOXs работает на Raspberry Pi 5 с Hailo-8L и обнаруживает людей и сумки на оживлённой улице](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

На Raspberry Pi обработка занимала 19.0 ms на кадр, что соответствует 52.6 FPS.

## Работа в реальном времени на Snapdragon

Для Qualcomm они квантизировали LibreYOLOXs до INT8 и запустили её через стек SNPE, QAIRT и AI Hub. Бенчмарк провели на QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs в реальном времени обнаруживает телевизор, ноутбук, клавиатуру, мышь и мобильный телефон в приложении EdgeVision AI на телефоне Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Задержка на HTP/DSP составила 6.69 ms.

## Благодарности

Большое спасибо Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera и Shan Ahmed Shaffi за то, что рассказали о проекте.

- Учебное занятие и слайды: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Квантизированные для Qualcomm модели LibreYOLOXs: [на Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Попробуйте

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO распространяется под MIT, работает в Linux, Mac и Windows, а также на GPU, Apple Silicon и обычном CPU без изменений кода. Один API охватывает YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, сегментацию, оценку позы, оценку глубины и другие задачи.

Поставьте звезду на GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Документация: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
