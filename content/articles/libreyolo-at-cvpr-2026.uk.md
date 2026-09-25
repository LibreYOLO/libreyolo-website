---
title: "LibreYOLO на CVPR 2026: запуск YOLOX на edge-пристроях"
description: "На CVPR 2026 у Денвері команда Jabra та IT University of Copenhagen використала LibreYOLOXs як приклад моделі в навчальному посібнику «Edge AI in Action», запустивши її на Hailo-8L і Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "З якою швидкістю LibreYOLOXs працює на edge-обладнанні?"
    a: "У навчальному посібнику CVPR 2026 LibreYOLOXs працювала зі швидкістю 19.0 ms на кадр (52.6 FPS) на Raspberry Pi 5 із Hailo-8L і 6.69 ms на HTP/DSP Qualcomm QCS6490 після квантування INT8."
  - q: "Як розгорнути модель LibreYOLO на edge-прискорювачі?"
    a: "Експортуйте модель у ONNX за допомогою функції експорту LibreYOLO, а потім скомпілюйте її для цільового обладнання: Hailo Dataflow Compiler створює HEF для чипів Hailo, а стек SNPE / QAIRT / AI Hub призначений для Qualcomm Snapdragon. Саме цей повний пайплайн демонстрували в навчальному посібнику CVPR."
---

![CVPR 2026, Денвер, Колорадо, 3-7 червня](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO завітала на CVPR 2026.

CVPR широко визнана найпрестижнішою конференцією з комп'ютерного зору у світі. Цього року вона проходила в Денвері, де команда Jabra та IT University of Copenhagen провела навчальний посібник під назвою «Edge AI in Action: Mastering On-Device Inference». Як приклад моделі для перенесення виявлення об'єктів на edge-чипи команда обрала LibreYOLOXs.

## Що вони створили

У навчальному посібнику розглянули повний edge-пайплайн від початку до кінця. Спершу взяли модель LibreYOLOXs, експортували її в ONNX, а потім скомпілювали цю модель ONNX для двох зовсім різних пристроїв: прискорювача Hailo-8L і Qualcomm Snapdragon.

## Робота в реальному часі на Hailo-8L

Вони скомпілювали ONNX у HEF за допомогою Hailo Dataflow Compiler, а потім запустили модель на Raspberry Pi 5 із Hailo-8L AI HAT.

![LibreYOLOXs працює на Raspberry Pi 5 із Hailo-8L та виявляє людей і сумки на жвавій вулиці](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

На Raspberry Pi модель обробляла кадр за 19.0 ms зі швидкістю 52.6 FPS.

## Робота в реальному часі на Snapdragon

На обладнанні Qualcomm вони квантували LibreYOLOXs до INT8 і запускали її через стек SNPE, QAIRT та AI Hub; бенчмарк виконали на QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs наживо виявляє телевізор, ноутбук, клавіатуру, мишу та мобільний телефон у застосунку EdgeVision AI на телефоні Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

На HTP/DSP затримка становила 6.69 ms.

## Подяка

Велика подяка Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera та Shan Ahmed Shaffi за те, що розповіли про проєкт.

- Навчальний посібник і слайди: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Моделі LibreYOLOXs із квантуванням Qualcomm: [на Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Спробуйте

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO поширюється за ліцензією MIT, працює на Linux, Mac і Windows, а також на GPU, Apple Silicon і звичайному CPU без змін у коді. Один API охоплює YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, сегментацію, оцінювання пози, глибину та інші задачі.

Поставте зірку на GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Документація: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
