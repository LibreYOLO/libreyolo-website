---
title: "Альтернатива Real-ESRGAN в 2026 году: апскейл изображений в LibreYOLO"
description: "Запускайте апскейл изображений в 2x и 4x через Real-ESRGAN в LibreYOLO, в том числе потайловый инференс для изображений большого размера."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN забросили?"
    a: "Репозиторий не архивирован, но последний релиз на GitHub — версия 0.3.0 от сентября 2022 года."
  - q: "Можно ли обучать Real-ESRGAN в LibreYOLO?"
    a: "Нет. LibreYOLO поддерживает предсказание, валидацию и экспорт Real-ESRGAN, но не обучение."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) официально не архивирован, но последний релиз на GitHub — версия 0.3.0 — вышел в сентябре 2022 года. LibreYOLO запускает его чекпойнты 2x и 4x через стандартный API предсказания.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO поддерживает чекпойнты 2x, 4x и fast 4x, потайловый инференс, валидацию с PSNR/SSIM и экспорт. Обучение не реализовано. В этом порте также нет доступного в upstream смешивания с настройкой силы шумоподавления.

Установите пакет командой `pip install libreyolo`. Все параметры восстановления описаны в [документации Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Подробнее об оценке глубины читайте в статье [альтернатива MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация](https://www.libreyolo.com/docs)
