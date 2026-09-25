---
title: "Альтернатива Real-ESRGAN у 2026 році: збільшення зображень у LibreYOLO"
description: "Запускайте збільшення зображень у 2x і 4x за допомогою Real-ESRGAN через LibreYOLO, зокрема тайловий інференс для великих зображень."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN покинули?"
    a: "Репозиторій не архівовано, але останній реліз на GitHub, версія 0.3.0, датується вереснем 2022 року."
  - q: "Чи може LibreYOLO навчати Real-ESRGAN?"
    a: "Ні. LibreYOLO підтримує передбачення, валідацію та експорт Real-ESRGAN, але не навчання."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) офіційно не архівовано, але останній реліз на GitHub, версія 0.3.0, вийшов у вересні 2022 року. LibreYOLO запускає контрольні точки 2x і 4x через стандартний API передбачення.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO підтримує контрольні точки 2x, 4x і fast 4x, тайлове передбачення, валідацію PSNR/SSIM та експорт. Навчання не реалізовано. Змішування відновленого зображення з оригіналом із регулюванням сили шумозаглушення, як у upstream-проєкті, також не підтримується в цьому порті.

Щоб спробувати, встановіть пакет командою `pip install libreyolo`. Повний перелік параметрів відновлення наведено в [документації Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Докладніше дивіться в статті про [альтернативу MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документація](https://www.libreyolo.com/docs)
