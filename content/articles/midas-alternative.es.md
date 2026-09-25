---
title: "Alternativa a MiDaS en 2026: estimación de profundidad con LibreYOLO"
description: "El repositorio oficial de MiDaS está archivado. Ejecuta sus modelos de profundidad Small y DPT-Large con LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "¿Está abandonado MiDaS?"
    a: "El repositorio oficial isl-org/MiDaS se archivó el 25 de agosto de 2025."
  - q: "¿MiDaS devuelve profundidad métrica?"
    a: "No. MiDaS devuelve profundidad inversa relativa, sin unidad métrica ni escala fija entre imágenes."
---

El [repositorio oficial de MiDaS](https://github.com/isl-org/MiDaS) se archivó el 25 de agosto de 2025. LibreYOLO ofrece una forma mantenida de ejecutar sus checkpoints Small y DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Se admiten la predicción, la validación zero-shot y la exportación a resolución fija. El entrenamiento no está disponible. La salida es profundidad inversa relativa: los valores más altos están más cerca, pero no hay una unidad métrica ni una escala fija entre imágenes.

Basta con la instalación básica: `pip install libreyolo`. Continúa con la [documentación de MiDaS](https://www.libreyolo.com/docs/models/midas) o consulta la [alternativa a Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
