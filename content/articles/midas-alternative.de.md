---
title: "MiDaS-Alternative 2026: Tiefenschätzung mit LibreYOLO"
description: "Das offizielle MiDaS-Repository ist archiviert. Führe die Small- und DPT-Large-Tiefenmodelle mit LibreYOLO aus."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "Wird MiDaS nicht mehr weiterentwickelt?"
    a: "Das offizielle Repository isl-org/MiDaS wurde am August 25, 2025 archiviert."
  - q: "Liefert MiDaS metrische Tiefe?"
    a: "Nein. MiDaS liefert relative inverse Tiefe ohne metrische Einheit oder feste Skalierung zwischen Bildern."
---

Das [offizielle MiDaS-Repository](https://github.com/isl-org/MiDaS) wurde am August 25, 2025 archiviert. LibreYOLO bietet eine gepflegte Möglichkeit, die Small- und DPT-Large-Checkpoints auszuführen.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Vorhersage, Zero-Shot-Validierung und Export mit fester Auflösung werden unterstützt. Training wird nicht unterstützt. Die Ausgabe ist eine relative inverse Tiefe: Höhere Werte entsprechen näher gelegenen Bereichen, aber es gibt weder eine metrische Einheit noch eine feste Skalierung zwischen Bildern.

Die Basisinstallation genügt: `pip install libreyolo`. Weiter geht es mit der [MiDaS-Dokumentation](https://www.libreyolo.com/docs/models/midas) oder der [Real-ESRGAN-Alternative](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentation](https://www.libreyolo.com/docs)
