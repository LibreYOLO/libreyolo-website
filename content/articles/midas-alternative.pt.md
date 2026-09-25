---
title: "MiDaS Alternative em 2026: estimativa de profundidade com LibreYOLO"
description: "O repositório oficial do MiDaS foi arquivado. Execute os modelos de profundidade Small e DPT-Large pelo LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "O MiDaS foi abandonado?"
    a: "O repositório oficial isl-org/MiDaS foi arquivado em 25 de agosto de 2025."
  - q: "O MiDaS retorna profundidade métrica?"
    a: "Não. O MiDaS retorna profundidade inversa relativa, sem unidade métrica nem escala fixa entre imagens."
---

O [repositório oficial do MiDaS](https://github.com/isl-org/MiDaS) foi arquivado em 25 de agosto de 2025. O LibreYOLO oferece uma forma mantida de executar seus checkpoints Small e DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Há suporte para predição, validação zero-shot e exportação com resolução fixa. Treinamento não é compatível. A saída é uma profundidade inversa relativa: valores mais altos indicam objetos mais próximos, mas não há unidade métrica nem escala fixa entre imagens.

A instalação básica é suficiente: `pip install libreyolo`. Continue com a [documentação do MiDaS](https://www.libreyolo.com/docs/models/midas) ou veja a [alternativa ao Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
