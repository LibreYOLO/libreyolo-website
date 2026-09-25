---
title: "Alternativa ao MMYOLO em 2026: use RTMDet com LibreYOLO"
description: "A versão mais recente do MMYOLO é de 2023. Execute detectores com suporte, como o RTMDet, pela API direta do LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "O MMYOLO foi abandonado?"
    a: "O MMYOLO não foi arquivado oficialmente, mas sua versão mais recente no GitHub é a 0.6.0, de agosto de 2023."
  - q: "O LibreYOLO consegue carregar qualquer checkpoint do MMYOLO?"
    a: "Não. O LibreYOLO oferece suporte a famílias de modelos específicas e não garante o carregamento direto de checkpoints arbitrários do MMYOLO."
---

O [MMYOLO](https://github.com/open-mmlab/mmyolo) não foi arquivado, mas sua versão mais recente, a 0.6.0, é de agosto de 2023. Se você só precisa de um detector com suporte, como o RTMDet, talvez não precise da pilha completa do OpenMMLab.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

O LibreYOLO não é um substituto completo do MMYOLO. Ele não reproduz todas as configurações, receitas ou checkpoints. Mas oferece a famílias com suporte, como RTMDet e YOLOX, uma API direta para predição, treinamento, validação e exportação.

Comece com `pip install libreyolo`. A [documentação do RTMDet](https://www.libreyolo.com/docs/models/rtmdet) lista as operações com suporte. O [guia rápido do RTMDet](/articles/rtmdet-without-mmdetection) explica a migração.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
