---
title: "YOLOX no LibreYOLO: alternativa para executar, treinar e exportar em 2026"
description: "A versão mais recente do YOLOX é de 2022. Execute, treine, valide e exporte YOLOX pela API do LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "O YOLOX foi abandonado?"
    a: "O repositório oficial não foi arquivado, mas sua versão mais recente é o YOLOX 0.3.0, de abril de 2022. É mais preciso dizer que o projeto está parado do que afirmar que foi formalmente abandonado."
  - q: "O LibreYOLO consegue treinar YOLOX?"
    a: "Sim. O LibreYOLO oferece suporte a predição, treinamento, validação e exportação com YOLOX."
---

O YOLOX ainda é um detector útil sob a licença Apache-2.0. Seu [repositório oficial](https://github.com/Megvii-BaseDetection/YOLOX) não foi arquivado, mas a versão mais recente, 0.3.0, é de abril de 2022. O problema de manutenção é real. O modelo continua funcionando.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

O LibreYOLO oferece suporte a seis tamanhos do YOLOX, de Nano a X, com predição, treinamento, validação e exportação em uma única API. Use o projeto original se você precisar do fluxo de configuração exato de `Exp`.

Comece com `pip install libreyolo`. A [documentação do YOLOX](https://www.libreyolo.com/docs/models/yolox) cobre toda a API. Há também uma nota curta sobre como [o YOLO-NAS continua sendo mantido](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
