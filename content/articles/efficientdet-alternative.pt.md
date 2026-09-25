---
title: "Alternativa ao EfficientDet em 2026: execute D0-D4 com LibreYOLO"
description: "Faça predição, validação e exportação do EfficientDet D0-D4 pela API de detecção de objetos do LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "O EfficientDet foi abandonado?"
    a: "O repositório PyTorch de rwightman não foi arquivado formalmente. É mais correto dizer que está pouco ativo do que abandonado."
  - q: "O LibreYOLO pode treinar o EfficientDet?"
    a: "Não. O LibreYOLO oferece suporte à inferência, validação e exportação do EfficientDet, mas não ao treinamento."
---

O conhecido [repositório PyTorch do EfficientDet](https://github.com/rwightman/efficientdet-pytorch) não foi arquivado, então seria incorreto dizer que foi abandonado. Ele está pouco ativo. O LibreYOLO oferece outro caminho para fazer deploy.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

O LibreYOLO oferece suporte à predição e validação do EfficientDet D0 ao D4. As exportações para ONNX, TorchScript, OpenVINO e TensorRT têm cobertura de paridade. O treinamento não foi implementado, e cada tamanho mantém sua resolução de entrada nativa, que é fixa.

Instale com `pip install libreyolo`. Consulte a [documentação do EfficientDet](https://www.libreyolo.com/docs/models/efficientdet) para ver cada tamanho ou compare-o com a [alternativa ao Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
