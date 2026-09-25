---
title: "Alternativa ao Facebook DETR em 2026: execute com LibreYOLO"
description: "O repositório original do Facebook Research DETR foi arquivado. Execute quatro variantes oficiais do DETR com o LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "O Facebook DETR foi abandonado?"
    a: "O repositório original facebookresearch/detr foi arquivado em 12 de março de 2024."
  - q: "O LibreYOLO consegue treinar o DETR original?"
    a: "Não. O LibreYOLO oferece suporte a predição, validação e exportação do DETR, mas não à receita de treinamento de 500 épocas."
---

O [repositório original do Facebook Research DETR](https://github.com/facebookresearch/detr) foi arquivado em 12 de março de 2024. O LibreYOLO mantém o detector original utilizável pela mesma API das outras famílias de modelos.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Há quatro variantes com ResNet-50 e ResNet-101 disponíveis para predição, validação e exportação para ONNX e TorchScript. O treinamento não está implementado. O LibreYOLO também usa uma entrada fixa de 800 por 800, em vez de reproduzir o pipeline de avaliação original, que preserva a proporção da imagem.

Experimente depois de executar `pip install libreyolo`. A [documentação do DETR](https://www.libreyolo.com/docs/models/detr) lista os quatro checkpoints. Para uma comparação com CNN, veja a [alternativa ao EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
