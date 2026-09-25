---
title: "YOLO-NAS ainda é mantido: como usar no LibreYOLO"
description: "Execute, treine, valide e exporte YOLO-NAS no LibreYOLO. A última versão do SuperGradients saiu em abril de 2024. Planejamos treinar novos pesos do zero para eliminar a licença da Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "O YOLO-NAS foi abandonado?"
    a: "A última versão do SuperGradients, a 3.7.1, saiu em abril de 2024. O LibreYOLO oferece suporte a predição, treinamento, validação e exportação do YOLO-NAS no PyTorch atual."
  - q: "O Ultralytics mantém o YOLO-NAS?"
    a: "O Ultralytics oferece suporte a inferência, validação e exportação. Ele não oferece suporte a treinamento."
  - q: "Posso usar comercialmente os pesos pré-treinados do YOLO-NAS?"
    a: "Os pesos pré-treinados da Deci continuam sujeitos aos termos não comerciais da licença original, independentemente da biblioteca que os carrega. Treinar com inicialização aleatória evita esses checkpoints. Também planejamos publicar pesos treinados do zero no COCO."
---

O YOLO-NAS ainda é um detector útil. Seu projeto original, o [SuperGradients](https://github.com/Deci-AI/super-gradients), lançou a versão 3.7.1 em abril de 2024. O LibreYOLO mantém o suporte a predição, treinamento, validação e exportação para ele.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

O LibreYOLO oferece suporte aos tamanhos S, M e L para detecção, além de pose. Os checkpoints são baixados do CDN público da Deci com nomes como `LibreYOLONASs.pt`. O LibreYOLO não hospeda nenhum deles, e os termos não comerciais da Deci continuam valendo. O SuperGradients fixa o PyTorch entre as versões 1.9 e 1.13; o LibreYOLO exige a versão 2.4 ou superior. A exportação para detecção inclui ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch e Core AI. CoreML não é compatível.

Para treinar sem um checkpoint da Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Também planejamos treinar YOLO-NAS do zero com o COCO e publicar esses pesos.

Instale com `pip install libreyolo`. A [documentação do YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) traz os detalhes do modelo. Também temos um artigo mais longo sobre a [alternativa ao SuperGradients](/articles/supergradients-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
