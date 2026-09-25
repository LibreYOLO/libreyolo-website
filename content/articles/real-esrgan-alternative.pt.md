---
title: "Alternativa ao Real-ESRGAN em 2026: aumente a resolução com LibreYOLO"
description: "Execute o aumento de escala de imagens em 2x e 4x do Real-ESRGAN com o LibreYOLO, incluindo inferência por blocos para imagens grandes."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "O Real-ESRGAN foi abandonado?"
    a: "O repositório não foi arquivado, mas sua versão mais recente no GitHub é a 0.3.0, de setembro de 2022."
  - q: "O LibreYOLO pode treinar o Real-ESRGAN?"
    a: "Não. O LibreYOLO oferece suporte a predição, validação e exportação do Real-ESRGAN, mas não ao treinamento."
---

O [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) não foi formalmente arquivado, mas sua versão mais recente no GitHub, a 0.3.0, é de setembro de 2022. O LibreYOLO executa seus checkpoints de 2x e 4x pela API padrão de predição.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

O LibreYOLO oferece suporte a checkpoints de 2x, 4x e 4x rápido, com predição por blocos, validação com PSNR/SSIM e exportação. O treinamento não foi implementado. A mistura de intensidade de remoção de ruído do projeto original também não faz parte desta implementação.

Use `pip install libreyolo` para experimentar. As opções completas de restauração estão na [documentação do Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Para estimativa de profundidade, veja a [alternativa ao MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
