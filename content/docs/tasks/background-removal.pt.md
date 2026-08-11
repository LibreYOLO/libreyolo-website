---
title: Remoção de fundo
seo_title: Remoção de fundo no LibreYOLO
description: >-
  Recorte um sujeito do fundo no LibreYOLO. Preveja um matte alfa suave, escreva
  um PNG transparente e valide com MAE e S-measure.
lead: >-
  A remoção de fundo separa um sujeito de tudo o que está atrás dele. O
  LibreYOLO a expõe como a tarefa matte, que retorna um valor alfa suave por
  pixel em vez de uma máscara binária de primeiro plano.
keywords:
  - remover fundo de imagem python
  - modelo alpha matting
  - segmentação dicotômica de imagens
  - recorte png com fundo transparente
  - matte alfa suave
last_verified: 1.5.0
snippets:
  predict:
    - label: Prever um matte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 em [0, 1]
    - label: Escrever um PNG transparente
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() compõe a imagem original com o matte como canal alfa.
        result.save("subject.png")

        rgba = result.cutout()   # o mesmo array (H, W, 4) uint8 em memória
        print(rgba.shape)
    - label: Compor sobre um novo fundo
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # branco

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Validar e ler as chaves das métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Um diretório contendo images/ e um diretório de mattes funciona no
        # lugar de um YAML de dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # quanto menor, melhor
        print(metrics["metrics/Smeasure"])   # fitness, quanto maior, melhor
  export:
    - label: Exportação
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Executar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Definição

A tarefa `matte` prevê um valor alfa por pixel a partir de uma única imagem RGB:
`1` é primeiro plano total e `0` é fundo total. O valor é contínuo em vez de
binário, e é justamente esse o ponto da tarefa. Uma máscara binária está a um
limiar de distância, em 0.5, enquanto o matte suave carrega também a cobertura
parcial em cabelo, pelo e bordas com motion blur que uma máscara binária
descarta.

Uma predição preenche `result.matte`, um payload `Matte` que guarda um array
float32 `(H, W)` em `[0, 1]` no canvas da imagem original, acessível como NumPy
através de `.array`. `result.cutout()` compõe a imagem original com esse alfa em
um array RGBA `(H, W, 4)` uint8, e `result.save(path)` escreve a mesma coisa em
um PNG de fundo transparente. `result.boxes` fica vazio, então `conf`, `iou` e
`max_det` não têm efeito.

## Modelos

Duas famílias atendem `matte`, e elas compartilham o mesmo forward path.

[BiRefNet](/docs/models/birefnet) é a rede de referência bilateral em torno da
qual a tarefa foi construída, publicada aqui como um checkpoint do nível Swin-L.

[FeyNobg](/docs/models/feynobg) é a variante mais profunda da Feyn Inc.: a
arquitetura do BiRefNet com o terceiro estágio Swin ampliado de 18 para 24
blocos e depois retreinada. O LibreYOLO reaproveita o forward path, o
pré-processamento e a saída de logit único do BiRefNet para ela, então predição,
validação e o tratamento de checkpoints se comportam de forma idêntica; os pesos
e a identidade da família são próprios do FeyNobg.

As duas têm licenças de pesos diferentes. Ambas estão indicadas nas páginas dos
modelos, e a licença do repositório do checkpoint específico no Hugging Face é a
que vale.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

Ambas as famílias rodam em um canvas nativo fixo de 1024x1024 e redimensionam o
matte de volta para a imagem original. Outra resolução não é suportada, porque as
tabelas de posição relativa do backbone Swin estão presas a esse tamanho, e um
descompasso as interpola mal em vez de levantar um erro. `Results.save()` é
definido apenas para resultados de matte e precisa da imagem original, que ele
recarrega de `Results.path` a menos que você passe uma. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Formato do dataset

A validação de matte emparelha cada imagem RGB com um matte alfa de ground truth
de canal único e de mesmo nome base, onde 0 é fundo e 255 é primeiro plano.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Basta passar essa raiz como `data=`: o diretório de mattes é detectado
automaticamente entre `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` e `alpha/`.
A alternativa é um YAML de dataset, com `path` mais `val_images` e `val_mattes`
nomeando diretórios relativos a ele:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` e `names` são marcadores de posição do esquema; um modelo de matte retorna
`Results.matte`, não detecções. Os valores do matte são lidos como alfa em
`[0, 1]` dividindo por 255, e um matte cuja forma difira do canvas de predição é
redimensionado bilinearmente para coincidir. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

Nenhuma das duas famílias de matte tem implementação de treinamento: `train()`
levanta `NotImplementedError` em ambas, e o suporte a matte cobre apenas
predição, validação e exportação. Cada página de modelo nomeia o projeto upstream
que publica o código de treinamento e o script de conversão que traz um
checkpoint de volta.

## Validação

`val()` executa o próprio `predict` do modelo, então a validação usa o
pré-processamento exato da família, e ambas as métricas são calculadas no canvas
da imagem original.

<code-tabs name="val" />

`metrics/MAE` é o erro absoluto médio em relação ao alfa de ground truth, em
`[0, 1]`, e quanto menor, melhor. `metrics/Smeasure` é a S-measure de Fan et al.
(ICCV 2017), uma similaridade estrutural que valoriza acertar a forma do sujeito
e seus vazios, algo que uma média por pixel sozinha deixa passar; quanto maior,
melhor. A S-measure também é o `fitness`, o número que a seleção do melhor
checkpoint lê. Nenhuma das métricas depende da resolução.

## Exportação

Um modelo de matte exportado é carregado de volta por `LibreYOLO()` pelo sufixo
do seu arquivo, então o artefato se comporta como um checkpoint e retorna o mesmo
`Results`.

<code-tabs name="export" />

TorchScript é o caminho validado para essa tarefa. A conversão para ONNX funciona,
mas não atingiu o mesmo nível de paridade, e os formatos restantes não estão
disponíveis. A cobertura por formato está nas páginas
[BiRefNet](/docs/models/birefnet) e [FeyNobg](/docs/models/feynobg) e na
[matriz de exportação completa](/docs/reference/export-matrix).
