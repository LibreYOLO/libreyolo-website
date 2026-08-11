---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: remoção de ruído, treinamento e exportação sob licença MIT'
description: >-
  Use o NAFNet no LibreYOLO para remoção de ruído e restauração de imagens.
  Instale, faça predições, treine, valide e exporte o checkpoint SIDD, com
  licença MIT.
lead: >-
  O NAFNet é uma rede convolucional para restauração de imagens que remove as
  funções de ativação não lineares de um bloco UNet típico, substituindo-as por
  multiplicação elemento a elemento. O LibreYOLO oferece suporte a uma tarefa,
  restauração, com um checkpoint publicado de remoção de ruído em imagens reais
  treinado no SIDD.
keywords:
  - NAFNet
  - restauração de imagens
  - remover ruído de imagem
  - reduzir ruído de foto python
  - deblur de imagem
  - image denoising
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Salvar a imagem restaurada
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Procedência do checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation e dataset são registrados no checkpoint salvo; eles
        # não mudam o que é treinado.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() devolve um dict simples, não um objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Usar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A factory roteia pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e devolve o mesmo objeto Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Instalação

O NAFNet não precisa de nenhum extra opcional. Tudo o que ele importa já vem na
instalação base.

```bash
pip install libreyolo
```

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache local.

<code-tabs name="predict" />

O objeto `Results` devolvido carrega um único campo para esta família,
`restored`, uma imagem RGB uint8 densa em HWC no canvas original; não há boxes
para percorrer. `save=True` grava essa imagem restaurada direto no disco, em vez
de desenhar uma anotação sobre a entrada. `conf`, `iou` e `max_det` são aceitos
por paridade de assinatura com todas as outras famílias, mas não têm efeito, já
que a restauração não produz detecções para filtrar. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Duas larguras compartilham esta arquitetura: `s` (largura 32) e `l` (largura
64), ambas construídas em torno de um patch de treinamento de 256 px. Predição e
validação rodam na resolução nativa da imagem, seja qual for o tamanho, com
padding apenas até o fator de downsample da rede. Só a largura `l` está
publicada no momento, como um checkpoint de remoção de ruído em imagens reais
treinado no SIDD.

## Treinamento

O NAFNet faz fine-tuning nos seus próprios pares de imagens degradada/limpa: um
YAML de dataset apontando para uma pasta `inputs/<split>/` de imagens degradadas
e uma pasta `targets/<split>/` de alvos limpos, pareadas pelo radical do nome do
arquivo. `degradation` e `dataset` são strings opcionais registradas no
checkpoint salvo para fins de procedência; elas não participam do treinamento.

<code-tabs name="train" />

Sem ajustes, o trainer roda 100 épocas com AdamW em `lr0=1e-3`, batch de 16,
crops de 256 px e early stopping depois de 50 épocas sem melhora de PSNR. Não
existe caminho LoRA para esta família: `lora=True` gera um erro em vez de rodar,
já que o `NAFNetTrainer` nunca adere ao fine-tuning por adaptadores.

Durante o treinamento a rede roda com global average pooling comum. O pooling
local por janelas do NAFNet, exclusivo de inferência (Test-time Local
Converter), é desacoplado antes da primeira época e reacoplado quando o
treinamento termina, já que retropropagar por um pooling local de janela fixa
não corresponderia à forma como o checkpoint é usado na inferência.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` devolve um dicionário com `metrics/PSNR` e `metrics/SSIM`, calculados em
RGB sobre todo o canvas válido: o SSIM usa uma janela gaussiana 11x11 com sigma
1.5, e o `fitness` para seleção do melhor checkpoint é o valor de PSNR. `data`
aponta para o mesmo formato de dataset de imagens pareadas usado no treinamento.

<code-tabs name="val" />

## Exportação

<export-matrix />

Um artefato exportado é carregado de volta por `LibreYOLO()` pelo sufixo do
arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um checkpoint e
devolve o mesmo `Results`, com `restored` carregando a imagem de saída. O NAFNet
exporta em uma resolução espacial fixa: `imgsz` precisa ser divisível pelo fator
de downsample da rede (16 para as duas larguras de arquitetura), e apenas a
dimensão de batch é dinâmica quando `dynamic=True`; altura e largura ficam fixas
no momento da exportação.

<code-tabs name="export" />

## Checkpoints

Todos os arquivos de pesos publicados desta família.

<checkpoint-table />

## Licenciamento

<provenance-box></provenance-box>

## Citação

<citation-block />
