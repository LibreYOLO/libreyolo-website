---
title: Restauração de imagens
seo_title: Restauração e aumento de escala de imagens no LibreYOLO
description: >-
  Remova ruído, corrija o desfoque e aumente a escala de imagens no LibreYOLO.
  Faça a predição de uma imagem RGB restaurada, treine o NAFNet com dados pareados e leia
  as chaves PSNR e SSIM.
lead: >-
  A restauração de imagens pega uma imagem degradada e devolve uma limpa. O
  LibreYOLO expõe isso como a tarefa restore, que cobre remoção de ruído,
  correção de desfoque e super-resolução atrás de um único contrato de saída:
  entra uma imagem RGB, sai uma imagem RGB.
keywords:
  - restauração de imagens python
  - remover ruído de imagem python
  - super resolução de imagem python
  - aumentar resolução de imagem ia
  - modelo para tirar desfoque de foto
  - validação PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Aumentar a escala de uma imagem
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # O gerador 4x compacto; tile limita o pico de memória numa origem
        grande.

        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")

        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)


        result.restored.save("upscaled.png")

        print(result.restored.array.shape)   # 4x a entrada em cada eixo
    - label: Remover o ruído de uma imagem
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Treinado com ruído real de imagem do SIDD; a saída mantém o tamanho de
        entrada.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")

        print(result.restore_scale)   # 1: este checkpoint não aumenta a escala
  train:
    - label: Fazer fine-tuning do NAFNet com imagens pareadas
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Registrar a procedência no checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation e dataset são gravados no checkpoint salvo como
        # procedência; não participam do treinamento.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Validar e ler as chaves de métricas
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() retorna um dict simples, não um objeto.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz é fixado no grafo, então passe o tamanho que o seu deploy
        # realmente entrega ao modelo.
        model.export(format="onnx", imgsz=256)
    - label: Rodar o arquivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A factory decide pelo sufixo do arquivo, então um artefato exportado
        # carrega como qualquer checkpoint e retorna o mesmo objeto Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Definição

A tarefa `restore` mapeia uma imagem para outra imagem. Remoção de ruído,
correção de desfoque e super-resolução são todas a mesma tarefa aqui, porque
compartilham um contrato: o modelo consome uma imagem RGB e retorna uma imagem
RGB, e a degradação que ele aprendeu a desfazer é uma propriedade do checkpoint,
não da API.

Uma predição preenche `result.restored`, um payload `RestoredImage` que guarda um
array RGB uint8 de `(H, W, 3)`. `.array` devolve esse array como NumPy e
`.save(path)` grava em disco. `result.restore_scale` registra o fator de aumento
de escala que o canvas de saída carrega, e vale `1` para um checkpoint que
preserva a resolução. `result.boxes` fica vazio, então `conf`, `iou` e `max_det`
são aceitos por paridade de assinatura, mas não têm efeito, e `save=True` grava a
imagem restaurada diretamente em vez de uma foto anotada.

## Modelos

Três famílias atendem `restore`, separadas pela degradação que desfazem.

[NAFNet](/docs/models/nafnet) é o modelo de remoção de ruído, e a única família
de restauração que o LibreYOLO consegue treinar. Sua arquitetura troca as
ativações não lineares de um bloco UNet por multiplicação elemento a elemento, e
o checkpoint publicado é treinado com ruído real de imagem do SIDD. A saída
mantém a resolução de entrada.

[Real-ESRGAN](/docs/models/real-esrgan) é o modelo prático de aumento de
escala: três checkpoints treinados contra degradações sintéticas, e não apenas
contra redução de escala bicúbica, em
4x, 2x e um gerador 4x menor e mais rápido, feito para menor latência.

[SwinIR](/docs/models/swinir) aumenta a escala em 4x com um backbone Swin
Transformer, em três tamanhos que cobrem o gerador leve oficial e dois geradores
para imagens reais.

## Predição

Os pesos são baixados do Hugging Face no primeiro uso e ficam em cache
localmente.

<code-tabs name="predict" />

A restauração roda na resolução da própria imagem de origem, e não em um canvas
fixo da rede, aplicando padding apenas até o fator de subamostragem da rede, então
tanto o tempo quanto a memória escalam com a quantidade de pixels da sua entrada.
`tile` divide o forward pass em tiles sobrepostos e mistura as emendas de volta,
e `tile_pad` é o halo adicionado em volta de cada tile antes de ele ser recortado
de novo; os dois são argumentos nomeados do Python. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Formato do dataset

A restauração pareia cada imagem de entrada degradada com uma imagem-alvo limpa
exatamente na mesma resolução, casadas pelo nome-base do arquivo.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` e `names` são marcadores de posição do schema; um modelo de restauração
retorna `Results.restored`, não detecções. `degradation` e `dataset` são rótulos
opcionais de procedência. `target_stem_suffix` cobre datasets que nomeiam a
imagem limpa de forma diferente do seu par degradado. A validação mantém a
resolução nativa e aplica padding apenas o suficiente para empilhar um batch,
então as métricas são calculadas no canvas original. Veja
[formatos de dataset](/docs/reference/dataset-formats) para o contrato completo.

## Treinamento

NAFNet é a única família de restauração com implementação de treinamento.
Tanto `Real-ESRGAN.train()` quanto `SwinIR.train()` lançam `NotImplementedError`:
esses
checkpoints vêm de treinamento GAN sobre pipelines de degradação sintética, e o
treinador de restauração pareada rodaria sem reproduzir essa receita.

<code-tabs name="train" />

O treinador tira recortes acoplados do par de entrada e alvo, então os dois lados
ficam alinhados. Veja [treinamento](/docs/train) para datasets, multi-GPU e
loggers, e a [página do NAFNet](/docs/models/nafnet) para os valores padrão desta
família e o pooling de inferência que ele desativa durante o treinamento.

## Validação

`val()` compara a saída restaurada com o alvo limpo, em RGB, no canvas original,
sem recorte de borda e sem redimensionamento.

<code-tabs name="val" />

`metrics/PSNR` é a relação sinal-ruído de pico em decibéis, e também é o
`fitness`, o número que a seleção do melhor checkpoint lê. `metrics/SSIM` é a
similaridade estrutural em `[0, 1]`, calculada com uma janela gaussiana 11x11 com
sigma 1.5 e com média sobre os três canais de cor. Nos dois casos, quanto maior,
melhor.

## Exportação

Um modelo de restauração exportado é carregado de volta por `LibreYOLO()` pelo
sufixo do arquivo, então um arquivo `.onnx` ou `.engine` se comporta como um
checkpoint e retorna o mesmo `Results`, com `restored` carregando a imagem de
saída.

<code-tabs name="export" />

A exportação de restauração fixa a resolução espacial no grafo, então passe o
`imgsz` que o seu deploy vai realmente entregar ao modelo. Para o NAFNet esse
tamanho tem que ser divisível pelo fator de subamostragem da rede, e só a
dimensão de batch continua dinâmica com `dynamic=True`. Para Real-ESRGAN e
SwinIR, omitir `imgsz` recorre a um tamanho de patch interno pequeno em vez da
sua resolução de trabalho. A cobertura por formato está em cada página de modelo
e na [matriz completa de exportação](/docs/reference/export-matrix).
[Exportação](/docs/export) lista os argumentos que todo formato aceita.
