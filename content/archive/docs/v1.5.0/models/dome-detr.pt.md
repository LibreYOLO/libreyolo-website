---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: detecção de objetos minúsculos no LibreYOLO'
description: >-
  Use o Dome-DETR no LibreYOLO para detecção de objetos minúsculos em imagens
  aéreas e de drone. Converta os pesos do upstream, rode predições, faça
  fine-tuning e valide com código licenciado sob a MIT.
lead: >-
  Um especialista em objetos minúsculos construído sobre o D-FINE: uma cabeça de
  densidade decide onde estão os objetos, a atenção do encoder fica restrita às
  janelas que os contêm, e a quantidade de queries é dimensionada a partir dessa
  densidade em vez de ser fixa. O LibreYOLO o suporta para detecção.
keywords:
  - Dome-DETR
  - detecção de objetos pequenos
  - detecção de objetos minúsculos
  - detecção em imagens de drone
  - imagens aéreas
  - sensoriamento remoto
  - VisDrone
  - AI-TOD
  - DETR
  - queries adaptativas por densidade
last_verified: 1.5.0
snippets:
  predict:
    - label: 'Converter, depois prever'
      language: bash
      code: |
        # O LibreYOLO não hospeda nenhum peso do Dome-DETR, então o checkpoint é
        # baixado do repositório upstream e convertido uma única vez.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Um caminho local, não um nome simples: nada é baixado para esta
        família.

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        result = model("drone-frame.jpg", save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: Nomes das classes
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Não existe checkpoint COCO, então as classes vêm do dataset em que os
        # pesos foram treinados e são lidas dos metadados do checkpoint.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 classes do AI-TOD-V2

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 classes do VisDrone
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 381f01d769e7c420
---

## Instalação

O Dome-DETR não precisa de nenhum extra opcional. Tudo o que ele importa já está
na instalação base.

```bash
pip install libreyolo
```

## Predição

Não há nada para baixar automaticamente. O LibreYOLO não hospeda esses pesos,
então o fluxo é: buscar o checkpoint do upstream, convertê-lo uma vez e depois
carregar o arquivo convertido por caminho. [Licenciamento](#licensing) explica o
porquê.

<code-tabs name="predict" />

O objeto `Results` retornado é o mesmo que todas as famílias retornam, então
trocar por outro detector é uma mudança de uma linha. `conf` e `max_det` filtram
a seleção de queries; `iou` é aceito por paridade de API mas não tem efeito,
porque o decoder é um preditor de conjuntos sem etapa de NMS. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

Duas capacidades ficam desligadas nesta família. A captura de grafos CUDA está
desativada, porque a quantidade de queries do PAQI depende dos dados e a passada
forward, portanto, muda de forma de imagem para imagem, que é exatamente o que a
captura de grafos não consegue absorver. O test-time augmentation roda em um
único tamanho quadrado fixo, então um pedido de TTA multiescala não faz nada.

## Variantes

Três tamanhos, s, m e l, todos a 800 por 800. O tamanho seleciona o backbone, e
o dataset de onde vieram os pesos seleciona a profundidade do decoder e o
orçamento de queries, então um código de tamanho sozinho não identifica um
grafo. Os pesos de AI-TOD-V2 selecionam entre 300 e 1500 queries por imagem, os
de VisDrone entre 250 e 500, e o modelo large roda quatro camadas de decoder no
AI-TOD-V2 contra seis no VisDrone.

O Dome-DETR é o D-FINE com três acréscimos. O DeFE prevê um mapa de densidade. O
MWAS usa esse mapa para restringir a atenção do encoder às janelas que de fato
contêm objetos, em vez de atender a tudo. O PAQI dimensiona o conjunto de
queries a partir da mesma densidade em vez de decodificar um valor fixo de 300.
O ganho se concentra onde os objetos são menores e encolhe conforme eles
crescem: a própria ablação do upstream leva o AP em objetos muito minúsculos de
14.0 para 17.8, enquanto o AP em objetos médios vai apenas de 45.4 para 46.4.
Pense nele como um companheiro do [D-FINE](/docs/models/d-fine) para imagens
aéreas, de drone e de sensoriamento remoto, não como um substituto.

O LibreYOLO não publica nenhuma linha de benchmark para esta família, porque não
publica checkpoints para medir.

## Treinamento

O Dome-DETR é treinável. O treinamento roda o objetivo completo do upstream: as
losses do D-FINE mais a supervisão de densidade e de contagem do DeFE, com as
queries de padding mascaradas para fora dos termos de classificação e máscaras
de atenção de denoising por imagem, para que o padding de uma imagem não vaze
para o de outra.

<code-tabs name="train" />

A configuração herda a receita do D-FINE e muda o que o MWAS exige. `imgsz` é
800, `lr0` é `2e-4`, o grupo de parâmetros do backbone é escalado por
`backbone_lr_mult=0.1`, e `multi_scale` é forçado a ficar desligado, porque as
janelas do MWAS precisam que a entrada continue divisível pelo stride 8. `batch`
tem padrão 4 em vez dos 16 do D-FINE: o PAQI aplica padding em todo batch até o
seu membro mais largo, então a memória acompanha a imagem mais carregada do
batch em vez da média.

Uma ressalva honesta sobre a acurácia. O upstream treina por 160 épocas com
`MultiStepLR(milestones=[80, 120], gamma=0.8)`, enquanto esses padrões rodam o
schedule flat-cosine do D-FINE pelas mesmas 160 épocas. Esse schedule não foi
reproduzido aqui, e os números de AP do artigo também não foram, então encare
esses números como resultados dos autores do upstream e não como uma promessa de
que esta receita os alcança. Forneça o schedule do upstream se o objetivo for
igualar o artigo.

Veja [treinamento](/docs/train) para datasets, data augmentation, multi-GPU e
loggers.

## Validação

`val()` retorna um dicionário indexado por nome de métrica e imprime resultados
por classe quando `verbose` fica ligado.

<code-tabs name="val" />

A validação roda contra o seu próprio dataset, no formato em que você treinou. O
portão de validação COCO da biblioteca não se aplica aqui, já que não existe
checkpoint COCO desta família para servir de medida.

## Exportação

A exportação não é suportada, para nenhum formato, e pedir uma levanta um erro
em vez de produzir um arquivo.

O motivo é o PAQI. Ele decide a quantidade de queries por imagem, a partir de
propostas filtradas por densidade e de um laço guloso de supressão adaptativa à
densidade, então o comprimento da saída do decoder é uma propriedade da entrada
e não do grafo. O tracing fixa a quantidade que a imagem usada no tracing por
acaso produziu, o que gera um artefato que devolve silenciosamente resultados
errados para todas as outras imagens. Uma formulação estática teria que
desenrolar essa supressão sobre todos os 250 a 1500 candidatos, e reduzir a um
top-k fixo removeria exatamente o recall em objetos minúsculos que justifica a
existência desta família. Se você precisa de um transformer de detecção
exportável, o [D-FINE](/docs/models/d-fine) é a escolha.

## Checkpoints

Não há nenhum para listar. O LibreYOLO não publica pesos do Dome-DETR, e nenhum
nome no formato `LibreDOMEDETR<size>-<dataset>.pt` resolve para um download.

O upstream publica seis checkpoints, s, m e l para cada um de dois datasets:
AI-TOD-V2 com 9 classes e VisDrone com 12. Não existe checkpoint COCO, então um
nome de arquivo canônico sempre carrega o sufixo do dataset, e os nomes das
classes viajam nos metadados do checkpoint em vez de virem de uma constante da
família. Pedir um `LibreDOMEDETRs.pt` puro levanta um erro imediatamente, com
uma mensagem que nomeia os dois arquivos reais e o comando de conversão, em vez
de tentar um download que daria 404.

`weights/convert_domedetr_weights.py` faz a conversão. Ele reconstrói o grafo do
LibreYOLO, carrega os tensores do upstream nele e se recusa a escrever qualquer
coisa se uma única chave estiver faltando, sobrando ou com a forma errada, então
um arquivo convertido ou é uma correspondência exata ou não existe. Aponte o
script para um `.pth` do upstream e passe o tamanho e a variante:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Sobre fidelidade numérica, `weights/parity_domedetr.py` compara esta
implementação com a do upstream nos seis checkpoints e reporta
`max_abs_diff == 0.0` tanto em `pred_logits` quanto em `pred_boxes`, depois de
primeiro conferir a máscara de janelas do MWAS bit a bit, e separadamente
compara cada termo de loss com o critério do upstream. Vale ser claro sobre o
que isso é: um script manual, que precisa do checkout do upstream e dos
checkpoints publicados em disco, rodado à mão. Ele não faz parte da integração
contínua, e nenhum job de CI o reproduz.

## Licenciamento

<provenance-box>

Os pesos são o motivo de esta família não ser espelhada. O model card do
upstream não traz campo de licença nos metadados, e o texto dele afirma que o
projeto é Apache-2.0 ao mesmo tempo que restringe o material apenas a fins de
pesquisa acadêmica. Essas duas leituras não combinam, e a mais restritiva não é
uma concessão de redistribuição, então o LibreYOLO aponta para o repositório
upstream em vez de copiar os arquivos, enquanto não há esclarecimento. O mesmo
raciocínio é o que rege o [YOLO-NAS](/docs/models/yolo-nas) aqui.

O código é uma questão separada e mais clara. O repositório upstream é
Apache-2.0, o código portado pelo LibreYOLO é MIT, e os pesos que você mesmo
treina com os seus dados são seus.

</provenance-box>

## Citação

O Dome-DETR foi publicado no ACM Multimedia 2025 como "Dome-DETR: DETR with
Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection". O preprint está em
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). Os autores não
publicam nenhum bloco BibTeX no repositório deles, então nenhum é reproduzido
aqui, em vez de ser montado à mão.

<citation-block />
