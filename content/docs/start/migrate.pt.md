---
title: Importar pesos existentes
seo_title: Carregar pesos upstream no LibreYOLO
description: >-
  Aponte o LibreYOLO para um checkpoint de um projeto upstream. A autoconversão
  o reempacota na hora do carregamento, preservando seu número de classes e seus
  nomes.
lead: >-
  O LibreYOLO porta suas famílias de modelos de projetos upstream, então os
  checkpoints que eles publicam já são quase carregáveis. O que falta neles são
  os metadados. A autoconversão os fornece na hora do carregamento.
keywords:
  - converter pesos libreyolo
  - carregar checkpoint upstream
  - migrar pesos para libreyolo
  - converter pth para libreyolo
  - autoconversão de checkpoint
last_verified: 1.5.0
meta:
  - label: Ponto de entrada
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Escrito ao lado da origem como
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Conversores em script
    value: weights/ no repositório
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Substitua pelo caminho de um checkpoint que você já tenha. Um layout
        # upstream reconhecido é convertido na hora, escrito ao lado da
        # origem, e então carregado.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # O número de classes e os nomes vêm dos tensores e dos metadados do
        # próprio arquivo, então um fine-tune mantém seu conjunto de rótulos
        # em vez do da COCO.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Conferir o resultado
      language: bash
      code: |
        # O arquivo convertido cumpre o mesmo esquema que um publicado.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Esta página trata de checkpoints de outros projetos. Se o que você está movendo é
o seu próprio código a partir de um LibreYOLO mais antigo, veja
[atualizar para 1.5.0](/docs/upgrade).

## O que acontece quando você carrega um arquivo alheio

`LibreYOLO()` carrega qualquer arquivo de pesos passando primeiro pelo caminho
restrito, somente de pesos. Se o resultado trouxer metadados completos do
LibreYOLO, ele é usado diretamente. Se não trouxer, o arquivo vai para o
autoconversor antes que qualquer outra coisa seja tentada. Se o carregamento
restrito falhar de vez, o que acontece quando um checkpoint tem um objeto de
terceiros serializado dentro dele, o autoconversor é tentado com um carregador
que neutraliza esses objetos.

A autoconversão faz quatro coisas. Ela desempacota o dicionário de tensores do
layout que o projeto upstream tiver usado. Pergunta a cada família registrada se
ela reconhece as chaves resultantes, remapeando nomes onde a nomenclatura
upstream difere da do port do LibreYOLO. Envolve a vencedora em um checkpoint que
cumpre a versão 1.0 do esquema de metadados, lendo tamanho, tarefa e número de
classes dos próprios tensores. Depois escreve o resultado ao lado do arquivo de
origem e carrega esse.

<code-tabs name="convert" />

A conversão não é silenciosa. Um arquivo convertido é registrado no log com a
família, o nome de origem, o nome de saída e o número de classes resultante, de
modo que o log de uma execução registra exatamente o que foi carregado.

## Os layouts que ele desempacota

Checkpoints upstream aninham seus pesos em um punhado de lugares convencionais, e
o conversor os tenta em ordem até que um deles contenha tensores: um bloco EMA
sob `ema.module` ou um `ema` plano, um `ema_state_dict` com seu prefixo `module.`
removido, depois `params_ema`, `params`, `ema_net`, `net`, `model`, `state_dict`
e, por fim, o próprio objeto. Tentar vários em vez de só o primeiro significa que
um bloco `ema` contendo apenas contadores não esconde os pesos reais abaixo dele.

Os prefixos de wrapper também saem: `module.` do treinamento distribuído,
`_orig_mod.` de um modelo compilado e um aninhamento `model.model.` que algumas
redistribuições acrescentam.

## O que ele lê, e de onde

Tamanho, tarefa e número de classes vêm dos tensores, não do nome do arquivo, que
é a razão pela qual um checkpoint com fine-tuning é convertido com seu próprio
número de classes em vez do padrão da arquitetura. Os nomes das classes são
tirados dos metadados do próprio checkpoint quando estão presentes, de um bloco
`args` ou `hyper_parameters` se os nomes estiverem ali, e são cortados até o
número de classes detectado, para que um fine-tune que manteve seu conjunto de
rótulos base não carregue índices que sua cabeça já não tem.

Tarefas densas são tratadas explicitamente em vez de receberem rótulos
inventados. Um checkpoint de profundidade ganha uma classe chamada `depth`; um
checkpoint de restauração, uma classe chamada `image`. Um checkpoint de pose
precisa fornecer um número de keypoints, seja pelos tensores, seja pela família;
se nenhum dos dois produzir um, a conversão é recusada em vez de escrever um
arquivo incompleto.

O RF-DETR tem seu próprio reconhecedor, porque a detecção do tamanho precisa do
checkpoint inteiro e porque sua cabeça tem 91 saídas onde o LibreYOLO usa a
convenção COCO de 80 classes. Um checkpoint é normalizado para 80 classes quando
carrega exatamente 80 nomes, ou declara um número de classes igual a 80, ou
indica COCO como seu dataset, ou não carrega metadado nenhum de classes ou de
dataset. Um modelo genuíno de 90 classes, identificado pelos seus nomes, por uma
contagem explícita diferente de 80 ou por uma pista de dataset que não seja COCO,
é preservado como está.

## Para onde vai o arquivo convertido

A saída é escrita ao lado da origem, com o nome tirado dela:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Um detector YOLOv9 tiny salvo como `upstream-checkpoint.pth` vira, portanto,
`upstream-checkpoint-LibreYOLO9t.pt`. Nomeá-lo a partir da origem em vez de a
partir da família significa que dois fine-tunes da mesma família e do mesmo
tamanho em um mesmo diretório não sobrescrevem um ao outro, e nenhum deles colide
com um checkpoint oficial. O arquivo é reescrito a cada carregamento, então nunca
fica desatualizado em relação à sua origem. Se o diretório for somente leitura, o
arquivo convertido vai para um diretório temporário privado recém-criado, e o log
diz onde.

Daí em diante ele é um checkpoint comum do LibreYOLO: carrega pelo caminho dos
metadados, e `libreyolo metadata` o reporta como válido.

## Casos que precisam de ajuda

Duas famílias ficam fora do reconhecedor genérico. A família de gaze (olhar) é
excluída de vez: ela é somente de inferência e os pesos que publica carregam
restrições de redistribuição. O RF-DETR é excluído porque tem o reconhecedor
dedicado descrito acima, que é o que cuida dele no lugar.

Checkpoints PIDNet upstream em estado bruto são recusados, com um erro que aponta
para `weights/convert_pidnet_weights.py`. Esse script escreve os metadados
semânticos do Cityscapes de que o checkpoint precisa.

D-FINE e DEIM compartilham as mesmas chaves de arquitetura, então só os tensores
não bastam para separá-los. Quando as duas reivindicam um arquivo e não há em
jogo nenhuma família irmã com um marcador que as distinga, quem decide é o nome
do arquivo: um nome no formato de `dfine_hgnetv2_n_coco.pth` ou
`deim_hgnetv2_n_coco.pth` resolve a questão, e um nome que não diz nada é
recusado com essa explicação em vez de ser adivinhado. Instanciar `LibreDFINE` ou
`LibreDEIM` diretamente também resolve.

Quando várias famílias reivindicam legitimamente um mesmo arquivo, uma subclasse
vence a classe base que ela refina, e a ordem do registro decide o resto, já que
essa ordem codifica quão específica é a checagem de cada família. O nome do
arquivo só é consultado para o empate entre D-FINE e DEIM, de modo que o nome de
um arquivo nunca pode promover uma correspondência ampla acima de uma precisa.

## Os conversores em script

O repositório traz scripts de conversão por família em `weights/`, mais helpers
compartilhados para a parte repetitiva do encanamento. Eles são o caminho para um
arquivo que a rota em tempo de execução recusa, para produzir um checkpoint com
antecedência em vez de na hora do carregamento, e para as famílias cujos
metadados precisam ser fornecidos em vez de inferidos dos tensores.

Esses scripts fazem parte do repositório, não do pacote instalado, então usar um
deles implica clonar:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Todo script escreve um checkpoint que cumpre a versão 1.0 do esquema, que é a
mesma régua que a autoconversão atinge e a mesma que os pesos publicados atingem.
Veja [checkpoints e pesos](/docs/weights) para saber o que esse esquema contém.
