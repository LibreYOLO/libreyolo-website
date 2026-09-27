---
title: 레이어 동결
seo_title: LibreYOLO에서 학습 중 레이어 고정하기
description: '전이 학습을 위해 모델의 일부를 동결: 계열 동결 그룹의 정수 개수, 명시적인 인덱스 목록, 또는 모듈 및 파라미터 이름 선택기.'
lead: >-
  프리징은 선택된 가중치를 고정시키는 동안 나머지 모델이 학습되도록 합니다. 선택자는 YAML 그래프의 원시 레이어 번호가 아니라 계열 자체의
  순서화된 프리즈 그룹이나 모듈 이름을 다룹니다.
keywords:
  - 레이어 동결
  - 전이 학습
  - 백본 동결
  - 동결 배치 정규화
  - 그룹 동결
  - 헤드만 파인튜닝
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 첫 10개의 그룹이 전체 YOLOv9 백본입니다.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: 이름으로
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: 여러 선택자
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: 계열의 냉동 그룹을 순서대로 나열하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## 무언가를 얼리다

`freeze`는 선택 사항이며 기본값은 동결하지 않음입니다.

<code-tabs name="train" />

모델이 구축된 후와 새로운 클래스 수를 위해 어떤 헤드를 다시 구축한 후, 그리고 옵티마이저가 생성되기 전에 동결이 수행되므로, 옵티마이저는 학습 가능한 매개변수만 받습니다.

## 셀렉터가 될 수 있는 것

| 값 | 의미 |
|---|---|
| `None`, `False`, `""`, `"none"` | 모든 매개변수를 학습시키다 |
| `10` 또는 `"10"` | 첫 번째 열 계열 냉동 그룹을 얼리십시오 |
| `[0, 3, 7]` | 그 0부터 시작하는 그룹을 고정하십시오 |
| `"backbone"` | 일치하는 그룹, 모듈 또는 매개변수 접두사를 고정합니다 |
| `["backbone", "neck"]` | 나열된 각 선택기를 고정하십시오 |
| `["backbone", 3]` | 혼합 목록이 작동합니다 |

문자열은 해석되기 전에 구문 분석되므로 CLI와 YAML 구성은 Python과 동일한 형태를 허용합니다. `freeze="[0, 3, 'head']"`는 리터럴 리스트로 구문 분석되고, `freeze="backbone,neck"`는 쉼표로 분할되며, 일반 숫자 문자열은 개수가 됩니다.

`freeze=True`는 애매하므로 거부되었습니다.

이름 선택자는 동결 그룹 이름, 모듈 이름 또는 매개변수 이름 접두사와 일치하며, 글로브 문자 `*`, `?` 및 `[`가 작동합니다. 선행 `model.`는 유연하게 처리되므로 `backbone`와 `model.backbone`는 모두 내부에서 계열이 사용하는 철자법과 일치합니다.

## 계열별 그룹 정의

정수는 공유된 그래프의 위치가 아니라 계열 자체의 정렬된 동결 그룹 목록을 가리킵니다. LibreYOLO의 계열들은 모두 하나의 YAML로 색인된 순차 모델이 아니므로, 원시 레이어 번호는 각 모델에서 서로 다른 의미를 갖게 됩니다.

YOLOv9는 입력 쪽에서 그룹을 정렬합니다: 열 개의 백본 단계, 그 다음 여섯 개의 넥 단계, 그리고 헤드. 그래서 `freeze=10`가 정확히 백본입니다. `backbone`, `neck`, `head`는 그 위에 있는 안정적인 이름 선택기입니다.

RF-DETR의 그룹은 `backbone.encoder`, `backbone.projector`, `decoder`, `queries`, `transformer.encoder_output` 및 `head`입니다. 여기서는 이름이 더 나은 선택입니다. 왜냐하면 트랜스포머 구성 요소는 계층 수와 일치하지 않기 때문입니다. `backbone`는 접두사로 두 백본 그룹과 모두 일치합니다.

의미 그룹을 정의하지 않는 계열는 보수적인 기본값으로 돌아갑니다: 적어도 하나의 매개변수를 가진 모델의 각 직접 자식, 선언 순서대로입니다. 보통 이것은 짧은 목록이므로, 큰 정수는 충분한 그룹을 찾지 못할 것입니다:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

추측하지 않고 실제 목록을 보려면:

<code-tabs name="groups" />

## 실패는 크게 들린다

이것을 잘못하는 모든 방법은 요청하지 않은 것을 학습하기보다는 오히려 문제를 일으킵니다.

아무 것도 일치하지 않는 선택기는, 일치하지 않은 선택기를 이름을 지정하며 오류를 발생시킵니다:

```text
freeze selector(s) matched no parameters: 'backbon'
```

동결 시점과 옵티마이저가 다시 구성될 때 모두 학습 가능한 항목을 남기지 않는 동결:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

`all`가 모든 매개변수와 일치하므로, `freeze="all"`가 하는 것이 바로 그것입니다.

냉동이 성공하면 한 줄이 발생한 일을 기록합니다:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## 동결된 배치 정규화는 업데이트를 중지

고정된 파라미터는 여전히 실행 통계가 계속 변하는 모듈 안에 있습니다. 파라미터가 고정 세트에 속한 모든 BatchNorm 스타일 모듈은 평가 모드로 전환되며, 트레이너는 각 에포크의 `model.train()` 호출 후에 이를 다시 적용하여 통계가 전체 실행 동안 고정되도록 합니다.

이것은 기본적으로 켜져 있으며, 실제로 골격을 얼리게 만드는 것이 바로 이것입니다.

## LoRA로 작곡하기

`freeze`와 `lora=True`는 함께 작동합니다. RF-DETR, DEIM 및 ConvNeXt에서는 부모 그룹이 고정되어 있어도 어댑터 매개변수가 학습 가능 상태로 유지되며, 이는 원하는 조합입니다: 어댑터가 그 위에서 학습하는 고정된 백본. [LoRA 파인튜닝](/docs/train/lora)을 참조하십시오.

## 범위

이것은 시작 시 결정되는 정적 동결입니다. 예약된 해동과 점진적 동결은 인터페이스의 일부가 아닙니다.

## 관련된

- 나머지 `train()`에 대한 [하이퍼파라미터](/docs/train/hyperparameters).
- [증류](/docs/train/distillation)은 대형 모델의 지식을 학습 실행으로 옮기는 다른 방법입니다.
