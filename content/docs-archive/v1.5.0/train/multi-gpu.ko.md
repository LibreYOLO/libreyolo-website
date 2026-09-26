---
title: 다중 GPU 학습
seo_title: LibreYOLO에서의 다중 GPU 학습
description: >-
  device="0,1"로 여러 GPU에서 학습합니다. 라이브러리가 DDP 워커를 생성하는 방법, 배치가 글로벌 배치인 이유, sync_bn을
  설정해야 하는 시점, 그리고 torchrun 경로입니다.
lead: >-
  LibreYOLO에서의 다중 GPU 학습은 PyTorch DistributedDataParallel입니다: GPU마다 하나의 프로세스가
  있으며, 각 프로세스는 전체 모델 복사본과 각 배치의 일부를 보유하고, 단계마다 랭크 간에 그래디언트를 평균화합니다.
keywords:
  - 파이토치 DDP 학습
  - 멀티 GPU 학습
  - torchrun nproc_per_node
  - 분산 데이터 병렬
  - 동기 배치 정규화
  - 글로벌 배치 크기
  - nccl 글루 백엔드
  - 멀티 GPU 윈도우
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # __main__ 가드는 필요합니다: 생성된 각 워커가 이것을 다시 임포트합니다
        # 모듈이며, 가드가 없으면 학습을 재귀적으로 다시 시작할 것입니다.
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # 글로벌 배치: 두 개의 GPU에서 GPU당 16개의 이미지
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: 출시
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # GPU 0에서 한 번 탐침되었으며, 세계 크기의 배수로 조정되었습니다.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## 두 개의 GPU에서 실행

장치 목록을 전달하십시오. 다른 것은 변경되지 않습니다.

<code-tabs name="train" />

장치가 하나 이상 있고 torchrun 환경이 없는 경우, 모델 `train()`는 가중치를 임시 파일에 저장하고 요청된 경우 autobatch를 해결하며, `torch.multiprocessing.spawn`와 함께 GPU당 하나의 작업자 프로세스를 생성합니다. 각 작업자는 모델 클래스를 다시 가져오고 저장된 가중치로 모델을 재구성하며, 일반 단일 장치 경로를 실행합니다. 이는 생성된 작업자 내부에서는 torchrun 환경 변수가 설정되어 있기 때문입니다. 실행이 끝나면 Rank 0의 최적 체크포인트가 호출자의 모델 인스턴스로 다시 로드됩니다.

`device`는 `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` 및 `"auto"`를 허용합니다. 두 개 이상의 CUDA 인덱스 목록만 생성이 시작됩니다.

## 필수 `__main__` 가드

생성된 워커는 자신이 온 모듈을 다시 임포트합니다. `if __name__ == "__main__":` 가드가 없으면, 그 임포트가 학습 호출을 다시 실행하고 각 워커가 자신의 워커를 생성합니다. 라이브러리는 이 경우를 탐지하고 재귀 실행을 허용하지 않은 채 예외를 발생시킵니다:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

작업자에게 전달되는 모든 것은 피클로 직렬화되므로, `callbacks=`는 피클 가능해야 합니다. 모듈 레벨 클래스는 작동하지만, 클로저나 람다는 작동하지 않으며, 오류 메시지가 이를 알려주고 내장 로거를 대안으로 가리킵니다.

## 글로벌 배치

`batch`는 모든 GPU에서 옵티마이저 스텝당 이미지 수입니다. 각 랭크의 데이터로더는 `batch // world_size`에서 `DistributedSampler`로 구축되므로, 두 개의 GPU에서 `batch=32`는 GPU당 16개의 이미지를 의미하며 32개가 아닙니다.

월드 크기로 나누어떨어지지 않는 배치는 조용히 다른 크기로 학습하는 대신 오류를 발생시킵니다:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, 그래서 이 값은 조용히 학습될 것입니다
different global batch than requested. Use batch=4 or batch=8.
```

그래디언트는 DDP 자체에 의해 평균 처리되므로 손실은 스케일 조정 없이 전달됩니다. 그 위에 월드 사이즈를 곱하면 효과적인 학습률이 대략 GPU 수만큼 증가하게 됩니다.

## DDP 하에서의 자동 배치

`batch=-1`는 작동하며, 전 세계 크기로 나눌 수 있는 글로벌 배치를 반환합니다.

<code-tabs name="autobatch" />

스폰 경로에서 프로브는 어떤 워커도 존재하기 전에 첫 번째 장치에서 부모 프로세스에서 실행되므로, 모든 워커는 구체적인 정수를 받으며 프로세스 간 조정이 필요하지 않습니다. torchrun 하에서는, 랭크 0이 프로브를 실행하고 그 결과를 단일 롱 텐서로 브로드캐스트합니다.

프로브는 한 GPU의 용량을 측정하고 세계 크기만큼 곱합니다. `nbs`가 설정되면, 글로벌 배치는 `nbs`로 제한되고 세계 크기의 배수로 내림하여 조정되므로, GPU를 추가하면 GPU당 배치를 줄이는 대신 누적 단계 수가 감소합니다. 프로브 자체의 동작 메커니즘은 [하이퍼파라미터](/docs/train/hyperparameters)에서 확인할 수 있습니다.

## 동기 배치 정규화

DDP에서는 각 랭크의 BatchNorm 레이어가 자신의 샤드만 봅니다. `batch // world_size`에서는 해당 샤드가 단일 GPU 실행에 비해 수렴된 모델의 러닝 통계를 손상시킬 만큼 작을 수 있습니다.

`sync_bn=True`는 모든 BatchNorm을 SyncBatchNorm으로 변환하여 통계가 전역 배치에 걸쳐 계산되도록 합니다. 변환은 distributed가 활성화되어 있을 때만 발생하므로, 단일 GPU 실행에는 해당 플래그가 어떤 경우에도 영향을 주지 않습니다.

BatchNorm이 많은 컨볼루션 계열 모델에서는 기본적으로 이미 켜져 있습니다: YOLOX, YOLOv7, YOLOv9 및 그 변형, YOLO-NAS, PicoDet, RTMDet, FOMO. 다른 모든 계열 모델은 기본적으로 꺼져 있습니다. 모델에 BatchNorm이 포함되어 있고, `sync_bn`가 꺼져 있으며, per-rank 배치가 16 미만일 경우, 트레이너가 경고를 표시합니다.

<code-tabs name="syncbn" />

`sync_bn`에 대한 CLI 플래그는 없습니다. 이는 Python 인수입니다.

## torchrun으로 실행

torchrun도 작동하며, 클러스터 스케줄러가 이미 프로세스 실행을 관리할 때 올바른 선택입니다. 단일 장치를 위한 스크립트를 작성하고 torchrun이 랭크 환경을 설정하도록 하십시오.

<code-tabs name="torchrun" />

두 가지를 결합하지 마십시오. torchrun 환경이 존재하면 `device="0,1"`는 생성되지 않습니다. 트레이너는 `cuda:LOCAL_RANK`를 가져가고 torchrun이 프로세스 수를 소유합니다.

## 계급 행동

랭크 0은 모든 부작용을 소유합니다. 실행 디렉토리를 해결하고 해결된 이름을 브로드캐스트하여 모든 랭크가 동의하도록 하며, 체크포인트와 아티팩트를 기록하고, 사용자 콜백과 로거를 실행합니다. 다른 랭크는 학습하고 기울기를 기여합니다.

각 랭크는 구성된 `seed`에서 파생된 난수 생성기를 사용하여 데이터로더와 증강 RNG의 시드를 다르게 설정하므로, 랭크들은 동일한 증강을 적용하지 않습니다.

## 플랫폼 및 백엔드

백엔드는 자동으로 선택됩니다: CUDA와 NCCL이 모두 사용 가능한 경우 NCCL, 그렇지 않으면 Gloo가 사용됩니다. NCCL은 Windows에서 빌드되지 않으므로 Windows 실행은 별도의 구성 없이 Gloo를 사용합니다. 프로세스 그룹은 세 시간 제한으로 초기화됩니다.

## DDP에서 실행되지 않는 것

- CUDA 그래프 캡처. `cuda_graph=True`는 한 줄을 기록하고 열심히 학습합니다. [학습 성능](/docs/train/performance)을 참조하십시오.
- 트레이닝 프로파일러. `profile=True`는 경고와 함께 무시됩니다.

모든 계열가 자동 스폰을 지원하는 것은 아닙니다. 스물네 개 계열는 지원하며, 이는 학습하는 탐지, 분류, 의미론 및 복원 계열를 포함합니다. 자동 스폰이 없는 계열는 멀티 GPU 장치를 받으면 조용히 하나의 GPU에서 학습하는 대신 모델 API와 torchrun 명령어를 이름으로 하는 오류를 발생시킵니다.

## 관련된

- `batch`, `nbs` 및 이력서에 대한 [하이퍼파라미터](/docs/train/hyperparameters).
- 콜백의 직렬화 가능성 제약을 위한 [실험 로거](/docs/train/loggers).
- [클라우드 GPU](/docs/train/cloud-gpus)로 다중 GPU 박스를 임대하십시오.
