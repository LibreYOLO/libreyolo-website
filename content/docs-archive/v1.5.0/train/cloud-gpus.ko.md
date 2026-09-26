---
title: 렌트한 GPU에서 학습
seo_title: 렌트한 클라우드 GPU에서 LibreYOLO 학습하기
description: >-
  대여한 GPU나 서버리스 GPU에서 LibreYOLO 학습 작업 실행: 데이터를 준비하고, 설치하고, 시작하고, 실시간으로 보고, 가중치를
  가져오고 비용 지불을 중지하십시오.
lead: >-
  렌트한 GPU는 학습 실행을 시작, 종료, 요금이 발생하는 작업으로 바꿉니다. 작업은 로컬에서 학습하는 것과 동일하지만, 달라지는 것은
  데이터를 가져오고, 외부에서 지켜보며, 가중치를 꺼내고, 머신을 종료하는 것입니다.
keywords:
  - 클라우드 GPU 학습
  - GPU를 임대하다
  - vast.ai 학습
  - 모달 서버리스 GPU
  - 빔 GPU
  - 원격 학습
  - 허깅 페이스 데이터셋 스테이징
  - 에폭당 GPU 비용
last_verified: 1.5.0
snippets:
  install:
    - label: 상자 위에
      language: bash
      code: |
        pip install libreyolo

        # 실행에 필요한 추가 항목만 추가하십시오. RF-DETR 학습을 위한 rfdetr,
        # 파라미터 효율적인 파인튜닝을 위한 로라, 이후 내보내기를 위한 ONNX.
        pip install "libreyolo[rfdetr,lora]"
    - label: 무엇보다 먼저 GPU를 확인하십시오
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # 다른 아키텍처용으로 만들어진 휠이 True를 보고한 후 실패합니다
        # 첫 번째 실제 커널에서, 그러니 하나 실행해라.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: '한 번에 패키지하고 업로드하십시오, 컴퓨터에서'
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: 상자 위의 무대
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 분리되어 있어 연결이 끊겨도 작업이 유지됨
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 파이썬 파일에서 멀티 GPU
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # 모든 GPU에 걸친 글로벌 배치
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: 저렴한 책 한 권
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: 스크립트에서
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 브라우저에서 SSH 터널을 통해
      language: bash
      code: |
        # 박스에서(기본적으로 127.0.0.1:8420에 바인딩됨):
        libreyolo monitor /root/runs/run1 --no-browser

        # 그런 다음, 컴퓨터에서 http://localhost:8420를 로컬로 열십시오:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: 웨이트를 영구적으로 둘 곳으로 밀어라
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## 무언가를 빌리기 전에

두 가지 결정은 지금보다 나중에 더 큰 비용이 듭니다.

먼저 데이터셋을 CDN에 올리십시오. Hugging Face 데이터셋 저장소에 단일 tar로 패킹하는 것은 모든 제공자에서 동일하게 작동하며, 모두에게 빠르게 제공되고, 저장소가 비공개일 때 작업 환경에서 `HF_TOKEN`만 필요합니다. 집 인터넷에서 데이터셋을 업로드하거나 느린 원본에서 박스로 가져오는 경우, 대기하는 동안 사용된 GPU 시간이 청구됩니다.

<code-tabs name="stage" />

그런 다음 디스크 크기를 정하십시오. 스토리지를 과금하는 제공업체는 사용된 용량이 아니라 할당된 용량에 대해 과금하며, 디스크는 생성 후 축소할 수 없습니다. 준비된 데이터, 체크포인트, 그리고 대략 30%의 여유 공간을 합산하고 거기서 멈추십시오.

## 상자에 설치

<code-tabs name="install" />

이미지가 카드와 일치하는 CUDA 빌드를 이미 포함하고 있지 않다면 먼저 PyTorch를 설치한 다음 LibreYOLO를 설치하십시오. 그래야 pip가 자체적으로 CPU 전용 torch를 설치하지 않습니다. 두 번째 코드 조각은 선택적 의식이 아닙니다: 잘못된 GPU 아키텍처용으로 빌드된 휠은 `torch.cuda.is_available() == True`를 보고하고 실제 첫 작업에서 `CUDA error: no kernel image is available for execution on the device`로 실패합니다. 한 번의 행렬 곱셈은 한 시간의 설정보다 먼저 이를 탐지합니다.

제공자가 볼륨을 제공하는 경우 지속적 저장소에 `HF_HOME` 포인트를 지정하여 체크포인트와 데이터세트 다운로드가 실행 사이에 유지되도록 합니다.

## 출시

작업을 분리 모드로 실행하십시오. 네트워크 연결이 끊어지면 함께 종료되는 대화형 세션은 학습도 같이 종료됩니다.

<code-tabs name="launch" />

`batch=-1`은 여기서 사용하기에 특히 가치가 있습니다. 왜냐하면 보통 이전에 학습하지 않은 카드에서 작업하기 때문입니다. 이 방법은 실제 역전파와 함께 모델을 학습 모드에서 탐색하고, 맞는 가장 큰 2의 거듭제곱을 선택합니다. 이는 20분 후에 메모리 부족 오류로 상한을 발견하는 것보다 빠릅니다. [하이퍼파라미터](/docs/train/hyperparameters)를 참조하십시오.

멀티 GPU 장치에서, `device="0,1,2,3"`는 각 GPU마다 하나의 워커를 자동으로 생성하고, `batch`는 모든 GPU 전체에서 글로벌 배치를 유지합니다. `__main__` 가드는 필수적이며, 각 워커가 스크립트를 다시 가져오기 때문입니다. 그것과 나머지 분산 동작은 [멀티 GPU 학습](/docs/train/multi-gpu)에서 다룹니다.

## 밖에서 지켜봐

모든 실행은 `status.json`를 자신의 실행 디렉토리에 기록하며, 각 에포크마다 원자적으로 다시 작성됩니다. 이것은 저렴한 읽기 방식으로, 로그를 파싱하지 않고 상태, 현재 에포크, ETA 및 최신 지표를 몇 백 바이트 정도만으로 전달합니다.

<code-tabs name="watch" />

`metrics.jsonl`는 전체 에포크별 기록을 포함하고 있으며, `train.log`는 콘솔 출력을 가지고 있습니다. `libreyolo monitor`는 표준 라이브러리만 사용하여 세 가지 전체를 브라우저 대시보드로 제공하므로, LibreYOLO 자체 외에 상자에 아무 것도 설치할 필요가 없습니다. SSH 포트 포워드를 통해 접근하십시오.

이 중 어느 것도 학습 과정을 건드리지 않으므로, 실시간 실행에 첨부하거나 완료된 것을 다시 열거나 충돌한 것을 검사합니다.

## 비용 지불 중단 전 가중치 가져오기

이 상자는 일회용입니다. 체크포인트는 끝뿐만 아니라 중요한 시점마다 설정하십시오. 그렇지 않으면 충돌, 선점, 또는 크레딧 부족으로 전체 실행이 손실될 수 있습니다.

<code-tabs name="push" />

`weights/best.pt`와 `weights/last.pt`는 매 에포크와 모든 개선 시에 기록됩니다. `save_period=N`는 그 위에 `weights/epoch_<N>.pt` 스냅샷을 추가하며, 이것이 중간 실행 푸시를 저렴하게 만드는 이유입니다. `summary.json`와 `results.csv`, 즉 계열이 이를 기록하는 항목들은 작고, 가져갈 가치가 있습니다.

`on_train_epoch_end`에서의 콜백은 푸시를 자동화하는 깨끗한 방법입니다. 호스팅된 백엔드가 전혀 상자에 접근하지 않고도 메트릭을 제공하는 [실험 로거](/docs/train/loggers)를 참조하십시오.

## 비용 지불 중단

이 부분은 잘못되면 실제 비용이 발생하는 부분이며, 규칙은 제공자 모델에 따라 다릅니다.

원시 머신을 대여하는 마켓플레이스에서는 인스턴스가 종료될 때까지 요금이 실제 시간 기준으로 청구됩니다. 유휴 GPU도 사용 중인 GPU와 똑같이 요금이 청구되므로, 학습 프로세스를 종료하는 것만으로는 비용을 절약할 수 없습니다. 중지된 인스턴스도 여전히 디스크 요금이 청구됩니다.

작업이 데코레이터된 함수인 서버리스 플랫폼에서는 함수가 반환되면 컨테이너가 0으로 스케일되므로 잊혀진 박스가 있을 가능성은 훨씬 적습니다. 타임아웃이 없는 멈춘 작업은 여전히 비용이 청구되므로 항상 타임아웃을 설정하십시오.

파괴하는 대신 중지하는 것은 진짜 지렛대이자 진짜 함정입니다. 2026-07-31 기준으로 8x RTX 4090과 250GB 디스크를 임대한 환경에서 측정한 결과: 실행 중일 때 시간당 $3.4828 청구, 중지 시 디스크만으로 시간당 $0.0694 청구, 파괴 시에는 아무것도 청구되지 않았습니다. 이는 환경, 단계별 데이터 및 체크포인트를 유지하면서 98%를 절약하는 것입니다.

멈춘 비율은 임대하기 전에 계산할 수 있는 산술입니다:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

재구축 비용과 비교해 보십시오: 다시 임대하고, 이미지를 가져오고, 설치하고, 데이터를 다시 스테이징하는 작업입니다. 같은 서버에서 재구축은 약 15분의 설정 시간과 43GB의 인바운드 전송량이 필요하며, 대략 총 $1.00 정도입니다. 시간당 $0.0694와 비교하면, 약 14시간 내에 돌아올 경우 중단하는 것이 유리하고, 더 긴 간격이면 스테이징된 복사본에서 파괴하고 재구축하는 것이 유리합니다.

하나의 위험 요소로 인해 희소한 하드웨어에서는 중지가 안전하지 않습니다: 중지를 하면 GPU가 해제됩니다. 이를 예약하는 것이 없기 때문에, 호스트가 여전히 GPU를 자유롭게 가지고 있는 경우에만 재시작이 성공합니다. 디스크는 안전하지만, GPU는 안전하지 않습니다.

## 서버리스, 함수로서

만약 기계를 직접 관리하고 싶지 않다면, Modal과 Beam 모두 GPU에서 장식된 Python 함수를 실행하며 함수가 반환되면 0으로 스케일 다운됩니다. LibreYOLO의 자체 야간 테스트 스위트는 Modal에서 실행되며, 라이브러리 저장소의 `tools/ci/modal_nightly.py`는 복사할 수 있는 동작 중인 레포 내 예제입니다.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # OpenCV 시스템 라이브러리
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # 실행 간 캐시 가중치

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # 볼륨을 유지하다


@app.local_entrypoint()
def main():
    train.remote()
```

`modal run modal_train.py`로 실행하십시오. 컨테이너 파일 시스템은 임시적이므로, 보관할 가치가 있는 것은 모두 볼륨에 넣거나 푸시해야 합니다. `timeout=`를 명시적으로 설정하십시오. 그것이 멈춘 실행과 무한 요금 사이에서 유일한 차이입니다.

빔은 `__main__`에서 호출된 `@function` 데코레이터, `Volume`, `train.remote()`와 같은 형태를 취합니다.

## 작업당 비용에 따라 적정 규모 조정

$/시간은 최적화할 잘못된 수치입니다. 작은 모델은 큰 GPU를 절반만 사용하므로, 저렴하고 느린 GPU가 종종 에포크당 비용이 더 저렴합니다. 장기 실행을 시작하기 전에 대여한 카드에서 몇 단계 동안 프로파일러를 실행해보십시오: 판정이 `dataloader` 또는 `host / launch`라면, 더 빠른 GPU는 아무런 이득이 없고, 더 많은 작업자나 더 큰 배치가 큰 이득을 줍니다. 자세한 내용은 [학습 성능](/docs/train/performance)을 참고하십시오.

## 관련된

- [데이터셋](/docs/train/datasets)은 스테이지된 아카이브가 가져야 할 레이아웃과, GPU가 과금되기 전에 문제를 잡는 doctor 명령을 위한 것입니다.
- 멀티 카드 박스를 위한 [멀티 GPU 학습](/docs/train/multi-gpu).
