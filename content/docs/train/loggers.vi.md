---
title: Logger thí nghiệm
seo_title: Logger thí nghiệm và callback trong LibreYOLO
description: >-
  Gửi metric huấn luyện tới TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune hoặc DVCLive, và viết callback riêng trên bốn hook huấn
  luyện.
lead: >-
  Mọi family có thể huấn luyện đều phát ra bốn sự kiện huấn luyện. Các logger
  tích hợp là đối tượng callback lắng nghe cùng các sự kiện đó, vì vậy tích hợp
  backend và hook tùy chỉnh dùng chung một interface.
keywords:
  - tensorboard huấn luyện
  - mlflow tracking
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callback huấn luyện
  - csv metric huấn luyện
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: Theo tên
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Instance đã cấu hình
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: Hàm thuần
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Đối tượng có nhiều hook
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: Theo dõi lượt chạy trong trình duyệt
      language: bash
      code: |
        libreyolo monitor                     # lượt chạy gần nhất trong runs/
        libreyolo monitor runs/train/exp      # một lượt chạy cụ thể
source_hash: de035acbaed32804
---

## Bật logger

`loggers=` nhận tên đã đăng ký, instance đã cấu hình hoặc iterable trộn cả hai.

<code-tabs name="logger" />

Tên không phân biệt chữ hoa chữ thường. Tập đã đăng ký là `tensorboard`,
`mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` và `dvc`, trong đó
tên cuối là alias của `dvclive`. Mọi giá trị khác phát sinh lỗi ngay và liệt kê
các tên hợp lệ. Không có giá trị bật tất cả, và không có flag CLI: `loggers=` là
đối số Python.

## Nội dung mọi backend ghi lại

Tất cả đều ghi cùng tên metric, vì vậy dashboard có cùng hình thức bất kể lựa
chọn backend nào:

| Key | Giá trị |
|---|---|
| `train/loss` | loss huấn luyện trung bình của epoch |
| `train/loss/<component>` | từng thành phần loss mà family báo cáo |
| `lr/<group>` | learning rate của từng nhóm tham số optimizer |
| `val/<metric>` | từng metric xác thực, đã bỏ tiền tố `metrics/` |
| `time/epoch_seconds` | thời gian thực của epoch |

Step là số epoch bắt đầu từ 1. Cấu hình huấn luyện đã phân giải đầy đủ được ghi
làm tham số khi bắt đầu huấn luyện, còn tên lượt chạy mặc định là
`<family><size>-<task>`, ví dụ `yolo9s-detect`.

Khi huấn luyện kết thúc, các backend hỗ trợ artifact tải lên `results.csv`,
`train_config.yaml` và `summary.json` khi những tệp đó tồn tại, cùng
`weights/best.pt` khi `log_checkpoints=True`. TensorBoard không tải gì lên vì
không có khái niệm artifact. Không logger nào tải ảnh biểu đồ xác thực lên.

## Hành vi khi lỗi

Package backend bị thiếu sẽ phát sinh lỗi khi khởi tạo và nêu rõ lệnh cài đặt,
vì yêu cầu logger nhưng âm thầm không nhận gì sẽ che giấu bug.

Lỗi backend trong lượt chạy được xử lý ngược lại. Exception đầu tiên từ handler
sẽ vô hiệu hóa logger đó cho phần còn lại của lượt chạy, ghi lỗi vào log, kết
thúc lượt chạy backend ở trạng thái thất bại, còn quá trình huấn luyện vẫn tiếp
tục. Tracking server bị ngừng không làm mất lượt huấn luyện.

## Các backend

Mỗi backend cần thành phần bổ sung riêng.

| Tên | Thành phần bổ sung | Constructor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Import các class từ `libreyolo.training`.

Cần biết các ghi chú theo backend trước lượt chạy đầu tiên:

Tệp sự kiện TensorBoard mặc định nằm tại `<save_dir>/tensorboard`. Xem bằng
`tensorboard --logdir runs/train`.

MLflow 3.x đã ngừng khuyến nghị file store cục bộ `./mlruns` và phát sinh lỗi
trừ khi `MLFLOW_ALLOW_FILE_STORE=true`. Để tracking cục bộ không cần server, hãy
truyền URI database như snippet ở trên và đọc bằng
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases quay về biến môi trường `WANDB_PROJECT`, sau đó tới
`libreyolo`. Comet quay về `COMET_PROJECT_NAME`, sau đó tới `libreyolo`, và lấy
credential từ cấu hình riêng; `online=False` tạo thí nghiệm offline. ClearML
tạo task mới, báo cáo cấu hình dưới `TrainConfig` và tắt cơ chế tự động capture
framework để metric không bị báo cáo hai lần. Neptune dùng client
`neptune-scale` hiện tại thay vì package cũ, còn `mode="offline"` ghi log cục bộ.

DVCLive ghi vào `<save_dir>/dvclive`. Nó dựng cây summary từ `/` và không thể
giữ float tại đường dẫn đồng thời là parent, nên `train/loss/box` được ghi thành
`train/loss.box` trong khi `train/loss` giữ nguyên tên. LibreYOLO cũng tắt các
giá trị mặc định thường dùng của DVCLive về lưu thí nghiệm DVC và ghi
`dvc.yaml` ở thư mục gốc, vì vậy logger tự chọn không tạo trạng thái kiểm soát
phiên bản bên ngoài thư mục lượt chạy; truyền `save_dvc_exp=True` hoặc
`dvcyaml=` tường minh để bật lại.

Neptune được chủ ý loại khỏi `libreyolo[all]`: client ổn định của nó cần protobuf
dưới phiên bản 7, trong khi thành phần bổ sung TFLite cần protobuf 7. Hãy cài
`libreyolo[neptune]` trong môi trường không có thành phần bổ sung TFLite.

## Viết callback

Cùng bốn sự kiện điều khiển mọi thứ.

<code-tabs name="callback" />

| Sự kiện | Thời điểm | Dữ liệu mang theo |
|---|---|---|
| `TrainStartEvent` | sau thiết lập, trước epoch 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | sau mỗi epoch, huấn luyện và xác thực | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | sau khi huấn luyện hoàn tất | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | nếu huấn luyện phát sinh lỗi | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Callable thuần chỉ nhận `TrainEpochEvent`. Đối tượng có thể triển khai tập con
bất kỳ của `on_train_start`, `on_train_epoch_end`, `on_train_end` và
`on_train_exception`; phương thức bị thiếu được bỏ qua.

`TrainStartEvent.config` là cấu hình đã phân giải đầy đủ, gồm kwarg người dùng
được merge với mặc định family, dưới dạng mapping chỉ đọc. Các sự kiện là
dataclass bị đóng băng và mapping cũng chỉ đọc, vì vậy callback không thể thay
đổi lượt chạy bằng cách ghi vào chúng.

Exception phát ra từ `on_train_start`, `on_train_epoch_end` hoặc `on_train_end`
được truyền lên và kết thúc lượt chạy. Chỉ `on_train_exception` được bảo vệ để
không thể che mất lỗi ban đầu.

Trong huấn luyện multi-GPU, callback chỉ kích hoạt trên rank 0. Với cơ chế spawn
DDP tự động, chúng cũng phải có thể pickle, nghĩa là class hoặc hàm cấp module
thay vì closure hay lambda. Xem [Huấn luyện multi-GPU](/docs/train/multi-gpu).

## Nội dung mọi lượt chạy luôn ghi

Ba tệp được ghi vào thư mục lượt chạy mà không cần cấu hình, trên mọi family:

| Tệp | Thời điểm ghi | Nội dung |
|---|---|---|
| `status.json` | theo cách atomic sau mỗi epoch và khi bắt đầu, kết thúc hoặc thất bại | `state` là `running`, `completed` hoặc `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, `metrics` mới nhất, `best_metric`, `best_epoch` và đối tượng `error` khi thất bại |
| `metrics.jsonl` | thêm một lần mỗi epoch | mỗi epoch một dòng JSON, cùng schema như `results.csv` |
| `train.log` | trực tiếp | đầu ra console của lượt chạy |

`status.json` là lượt đọc nhẹ cho script hoặc agent đang poll lượt chạy, còn
thao tác ghi atomic bảo đảm bên đọc không bao giờ thấy tệp mới được ghi một nửa.

`results.csv` và `summary.json` là các tệp riêng, có điều kiện theo family. Chúng
được ghi cho YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC và
DINOv2, nhưng không được ghi cho các family khác. `results.csv` có mỗi epoch một
dòng với các thành phần loss, metric xác thực và learning rate làm cột, còn
header được mở rộng khi xuất hiện cột mới. Khi tiếp tục, tệp được cắt về các
dòng trước epoch tiếp tục thay vì lặp lại chúng.

Bên cạnh đó, trainer luôn ghi `train_config.yaml` khi thiết lập và các checkpoint
dưới `weights/`.

## Theo dõi trực tiếp một lượt chạy

<code-tabs name="monitor" />

`libreyolo monitor` cung cấp dashboard trình duyệt trên các tệp ở trên chỉ bằng
thư viện chuẩn: biểu đồ metric, phần cuối log và mọi ảnh xác thực, đồng thời làm
mới khi lượt chạy đang hoạt động. Công cụ chỉ đọc và không bao giờ chạm tới
process huấn luyện, vì vậy có thể gắn vào lượt chạy đang hoạt động, mở lại lượt
chạy đã hoàn tất hoặc kiểm tra lượt chạy bị crash.

## Nội dung liên quan

- Xem [Xác thực và metric](/docs/train/validation) để biết ý nghĩa của các key
  `val/` và cách thêm validation loss.
- Xem [Hiệu năng huấn luyện](/docs/train/performance) để biết profiler, một công
  cụ khác trả lời một câu hỏi khác.
