---
title: Siêu tham số
seo_title: Siêu tham số huấn luyện trong LibreYOLO
description: >-
  Các đối số train() quan trọng: epochs, batch, lr0, optimizer, EMA, autobatch,
  tích lũy gradient và tiếp tục huấn luyện, cùng lý do giá trị mặc định khác
  nhau theo family.
lead: >-
  Mọi đối số huấn luyện là một trường trên dataclass TrainConfig. Class cơ sở
  định nghĩa trường và giá trị mặc định; mỗi model family tạo subclass và ghi đè
  những giá trị mặc định mà recipe đã công bố của nó thay đổi.
keywords:
  - đối số train
  - learning rate
  - kích thước batch
  - autobatch
  - exponential moving average
  - tích lũy gradient
  - tiếp tục huấn luyện
  - patience early stopping
  - amp bfloat16
  - train config yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Đọc các giá trị mặc định đã phân giải của family
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # In giá trị mặc định của train, val và predict, gồm cả ghi đè theo
        family.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 thăm dò bộ nhớ GPU và phân giải thành một lũy thừa cụ thể của
        hai.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 4 micro-batch gồm 16 ảnh trên mỗi bước optimizer, batch hiệu dụng là
        64.

        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Nạp checkpoint của lượt chạy bị gián đoạn rồi yêu cầu tiếp tục.
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Các key trong yaml là tên trường TrainConfig. Kwarg tường minh được ưu
        tiên.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Thiết lập đối số

`train()` nhận đối số keyword, còn CLI nhận cùng tên ở dạng `key=value`.

<code-tabs name="train" />

Cả hai đường dẫn đi tới cùng một nơi. Các kwarg được truyền cho
`TrainConfig.from_kwargs()`, phương thức dựng dataclass cấu hình của family.

## Lỗi chính tả không phát sinh lỗi

`from_kwargs()` loại mọi key không phải trường trong cấu hình và phát
`UserWarning` nêu tên key đó. Quá trình huấn luyện sau đó bắt đầu với giá trị
mặc định:

```python
# UserWarning: Key cấu hình huấn luyện không xác định (bị bỏ qua): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Không có gì thất bại, lượt chạy hoàn tất, còn learning rate chưa bao giờ mang
giá trị bên gọi yêu cầu. Hãy đọc cảnh báo ở epoch đầu tiên của recipe mới. CLI
nghiêm ngặt hơn vì xác thực tên flag trước khi dựng cấu hình, nên flag CLI sai
chính tả bị từ chối ngay.

## Giá trị mặc định theo từng family

`TrainConfig` định nghĩa trường và giá trị mặc định cơ sở. Mỗi family tạo
subclass và ghi đè những gì recipe đã công bố của nó thay đổi, vì vậy không có
một câu trả lời đúng duy nhất cho câu hỏi "learning rate mặc định là bao nhiêu".

Các giá trị mặc định cơ sở là `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` và `amp=True`. Ba ví dụ cho thấy family khác giá trị đó đến đâu:

| Trường | Cơ sở | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE và DEIM được phân phối với `amp=False` vì decoder D-FINE clamp activation
ở 65504, giá trị float16 hữu hạn lớn nhất. YOLO-NAS và FOMO cũng mặc định tắt.
Flag `--amp` của CLI mặc định là `True` cho mọi family, vì vậy được tính là do
người dùng cung cấp và ghi đè mặc định family; hãy giữ nguyên trừ khi thực sự
muốn thay đổi.

Để đọc giá trị mặc định thực của family thay vì đoán:

<code-tabs name="defaults" />

## Kích thước batch

`batch` là batch toàn cục. Trong huấn luyện multi-GPU, mỗi rank nạp
`batch // world_size`, vì vậy số được truyền là số ảnh trên mỗi bước optimizer,
bất kể có bao nhiêu GPU. Xem [Huấn luyện multi-GPU](/docs/train/multi-gpu).

`batch=-1` bật autobatch. Trainer thăm dò mô hình ở training mode bằng backward
pass thực theo các lũy thừa của hai, khớp một đường thẳng với đường cong bộ nhớ
và chọn lũy thừa lớn nhất của hai nằm hẳn dưới giá trị ngoại suy có thể vừa trong
60 phần trăm tổng VRAM.

<code-tabs name="autobatch" />

Điểm cốt lõi là thăm dò ở training mode bằng backward pass: thăm dò ở inference
mode bỏ qua activation được giữ lại và tensor gradient, vốn chiếm gấp nhiều lần
inference trên CNN sâu. RF-DETR giảm tỷ lệ mục tiêu xuống 45 phần trăm vì
backward tổng hợp của phép thăm dò vẫn đánh giá thấp chi phí criterion và các
lớp decoder phụ trợ.

Autobatch là tính năng CUDA. Trên CPU hoặc MPS, nó ghi một dòng log và giữ batch
mặc định.

## Tích lũy gradient

`nbs` đặt kích thước batch danh nghĩa, hay hiệu dụng. Trainer tích lũy
`round(nbs / batch)` micro-batch trên mỗi bước optimizer.

<code-tabs name="accumulate" />

Khi để `None`, giá trị mặc định, cơ chế tích lũy bị tắt và quá trình huấn luyện
không thay đổi.

## Learning rate và lịch

`lr0` là learning rate ban đầu, còn `optimizer` chấp nhận `sgd`, `adam` và
`adamw`. `momentum` là momentum SGD hoặc beta1 của Adam, `weight_decay` là thành
phần L2, còn `nesterov` áp dụng cho SGD.

Lịch được định hình bởi `scheduler`, `warmup_epochs`, `warmup_lr_start` và
`min_lr_ratio`. `no_aug_epochs` đặt số epoch cuối chạy không có augmentation
mạnh, và một số lịch cũng dùng nó để định hình phần đuôi, vì vậy đây không chỉ
là nút augmentation. Cách từng family dùng nửa augmentation của nó được trình
bày tại [Tăng cường dữ liệu](/docs/train/augmentations).

Một số family thêm nút learning rate riêng. `backbone_lr_mult` điều chỉnh tỷ lệ
nhóm backbone so với head, `clip_max_norm` đặt gradient clipping, còn SegFormer
dùng `head_lr_mult` để chạy decode head ở tốc độ gấp mười backbone. Các nút này
nằm trên subclass cấu hình của family, không phải class cơ sở.

## EMA

`ema=True` duy trì exponential moving average của trọng số bên cạnh trọng số
được huấn luyện. Tùy chọn này mặc định bật ở mọi nơi trừ FOMO.

`ema_decay` là decay mục tiêu. Decay tăng dần thay vì bắt đầu ở mục tiêu: giá
trị hiệu dụng tại lần cập nhật `n` là `ema_decay * (1 - exp(-n / tau))`, với
`tau` mặc định là 2000, nhờ vậy các cập nhật ban đầu bám sát mô hình hơn còn cập
nhật muộn được làm mượt. Giá trị mặc định theo family trải từ `0.997` trên
YOLO-NAS pose, qua `0.9998` trên YOLOX, đến `0.9999` trên YOLOv9 và dòng DETR.

Trọng số EMA được dùng để xác thực và được lưu trong `best.pt` cùng `last.pt`.
Trọng số thô đã huấn luyện cũng được lưu dưới key `train_model`, để lượt tiếp
tục đi theo quỹ đạo đã huấn luyện thay vì giá trị trung bình.

## Precision

`amp=True` chạy forward pass trong CUDA autocast. `amp_dtype` chọn `float16`
(mặc định) hoặc `bfloat16`; `fp16` và `bf16` là cách viết được chấp nhận.

Float16 cần dynamic loss scaling và nhận `GradScaler` hoạt động. Phạm vi số mũ
rộng hơn của bfloat16 không cần, vì vậy scaler của nó được dựng nhưng tắt, giúp
đường dẫn optimizer giữ nguyên. Yêu cầu bfloat16 trên thiết bị CUDA không hỗ trợ
bfloat16 sẽ phát sinh lỗi khi thiết lập thay vì âm thầm giảm cấp.

## Đầu ra, checkpoint và dừng

Các lượt chạy được ghi vào `project/name`. `project` mặc định là `runs/train` ở
mọi nơi, nhưng `name` là một trong các giá trị ghi đè theo family: mặc định cơ
sở là `exp`, YOLOv9 dùng `yolo9_exp`, còn D-FINE dùng `dfine_exp`. Với
`exist_ok=False`, giá trị mặc định, thư mục đã tồn tại sẽ nhận hậu tố tăng dần
thay vì bị ghi đè.

`save_period` ghi thêm `weights/epoch_<N>.pt` sau mỗi N epoch, bên cạnh
`weights/last.pt` sau từng epoch và `weights/best.pt` mỗi khi metric được theo
dõi cải thiện. `eval_interval` đặt tần suất xác thực, còn `patience` dừng lượt
chạy sau số epoch đó mà không cải thiện; `0` tắt early stopping.

`cache` tăng tốc các epoch lặp lại bằng cách giữ ảnh đã decode trong RAM (`True`
hoặc `"ram"`) hoặc dưới dạng tệp `.npy` bên cạnh nguồn (`"disk"`). Lượt đọc từ
cache giống từng byte với lượt đọc mới. Khi có worker dataloader, `"disk"` là
lựa chọn an toàn hơn.

## Tiếp tục huấn luyện

`resume=True` tiếp tục một lượt chạy bị gián đoạn. Checkpoint phải được nạp
trước vì thao tác tiếp tục đọc nó từ mô hình, không phải từ đối số riêng.

<code-tabs name="resume" />

Thao tác tiếp tục khôi phục trọng số đã huấn luyện, trạng thái optimizer, trọng
số EMA và số lần cập nhật, thông tin theo dõi metric tốt nhất, scale của
`GradScaler`, cùng trạng thái ngẫu nhiên PyTorch, CUDA và NumPy. Nó bắt đầu ở
epoch sau epoch trong checkpoint và tua nhanh lịch tới vị trí đó.

Có hai điều nó không thực hiện. Không thể kết hợp `resume=True` với `pretrained`,
và yêu cầu như vậy sẽ phát sinh lỗi. Khi key metric tốt nhất của checkpoint khác
lượt chạy hiện tại, thông tin theo dõi metric tốt nhất được đặt lại về 0 kèm
cảnh báo thay vì so sánh các giá trị không cùng ý nghĩa.

## Recipe trong tệp

`cfg=` nạp mapping YAML gồm tên trường `TrainConfig` và merge bên dưới các đối
số keyword tường minh, vì vậy kwarg luôn được ưu tiên hơn tệp.

<code-tabs name="cfg" />

`size` và `num_classes` bị loại khỏi tệp vì instance mô hình đã sở hữu chúng.
CLI không có flag `--cfg`; đường dẫn tệp là đối số Python.

## Nội dung liên quan

- Xem [Dataset](/docs/train/datasets) để biết `data=` chấp nhận gì.
- Xem [Tăng cường dữ liệu](/docs/train/augmentations) để biết các nút
  augmentation và family nào áp dụng chúng.
- Xem [Đóng băng lớp](/docs/train/layer-freezing) và
  [LoRA](/docs/train/lora) để huấn luyện một tập con của trọng số.
- Xem [Xác thực và metric](/docs/train/validation) để biết lượt chạy báo cáo gì.
