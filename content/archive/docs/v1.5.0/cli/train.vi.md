---
title: libreyolo train
seo_title: Tham chiếu lệnh libreyolo train
description: >-
  Huấn luyện một mô hình từ dòng lệnh: toàn bộ 59 tham số cùng giá trị mặc định,
  cách mặc định của từng dòng mô hình ghi đè chúng, và những tham số mà một dòng
  mô hình bỏ qua.
lead: >-
  Huấn luyện một mô hình trên một tập dữ liệu (dataset) và ghi checkpoint, chỉ
  số cùng log vào một thư mục chạy. Mọi tham số bên dưới đều có giá trị mặc định
  lấy từ định nghĩa lệnh, và cấu hình huấn luyện riêng của từng dòng mô hình có
  thể thay thế giá trị đó.
keywords:
  - libreyolo train cli
  - lệnh huấn luyện libreyolo
  - huấn luyện yolo bằng cli
  - tham số libreyolo train
  - libreyolo dry run
  - đóng băng lớp libreyolo
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo train
    mono: true
  - label: Bắt buộc
    value: data
    mono: true
  - label: Đầu ra
    value: 'Checkpoint, chỉ số và log nằm dưới runs/train/exp'
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: >
        # coco8.yaml có sẵn trong gói và tự tải 8 ảnh của nó ở lần chạy đầu tiên

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Kiểm tra cấu hình đã phân giải trước
      language: bash
      code: >
        # In ra cấu hình mà lần chạy sẽ dùng, gồm cả mặc định của family, rồi
        thoát

        # mà không huấn luyện hay nạp dữ liệu

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Lần chạy có tên với công thức chỉ định rõ
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Cú pháp

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Tham số là các cặp `key=value`, và dạng POSIX cũng dùng được, nên `epochs=50` và
`--epochs 50` là cùng một tham số. Giá trị boolean nhận `true` và `false`:
`amp=false` trở thành `--no-amp` ở những cờ có dạng phủ định.

## Tham số

### Mô hình và dữ liệu

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `data` | | Đường dẫn tới YAML của dataset (định dạng YOLO, ví dụ `coco8.yaml`). Bắt buộc |
| `model` | `yolox-s` | Tên mô hình hoặc đường dẫn tới trọng số |
| `task` | | Ghi đè task một cách tường minh: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Dùng trọng số được huấn luyện sẵn (pretrained). `false` sẽ dựng kiến trúc và huấn luyện từ đầu |
| `allow_download_scripts` | `false` | Cho phép mã Python nhúng trong khối download của YAML dataset |

### Vòng lặp huấn luyện

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `epochs` | `300` | Số epoch huấn luyện |
| `batch` | `16` | Kích thước batch trên mỗi thiết bị |
| `imgsz` | `640` | Kích thước ảnh huấn luyện: `640` (vuông) hoặc `480x640` (HxW) |
| `device` | `auto` | Thiết bị: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Số worker của dataloader |
| `cache` | `false` | Cache ảnh để tăng tốc nạp dữ liệu: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Seed ngẫu nhiên |
| `resume` | | Tiếp tục huấn luyện: `true`, hoặc đường dẫn tới một checkpoint |
| `amp` | `true` | Automatic Mixed Precision |
| `amp_dtype` | `float16` | Kiểu dữ liệu AMP trên CUDA: `float16` hoặc `bfloat16` |
| `cuda_graph` | `false` | Thu lượt forward và backward của quá trình huấn luyện vào CUDA graph. Chỉ một GPU và các family được hỗ trợ; phần còn lại chạy ở chế độ eager |
| `lora` | `false` | Tinh chỉnh (fine-tuning) bằng LoRA, dành cho các family transformer liệt kê ở mục Ghi chú |
| `freeze` | | Đóng băng các lớp: một số nguyên đếm, một danh sách chỉ số, hoặc tên module |

### Distillation

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `distill_model` | | Teacher: một checkpoint detector, hoặc id của foundation teacher như `dinov2` để distillation đặc trưng từ backbone |
| `dis` | | Trọng số loss của distillation. Khi không đặt, dùng giá trị mặc định đã công bố cho kiểu loss tương ứng |
| `distill_loss_type` | `mgd` | Loss đặc trưng cho teacher là detector: `mgd`, `cwd`. Foundation teacher luôn dùng `feat_mse` |

### Optimizer

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `optimizer` | `sgd` | Optimizer: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Learning rate ban đầu |
| `momentum` | `0.937` | Momentum của SGD, và hệ số moment bậc nhất cho các optimizer Adam |
| `weight_decay` | `0.0005` | Điều chuẩn L2 |
| `nesterov` | `true` | Momentum Nesterov |

### Scheduler

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Kiểu lịch trình LR |
| `warmup_epochs` | `5` | Thời lượng warmup |
| `warmup_lr_start` | `0.0` | LR khởi đầu của warmup |
| `min_lr_ratio` | `0.05` | Tỉ lệ LR tối thiểu |
| `lr_drop` | `100` | Epoch giảm LR theo bậc của RF-DETR |

### Tăng cường dữ liệu

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `mosaic` | `1.0` | Xác suất mosaic |
| `mixup` | `1.0` | Xác suất mixup |
| `hsv_prob` | `1.0` | Xác suất jitter HSV |
| `flip_prob` | `0.5` | Xác suất lật ngang |
| `degrees` | `10.0` | Biên độ xoay, cộng và trừ, tính bằng độ |
| `translate` | `0.1` | Tỉ lệ tịnh tiến |
| `shear` | `2.0` | Góc xiên (shear) |
| `mosaic_scale` | `(0.1,2.0)` | Khoảng tỉ lệ của mosaic |
| `mixup_scale` | `(0.5,1.5)` | Khoảng tỉ lệ của mixup |
| `no_aug_epochs` | `15` | Tắt tăng cường dữ liệu (data augmentation) trong N epoch cuối |

### EMA

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `ema` | `true` | Exponential Moving Average |
| `ema_decay` | `0.9998` | Hệ số decay của EMA |

### Đánh giá trong lúc huấn luyện

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `val` | `true` | Đánh giá trong lúc huấn luyện |
| `eval_interval` | `10` | Đánh giá sau mỗi N epoch |
| `max_det` | `300` | Số dự đoán tối đa trên mỗi ảnh sau NMS của bước đánh giá |
| `eval_max_det` | | Giới hạn của bộ đánh giá COCO. Khi không đặt, dùng quy ước AP@100 của pycocotools |
| `faster_coco_eval` | `true` | Dùng backend C++ faster-coco-eval cho các chỉ số COCO khi đã cài; nếu không thì quay về pycocotools |
| `save_plots` | `false` | Lưu các biểu đồ đánh giá cuối cùng trong lúc huấn luyện |
| `patience` | `50` | Patience của early stopping. `0` sẽ tắt nó |

### Đầu ra

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `project` | `runs/train` | Thư mục gốc cho đầu ra |
| `name` | `exp` | Tên thí nghiệm |
| `exist_ok` | `false` | Dùng lại thư mục đầu ra đã có |
| `save_period` | `10` | Lưu checkpoint sau mỗi N epoch |
| `log_interval` | `10` | Ghi log loss sau mỗi N batch |

### Cờ dành cho agent

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Chặn stderr |
| `dry_run` | `false` | Phân giải và in cấu hình mà không thực thi |
| `help_json` | `false` | Xuất schema của lệnh dưới dạng JSON rồi thoát |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Mặc định ở trên không phải lúc nào cũng là giá trị được dùng

Mỗi dòng mô hình (family) đều mang cấu hình huấn luyện riêng, và ở đâu cấu hình
đó khác với cấu hình cơ sở, giá trị của nó sẽ thay thế mặc định của lệnh cho mọi
tham số mà bạn không đặt tường minh. Tự đặt tham số thì luôn thắng.
`libreyolo cfg` in ra các mặc định cơ sở và phần ghi đè theo từng family, đó là
cách để xem một family cụ thể thực sự sẽ dùng gì.

`imgsz` là tham số mà điều này quan trọng nhất. Mặc định của lệnh là `640`, vốn
không phải đầu vào gốc của mọi checkpoint: các kích thước phát hiện đối tượng đã
công bố của RF-DETR là 384, 512, 576 và 704, còn checkpoint YOLOX `n` và `t` là
416. RF-DETR và DEIMv2 được xử lý bằng cách chỉ chuyển tiếp `imgsz` khi nó được
đặt tường minh, nên nếu không thì kích thước riêng của chúng vẫn giữ nguyên hiệu
lực. Các family khác nhận đúng giá trị được đưa vào và huấn luyện ở kích thước
đó. FOMO là trường hợp nghiêm ngặt: mỗi kích thước chỉ chấp nhận đầu vào gốc của
nó (96, 192 và 224), nên một lần chạy FOMO cần đặt `imgsz` cho khớp, nếu không
nó sẽ dừng với lỗi. RF-DETR còn yêu cầu giá trị phải chia hết cho patch size
nhân với số lượng window của nó, và báo hai kích thước hợp lệ gần nhất khi giá
trị không thỏa.

### Những tham số mà family bỏ qua

Không phải family nào cũng đọc mọi tham số, và nhóm data augmentation là chỗ
điều đó lộ rõ nhất. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 và DINOv2 huấn
luyện qua các pipeline pass-through không có mosaic, không mixup và không biến
đổi affine, nên `mosaic`, `mixup`, `hsv_prob`, `degrees`, `translate`, `shear`,
`mosaic_scale` và `mixup_scale` không tác động tới đâu cả ở đó. EC dùng chung
pipeline đó nhưng vẫn đọc `hsv_prob`, `degrees` và `translate` khi task của nó
là pose. Các family phân loại, SegFormer và NAFNet bỏ qua toàn bộ nhóm đó cùng
với `flip_prob`, vì phép lật của chúng chạy ở một xác suất cố định chứ không
cấu hình được. YOLO-NAS chỉ bỏ qua riêng `mosaic`, vì thay vào đó nó tăng cường
bằng một phép affine trên từng mẫu luôn bật. RF-DETR bỏ qua thêm ba tham số nữa
ngoài danh sách đó: `optimizer`, `momentum` và `nesterov`.

Đặt một trong số này không phải là lỗi. Lần chạy sẽ ghi ra stderr một dòng nêu
tên family và các tham số nó sẽ bỏ qua, rồi huấn luyện, và dòng đó là danh sách
chuẩn cho phiên bản đang cài. Nó cũng là tín hiệu duy nhất, nên một lần chạy
trong script với `quiet=true` sẽ chặn luôn cảnh báo đó cùng mọi thứ khác trên
stderr.

`val=false` là một trường hợp liên quan. Nó đặt `eval_interval` về `0` cho hầu
hết các family; RF-DETR không thể tắt việc đánh giá theo cách đó và ghi log rằng
nó đã bỏ qua yêu cầu.

### Những hành vi khác đáng biết

`lora=true` được RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 và v4, EC và
ConvNeXt chấp nhận. Mọi family khác sẽ thoát với `config_unsupported` thay vì
huấn luyện mà không có nó.

`pretrained=false` kết hợp với `resume` bị từ chối ở những family có hỗ trợ huấn
luyện từ đầu, vì hai thứ này yêu cầu những điều trái ngược nhau.

`mosaic` và `mixup` là cách viết trên dòng lệnh của các trường cấu hình
`mosaic_prob` và `mixup_prob`. Ở những family mà mixup chỉ áp dụng cho các mẫu
mosaic, `mixup` lớn hơn không đi cùng `mosaic` bằng không sẽ không bao giờ kích
hoạt, và lần chạy sẽ báo như vậy.

`dry_run=true` phân giải tham chiếu mô hình, áp dụng mặc định của family, và in
ra cấu hình mà nó sẽ dùng để huấn luyện. Nó không nạp dataset, nên đây là cách
rẻ để xác nhận một tham số đã nhận đúng giá trị bạn mong đợi.

stdout mang đối tượng kết quả cuối cùng; tiến trình và cảnh báo đi ra stderr.
Mã thoát là `0` khi thành công, `2` khi lỗi cách dùng hoặc cấu hình, `3` khi
không tìm thấy hoặc không đọc được dataset, `4` khi không nạp được mô hình, và
`1` cho các lỗi runtime khác.

Liên quan: [`libreyolo doctor`](/docs/cli/doctor) để kiểm tra dataset trước khi
bắt tay vào một lần chạy, [`libreyolo monitor`](/docs/cli/monitor) để theo dõi
lần chạy trên trình duyệt, [`libreyolo val`](/docs/cli/val) để đo kết quả.
