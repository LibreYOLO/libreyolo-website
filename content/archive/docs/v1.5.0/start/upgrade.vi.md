---
title: Nâng cấp lên 1.5.0
seo_title: Nâng cấp LibreYOLO 1.4.0 lên 1.5.0
description: >-
  Bốn thay đổi mã mà 1.5.0 yêu cầu, ba thay đổi làm dịch chuyển metric và những
  thay đổi hành vi nhỏ cần biết trước khi so sánh các lần chạy.
lead: >-
  Không có gì bị xóa khỏi model API công khai: mọi lớp và hàm hoạt động trong
  1.4.0 vẫn import được. Bốn đối số đổi dạng và ba giá trị mặc định làm thay đổi
  con số bạn có thể đang so sánh.
keywords:
  - nâng cấp libreyolo
  - chuyển sang libreyolo 1.5.0
  - allow_experimental bị xóa
  - thay đổi không tương thích libreyolo
  - yolox bn eps
  - faster coco eval mặc định
last_verified: 1.5.0
meta:
  - label: Áp dụng cho
    value: 1.4.0 đến 1.5.0
  - label: Thay đổi mã bắt buộc
    value: 'Bốn thay đổi, đều có phạm vi hẹp'
  - label: Kết quả thay đổi
    value: 'Backend COCO, BN eps YOLOX, multi-scale D-FINE'
  - label: Thành phần API công khai bị xóa
    value: Không có
source_hash: ab38d8ef7b53f596
---

Trang này nói về nâng cấp chính LibreYOLO. Nếu bạn đang tìm cách nạp checkpoint
từ một dự án upstream, hãy xem [nhập trọng số có sẵn](/docs/migrate), đây là
chủ đề khác.

Nội dung phát hành đầy đủ nằm trong [changelog](/docs/changelog). Phần sau chỉ
trình bày những thay đổi đòi hỏi bạn thực hiện thao tác.

## Những thay đổi mã bạn phải thực hiện

### `allow_experimental=True` không còn tồn tại

Cổng xác nhận đã bị xóa cùng cơ chế `ddp_aware(experimental_key=...)` đứng sau.
Trước đây, huấn luyện và xuất EC, RTMDet, PicoDet và FOMO cần đối số này, nên
mọi script huấn luyện một trong các họ đó đều bị ảnh hưởng.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: xóa đối số
model.train(data="data.yaml", epochs=100)
```

Không có lớp tương thích deprecation. Lệnh gọi vẫn truyền đối số sẽ phát
`TypeError`. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` cũng bị xóa theo. Hook
`get_download_notice()` vẫn còn và vẫn được MiDaS, SegFormer cùng YOLO9-P2
override.

Các mức hỗ trợ vẫn được công bố, chúng chỉ không còn là một đối số: xem [cấp
ổn định](/docs/reference/stability-tiers).

### Cấp xuất `"experimental"` không còn tồn tại

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Mã rẽ nhánh theo chuỗi cấp phải đọc `"available"` tại nơi trước đây đọc
`"experimental"`. `BaseExporter` không còn phát `RuntimeWarning` cho các định
dạng này. Trạng thái theo từng định dạng được liệt kê trong [ma trận
xuất](/docs/reference/export-matrix).

### `pretrained=False` cùng `resume` giờ bị từ chối

Trước đây tổ hợp này tiếp tục trong trạng thái không nhất quán. Giờ nó phát:

```
ValueError: pretrained=False cannot be combined with resume.
```

Hãy chọn một. `pretrained=False` bắt đầu từ khởi tạo mới có seed, trong 1.5.0
hoạt động với mọi họ có thể huấn luyện thay vì chỉ ba họ, còn `resume` tiếp tục
lần chạy bị gián đoạn từ checkpoint. Cả hai được ghi lại trong phần [huấn
luyện](/docs/train).

### `--imgsz` của CLI là chuỗi, không phải số nguyên

Phạm vi ảnh hưởng hẹp hơn vẻ ngoài. Hai cách sau không bị ảnh hưởng:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # vẫn hoạt động
```

```python
model.predict("img.jpg", imgsz=640)   # vẫn hoạt động
```

Chỉ mã gọi trực tiếp các hàm lệnh [CLI](/docs/cli) từ Python cần thay đổi, vì
`predict`, `train` và `val` đã mở rộng `--imgsz` từ `int` sang `str` để nhận
kích thước chữ nhật:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, và "480x640" giờ cũng hoạt động
```

Giá trị mặc định của `train` giờ là chuỗi `"640"`. `export --imgsz` vốn đã là
chuỗi và `profile` không đổi.

## Các con số thay đổi

Ba thay đổi làm dịch chuyển metric ở thiết lập mặc định. Nếu bạn theo dõi kết
quả qua nhiều phiên bản, hãy đọc phần này trước khi so sánh lần chạy 1.5.0 với
1.4.0.

### faster-coco-eval là backend metric COCO mặc định

`val()` và xác thực huấn luyện theo epoch giờ tính metric COCO bằng backend C++
faster-coco-eval thay vì pycocotools.

Quyết định chuyển đổi dựa trên parity đo được trên toàn bộ 100 tập test
RF100-VL: 1381 trong 1400 giá trị metric giống hệt từng bit, độ lệch tối đa
2.22e-16, chênh lệch metric chính bằng đúng 0, tổng thể nhanh hơn 15.6 lần và
nhanh hơn 56 lần trên dataset dày đặc detection. Các con số của bạn không nên
thay đổi. Dù vậy, chúng được tạo bởi bản triển khai khác, đó là lý do thay đổi
này có trong danh sách.

pycocotools vẫn là fallback tự động khi faster-coco-eval chưa được cài. Để ép
dùng nó:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` thực hiện cùng việc trên toàn cục. Backend thực
sự được dùng được log ở mức INFO, xuất hiện dưới `model.last_eval_backend` sau
`val()` và nằm trong trường `eval_backend` của payload JSON
[CLI](/docs/cli/val). Cài pipeline nhanh bằng `pip install
libreyolo[fast-eval]`.

### Checkpoint YOLOX huấn luyện trước 1.5.0 cần override eps

Đây là điểm dễ mắc lỗi trong bản phát hành. Hãy đọc nếu bạn đã tinh chỉnh
[YOLOX](/docs/models/yolox).

YOLOX quy định BatchNorm `eps=1e-3` và `momentum=0.03`. Trước 1.5.0, các giá
trị này được áp dụng như bước sửa sau cùng và không tồn tại qua quá trình dựng
lại số lớp mà `train()` thực hiện khi `nc` của dataset khác checkpoint. Bản
tinh chỉnh như vậy được huấn luyện và báo xác thực trong lúc huấn luyện với
`eps=1e-5` mặc định của torch, rồi được nạp lại cho inference với `1e-3`: cùng
tensor nhưng normalization khác nhau.

Các kích thước regular-conv gần như không đổi. Bản depthwise `n` thay đổi lớn
vì `running_var` theo kênh đủ nhỏ để eps chi phối. Trên RF100-VL `ball`, cùng
checkpoint nano đạt **0.566** mAP50-95 khi đánh giá ở eps dùng lúc huấn luyện và
**0.151** sau khi nạp lại theo mặc định.

Checkpoint huấn luyện trước 1.5.0 mang ngữ nghĩa eps=1e-5. Để báo con số trung
thực cho checkpoint đó, hãy đánh giá với BN eps được override thành 1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Hoặc gộp `sqrt((var + 1e-3) / (var + 1e-5))` vào trọng số BN một lần rồi lưu
kết quả. Checkpoint huấn luyện trên 1.5.0 trở lên không cần cả hai cách này.

### Huấn luyện multi-scale D-FINE dùng công thức theo kích thước từ upstream

`base_size_repeat` từng được cố định là 3 cho mọi kích thước. Giờ nó phân giải
theo kích thước như upstream chỉ định: **n** huấn luyện ở kích thước cố định và
tắt multi-scale, **s** 20, **m** 6, **l** 4, **x** 3. Trước đây chỉ x khớp, nên
n, s, m và l thấy phân phối scale khác và hội tụ đến metric khác.

Để khôi phục hành vi cũ, hãy đặt tường minh:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM vẫn dùng giá trị cố định 3. Chi tiết về họ nằm tại
[D-FINE](/docs/models/d-fine).

## Những điều nên biết, không cần hành động

- **Kết quả `imgsz` chữ nhật thay đổi vì trước đây chúng bị sai.** Tọa độ box,
  đổi kích thước mask RTMDet, rescale YOLO-NAS và scale ground truth của
  validator giờ dùng riêng chiều cao và chiều rộng thay vì một scalar. `imgsz`
  vuông không đổi từng bit. Inference hoặc xác thực chữ nhật chạy trên 1.4.0 bị
  scale sai. YOLO-NAS giờ từ chối thẳng `imgsz` chữ nhật thay vì âm thầm tạo
  đầu ra sai.
- **Dictionary metric có thêm khóa.** `max_det`, `ar_max_det` và `AR_max_det`
  từ evaluator COCO, cùng `metrics/loss` và `metrics/loss/ce` từ FOMO. Giá trị
  mặc định không đổi, nhưng mọi mã duyệt khóa metric, gồm [logger tùy
  chỉnh](/docs/train/loggers) và header CSV, sẽ thấy cột mới.
- **Các lần chạy YOLO9 có seed kích hoạt dựng lại head** bắt đầu từ khởi tạo
  khác, vì seed giờ được áp dụng trước bước dựng lại thay vì sau. Bản tinh chỉnh
  1.4.0 có seed sang số lớp khác không thể tái lập từng bit trên 1.5.0.
- **`libreyolo[hub-kernels]` trên CUDA giờ thực sự dùng kernel
  MS-deform-attn native.** 1.4.0 đặt nó sau điều kiện mà RF-DETR không bao giờ
  đi qua nên kernel chưa từng chạy. Dự đoán có thể dịch chuyển trong sai số
  float với RF-DETR và các họ deformable-attention khác. Bản cài thông thường
  không bị ảnh hưởng, còn `LIBREYOLO_HUB_KERNELS=0` sẽ tắt nó.
- **`libreyolo predict` bỏ tùy chọn không được hỗ trợ thay vì báo lỗi.** CLI
  lọc kwarg theo signature `__call__` của mô hình, nên tùy chọn mà một họ không
  nhận sẽ bị bỏ qua thay vì phát `TypeError`. Lỗi chính tả trong tên cờ giờ bị
  âm thầm bỏ qua.
- **Source trực tiếp thay đổi dạng đầu ra JSON.** Webcam, stream RTSP và chụp
  màn hình ngầm bật streaming, phát một bản ghi mỗi frame thay vì một bản ghi
  cho cả lệnh gọi. Các [source](/docs/predict/sources) này mới trong 1.5.0 nên
  không script 1.4.0 nào bị ảnh hưởng.
- **Xuất lại `rfdetr-pose` hoặc `yolonas-pose` sang ONNX tạo tên đầu ra khác.**
  1.4.0 đọc nhầm head pose nhiều tensor của chúng thành segmentation qua
  heuristic dựa trên số đầu ra. File `.onnx` hiện có trên ổ đĩa không đổi.
- **Trên bản cài không có torch**, kết quả chứa mảng numpy thay vì
  `torch.Tensor`, nên `.boxes.data` trả về kiểu khác và cách phân xử hòa của NMS
  có thể khác torchvision. Khi torch được cài, hành vi không đổi từng byte. Xem
  [cài đặt gọn nhẹ](/docs/lightweight-install).
- **Object cấu hình xác thực nhiều hơn lúc khởi tạo.** `TrainConfig` có thêm
  `__post_init__` tại nơi trước đây không có, nên cấu hình vốn đã không hợp lệ
  giờ báo lỗi ngay thay vì thất bại sâu trong lần chạy. Serialization
  `ValidationConfig` có thêm khóa `edge_thresholds`, làm hỏng round-trip nghiêm
  ngặt `ValidationConfig(**dump)` từ dump 1.4.0.
- **Tên file trọng số cho họ có hậu tố tác vụ được phân giải khác.**
  `segformer-b0` giờ phân giải thành `LibreSegformerb0-sem.pt`. Điều này sửa lỗi
  404 khi tự động tải và làm hỏng script đã hardcode tên cũ không có hậu tố.
- **Pytest marker `experimental_backend` giờ là `extended_backend`.** Chỉ liên
  quan nếu bạn chạy bộ kiểm thử với `-m`.

## Checkpoint và dataset

Checkpoint được ghi bởi 1.4.0 vẫn nạp không thay đổi.
[Schema](/docs/reference/checkpoint-schema) có thêm `imgsz_h` và `imgsz_w` cho
mô hình chữ nhật, đồng thời vẫn ghi scalar `imgsz = max(h, w)` cho reader cũ.
Bản xuất [ExecuTorch](/docs/export/executorch) và [MNN](/docs/export/mnn) giờ
yêu cầu sidecar, lần lượt là `<program>.pte.json` và `<model>.mnn.json`, còn
bản xuất HRNet mang `pose_input: "person_crop"`. Định dạng dataset không đổi.
