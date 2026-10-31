---
title: Nâng cấp lên 1.7.0
seo_title: Nâng cấp LibreYOLO 1.6.0 lên 1.7.0
description: >-
  Các bước chuyển đổi cho checkpoint, tên module và công thức huấn luyện YOLO9,
  thư mục chạy của CLI, confusion matrix và các lỗi mới trong LibreYOLO 1.7.0.
lead: >-
  Phiên bản 1.7.0 vẫn nạp được các checkpoint YOLO9 cũ với cùng box dạng float.
  Nó thay đổi huấn luyện họ YOLO9, các tên mà mã nhìn thấy bên trong mạng, và
  một vài hợp đồng của validation và predict.
keywords:
  - nâng cấp libreyolo
  - chuyển sang libreyolo 1.6.0
  - thay đổi không tương thích libreyolo
  - tên module yolo9
  - clip_max_norm
  - confusion matrix
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---
## 1.6.0 lên 1.7.0

- Checkpoint YOLO9, YOLO9-E2E và YOLO9-P2 do 1.7.0 lưu dùng tên tensor mới và không nạp được trong 1.6.x: 1.6.x nạp checkpoint YOLO9 và YOLO9-P2 kèm cảnh báo thiếu khóa rồi không dự đoán ra box nào, và phát sinh lỗi không khớp `model_family` với checkpoint YOLO9-E2E. Hãy nâng cấp mọi môi trường đọc chúng. Checkpoint float, lượng tử hóa và huấn luyện do các bản phát hành trước ghi vẫn nạp được trong 1.7.0, và bản xuất ONNX không bị ảnh hưởng.

- Mã truy cập vào `model.model` của một mô hình họ YOLO9 sẽ thấy tên module của MultimediaTechLab/YOLO, không có alias. `DDetect` là `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` là `RepConv`, `RepNBottleneck` là `Bottleneck`, `DFL` là `Anchor2Vec` (`anchor2vec`), `nc` là `num_classes`, và các lớp con của block là `conv1`/`conv2`/`bottleneck`. Trong YOLO9-E2E, `YOLO9E2EDetect` là `YOLO9E2EHead` và `one2one_cv2` / `one2one_cv3` là `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat` và `yolo9.utils.decode_boxes` đã bị xóa. [Changelog](/docs/changelog) liệt kê mọi lần đổi tên.

- Tên node ONNX theo tên module mới trong head, backbone và neck, chẳng hạn `/head/anchor_convs.0/...`. Hãy cập nhật các công cụ dựa vào tên node, như cấu hình end-node của Hailo. Đầu vào, đầu ra và tên đầu ra của mô hình không đổi.

- Một lần tinh chỉnh YOLO9 từ trọng số không có tensor PGI, bao gồm `LibreYOLO9{t,s,m,c}.pt` đã công bố, chỉ huấn luyện head chính, như trong 1.5.0; 1.6.0 đã gắn một nhánh PGI khởi tạo ngẫu nhiên ở `aux_weight=0.25`. Trọng số có chứa tensor PGI, và các lần chạy từ đầu, vẫn huấn luyện với PGI. Truyền `aux_weight=0.25` để vẫn gắn một nhánh mới, hoặc `aux_weight=0` để buộc tắt nó.

- Khi PGI được gắn, YOLO9-m và YOLO9-c dựng nhánh phụ của MultimediaTechLab: 12.8M (m) và 25.7M (c) tham số chỉ dùng khi huấn luyện ở 80 lớp đối tượng, tăng từ 7.5M và 10.2M, nên huấn luyện PGI ở các kích thước này cần nhiều bộ nhớ và thời gian hơn cho mỗi bước. Một checkpoint huấn luyện m/c của 1.6.0 tiếp tục và tinh chỉnh với nhánh mà nó đã được huấn luyện. Inference và YOLO9-t/s không đổi.

- YOLO9, YOLO9-E2E và YOLO9-P2 cắt chuẩn L2 của gradient ở 10, và bộ gán trong huấn luyện của chúng chấm điểm anchor ở fp32, nên các mẫu dương được chọn dưới fp16 AMP có thể khác. Truyền `clip_max_norm=0` để có cách cắt của 1.6.0; một lần chạy bắt đầu trong 1.6.0 và tiếp tục trong 1.7.0 cũng cắt trừ khi bạn truyền tham số này. Việc gán ở fp32 không có công tắc.

- Một giá trị `perspective` khác không làm méo theo độ dịch góc ở mọi họ dùng phép affine warp dùng chung (YOLO9 cùng các biến thể E2E và P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR và RT-DETRv2), kể cả các lần chạy tiếp tục từ checkpoint 1.6.0 có đặt giá trị này. Các giá trị từ `0.002` trở lên cho cùng mức méo tối đa, và giá trị âm phát sinh `ValueError`. Không thiết lập nào tái tạo được phép méo của 1.6.0; mặc định `perspective=0` không đổi.

- Pipeline mosaic của YOLO9 lấy tỉ lệ mixup từ Beta(1, 1) thay vì Beta(32, 32), không có thiết lập nào cho giá trị cũ. Huấn luyện YOLOv7 mặc định thay đổi, vì YOLOv7 bật mixup theo mặc định. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR và RT-DETRv2 chỉ thay đổi khi mixup được bật.

- `libreyolo train` không có `name=` ghi các lần chạy EC, D-FINE, SegFormer và LingBot-Vision vào `runs/train/ec_exp`, `dfine_exp`, `segformer_exp` và `lingbotvision_exp` thay vì `runs/train/exp`, các probe V-JEPA 2 vào `runs/train/train`, và các lần chạy YOLO9-P2 vào `runs/train/yolo9_exp`. Truyền `name=exp` để dùng đường dẫn cũ, hoặc `name=yolo9_p2_exp` cho YOLO9-P2.

- `libreyolo train` khởi động D-FINE, SegFormer và LingBot-Vision từ công thức riêng của chúng. Với D-FINE đó là 132 epoch, `lr0=2e-4`, `weight_decay=1e-4`, 2 epoch warmup và không có tăng cường mosaic, mixup, HSV hay affine. Để gửi các giá trị mà 1.6.0 đã gửi, truyền `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` lập chỉ mục `matrix[predicted, true]` (trước đây là `[actual, predicted]`), đếm các detection ở `conf_thres=0.25` (trước đây là 0.15), chỉ nhận `names=` và `task=` dưới dạng đối số từ khóa, và `matrix` của nó không còn gán lại được. Hãy chuyển vị mã đã lập chỉ mục ma trận cũ, và truyền `conf_thres=0.15` để có số đếm cũ. `plots=True` ghi số đếm vào `confusion_matrix.png` và biểu đồ đã chuẩn hóa vào `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` loại bỏ các box chồng lấn xuyên các lớp đối tượng thay vì phát cảnh báo. Các tác vụ không có box phát sinh `ValueError`, `LibreEnsemble` và HRNet phát sinh `NotImplementedError`, và tầng SAM từ chối từ khóa này bằng `TypeError`; hãy bỏ đối số khỏi các lời gọi đó.

- Một `interpolation` của bộ phân loại không xác định phát sinh `ValueError`, kể cả các chế độ torchvision `LANCZOS`, `BOX`, `HAMMING` và `NEAREST_EXACT`. Hãy dùng `bilinear`, `bicubic`, `nearest` hoặc `bilinear_no_antialias`.

- Hãy chạy lại validation trước khi so sánh với số liệu của 1.6.0 cho `val(single_cls=True)` trên một mô hình được huấn luyện với nhiều lớp đối tượng, các dataset phân đoạn trộn lẫn hàng chỉ có box và hàng polygon, và RTMDet-Ins, vốn giữ tối đa `max_det=300` mask mỗi ảnh theo mặc định; truyền `max_det=100` để có thiết lập của MMDetection.

- Khi không có `libreyolo[rfdetr]`, các lời gọi CLI với tên mô hình RF-DETR hoặc DINOv2 thoát với mã lỗi JSON `model_load_failed` thay vì `model_not_found`, và thông báo nêu tên `pip install libreyolo[rfdetr]`.

Xem [changelog](/docs/changelog) để biết toàn bộ bản phát hành.

## 1.5.0 lên 1.6.0

- Nâng môi trường đang ghim dưới ONNX Runtime 1.18 lên `onnxruntime>=1.18.0` trước khi cài extra ONNX.

- SAM 3D Body chấp nhận bộ tài nguyên snapshot đã được rà soát. Cài `libreyolo[hf]`, xin quyền truy cập và dùng tải tự động, hoặc cung cấp thư mục snapshot nguyên vẹn và tài nguyên MHR đã ghim trên hệ thống tệp cục bộ đáng tin cậy.

- Chạy lại đánh giá sau các sửa đổi đổi kích thước RF-DETR ngoài tác vụ tư thế và chuẩn hóa bộ phân loại. Kiểm tra lại ngưỡng độ tin cậy đã triển khai trước khi so sánh với kết quả 1.5.0; không có cờ tiền xử lý cũ.

- Phát hiện D-FINE, DEIM, RT-DETRv4 và YOLO-NAS mặc định bật FP16 AMP. Truyền `amp=False` để giữ FP32.

- Để dùng các lựa chọn tinh chỉnh YOLO9 trước đây, đặt `aux_weight=0`, `max_labels=100` và `warmup_momentum=0.937`. Đặt `letterbox_pad="topleft"` khi bản chuyển đổi mới có dấu center phải tái tạo hình học cũ. Checkpoint một head cũ tiếp tục với đồ thị gốc. Các lần chạy YOLO9 mới dùng nhiều bộ nhớ GPU hơn, nên hãy giảm `batch` nếu kích thước batch của 1.5.0 bị hết bộ nhớ.

- RF-DETR và DINOv2 tạo thư mục chạy mang tên họ mô hình với hậu tố tăng dần. Cập nhật nơi sử dụng đường dẫn tệp đầu ra, hoặc đặt `output_dir="runs/train", exist_ok=True` để giữ vị trí và hành vi tái sử dụng cũ.

- Giữ `mixup + cutmix <= 1` khi phân loại; tổ hợp không hợp lệ nay phát sinh lỗi trong thiết lập.

- QAT tắt EMA, SyncBatchNorm và lấy trung bình checkpoint. Dùng checkpoint best/last của QAT mà không dựa vào các trạng thái đó.

- Loader tùy chỉnh có hook thay đổi dataset phải dùng `persistent_workers=False` hoặc dựng lại worker sau thay đổi. Các bản sao nhiều worker duy trì liên tục không tương thích nay phát sinh lỗi.

- Mảng ảnh NumPy được đọc là BGR, thứ tự của OpenCV. Truyền `color_format="rgb"` ở những chỗ bạn truyền một mảng RGB như `np.asarray(pil_image)`; đầu ra của `cv2.imread()` và các frame video không cần thay đổi.

- Mảng NumPy hoặc tensor 4D là một batch: `predict()` trả về một danh sách gồm một `Results` cho mỗi ảnh thay vì chỉ dùng ảnh đầu tiên.

- `train(resume=True)` và `train(resume="path/to/last.pt")` khôi phục các đối số huấn luyện đã lưu của lần chạy và tiếp tục ghi vào thư mục chạy của nó; đối số bạn truyền tường minh được ưu tiên. Tiếp tục từ trọng số đã phát hành, hoặc từ một lần chạy đã đạt đủ `epochs`, sẽ phát sinh `ValueError`.

- Khi bật đánh giá, epoch cuối cùng luôn được đánh giá, nên các lần chạy ngắn hơn `eval_interval` nay báo cáo chỉ số và ghi ra `best.pt`. `val=False` tắt đánh giá, kể cả ở epoch cuối. Đánh giá trong lúc huấn luyện ghi vào `<run>/val` thay vì `runs/val/`.

- Lỗi của lệnh CLI `train` báo loại lỗi của chúng, nên lỗi cấu hình thoát với mã 2 thay vì mã 1 kèm `io_error`.

Xem [nhật ký thay đổi](/docs/changelog) để biết toàn bộ bản phát hành và [nhập trọng số](/docs/migrate) để chuyển đổi checkpoint.

## 1.4.0 lên 1.5.0

Trang này nói về nâng cấp chính LibreYOLO. Nếu bạn đang tìm cách nạp checkpoint
từ một dự án upstream, hãy xem [nhập trọng số có sẵn](/docs/migrate), đây là
chủ đề khác.

Nội dung phát hành đầy đủ nằm trong [changelog](/docs/changelog). Phần sau chỉ
trình bày những thay đổi đòi hỏi bạn thực hiện thao tác.

### Những thay đổi mã bạn phải thực hiện

#### `allow_experimental=True` không còn tồn tại

Cổng xác nhận đã bị xóa cùng cơ chế `ddp_aware(experimental_key=...)` đứng sau.
Trước đây, huấn luyện và xuất EC, RTMDet, PicoDet và FOMO cần đối số này, nên
mọi script huấn luyện một trong các họ đó đều bị ảnh hưởng.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: xóa đối số
model.train(data="data.yaml", epochs=100)
```

Không có lớp tương thích deprecation. Lệnh gọi vẫn truyền đối số sẽ nhận cảnh
báo `Unknown training config keys (ignored)`, và đối số đó không có tác dụng.
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` cũng bị xóa theo. Hook
`get_download_notice()` vẫn còn và vẫn được MiDaS, SegFormer cùng YOLO9-P2
override.

Các mức hỗ trợ vẫn được công bố, chúng chỉ không còn là một đối số: xem [cấp
ổn định](/docs/reference/stability-tiers).

#### Cấp xuất `"experimental"` không còn tồn tại

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Mã rẽ nhánh theo chuỗi cấp phải đọc `"available"` tại nơi trước đây đọc
`"experimental"`. `BaseExporter` không còn phát `RuntimeWarning` cho các định
dạng này. Trạng thái theo từng định dạng được liệt kê trong [ma trận
xuất](/docs/reference/export-matrix).

#### `pretrained=False` cùng `resume` giờ bị từ chối

Trước đây tổ hợp này tiếp tục trong trạng thái không nhất quán. Giờ nó phát:

```
ValueError: pretrained=False cannot be combined with resume.
```

Hãy chọn một. `pretrained=False` bắt đầu từ khởi tạo mới có seed, trong 1.5.0
hoạt động với mọi họ có thể huấn luyện thay vì chỉ ba họ, còn `resume` tiếp tục
lần chạy bị gián đoạn từ checkpoint. Cả hai được ghi lại trong phần [huấn
luyện](/docs/train).

#### `--imgsz` của CLI là chuỗi, không phải số nguyên

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

### Các con số thay đổi

Ba thay đổi làm dịch chuyển metric ở thiết lập mặc định. Nếu bạn theo dõi kết
quả qua nhiều phiên bản, hãy đọc phần này trước khi so sánh lần chạy 1.5.0 với
1.4.0.

#### faster-coco-eval là backend metric COCO mặc định

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

#### Checkpoint YOLOX huấn luyện trước 1.5.0 cần override eps

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
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Hoặc gộp `sqrt((var + 1e-3) / (var + 1e-5))` vào trọng số BN một lần rồi lưu
kết quả. Checkpoint huấn luyện trên 1.5.0 trở lên không cần cả hai cách này.

#### Huấn luyện multi-scale D-FINE dùng công thức theo kích thước từ upstream

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

### Những điều nên biết, không cần hành động

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
  nhận sẽ bị bỏ qua thay vì phát `TypeError`. Tên cờ không xác định vẫn bị từ
  chối với `No such option`.
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

### Checkpoint và dataset

Checkpoint được ghi bởi 1.4.0 vẫn nạp không thay đổi.
[Schema](/docs/reference/checkpoint-schema) có thêm `imgsz_h` và `imgsz_w` cho
mô hình chữ nhật, đồng thời vẫn ghi scalar `imgsz = max(h, w)` cho reader cũ.
Bản xuất [ExecuTorch](/docs/export/executorch) và [MNN](/docs/export/mnn) giờ
yêu cầu sidecar, lần lượt là `<program>.pte.json` và `<model>.mnn.json`, còn
bản xuất HRNet mang `pose_input: "person_crop"`. Định dạng dataset không đổi.
