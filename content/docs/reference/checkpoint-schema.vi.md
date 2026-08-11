---
title: Lược đồ checkpoint
seo_title: Lược đồ metadata checkpoint LibreYOLO v1.0
description: >-
  Metadata có trong mọi checkpoint .pt của LibreYOLO: các khóa bắt buộc, phần bổ
  sung theo tác vụ, khóa runtime khi xuất, manifest lượng tử hóa (quantization)
  và các trường huấn luyện.
lead: >-
  Tệp .pt của LibreYOLO là một dictionary phẳng được lưu bằng torch.save. Khóa
  model chứa state dict; các khóa cấp cao nhất khác là metadata dùng để xác định
  checkpoint mà không cần phân tích tên tệp hay dò xét state dict.
keywords:
  - lược đồ checkpoint libreyolo
  - schema_version 1.0
  - model_family
  - metadata checkpoint libreyolo
  - manifest quantization
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Phản ánh docs/checkpoint_schema.md trong repo libreyolo tại v1.5.0, được đối
  chiếu với libreyolo/utils/serialization.py và BaseModel.save.
snippets:
  usage:
    - label: Đọc metadata từ checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Tải checkpoint xuống, rồi lưu lại để có đường dẫn cục bộ

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Lược đồ v1.0

Mọi checkpoint `.pt` chính thức của LibreYOLO đều chứa:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Khóa | Kiểu | Ý nghĩa |
|---|---|---|
| `model` | state dict | Trọng số của mô hình |
| `schema_version` | str | Phiên bản hợp đồng metadata; v1.0 dùng chuỗi `"1.0"` |
| `libreyolo_version` | str | Phiên bản đã tạo ra checkpoint |
| `model_family` | str | Một họ đã đăng ký, chẳng hạn `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Biến thể trong họ, chẳng hạn `t`, `s`, `r18`, `atto` |
| `task` | str | Tên tác vụ chuẩn |
| `nc` | int | Số lớp đối tượng dương |
| `names` | dict | `dict[int, str]` với các khóa trong `0..nc-1` |
| `imgsz` | int | Độ phân giải đầu vào hình vuông dương, hoặc giá trị vô hướng cũ cho một hợp đồng hình chữ nhật |

`task` là một trong các giá trị `detect`, `segment`, `semantic`, `panoptic`, `pose`,
`classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` hoặc `mesh`.

Các checkpoint chính thức ghi mọi khóa `names`. Trình đọc có thể bổ sung các khóa
bị thiếu bằng nhãn `class_i` cho những ánh xạ thưa cũ, nhưng khóa nằm ngoài phạm vi
là không hợp lệ.

Checkpoint hình chữ nhật giữ một giá trị vô hướng `imgsz` cho trình đọc cũ, được đặt
thành `max(imgsz_h, imgsz_w)`, đồng thời ghi thêm `imgsz_h` và `imgsz_w` với
kích thước thực. Trình đọc hiểu các trường hình chữ nhật phải ưu tiên chúng
thay vì giá trị vô hướng. Các họ có hợp đồng hình chữ nhật cố định, chẳng hạn
HRNet pose, sẽ từ chối kích thước runtime không tương thích.

Lược đồ được thiết kế dạng phẳng, và `model` được thiết kế là một state dict.

<code-tabs name="usage" />

## Phần bổ sung cho pose

Pose thường chỉ có một lớp đối tượng, `nc: 1` với `person`, nhưng head pose của
YOLO-NAS cũng hỗ trợ pose nhiều lớp đối tượng với một bộ khung keypoint dùng chung.
Trong trường hợp đó, `nc` và `names` mô tả các lớp đối tượng như trong phát hiện.
Bản xuất runtime cho pose phát ra `scores` với shape `[batch, anchors, nc]`.

| Khóa | Ý nghĩa |
|---|---|
| `num_keypoints` | Số keypoint dương được head pose sử dụng |
| `keypoint_dim` | `2` cho nhãn `x,y` hoặc `3` cho nhãn `x,y,visibility`; đầu ra của mô hình luôn cung cấp `x,y,visibility` |
| `oks_sigmas` | Các sigma OKS tùy chọn theo từng keypoint; khi không có, giá trị mặc định của tác vụ cho `num_keypoints` được sử dụng |
| `num_keypoints_per_class` | Số keypoint tùy chọn theo từng lớp đối tượng cho các head kiểu GroupPose có tensor keypoint được đệm theo lớp đối tượng; `0` cho các lớp đối tượng không có keypoint |

## Phần bổ sung cho mesh

Checkpoint mesh dùng `task: "mesh"`, `nc: 1` và `names: {0: "person"}`.
Bố cục tham số khác nhau giữa các mô hình cơ thể, vì vậy các chiều được ghi lại
thay vì giả định.

| Khóa | Ý nghĩa |
|---|---|
| `body_model` | Cách tham số hóa, chẳng hạn `mhr`; bắt buộc và được dùng để diễn giải mọi trường bên dưới |
| `num_betas` | Số hệ số nhận dạng và hình dạng; 45 đối với MHR |
| `num_body_pose` | Độ rộng của khối tham số pose cơ thể; 130 đối với MHR. Đây là một vector phẳng, không phải một bộ ba cho mỗi khớp, vì các khớp rig có bậc tự do khác nhau |
| `num_vertices` | Số đỉnh mà decoder phát ra; 18439 đối với MHR |
| `num_joints` | Số khớp mà decoder phát ra; 127 đối với MHR |
| `rotation_format` | Cách mã hóa phép xoay, chẳng hạn `euler_zyx` cho MHR hoặc `axis_angle`. Không bao giờ suy ra từ shape tensor, vì một vector 3 chiều có thể mang nhiều nghĩa |

## Giá trị giữ chỗ cho tác vụ dense

Một số tác vụ dự đoán các map dense thay vì lớp đối tượng, vì vậy những vị trí
giống lớp đối tượng chỉ tồn tại để tương thích với lược đồ.

| Tác vụ | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Dự đoán edge là các map xác suất float32 dense trong `[0, 1]`.

Checkpoint restore có thể thêm `degradation`, một nhãn ngắn cho dạng suy giảm như
`deblur`, `denoise` hoặc `super-resolution`; `dataset`, một nhãn nguồn gốc như
`GoPro` hoặc `SIDD`; và `scale`, hệ số phóng to dương nguyên từ đầu vào đến đầu ra,
ví dụ `4` cho mô hình super-resolution x4. Nếu không có hoặc bằng `1`, ảnh được
phục hồi giữ nguyên độ phân giải đầu vào. Runtime cũng suy ra scale từ họ và kích
thước, vì vậy `scale` là metadata nguồn gốc chứ không phải yêu cầu khi tải.

## Phần bổ sung cho OCR

Họ `ppocr` cung cấp một checkpoint tổng hợp cho mỗi cấp, trong đó state dict của
`model` chứa hai mô hình con dưới các namespace khóa `det.*` và `rec.*`.

| Khóa | Ý nghĩa |
|---|---|
| `charset` | Toàn bộ bảng chữ cái CTC theo thứ tự chỉ mục đầu ra: chỉ mục 0 là ký hiệu trống CTC, tiếp theo là dictionary nhận dạng, rồi đến ký tự khoảng trắng. Trình tải phải đọc giá trị này từ checkpoint, không bao giờ từ tệp phụ |
| `pipeline` | Các giá trị mặc định của pipeline được ghi cố định tại thời điểm chuyển đổi: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Đối số runtime có thể ghi đè chúng cho từng lần gọi |
| `components` | Dành riêng cho các giai đoạn pipeline tùy chọn như định hướng tài liệu, khử biến dạng và xoay dòng văn bản. Trống trong v1 |

## Metadata runtime khi xuất

Các tạo tác đã xuất dùng cùng quy ước ghi kép cho hình chữ nhật: `imgsz_h`
và `imgsz_w` được ghi bên cạnh giá trị vô hướng cũ `imgsz`, và trình đọc không
hiểu các trường hình chữ nhật không được âm thầm coi giá trị vô hướng là một
hợp đồng hình vuông.

Hỗ trợ runtime hình chữ nhật phụ thuộc vào họ và định dạng. Bản xuất của họ YOLO9,
HRNet, NAFNet và Real-ESRGAN có thể dùng `imgsz_h` và `imgsz_w` không vuông trong
các định dạng được hỗ trợ; những họ hoặc định dạng không có hỗ trợ hình chữ nhật
tường minh sẽ từ chối metadata thay vì tiền xử lý các tạo tác đó thành hình vuông.
Bản xuất HRNet là các head FP32, batch một, đầu vào là vùng cắt người cố định,
trong đó W32 chấp nhận 256x192 và W48 chấp nhận 384x288, còn detector người không
được nhúng trong graph.

Bản xuất có NMS nhúng có thể thêm các khóa phẳng sau:

| Khóa | Ý nghĩa |
|---|---|
| `nms` | Boolean dạng chuỗi; `"true"` nghĩa là graph có chứa đầu ra hậu xử lý được nhúng |
| `nms_conf` | Ngưỡng độ tin cậy được ghi cố định vào đầu ra nhúng |
| `nms_iou` | Ngưỡng IoU được ghi cố định vào đầu ra nhúng |
| `max_det` | Số hàng phát hiện tối đa sau NMS mà đầu ra nhúng phát ra |
| `nms_raw_output` | Boolean dạng chuỗi; `"true"` nghĩa là graph cũng cung cấp một đầu ra detector thô phụ trợ |

Đối với bản xuất phát hiện YOLO9 sang ONNX với `nms=true`, đầu ra `0` (tên `output`)
là tensor hậu NMS độc lập tại các ngưỡng lúc xuất. Khi
`nms_raw_output=true`, đầu ra `1` (tên `raw`) được dành cho các backend LibreYOLO
để chúng có thể áp dụng thao tác cắt theo canvas gốc và ngữ nghĩa runtime gốc của
`predict(conf=..., iou=..., max_det=...)`. Bên tiêu thụ của bên thứ ba
nên dùng đầu ra thứ nhất.

Bản xuất pose có thể thêm `num_keypoints`; `keypoint_dim`, trong đó bản xuất thô
kiểu GroupPose có thể dùng các giá trị lớn hơn như `8` khi tensor bao gồm các
trường precision hoặc logit lớp đối tượng; `num_keypoints_per_class` dưới dạng
danh sách mã hóa JSON, trong đó phải giữ lại các vị trí lớp đối tượng không có
keypoint vì chúng xác định lược đồ; và `pose_input`, trong đó `"person_crop"`
nghĩa là graph nhận một vùng cắt đã được trích xuất và không chứa detector.
Bản xuất runtime HRNet yêu cầu giá trị đó.

Bản xuất phân loại có thể thêm `crop_pct`, tỷ lệ crop giữa dạng float có kích
thước đích trước khi crop là `round(imgsz / crop_pct)` và mặc định là `0.875`
khi không có, cùng với `interpolation`, `"bilinear"` hoặc `"bicubic"`,
mặc định là `"bilinear"`.

Bản xuất ExecuTorch ghi metadata phẳng vào một tệp phụ `<program>.pte.json`
bắt buộc. Hợp đồng v1 là CPU, FP32, batch 1 và canvas đầu vào cố định, đồng thời
yêu cầu thêm `executorch_version`, `executorch_delegate` bằng `"xnnpack"`, và
`executorch_delegate_partitions` dương. Trình tải từ chối tệp phụ khai báo delegate
khác, shape động hoặc precision không phải FP32.

Bản xuất MNN ghi metadata phẳng vào một tệp phụ `<model>.mnn.json` bắt buộc.
Hợp đồng v1 chỉ dành cho CPU, FP32 và phát hiện, với shape đầu vào NCHW cố định,
đồng thời yêu cầu thêm `mnn_version`, `mnn_backend` bằng `"cpu"`,
`mnn_input_names` và `mnn_output_names` có thứ tự, không rỗng, `mnn_input_shape`
là bốn số nguyên dương theo thứ tự `[batch, channels, height, width]`, và
`mnn_batch` bằng `mnn_input_shape[0]`. Trình tải từ chối metadata có shape động,
không phải FP32, không phải tác vụ phát hiện, thuộc họ không được hỗ trợ hoặc có
shape không nhất quán.

Tệp `.pte` và `.mnn` là các tạo tác dành riêng cho backend, không phải
checkpoint PyTorch.

## Checkpoint lượng tử hóa

Mô hình đã lượng tử hóa thêm một khóa phẳng tùy chọn là `quant`, chứa một
dictionary manifest với `schema`, `recipe`, `keep_high_precision`, `execution`,
nguồn gốc hiệu chuẩn, `module_count` và `state`. Manifest FP8 cũng có thể mang
`fp8_tensorwise_weights`, danh sách chính xác tên module `QuantLinear` có scale
trọng số theo toàn tensor thay vì theo từng kênh đầu ra. Trình tải thấy `quant`
sẽ dựng lại cấu trúc module đã lượng tử hóa và chính sách scale trước
`load_state_dict`.

`state` phân biệt hai dạng tạo tác.

`"prepared"`, giá trị mặc định, chứa trọng số chính FP32 cùng các buffer scale
`_q_*` và có thể huấn luyện. Trình đọc không hỗ trợ quantization
có thể bỏ qua khóa `quant` và tải trọng số chính như một mô hình float.

`"finalized"` là dạng triển khai được ghi bởi `export(format="pt")`.
Trọng số chính bị loại bỏ và mỗi module đã lượng tử hóa thay vào đó mang trọng số đóng gói:

| Recipe | Tensor đóng gói | Giải lượng tử |
|---|---|---|
| int8 | `weight_packed` int8 theo shape trọng số gốc, `_q_w_scale` FP32 theo từng kênh | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn theo shape gốc, `_q_w_scale` FP32 với một phần tử cho mỗi kênh đầu ra | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, hai mã 4 bit trên mỗi byte, nibble thấp trước, mã `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, nhóm 128 theo in_features | Scale theo nhóm |
| int2 | Bốn mã 2 bit trên mỗi byte, mã `q + 2`, nhóm 64 | Scale theo nhóm |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, mã `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 theo từng tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Giống nvfp4 nhưng dùng các khối 32 phần tử, cộng thêm `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Các buffer phạm vi kích hoạt `_q_act_lo`, `_q_act_hi` và `_q_calibrated` được
giữ lại cho int8. Manifest ghi `remainder`, `"fp16"` hoặc `"fp32"`,
cho các tensor không được lượng tử hóa. Việc giải đóng gói tái tạo bit mô phỏng
chính xác từng bit, vì vậy suy luận (inference) đã hoàn tất khớp hoàn toàn với
inference đã chuẩn bị trên thiết bị hoàn tất. Bố cục này là hợp đồng ổn định cho
các trình xuất và runtime bên ngoài.

## Checkpoint huấn luyện

Checkpoint của trainer dùng cùng lõi metadata bắt buộc và có thể thêm các trường
phẳng phục vụ huấn luyện và tiếp tục:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` khai báo liệu `model` cấp cao nhất có được làm mượt bằng EMA hay
không. Khi EMA được bật, `train_model`, `ema` và `ema_updates` bảo toàn trạng thái
để tiếp tục. Trọng số inference được phát hành nên gọn nhẹ và không nên chứa
optimizer, epoch, config, loss hoặc trạng thái tiếp tục EMA, trừ khi chúng được
chủ ý phân phối dưới dạng checkpoint huấn luyện.

Để tương thích với bản phát hành, trình đọc chấp nhận các bí danh metric tốt nhất cũ
`best_mAP50_95`, `best_mAP50`, `best_metric` và `best_metric_name`.

## Snapshot bên ngoài

Lược đồ điều chỉnh các tệp `.pt` do LibreYOLO tạo. Nó không đổi tên hay bao bọc
các snapshot upstream nhiều tệp được dùng bởi những cấp mô hình riêng biệt.

Kích thước LibreMODUS `14b-a7b` là một ngoại lệ tường minh: bí danh được phân giải
qua `LibreVLM(...)` thành một thư mục chứa các tệp upstream được ghim phiên bản,
và LibreYOLO không thêm metadata v1.0 vào đó cũng không phát hành lại dưới dạng `.pt`.

## Trọng số cũ và từ nguồn ngoài

Trình ghi mới xác thực nghiêm ngặt và phải phát ra metadata v1.0. Khi metadata
bị thiếu hoặc không đầy đủ, checkpoint cũ có vẻ thuộc LibreYOLO được tải qua
đường dẫn tương thích kèm cảnh báo và hướng dẫn chuyển đổi, còn checkpoint
upstream từ nguồn ngoài được chuyển đến quy trình tự động chuyển đổi. Xem
[checkpoint upstream](/docs/reference/upstream-checkpoints).

## Hàm hỗ trợ

Các hàm hỗ trợ lược đồ nằm trong `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` không sửa đổi dữ liệu và trả về danh sách
lỗi; với `strict=True`, hàm này sẽ phát sinh `CheckpointMetadataError`.
`model.save(path)` là cách được hỗ trợ để ghi một checkpoint tuân thủ lược đồ.
