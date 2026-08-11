---
title: Schema checkpoint
seo_title: Schema metadata checkpoint LibreYOLO v1.0
description: >-
  Metadata có trong mọi checkpoint .pt LibreYOLO: khóa bắt buộc, phần bổ sung
  theo tác vụ, khóa runtime xuất, manifest lượng tử hóa và trường huấn luyện.
lead: >-
  Tệp .pt LibreYOLO là từ điển phẳng được lưu bằng torch.save. Khóa model chứa
  state dict; các khóa cấp cao nhất khác là metadata nhận diện checkpoint mà
  không cần phân tích tên tệp hay dò state dict.
keywords:
  - schema checkpoint libreyolo
  - schema_version 1.0
  - model_family
  - metadata checkpoint libreyolo
  - manifest lượng tử hóa
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Phản ánh docs/checkpoint_schema.md trong repo libreyolo ở v1.5.0, được đối
  chiếu với libreyolo/utils/serialization.py và BaseModel.save.
snippets:
  usage:
    - label: Đọc metadata từ checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Tải checkpoint rồi lưu lại để có đường dẫn cục bộ.

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

## Schema v1.0

Mọi checkpoint `.pt` LibreYOLO chính thức đều chứa:

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

| Khóa | Loại | Ý nghĩa |
|---|---|---|
| `model` | state dict | Trọng số mô hình |
| `schema_version` | str | Phiên bản giao diện metadata; v1.0 dùng chuỗi `"1.0"` |
| `libreyolo_version` | str | Phiên bản tạo checkpoint |
| `model_family` | str | Họ đã đăng ký, chẳng hạn `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Biến thể trong họ, chẳng hạn `t`, `s`, `r18`, `atto` |
| `task` | str | Tên tác vụ chuẩn |
| `nc` | int | Số lớp dương |
| `names` | dict | `dict[int, str]` với khóa trong `0..nc-1` |
| `imgsz` | int | Độ phân giải đầu vào vuông dương hoặc giá trị vô hướng cũ cho giao diện hình chữ nhật |

`task` là một trong `detect`, `segment`, `semantic`, `panoptic`, `pose`,
`classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` hoặc `mesh`.

Checkpoint chính thức ghi mọi khóa `names`. Trình đọc có thể thêm nhãn `class_i`
cho khóa thiếu trong ánh xạ thưa cũ, nhưng khóa ngoài phạm vi không hợp lệ.

Checkpoint hình chữ nhật giữ `imgsz` vô hướng cho trình đọc cũ, đặt bằng
`max(imgsz_h, imgsz_w)`, đồng thời ghi `imgsz_h` và `imgsz_w` với kích thước thực.
Trình đọc hiểu các trường hình chữ nhật phải ưu tiên chúng hơn giá trị vô hướng.
Họ có giao diện hình chữ nhật cố định như tư thế HRNet từ chối kích thước runtime
không tương thích.

Schema được cố ý thiết kế phẳng và `model` được cố ý đặt là state dict.

<code-tabs name="usage" />

## Phần bổ sung cho tư thế

Tư thế thường là một lớp, `nc: 1` với `person`, nhưng head tư thế YOLO-NAS cũng
hỗ trợ tư thế đa lớp với một skeleton keypoint dùng chung; khi đó `nc` và `names`
mô tả các lớp như trong phát hiện. Bản xuất tư thế runtime phát `scores` có shape
`[batch, anchors, nc]`.

| Khóa | Ý nghĩa |
|---|---|
| `num_keypoints` | Số keypoint dương được head tư thế dùng |
| `keypoint_dim` | `2` cho nhãn `x,y` hoặc `3` cho nhãn `x,y,visibility`; đầu ra mô hình luôn cung cấp `x,y,visibility` |
| `oks_sigmas` | Sigma OKS tùy chọn theo keypoint; dùng mặc định tác vụ cho `num_keypoints` khi không có |
| `num_keypoints_per_class` | Số keypoint tùy chọn theo lớp cho head kiểu GroupPose có tensor keypoint được padding theo lớp; `0` cho lớp không có keypoint |

## Phần bổ sung cho mesh

Checkpoint mesh dùng `task: "mesh"`, `nc: 1` và `names: {0: "person"}`. Bố cục
tham số khác nhau giữa các mô hình cơ thể, nên kích thước được ghi lại thay vì giả định.

| Key | Meaning |
|---|---|
| `body_model` | Cách tham số hóa như `mhr`; bắt buộc và dùng để diễn giải mọi trường bên dưới |
| `num_betas` | Số hệ số nhận dạng và shape; 45 cho MHR |
| `num_body_pose` | Độ rộng block tham số tư thế cơ thể; 130 cho MHR. Là vector phẳng, không phải một bộ ba cho mỗi khớp vì các khớp rig có bậc tự do khác nhau |
| `num_vertices` | Số vertex decoder phát ra; 18439 cho MHR |
| `num_joints` | Số khớp decoder phát ra; 127 cho MHR |
| `rotation_format` | Cách mã hóa phép quay như `euler_zyx` cho MHR hoặc `axis_angle`. Không bao giờ suy ra từ shape tensor vì vector 3 phần tử nhập nhằng |

## Giá trị giữ chỗ cho tác vụ dày đặc

Một số tác vụ dự đoán bản đồ dày đặc thay vì lớp, nên các slot giống lớp chỉ tồn tại để tương thích schema.

| Tác vụ | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Dự đoán cạnh là bản đồ xác suất float32 dày đặc trong `[0, 1]`.

Checkpoint khôi phục có thể thêm `degradation`, nhãn suy giảm ngắn như `deblur`,
`denoise` hoặc `super-resolution`; `dataset`, nhãn nguồn gốc như `GoPro` hoặc
`SIDD`; và `scale`, hệ số upscale đầu ra so với đầu vào là số nguyên dương, chẳng
hạn `4` cho mô hình siêu phân giải x4. Không có hoặc bằng `1` nghĩa là ảnh khôi
phục giữ độ phân giải đầu vào. Runtime cũng suy ra scale từ họ và kích thước, nên
`scale` là metadata nguồn gốc thay vì yêu cầu khi tải.

## Phần bổ sung cho OCR

Họ `ppocr` phân phối một checkpoint tổng hợp cho mỗi tầng, có state dict `model`
chứa hai mô hình con trong namespace khóa `det.*` và `rec.*`.

| Key | Meaning |
|---|---|
| `charset` | Toàn bộ bảng chữ cái CTC theo thứ tự chỉ mục đầu ra: chỉ mục 0 là blank CTC, tiếp đến từ điển nhận dạng rồi ký tự khoảng trắng. Trình tải phải đọc từ checkpoint, không bao giờ từ tệp bên cạnh |
| `pipeline` | Giá trị mặc định pipeline được nhúng khi chuyển đổi: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Đối số runtime có thể ghi đè theo từng lời gọi |
| `components` | Dành riêng cho giai đoạn pipeline tùy chọn như định hướng tài liệu, khử biến dạng và xoay dòng văn bản. Rỗng trong v1 |

## Metadata runtime xuất

Artifact đã xuất dùng cùng quy ước ghi kép hình chữ nhật: `imgsz_h` và `imgsz_w`
được ghi cạnh `imgsz` vô hướng cũ; trình đọc không hiểu trường hình chữ nhật không
được âm thầm coi giá trị vô hướng là giao diện hình vuông.

Hỗ trợ runtime hình chữ nhật theo phạm vi họ và định dạng. Bản xuất họ YOLO9,
HRNet, NAFNet và Real-ESRGAN có thể dùng `imgsz_h`, `imgsz_w` không vuông trong
định dạng được hỗ trợ; họ hoặc định dạng không hỗ trợ hình chữ nhật rõ ràng sẽ
từ chối metadata thay vì tiền xử lý artifact như hình vuông. Bản xuất HRNet là
head crop người cố định, batch một, FP32, trong đó W32 nhận 256x192 và W48 nhận
384x288; detector người không được nhúng trong graph.

Bản xuất có NMS nhúng có thể thêm các khóa phẳng sau:

| Key | Meaning |
|---|---|
| `nms` | Boolean chuỗi; `"true"` nghĩa là graph có đầu ra hậu xử lý nhúng |
| `nms_conf` | Ngưỡng độ tin cậy được nhúng vào đầu ra |
| `nms_iou` | Ngưỡng IoU được nhúng vào đầu ra |
| `max_det` | Số hàng phát hiện tối đa sau NMS mà đầu ra nhúng phát ra |
| `nms_raw_output` | Boolean chuỗi; `"true"` nghĩa là graph còn cung cấp đầu ra detector thô phụ |

Với bản xuất phát hiện YOLO9 ONNX có `nms=true`, đầu ra `0` (tên `output`) là
tensor độc lập sau NMS ở các ngưỡng khi xuất. Khi `nms_raw_output=true`, đầu ra
`1` (tên `raw`) dành riêng cho backend LibreYOLO để áp dụng thao tác cắt canvas
gốc native và ngữ nghĩa runtime `predict(conf=..., iou=..., max_det=...)`. Bên
tiêu thụ thứ ba nên dùng đầu ra đầu tiên.

Bản xuất tư thế có thể thêm `num_keypoints`; `keypoint_dim`, nơi bản xuất thô kiểu
GroupPose có thể dùng giá trị lớn như `8` khi tensor chứa trường độ chính xác hoặc
class-logit; `num_keypoints_per_class` dưới dạng danh sách mã hóa JSON, nơi slot
lớp không keypoint phải được giữ vì định nghĩa schema; và `pose_input`, trong đó
`"person_crop"` nghĩa là graph dùng một crop đã trích xuất và không chứa detector.
Bản xuất runtime HRNet yêu cầu giá trị đó.

Bản xuất phân loại có thể thêm `crop_pct`, tỉ lệ crop giữa dạng số thực có đích
đổi kích thước trước crop là `round(imgsz / crop_pct)` và mặc định `0.875` khi
không có; cùng `interpolation`, `"bilinear"` hoặc `"bicubic"`, mặc định
`"bilinear"`.

Bản xuất ExecuTorch ghi metadata phẳng vào sidecar `<program>.pte.json` bắt buộc.
Giao diện v1 là CPU, FP32, batch 1 và canvas đầu vào cố định; đồng thời yêu cầu
`executorch_version`, `executorch_delegate` bằng `"xnnpack"` và
`executorch_delegate_partitions` dương. Trình tải từ chối sidecar khai báo
delegate khác, shape động hoặc độ chính xác số không phải FP32.

Bản xuất MNN ghi metadata phẳng vào sidecar `<model>.mnn.json` bắt buộc. Giao
diện v1 là CPU, FP32, chỉ phát hiện và shape đầu vào NCHW cố định; đồng thời yêu
cầu `mnn_version`, `mnn_backend` bằng `"cpu"`, `mnn_input_names` và
`mnn_output_names` có thứ tự, không rỗng; `mnn_input_shape` gồm bốn số nguyên
dương theo thứ tự `[batch, channels, height, width]`; `mnn_batch` bằng
`mnn_input_shape[0]`. Trình tải từ chối metadata động, không FP32, không phát
hiện, họ không hỗ trợ hoặc shape không nhất quán.

`.pte` và `.mnn` là artifact riêng theo backend, không phải checkpoint PyTorch.

## Checkpoint lượng tử hóa

Mô hình lượng tử hóa thêm một khóa phẳng tùy chọn `quant`, chứa dict manifest với
`schema`, `recipe`, `keep_high_precision`, `execution`, nguồn gốc hiệu chuẩn,
`module_count` và `state`. Manifest FP8 còn có thể chứa `fp8_tensorwise_weights`,
danh sách chính xác tên module `QuantLinear` có scale trọng số theo tensor thay
vì theo kênh đầu ra. Trình tải thấy `quant` sẽ dựng lại cấu trúc module lượng tử
hóa và chính sách scale trước `load_state_dict`.

`state` phân biệt hai dạng artifact.

`"prepared"`, mặc định, chứa trọng số master FP32 cùng buffer scale `_q_*` và có
thể huấn luyện. Trình đọc không hỗ trợ lượng tử hóa có thể bỏ qua khóa `quant` và
tải master dưới dạng mô hình số thực.

`"finalized"` là dạng triển khai được ghi bằng `export(format="pt")`. Master bị
loại và mỗi module lượng tử hóa thay vào đó chứa trọng số đóng gói:

| Công thức | Tensor đóng gói | Giải lượng tử hóa |
|---|---|---|
| int8 | `weight_packed` int8 ở shape trọng số gốc, `_q_w_scale` FP32 theo kênh | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn ở shape gốc, `_q_w_scale` FP32 một mục cho mỗi kênh đầu ra | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, hai mã 4 bit mỗi byte, nibble thấp trước, mã `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, nhóm 128 dọc in_features | Scale theo nhóm |
| int2 | Bốn mã 2 bit mỗi byte, mã `q + 2`, nhóm 64 | Scale theo nhóm |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, mã `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 theo tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Như nvfp4 nhưng block 32 phần tử, cùng `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Các buffer phạm vi activation `_q_act_lo`, `_q_act_hi` và `_q_calibrated` được
giữ cho int8. Manifest ghi `remainder`, `"fp16"` hoặc `"fp32"`, cho tensor không
lượng tử hóa. Giải nén tái tạo mô phỏng từng bit, nên suy luận finalized khớp
chính xác suy luận prepared trên thiết bị hoàn tất. Bố cục này là giao diện ổn
định cho trình xuất và runtime bên ngoài.

## Checkpoint huấn luyện

Checkpoint trình huấn luyện dùng cùng lõi metadata bắt buộc và có thể thêm các trường huấn luyện và tiếp tục phẳng:

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

`is_ema_weights` khai báo `model` cấp cao nhất có được làm mượt EMA hay không. Khi
bật EMA, `train_model`, `ema` và `ema_updates` giữ trạng thái tiếp tục. Trọng số
suy luận đã công bố nên gọn nhẹ, không gồm optimizer, epoch, config, loss hoặc
trạng thái tiếp tục EMA trừ khi cố ý phân phối dưới dạng checkpoint huấn luyện.

Để tương thích bản phát hành, trình đọc chấp nhận các bí danh chỉ số tốt nhất cũ
`best_mAP50_95`, `best_mAP50`, `best_metric` và `best_metric_name`.

## Snapshot bên ngoài

Schema quản lý tệp `.pt` do LibreYOLO tạo. Nó không đổi tên hay bọc snapshot
thượng nguồn nhiều tệp được các tầng mô hình riêng dùng.

LibreMODUS kích thước `14b-a7b` là ngoại lệ rõ ràng: bí danh phân giải qua
`LibreVLM(...)` thành thư mục tệp thượng nguồn đã ghim; LibreYOLO không thêm
metadata v1.0 hay tái công bố dưới dạng `.pt`.

## Trọng số cũ và bên ngoài

Trình ghi mới kiểm tra nghiêm ngặt và phải phát metadata v1.0. Khi metadata thiếu
hoặc không hoàn chỉnh, checkpoint cũ trông giống LibreYOLO được tải qua luồng
tương thích kèm cảnh báo và hướng dẫn chuyển đổi, còn checkpoint thượng nguồn
bên ngoài được định tuyến sang tự động chuyển đổi. Xem
[checkpoint thượng nguồn](/docs/reference/upstream-checkpoints).

## Hàm trợ giúp

Các hàm trợ giúp schema nằm trong `libreyolo.utils.serialization`:

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

`validate_checkpoint_metadata` không thay đổi dữ liệu và trả về danh sách lỗi;
với `strict=True`, phương thức phát sinh `CheckpointMetadataError` thay thế.
`model.save(path)` là cách được hỗ trợ để ghi checkpoint tuân thủ schema.

