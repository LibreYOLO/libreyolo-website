---
title: Model API
seo_title: Phương thức và signature object mô hình LibreYOLO
description: >-
  Mọi phương thức trên mô hình LibreYOLO đã nạp: predict, embed, track, val,
  train, export, save, quantize, info và điều khiển CUDA graph, với giá trị mặc
  định thực.
lead: >-
  Mô hình LibreYOLO đã nạp là instance của BaseModel. Trang này liệt kê các
  phương thức instance đó cung cấp, với signature và giá trị mặc định đọc từ
  libreyolo/models/base/model.py.
keywords:
  - phương thức mô hình libreyolo
  - đối số predict libreyolo
  - đối số val libreyolo
  - đối số export libreyolo
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Signature và giá trị mặc định được đọc từ libreyolo/models/base/model.py và
  libreyolo/models/base/inference.py ở v1.5.0. Lớp họ có thể thu hẹp hoặc mở
  rộng chúng; train() được định nghĩa theo họ và chỉ wrapper cfg= dùng chung
  được mô tả tại đây.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True trả về generator, mỗi frame hoặc ảnh một Results.
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Khởi tạo

Factory trả về instance lớp họ. Dựng trực tiếp lớp đó nhận cùng các đối số,
ngoại trừ `size` là bắt buộc:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` chọn CUDA khi khả dụng, sau đó MPS rồi CPU. Số nguyên hoặc chuỗi
chữ số được hiểu là thứ tự CUDA, nên `device=0` và `device="0"` đều có nghĩa
`cuda:0`. `task` được xác thực với `SUPPORTED_TASKS` của họ. Truyền
`model_path=None` dựng kiến trúc và để ở chế độ huấn luyện; truyền `dict` nạp
trực tiếp state dict đó.

## predict và \_\_call\_\_

`predict` là alias của `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Đối số | Mặc định | Ý nghĩa |
|---|---|---|
| `source` | `None` | Ảnh, danh sách hoặc tuple ảnh trong bộ nhớ, thư mục, file video hoặc source màn hình như `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Ngưỡng độ tin cậy |
| `iou` | `0.45` | Ngưỡng IoU cho NMS |
| `imgsz` | `None` | Override kích thước đầu vào; `None` dùng kích thước native của mô hình |
| `device` | `None` | Override thiết bị cho lệnh gọi này |
| `classes` | `None` | Chỉ giữ các ID lớp này |
| `max_det` | `300` | Số detection tối đa mỗi ảnh |
| `augment` | `False` | Test-time augmentation |
| `save` | `False` | Ghi ảnh hoặc video đã chú thích |
| `batch` | `1` | Số ảnh mỗi forward pass cho source thư mục và danh sách |
| `stream` | `False` | Trả về generator thay vì danh sách đã materialize |
| `stream_buffer` | `False` | Giữ mọi frame trực tiếp đã chụp thay vì chỉ frame mới nhất |
| `vid_stride` | `1` | Xử lý mỗi frame video hoặc màn hình thứ N |
| `show` | `False` | Hiển thị frame đã chú thích trong cửa sổ |
| `output_path` | `None` | Đường dẫn đầu ra khi `save=True` |
| `color_format` | `"auto"` | Gợi ý định dạng màu cho mảng trong bộ nhớ |
| `tiling` | `False` | Inference theo tile cho ảnh lớn |
| `overlap_ratio` | `0.2` | Tỷ lệ chồng lấp tile |
| `output_file_format` | `None` | `"jpg"`, `"png"` hoặc `"webp"` |
| `cuda_graph` | `False` | `True` capture ở lần dùng đầu theo shape đầu vào, `"auto"` chờ shape lặp lại |

Source là một ảnh trả về một `Results`. Danh sách, tuple hoặc thư mục trả về
danh sách, còn `stream=True` luôn trả về generator.

Source stream trực tiếp không giới hạn và cần `stream=True`. Không thể kết hợp
`tiling` với `augment`. Test-time augmentation báo lỗi cho tác vụ `embed`,
`point` và `edge`.

<code-tabs name="usage" />

Với `batch > 1`, các họ có `SUPPORTS_BATCHED_PREDICT` bằng true chạy một forward
xếp chồng cho mỗi chunk; `batch=1` giữ một forward mỗi ảnh.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Wrapper tiện ích trên `predict` xếp mọi hàng embedding vào một tensor
`(N_total, D)`. Mô hình phải được dựng với `task="embed"`, nếu không sẽ phát
`NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Tạo một `Results` mỗi frame với `track_id` được đặt. `tracker` là
`"bytetrack"`, `"botsort"`, `"ocsort"` hoặc `"deepocsort"`, và bị bỏ qua khi
có `tracker_config` vì kiểu cấu hình chọn tracker. `track_conf` ánh xạ sang
`track_high_thresh` cho ByteTrack và BoT-SORT, sang `det_thresh` cho OC-SORT và
Deep OC-SORT. `output_path` mặc định là `runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Trả về dictionary metric có khóa tùy tác vụ; detection trả về
`metrics/precision`, `metrics/recall`, `metrics/mAP50` và
`metrics/mAP50-95`. `imgsz` nhận số nguyên cho hình vuông hoặc tuple `(height,
width)` và mặc định theo kích thước đầu vào native của mô hình. `plots` là alias
của `save_plots`. `allow_download_scripts` kiểm soát Python nhúng mà YAML
dataset có thể chứa trong trường `download`.

`faster_coco_eval` được nhận qua `**kwargs`, mặc định `True`, và quay về
pycocotools khi package chưa được cài. Backend đã chạy được báo trong
`model.last_eval_backend`.

Xác thực có augmentation báo lỗi với tác vụ `obb` và `pose`.

## train

`train` được định nghĩa theo họ nên đối số khác nhau. Hai hành vi dùng chung vì
lớp cơ sở bọc `train` của mọi họ:

- `cfg=` nhận đường dẫn YAML có khóa được hợp nhất vào lệnh gọi. Keyword tường
  minh ưu tiên hơn file.
- `pretrained=False` trên họ thuộc nhóm phạm vi `g0` hoặc `g1` khởi tạo lại mô
  hình từ đầu trước khi huấn luyện và không thể kết hợp `resume=True`.

Nút augmentation nào thực sự được họ tuân theo là câu hỏi riêng theo họ; xem
[ma trận augmentation](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Trả về đường dẫn artifact đã ghi. `format` được phân giải qua registry exporter,
trong đó `engine` là alias cho `tensorrt`, còn `litert` là alias cho `tflite`.
Các đối số dùng chung cho mọi exporter:

| Đối số | Mặc định | Ý nghĩa |
|---|---|---|
| `output_path` | `None` | Đường dẫn file đầu ra; tạo dưới `weights/` khi bỏ qua |
| `imgsz` | `None` | Tuple `(height, width)` hoặc số nguyên; mặc định theo kích thước native |
| `opset` | `None` | Phiên bản opset ONNX |
| `simplify` | `True` | Chạy đơn giản hóa graph ONNX |
| `dynamic` | `True` | Bật trục dynamic |
| `half` | `False` | Độ chính xác FP16 |
| `int8` | `False` | Độ chính xác INT8 |
| `batch` | `1` | Batch size được cố định trong artifact |
| `device` | `None` | Thiết bị dùng để trace |
| `data` | `None` | data.yaml cho hiệu chuẩn INT8 |
| `fraction` | `1.0` | Tỷ lệ dataset hiệu chuẩn được sử dụng |
| `allow_download_scripts` | `False` | Cho phép Python nhúng trong phần tải YAML dataset |
| `verbose` | `False` | Logging exporter chi tiết |

Tổ hợp blocked phát `NotImplementedError` trong preflight trước tracing. Phạm
vi và quy tắc nằm trên trang [ma trận xuất](/docs/reference/export-matrix).
Khi có adapter LoRA trực tiếp, chúng được gộp vào trọng số dense, và phép gộp
chỉ diễn ra sau khi mọi bước từ chối yêu cầu đã hoàn tất.

## save

```python
model.save(path) -> str
```

Ghi checkpoint LibreYOLO schema v1.0: state dict cùng metadata mô tả trong
[schema checkpoint](/docs/reference/checkpoint-schema). Mô hình đã quantization
còn mang manifest `quant`, để `LibreYOLO(path)` khôi phục cấu trúc và scale đã
quantization.

## quantize, quant_info và dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Quantization tại chỗ và trả về mô hình. `recipe` là một trong các phép cast
`fp16` và `bf16`, công thức Conv và Linear `int8` cùng `fp8`, hoặc công thức chỉ
Linear `w4a16`, `w4a8`, `nvfp4`, `mxfp4` và `int2` mà các họ transformer như
RF-DETR hỗ trợ. `int2` cần QAT. `calib` nhận đường dẫn data.yaml hoặc tên dataset
tích hợp và chỉ đọc ảnh theo forward; nhãn không bao giờ được đọc. Truyền
`calib=None` để bỏ hiệu chuẩn. `algorithm` là `"minmax"`, `"percentile"` hoặc
`"auto"`.

`model.quant_info()` trả về tóm tắt trạng thái quantization hoặc `None` cho mô
hình float. `model.dequantize()` khôi phục module float tại chỗ trong khi giữ
trọng số chính đã huấn luyện quantization, là cầu nối từ QAT đến
`export(format="onnx", int8=True, data=...)`.

## info và layers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` trả về dictionary thân thiện với JSON và ghi log tóm tắt dễ đọc khi
`verbose` là true. `get_available_layer_names` liệt kê các lớp mà cấu hình
distillation hoặc trích xuất feature có thể nêu tên.

## CUDA graph

Khả dụng trên các họ có thuộc tính lớp `SUPPORTS_CUDA_GRAPH` bằng true. Replay
giống hệt eager execution từng bit.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # context manager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Graph đã capture chỉ hợp lệ cho đúng shape dùng lúc capture, nên `batch` và
`imgsz` phải khớp lệnh `predict` sau đó. `capture_graph` chuyển chi phí capture
ra khỏi yêu cầu đầu tiên. `mode` nhận `True` hoặc `"on"` để capture ở lần dùng
đầu, `"auto"` để chờ shape lặp lại, và `False` để không làm gì. `capture_graph`
phát `NotImplementedError` khi họ chưa opt-in và `CudaGraphUnavailable` khi
capture thất bại.

## Thiết bị và dtype

Object `Results` có `.to()`, `.cpu()`, `.cuda()` và `.numpy()`; xem [các kiểu
Results](/docs/reference/results-types). Bản thân mô hình được di chuyển bằng
cách truyền `device=` cho `predict` hoặc lúc khởi tạo.
