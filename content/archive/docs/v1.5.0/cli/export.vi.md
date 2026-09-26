---
title: libreyolo export
seo_title: Tham chiếu lệnh libreyolo export
description: >-
  Xuất một checkpoint sang định dạng triển khai: từng tham số kèm giá trị mặc
  định, nơi tệp kết quả được ghi ra, và những tổ hợp mà lệnh từ chối.
lead: >-
  Chuyển một checkpoint thành một định dạng triển khai và ghi tệp kết quả vào
  weights/. Định dạng quyết định những tham số nào bên dưới được áp dụng.
keywords:
  - xuất mô hình libreyolo
  - lệnh libreyolo export
  - xuất yolo sang onnx bằng cli
  - lệnh xuất tensorrt
  - tham số libreyolo export
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo export
    mono: true
  - label: Bắt buộc
    value: model
    mono: true
  - label: Đầu ra
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        # Ghi ra weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS bên trong graph
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Chạy tệp đã xuất
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # Factory định tuyến theo phần mở rộng tệp, nên tệp xuất ra được tải như
        một checkpoint

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Cú pháp

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Tham số là các cặp `key=value`, và dạng POSIX cũng dùng được, nên `format=onnx`
và `--format onnx` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `model` | | Trọng số mô hình `.pt`. Bắt buộc |
| `format` | `onnx` | Định dạng xuất: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Nền tảng đích của RKNN, hiện chỉ có `rk3588`. Bị từ chối với mọi định dạng khác |
| `imgsz` | | Kích thước ảnh đầu vào: `640` hoặc `480x640` (HxW). `480,640` cũng được chấp nhận. Dùng kích thước của chính mô hình khi không đặt |
| `batch` | `1` | Kích thước batch khi xuất |
| `half` | `false` | Độ chính xác FP16 |
| `int8` | `false` | Lượng tử hóa (quantization) INT8 |
| `dynamic` | `false` | Hình dạng đầu vào động (ONNX) |
| `simplify` | `true` | Đơn giản hóa graph ONNX |
| `nms` | `false` | Nhúng NMS vào mô hình. Chỉ ONNX và CoreML |
| `conf` | `0.25` | Ngưỡng độ tin cậy cho NMS nhúng |
| `iou` | `0.45` | Ngưỡng IoU cho NMS nhúng |
| `max_det` | `300` | Số phát hiện tối đa cho NMS nhúng trong ONNX |
| `opset` | | Phiên bản opset của ONNX. Tự chọn khi không đặt |
| `data` | | Dữ liệu hiệu chuẩn cho INT8 |
| `fraction` | `1.0` | Tỷ lệ dữ liệu hiệu chuẩn được dùng |
| `device` | `auto` | Thiết bị dùng để tracing |
| `allow_download_scripts` | `false` | Cho phép mã Python nhúng trong khối download của YAML tập dữ liệu (dataset) |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Tắt stderr |
| `verbose` | `false` | Ghi log xuất chi tiết |
| `verify` | `false` | Chạy trình mô phỏng PC của RKNN Toolkit2 và so sánh với ONNX Runtime. Chỉ RKNN |
| `help_json` | `false` | In schema của lệnh dưới dạng JSON rồi thoát |

`engine` là bí danh của `tensorrt`, còn `litert` là bí danh của `tflite`. Cả hai
đều được quy về tên chuẩn trước khi bất cứ thứ gì được ghi ra, nên đầu ra JSON và
dòng log luôn báo `tensorrt` hoặc `tflite`.

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Tệp được ghi ra ở đâu

Lệnh này không nhận đường dẫn đầu ra. Tệp kết quả được ghi vào `weights/`, đặt
tên theo phần gốc tên tệp của checkpoint nguồn cộng với phần mở rộng của định
dạng, kèm `_fp16` hoặc `_int8` chèn vào khi một trong hai độ chính xác đó được
yêu cầu. `LibreYOLO9s.pt` xuất sang ONNX ở FP16 sẽ thành
`weights/LibreYOLO9s_fp16.onnx`. Kết quả JSON mang theo `output_path` đã được
xác định, kích thước tệp tính bằng MB, và hình dạng đầu vào dưới dạng
`[batch, 3, height, width]`.

### Những tổ hợp bị từ chối

`nms=true` được chấp nhận với ONNX và CoreML, và bị từ chối với mọi định dạng
khác kèm `nms_unsupported_format`. Với ONNX, nó buộc `dynamic` phải tắt, vì graph
nhúng được cố định ở batch 1, và báo điều đó ra stderr. Với CoreML, nó nhận `conf`
và `iou` nhưng không nhận `max_det`, nên một giá trị `max_det` khác mặc định đi
cùng `format=coreml nms=true` sẽ thoát với `config_unsupported`.

`half=true` đi cùng `int8=true` không phải là lỗi. INT8 thắng, `half` bị bỏ, và
một cảnh báo được đưa ra stderr.

`name` và `verify` hiện là các tùy chọn của RKNN. Truyền một trong hai cùng với
định dạng khác sẽ thoát với `config_unsupported` thay vì bị bỏ qua.

### Mỗi họ mô hình hỗ trợ những định dạng nào

Mức hỗ trợ được tính theo từng họ mô hình và từng tác vụ, không phải toàn cục.
`libreyolo formats family=<family> task=<task>` in ra mức hỗ trợ của từng định
dạng cho tổ hợp đó, kèm lý do và mọi ràng buộc đi kèm. Xem
[`libreyolo formats`](/docs/cli/utilities) để biết các tham số.

Một số định dạng cần cài thêm gói tùy chọn và một số cần cả toolchain. Thiếu một
phụ thuộc Python sẽ thoát với `export_dep_missing`; một độ chính xác mà định dạng
không tạo ra được sẽ thoát với `format_precision_unsupported`.

### Chạy thứ bạn vừa xuất

Các tệp đã xuất được tải qua cùng một model factory như checkpoint, phân biệt
theo phần mở rộng tệp, nên `libreyolo predict model=weights/LibreYOLO9s.onnx`
chạy được mà không cần chuyển đổi gì thêm. Ba tùy chọn dự đoán là ngoại lệ và bị
từ chối trên các backend runtime: `tiling`, `overlap_ratio` và
`output_file_format`.

Hai đích triển khai có trang riêng:
[NVIDIA DeepStream](/docs/export/deepstream) và
[NVIDIA Jetson](/docs/export/jetson).

### Đầu ra và mã thoát

stdout mang kết quả; tiến trình đi ra stderr. Mã thoát là `0` khi thành công,
`2` khi dùng sai lệnh hoặc cấu hình sai, `4` khi không tải được mô hình, `5` khi
định dạng không xác định, thiếu phụ thuộc để xuất, độ chính xác không được hỗ trợ
hoặc yêu cầu nhúng NMS bị từ chối, và `1` cho các lỗi runtime khác.

Liên quan: [`libreyolo quantize`](/docs/cli/quantize), lệnh ở lại trong PyTorch
và ghi ra một checkpoint thay vì một tệp triển khai.
