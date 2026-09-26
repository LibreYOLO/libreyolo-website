---
title: libreyolo quantize
seo_title: tham chiếu lệnh libreyolo quantize
description: >-
  Lượng tử hóa (quantization) một checkpoint trong PyTorch từ dòng lệnh: các
  recipe, tham số hiệu chuẩn, giá trị mặc định và những họ mô hình mà mỗi recipe
  chấp nhận.
lead: >-
  Thay các module float của mô hình bằng module đã lượng tử hóa, hiệu chuẩn
  chúng trên ảnh không nhãn khi recipe cần thống kê, rồi lưu kết quả thành một
  checkpoint PyTorch.
keywords:
  - libreyolo quantize cli
  - lệnh lượng tử hóa int8
  - lượng tử hóa fp8
  - lượng tử hóa sau huấn luyện yolo
  - tham số libreyolo quantize
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo quantize
    mono: true
  - label: Bắt buộc
    value: model
    mono: true
  - label: Đầu ra
    value: >-
      Đường dẫn nguồn kèm -<recipe> trước phần mở rộng, ví dụ
      LibreYOLO9s-int8.pt
    mono: true
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        # Hiệu chuẩn trên coco128 rồi ghi ra LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Chỉ ép kiểu, không hiệu chuẩn'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 'Hiệu chuẩn rộng hơn, rồi khôi phục'
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # Quantization-aware training trên checkpoint đã lượng tử hóa giúp lấy
        lại độ chính xác

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Cú pháp

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Các tham số là cặp `key=value`, và dạng POSIX cũng dùng được, nên `recipe=int8`
và `--recipe int8` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `model` | | Trọng số mô hình `.pt`. Bắt buộc |
| `recipe` | `int8` | Recipe lượng tử hóa: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Ảnh hiệu chuẩn: một tệp YAML dữ liệu hoặc tên một tập dữ liệu (dataset) có sẵn. Không nhãn, chỉ chạy forward. `none` bỏ qua bước hiệu chuẩn |
| `samples` | `128` | Số ảnh hiệu chuẩn tối đa |
| `batch` | `8` | Kích thước batch khi hiệu chuẩn |
| `algorithm` | `auto` | Ước lượng dải giá trị activation: `auto`, tức chọn minmax, hoặc `minmax`, hoặc `percentile` |
| `out` | | Đường dẫn checkpoint đầu ra. Mặc định là đường dẫn nguồn kèm `-<recipe>` trước phần mở rộng |
| `device` | `auto` | Thiết bị |
| `allow_download_scripts` | `false` | Cho phép mã Python nhúng trong khối download của tệp YAML dataset |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Tắt stderr |
| `help_json` | `false` | Xuất schema của lệnh dưới dạng JSON rồi thoát |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Những họ mô hình nào chấp nhận lệnh này

Quantization áp dụng cho bốn họ mô hình: `yolo9`, `rfdetr`, `birefnet` và
`feynobg`. Mọi họ khác sẽ thoát với `quantize_failed` kèm theo danh sách đó.

### Mỗi recipe tác động tới những gì

`fp16` và `bf16` là các phép ép kiểu. Chúng chỉ đổi dtype, không cần hiệu chuẩn,
và `calib=none` là thiết lập đúng cho chúng.

`int8` và `fp8` lượng tử hóa các module `Conv2d` và `Linear`, đó là lý do chúng
hợp với các họ mô hình dùng convolution.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` và `int2` chỉ lượng tử hóa `nn.Linear`, nên
chúng nhắm tới các họ mô hình transformer. Yêu cầu một trong số đó trên `yolo9`
sẽ bị từ chối kèm giải thích, thay vì âm thầm tạo ra một mô hình chưa lượng tử
hóa, bởi ở đó việc tăng tốc dưới 8 bit chỉ có với GEMM và các convolution sẽ vẫn
nằm ở precision cao hơn.

`int8`, `fp8`, `w4a8` và `int2` cần thống kê hiệu chuẩn cho activation của
chúng. `int2` còn cần huấn luyện để khôi phục sau đó, nên nó bị từ chối trên
`birefnet` và `feynobg`, vốn không có trainer.

Mỗi họ mô hình giữ một số module ở dạng float bất kể recipe là gì: các lớp đầu
tiên, các head dự đoán, và trên YOLOv9 là convolution DFL, vốn là một toán tử
kỳ vọng tích phân cố định và không được phép lượng tử hóa.

### Dữ liệu hiệu chuẩn không phải dữ liệu huấn luyện

`calib` trỏ tới một tập ảnh nhỏ không có nhãn, chỉ dùng theo chiều forward, để
suy ra dải giá trị activation. Nó không được dùng để đánh giá và nhãn của nó
không bao giờ được đọc. Tệp `coco128.yaml` mặc định sẽ tải về từ một URL ở lần
dùng đầu tiên, nên không cần thêm quyền gì; một tệp YAML có script Python tải về
nhúng bên trong thì cần `allow_download_scripts=true`.

`algorithm=percentile` có sẵn và có thể làm giảm độ chính xác trên các họ mô
hình transformer, đó là lý do `auto` chọn minmax.

### Lấy lại độ chính xác

Đầu ra là một checkpoint PyTorch bình thường, nên
[`libreyolo train`](/docs/cli/train) nhận nó trực tiếp. Huấn luyện một checkpoint
đã lượng tử hóa chính là quantization-aware training; thêm
`distill_model=<teacher>` sẽ biến nó thành quantization-aware distillation.

### Đầu ra và mã thoát

Kết quả in ra đường dẫn đã lưu, recipe, chế độ thực thi, việc hiệu chuẩn có chạy
hay không, và số module được thay theo từng loại. Mã thoát là `0` khi thành
công, `4` khi không tải được mô hình, `5` khi lượng tử hóa hoặc việc lưu thất
bại, và `1` cho các lỗi runtime khác.

Liên quan: [`libreyolo export`](/docs/cli/export), lệnh rời khỏi PyTorch và thay
vào đó ghi ra một artifact để triển khai.
