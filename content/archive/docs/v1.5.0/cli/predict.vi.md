---
title: libreyolo predict
seo_title: Tham chiếu lệnh libreyolo predict
description: >-
  Chạy suy luận (inference) từ dòng lệnh: mọi tham số, giá trị mặc định đọc
  thẳng từ định nghĩa CLI, và các cờ quyết định những gì được in ra stdout.
lead: >-
  Chạy một mô hình đã tải trên một nguồn và in ra các dự đoán. Nguồn có thể là
  ảnh, thư mục, video, URL hoặc luồng (stream) trực tiếp; mô hình có thể là
  checkpoint hoặc một artifact đã xuất.
keywords:
  - libreyolo predict cli
  - lệnh inference libreyolo
  - dự đoán yolo bằng dòng lệnh
  - tham số libreyolo predict
  - xuất json từ libreyolo predict
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo predict
    mono: true
  - label: Bắt buộc
    value: source
    mono: true
  - label: Đầu ra
    value: >-
      Dự đoán in ra stdout. Với save=true, các tệp đã vẽ chú thích nằm trong
      runs/detect/predict
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Lưu ảnh đã vẽ chú thích
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Lọc lớp đối tượng, JSON trên stdout'
      language: bash
      code: >
        # lớp đối tượng 0 là person trong danh sách lớp COCO đi kèm checkpoint

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Cú pháp

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Các tham số là cặp `key=value`. Cùng một lệnh đó cũng chấp nhận dạng POSIX, nên
`conf=0.4` và `--conf 0.4` có thể dùng thay cho nhau, và một giá trị boolean
viết `save=true` sẽ trở thành `--save`. Tên có dấu gạch dưới chấp nhận cả hai
cách viết: `max_det=50` và `--max-det 50` cùng trỏ tới một tùy chọn.

`libreyolo detect predict ...` cũng được chấp nhận và hoạt động y hệt; từ chỉ
tác vụ được loại bỏ trước khi phân tích cú pháp.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `source` | | Đường dẫn ảnh, thư mục, hoặc URL. Bắt buộc |
| `model` | `yolox-s` | Tên hoặc đường dẫn mô hình |
| `conf` | `0.25` | Ngưỡng độ tin cậy |
| `iou` | `0.45` | Ngưỡng IoU cho NMS |
| `imgsz` | | Kích thước ảnh đầu vào: `640` (vuông) hoặc `480x640` (cao x rộng). Dùng kích thước đầu vào của chính mô hình khi không đặt |
| `classes` | | Lọc theo ID lớp đối tượng, ví dụ `[0,2,5]`. Chấp nhận một số nguyên đơn lẻ |
| `max_det` | `300` | Số phát hiện tối đa trên mỗi ảnh |
| `half` | `false` | Inference FP16 (chỉ CUDA, cần mô hình hỗ trợ) |
| `save` | `false` | Lưu ảnh đã vẽ chú thích |
| `batch` | `1` | Số ảnh mỗi lượt forward khi nguồn là thư mục. Lớn hơn 1 sẽ chạy inference theo batch thật sự trên các mô hình hỗ trợ |
| `stream` | `false` | Trả kết quả dần dần. Tự động bật cho webcam và luồng trực tiếp |
| `stream_buffer` | `false` | Đệm mọi khung hình trực tiếp thay vì chỉ giữ khung mới nhất |
| `vid_stride` | `1` | Xử lý mỗi khung hình thứ N của video hoặc luồng trực tiếp |
| `show` | `false` | Hiển thị kết quả video và luồng trực tiếp; `q` để dừng |
| `tiling` | `false` | Inference chia ô cho ảnh lớn |
| `overlap_ratio` | `0.2` | Tỉ lệ chồng lấn giữa các ô |
| `output_path` | | Đường dẫn đầu ra tường minh. Nếu không có thì dùng `project/name` khi `save=true` |
| `color_format` | `auto` | Màu đầu vào: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Định dạng đầu ra: `jpg`, `png`, `webp` |
| `device` | `auto` | Thiết bị: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Mô hình phát hiện khuôn mặt (đường dẫn hoặc tên CLI). Bắt buộc với các mô hình gaze |
| `gallery` | | Tệp `.npz` gallery khuôn mặt sinh từ `libreyolo enroll` để đối chiếu danh tính. Chỉ dành cho mô hình face-embedding |
| `gallery_threshold` | `0.4` | Ngưỡng cosine để khớp một danh tính trong gallery |
| `project` | `runs/detect` | Thư mục gốc chứa đầu ra |
| `name` | `predict` | Tên thí nghiệm |
| `exist_ok` | `false` | Dùng lại thư mục đầu ra đã có |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Tắt stderr |
| `verbose` | `false` | Xuất stderr chi tiết |
| `help_json` | `false` | In schema của lệnh dưới dạng JSON rồi thoát |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

Một artifact đã xuất được tải theo đúng cách như một checkpoint, nên
`model=weights/LibreYOLO9s.onnx` và `model=weights/LibreYOLO9s.engine` đều là
giá trị hợp lệ cho `model`. Có ba tùy chọn bị từ chối trên các runtime đó thay
vì bị bỏ qua: `tiling`, `overlap_ratio` và `output_file_format` sẽ thoát với
`config_unsupported` khi một backend runtime không thể đáp ứng chúng.

`half` thì ngược lại. Các runtime đã xuất nhận tham số này và chạy ở FP16;
inference PyTorch gốc ghi log rằng nó bị bỏ qua và tiếp tục chạy ở FP32.

Các mô hình gaze gồm hai giai đoạn và không có bộ phát hiện của riêng chúng,
nên `face_detector` là bắt buộc với chúng. `gallery` chỉ áp dụng cho các mô
hình có tác vụ là `embed`; truyền nó cho bất kỳ thứ gì khác sẽ thoát với
`config_unsupported`.

stdout chỉ mang kết quả và không mang gì khác; tiến trình, cảnh báo và lỗi đều
đi ra stderr. `json=true` in một đối tượng JSON cho mỗi lần gọi, hoặc một đối
tượng cho mỗi khung hình khi chạy ở chế độ stream, mỗi đối tượng đều mang
`schema_version`. `quiet=true` tắt stderr. Dùng cả hai cùng lúc sẽ cho chương
trình đọc tự động một luồng stdout sạch.

Mã thoát là `0` khi thành công, `2` khi dùng sai hoặc lỗi cấu hình, `3` khi
không tìm thấy nguồn, `4` khi không tải được mô hình, và `1` cho các lỗi
runtime khác.

`help_json=true` in ra các tham số, kiểu dữ liệu, giá trị mặc định và cờ của
lệnh dưới dạng JSON mà không chạy gì cả, đây là cách đáng tin cậy để đọc lại
bảng này từ phiên bản đã cài đặt.

Liên quan: [`libreyolo val`](/docs/cli/val) để đo các chỉ số trên một tập dữ
liệu (dataset), [`libreyolo export`](/docs/cli/export) để tạo ra các artifact
runtime nêu ở trên.
