---
title: tiện ích libreyolo
seo_title: Tham chiếu các lệnh tiện ích của CLI libreyolo
description: >-
  Các lệnh nhỏ của LibreYOLO: version, checks, models, formats, cfg, info,
  metadata, enroll và compare, mỗi lệnh kèm tham số và giá trị mặc định của nó.
lead: >-
  Chín lệnh dùng để báo cáo hoặc kiểm tra thay vì tính toán. Chúng in ra thông
  tin môi trường, danh mục mô hình và định dạng, các giá trị mặc định đã được
  phân giải, chi tiết của checkpoint, và chúng xây dựng cùng truy vấn một bộ sưu
  tập khuôn mặt (gallery).
keywords:
  - libreyolo version
  - libreyolo checks
  - liệt kê mô hình libreyolo
  - định dạng xuất libreyolo
  - xem metadata checkpoint yolo
  - libreyolo cfg mặc định
  - gallery khuôn mặt libreyolo enroll
  - so sánh khuôn mặt libreyolo compare
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: Đầu ra
    value: >-
      stdout, dạng văn bản hoặc với json=true là một đối tượng duy nhất mang
      schema_version
snippets:
  examples:
    - label: Môi trường
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Những gì có sẵn
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Kiểm tra một checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## Cú pháp

```bash
libreyolo <command> [key=value ...]
```

Các tham số là cặp `key=value`, và dạng POSIX cũng dùng được, nên `model=x` và
`--model x` là cùng một tham số. Mọi lệnh ở đây đều ghi kết quả ra stdout và
chấp nhận `json=true` cùng `quiet=true`.

Lệnh gốc có một cờ riêng của nó, `libreyolo --version`, in ra chuỗi phiên bản
rồi thoát. Đầu ra đó nhỏ hơn so với lệnh `version` bên dưới.

## version

In phiên bản LibreYOLO cùng các phiên bản Python, torch và CUDA mà nó đang chạy
trên đó.

```bash
libreyolo version
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

## checks

In môi trường chi tiết hơn: Python, torch, CUDA, cuDNN, mọi GPU phát hiện được
kèm tên và bộ nhớ của nó, và phiên bản đã cài của từng gói tùy chọn mà các
đường xuất mô hình sử dụng.

```bash
libreyolo checks
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

Danh sách gói bao gồm `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` và `scipy`. Gói chưa được cài sẽ được báo đúng như vậy chứ không
bị bỏ qua, nên một lần xuất mô hình thất bại có thể truy ngược về một phụ thuộc
còn thiếu chỉ từ một lệnh này.

## models

Liệt kê mọi họ mô hình cùng các tác vụ, các kích thước, những tên CLI phân giải
ra checkpoint của họ đó, và độ phân giải đầu vào của từng kích thước.

```bash
libreyolo models
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

Một họ mô hình có phụ thuộc tùy chọn chưa được cài sẽ được liệt kê là không khả
dụng, kèm theo dòng `pip install` giúp nó khả dụng. Các tên CLI chính là thứ mà
`model=` chấp nhận như một dạng viết tắt: `yolox-s` phân giải thành
`LibreYOLOXs.pt`, và những tác vụ không phải phát hiện đối tượng mang thêm hậu
tố tác vụ của chúng.

## formats

Liệt kê các định dạng xuất mà môi trường đã cài có thể tạo ra, kèm phần mở rộng
tệp của từng định dạng và việc nó có hỗ trợ FP16 và INT8 hay không.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `family` | | Hiển thị các mức hỗ trợ cho một họ mô hình. `model=` được chấp nhận như cùng một tùy chọn |
| `task` | | Tác vụ chuẩn của mô hình. Là tác vụ mặc định của họ mô hình khi không đặt |
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

Khi không có `family`, đầu ra chỉ là danh mục định dạng. Khi có, mỗi định dạng
được bổ sung mức hỗ trợ cho họ mô hình và tác vụ đó, lý do đằng sau mức đó, và
mọi ràng buộc gắn với nó. Một họ mô hình không xác định, hoặc một tác vụ mà họ
mô hình đó không hỗ trợ, là lỗi cách dùng.

Các bí danh định dạng xuất hiện bên cạnh tên chuẩn của chúng: `engine` cho
`tensorrt`, `litert` cho `tflite`.

## cfg

In ra cấu hình mặc định đã được phân giải: các mặc định của huấn luyện, các mặc
định của kiểm định (validation), các mặc định của dự đoán, và các ghi đè theo
từng họ mô hình.

```bash
libreyolo cfg
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

Các giá trị được đọc từ chính các dataclass cấu hình, không phải từ một bản sao,
nên đây là nguồn chuẩn cho biết một lần chạy huấn luyện sẽ dùng gì khi bạn không
truyền tham số. `family_overrides` là phần trả lời câu hỏi vì sao một họ mô hình
lại được huấn luyện với những thiết lập mà bạn không yêu cầu. Xem
[`libreyolo train`](/docs/cli/train) để biết các ghi đè đó được áp dụng ra sao.

## info

Tải một mô hình trên CPU và báo cáo họ mô hình, kích thước, số lượng tham số,
các lớp đối tượng, và mức hỗ trợ xuất cho từng định dạng.

```bash
libreyolo info model=<name|path>
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `model` | | Tên mô hình hoặc đường dẫn tới trọng số. Bắt buộc |
| `detailed` | `false` | Kèm chi tiết theo từng tham số |
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

## metadata

Đọc metadata của một checkpoint mà không dựng mô hình, và kiểm tra nó theo
schema checkpoint của LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `path` | | Đường dẫn tới một checkpoint `.pt`. Bắt buộc |
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

Những mục chứa tensor lớn được tóm tắt thay vì in ra, nên đầu ra vẫn dễ đọc với
một checkpoint huấn luyện đầy đủ. Một checkpoint không tồn tại sẽ thoát với
`checkpoint_not_found`, còn checkpoint có metadata không qua được kiểm tra sẽ in
ra các lỗi và thoát với mã `1`.

## enroll

Xây dựng một gallery khuôn mặt từ cây thư mục mỗi người một thư mục, để những
lần dự đoán sau có thể gọi tên các khuôn mặt mà chúng tìm thấy.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `model` | | Mô hình embedding khuôn mặt, đường dẫn hoặc tên. Bắt buộc |
| `source` | | Cây thư mục mỗi người một thư mục, `source/<identity>/*.jpg`. Bắt buộc |
| `gallery` | | Tệp gallery đầu ra `.npz`. Được mở rộng tại chỗ nếu đã tồn tại. Bắt buộc |
| `face_detector` | | Bộ phát hiện khuôn mặt: một tệp YuNet `.onnx` hoặc một bộ phát hiện của LibreYOLO. Là bộ phát hiện mặc định của họ mô hình khi không đặt |
| `device` | `auto` | Thiết bị: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

```bash
# people/ chứa mỗi danh tính một thư mục; tên thư mục trở thành danh tính
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

Tên thư mục con chính là danh tính. Một ảnh tham chiếu không phát hiện được
khuôn mặt nào sẽ bị bỏ qua kèm một dòng trên stderr và phần còn lại vẫn tiếp
tục; một nguồn không có thư mục con danh tính nào, hoặc một nguồn hoàn toàn
không tìm thấy khuôn mặt nào, là lỗi.

Truyền tệp kết quả cho
[`libreyolo predict`](/docs/cli/predict) dưới dạng `gallery=people.npz` để các
phát hiện mang theo một danh tính và một điểm khớp.

## compare

Báo cáo độ tương đồng cosine giữa hai ảnh khuôn mặt và việc nó có vượt ngưỡng
cùng danh tính hay không.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `model` | | Mô hình embedding khuôn mặt, đường dẫn hoặc tên. Bắt buộc |
| `source` | | Ảnh thứ nhất. Bắt buộc |
| `source2` | | Ảnh thứ hai để so sánh với ảnh đầu. Bắt buộc |
| `face_detector` | | Bộ phát hiện khuôn mặt: một tệp YuNet `.onnx` hoặc một bộ phát hiện của LibreYOLO |
| `threshold` | `0.4` | Ngưỡng độ tương đồng cosine cho quyết định cùng danh tính |
| `device` | `auto` | Thiết bị: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Đầu ra JSON ra stdout |
| `quiet` | `false` | Tắt stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` được đăng ký như một tên thứ hai cho lệnh này và nhận cùng
các tham số.

Cả `compare` lẫn `enroll` đều cần một mô hình có tác vụ là embedding khuôn mặt.
Bất kỳ thứ gì khác sẽ thoát với `config_unsupported`. Cả đường dẫn ảnh cục bộ
lẫn URL `http` hoặc `https` đều được chấp nhận làm nguồn.

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

stdout mang kết quả; tiến trình và cảnh báo đi ra stderr. `json=true` in ra một
đối tượng duy nhất có `schema_version`, đây là dạng nên đọc từ một script. Đầu
ra dạng văn bản là mặc định và được dành cho người đọc.

Mã thoát theo cùng bảng ánh xạ như phần còn lại của CLI: `0` khi thành công, `2`
cho lỗi cách dùng hoặc lỗi cấu hình, `3` khi không tìm thấy nguồn, `4` khi không
tải được mô hình hoặc checkpoint, và `1` cho các lỗi runtime khác.

Liên quan: [`libreyolo doctor`](/docs/cli/doctor), lệnh kiểm tra ở phía tập dữ
liệu (dataset), và [`libreyolo profile`](/docs/cli/profile), lệnh ở phía hiệu
năng.
