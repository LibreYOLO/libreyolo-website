---
title: libreyolo label
seo_title: Tham chiếu lệnh libreyolo label
description: >-
  Khởi chạy công cụ gán nhãn bounding box chạy cục bộ: các tham số cùng giá trị
  mặc định, công tắc hỗ trợ bằng AI, và việc gắn vào một network interface sẽ mở
  ra những gì.
lead: >-
  Khởi chạy một công cụ web cục bộ để vẽ và chỉnh sửa bounding box. Nó ghi ra
  các tệp nhãn theo định dạng gốc của LibreYOLO, nên tập dữ liệu (dataset) được
  gán nhãn ở đây huấn luyện được ngay mà không cần bước chuyển đổi nào.
keywords:
  - libreyolo label cli
  - công cụ gán nhãn bounding box
  - gán nhãn dataset yolo
  - tự động gán nhãn cli
  - chia sẻ libreyolo label
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo label
    mono: true
  - label: Đầu ra
    value: Một URL server trên stdout; nhãn được ghi thành labels/*.txt cạnh các ảnh
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        # Mở trang chủ dự án; chọn hoặc tạo một dataset trong trình duyệt
        libreyolo label
    - label: 'Chỉ thủ công, cổng cố định'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Cho đồng nghiệp cùng vào
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## Cú pháp

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Các tham số là những cặp `key=value`, và dạng POSIX cũng dùng được, nên
`port=9200` và `--port 9200` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `data` | | Tệp YAML hoặc thư mục dataset để mở thẳng. Bắt đầu ở trang chủ dự án khi không đặt |
| `host` | `127.0.0.1` | Host hoặc interface để bind |
| `port` | `8000` | Cổng để bind. Nhảy sang cổng trống kế tiếp nếu cổng này đã bị chiếm |
| `device` | `auto` | Thiết bị cho việc tự động gán nhãn bằng AI: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Tắt tự động gán nhãn bằng AI, chỉ còn công cụ gán nhãn thủ công |
| `no_browser` | `false` | Không tự động mở trình duyệt |
| `share` | `false` | Bind vào `0.0.0.0` để đồng nghiệp trong mạng của bạn có thể vào cùng |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Tắt stderr |
| `verbose` | `false` | Xuất stderr chi tiết |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Nó ghi ra những gì

Các box được lưu thành tệp `labels/*.txt` theo định dạng gốc của LibreYOLO, đúng
định dạng mà `libreyolo train` đọc, nên sau đó không phải chuyển đổi gì cả.
Phiên bản này chỉ xử lý bounding box. Các chỉnh sửa được lưu khi bạn chuyển qua
lại giữa các ảnh.

### Mở một dataset

Khi không có `data`, công cụ khởi động ở trang chủ dự án và dataset được chọn
hoặc tạo từ trình duyệt. Truyền `data=path/to/data.yaml` sẽ mở thẳng dataset đó,
và dòng khởi động báo số lượng ảnh, số lượng lớp đối tượng, và dataset có ghi
được hay không. Một dataset chỉ đọc vẫn mở được và cho biết vì sao không ghi vào
đó được.

### Chia sẻ, và `host` làm gì

`share=true` bind vào địa chỉ wildcard, nhờ đó các máy khác trong mạng của bạn
truy cập được công cụ, trong khi các thao tác quản trị (chuyển hoặc xóa dự án và
khởi động tính toán) vẫn nằm ở máy này.

Đặt `host` thành một interface cụ thể lại làm chuyện khác và kém an toàn hơn:
máy host trở nên không phân biệt được với một client trong mạng, nên mọi client
đều có quyền quản trị. Lệnh sẽ in một cảnh báo ra stderr khi bạn làm vậy. Hãy ưu
tiên `share=true`.

### Cổng và việc tắt

Một cổng đang bị chiếm sẽ chuyển sang cổng kế tiếp, tối đa hai mươi cổng sau
cổng được yêu cầu. Nếu cả hai mươi đều hỏng thì lệnh thoát với `io_error`. URL
in ra stdout chính là cổng đã thực sự được bind. Với `share=true`, kết quả còn
kèm theo `lan_url`, địa chỉ mà đồng nghiệp nên mở.

Lệnh phục vụ ở chế độ foreground cho đến khi Ctrl+C.

Liên quan: [`libreyolo doctor`](/docs/cli/doctor) để kiểm tra dataset đã gán nhãn
trước khi huấn luyện, và [`libreyolo train`](/docs/cli/train) để huấn luyện trên
dataset đó.
