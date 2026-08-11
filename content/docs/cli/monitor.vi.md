---
title: libreyolo monitor
seo_title: Tham chiếu lệnh libreyolo monitor
description: >-
  Phục vụ dashboard trực tiếp cho các lần chạy huấn luyện (run): các tham số
  cùng giá trị mặc định, những gì server đọc từ đĩa, và cách một server bao quát
  nhiều run.
lead: >-
  Phục vụ một dashboard web cho các run huấn luyện, đọc những artifact mà một
  run ghi ra đĩa. Nó không bao giờ gắn vào tiến trình huấn luyện, nên run đang
  chạy, run đã kết thúc và run bị crash đều hiển thị được.
keywords:
  - libreyolo monitor cli
  - dashboard huấn luyện yolo
  - theo dõi quá trình training
  - libreyolo monitor port
  - xem chỉ số training yolo
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo monitor
    mono: true
  - label: Đầu ra
    value: 'URL của server trên stdout, sau đó tiến trình tiếp tục chạy ở foreground'
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        # Theo dõi runs/ và liệt kê mọi run bên dưới nó
        libreyolo monitor
    - label: Thư mục gốc runs khác
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 'Một run, port cố định, không mở trình duyệt'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## Cú pháp

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

Thư mục là tham số vị trí. Mọi thứ còn lại là cặp `key=value`, và dạng POSIX
cũng dùng được, nên `port=9100` và `--port 9100` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `run_dir` | `runs` | Tham số vị trí. Thư mục gốc chứa các run cần theo dõi, hoặc một thư mục run đơn lẻ để mở thẳng. Dù theo cách nào thì mọi run nằm dưới thư mục gốc đều được liệt kê |
| `host` | `127.0.0.1` | Host hoặc interface để bind |
| `port` | `8420` | Port để bind. Nhảy sang port trống kế tiếp nếu đã bị chiếm |
| `no_browser` | `false` | Không tự động mở trình duyệt |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Ẩn stderr |
| `verbose` | `false` | Xuất stderr chi tiết |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Một server, nhiều run

Server theo dõi một thư mục gốc chứa các run chứ không phải một run đơn lẻ, và
định địa chỉ từng run bằng URL, nên nhiều run trên cùng một máy dùng chung một
port. Mở URL gốc để xem trang danh sách, hoặc mở mỗi run một tab; tham số
`?run=` trong từng URL cho biết đó là run nào.

Trỏ lệnh vào một thư mục run đơn lẻ sẽ đặt gốc của server tại thư mục cha của
nó, nên các run cùng cấp vẫn xuất hiện trong danh sách, đồng thời liên kết
thẳng tới run được nêu tên.

### Nó đọc những gì

Dashboard được dựng từ các tệp mà `libreyolo train` ghi ra: `status.json`,
`metrics.jsonl`, `train.log` và các ảnh của run đó. Không có gì được đọc từ
chính tiến trình huấn luyện, nên một run đã kết thúc, hoặc đã chết, hiển thị y
hệt một run đang chạy.

### Điều kiện tiên quyết và port

Phải có sẵn ít nhất một run. Khi không truyền tham số và cũng không có thư mục
`runs/`, lệnh thoát với `source_not_found`; điều tương tự xảy ra khi thư mục
được chỉ định không chứa run nào.

Một port đã bị chiếm sẽ được đẩy sang port kế tiếp, tối đa hai mươi port sau
port được yêu cầu. Nếu cả hai mươi đều thất bại, lệnh thoát với `io_error`. URL
in ra stdout chính là port đã thực sự được bind.

Lệnh chạy ở foreground cho đến khi nhấn Ctrl+C. `json=true` in ra URL, thư mục
gốc đang được theo dõi và số run tìm thấy, dưới dạng một đối tượng có
`schema_version`.

Liên quan: [`libreyolo train`](/docs/cli/train), với các tham số `project` và
`name` quyết định những thư mục run này nằm ở đâu.
