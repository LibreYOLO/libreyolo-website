---
title: libreyolo ui
seo_title: Tham chiếu lệnh libreyolo ui
description: >-
  Khởi chạy giao diện web chạy suy luận (inference) cục bộ: địa chỉ bind, cách
  xử lý cổng, cách chọn thiết bị, và cách lệnh kết thúc.
lead: >-
  Khởi động một web server cục bộ nhận ảnh được kéo thả hoặc dán vào, chạy mô
  hình đã chọn trên chúng, rồi hiển thị kết quả ngay trong trình duyệt.
keywords:
  - libreyolo ui cli
  - giao diện web libreyolo
  - chạy inference cục bộ
  - kéo thả ảnh để nhận diện
  - libreyolo ui đổi cổng
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo ui
    mono: true
  - label: Đầu ra
    value: 'URL của server trên stdout, sau đó tiến trình tiếp tục chạy ở foreground'
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        libreyolo ui
    - label: 'Cổng cố định, không mở trình duyệt'
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 'Trên CPU, đầu ra cho máy đọc'
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## Cú pháp

```bash
libreyolo ui [key=value ...]
```

Các tham số là cặp `key=value`, và dạng POSIX cũng dùng được, nên `port=9000`
và `--port 9000` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `host` | `127.0.0.1` | Host hoặc interface để bind |
| `port` | `8000` | Cổng để bind. Tự nhảy lên cổng trống kế tiếp nếu cổng này đã bị chiếm |
| `device` | `auto` | Thiết bị: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Không tự mở trình duyệt |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Ẩn stderr |
| `verbose` | `false` | Xuất stderr chi tiết |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

Mặc định lệnh bind vào loopback, nên giao diện chỉ truy cập được từ chính máy
này.

Nếu cổng được yêu cầu đang bị chiếm, lệnh thử cổng kế tiếp và tiếp tục như vậy
tới hai mươi cổng sau cổng đã yêu cầu. Hỏng cả hai mươi cổng thì lệnh thoát với
`io_error` kèm gợi ý truyền vào một cổng khác. URL in ra stdout là cổng thực sự
được bind, nên hãy đọc URL đó thay vì mặc định cho rằng đó là cổng bạn đã yêu
cầu.

Trừ khi `no_browser=true`, một tab trình duyệt sẽ mở ở URL đó ngay sau khi bind
xong.

Sau đó lệnh phục vụ ở foreground cho tới khi nhấn Ctrl+C, thao tác này tắt
server một cách sạch sẽ. Không có chế độ chạy tách rời; hãy đẩy nó xuống nền
bằng shell của bạn nếu muốn lấy lại terminal.

`json=true` in ra URL và thiết bị dưới dạng một đối tượng kèm `schema_version`
trước khi server khởi động, đó là cách một script lấy được cổng đã bind.

Liên quan: [`libreyolo label`](/docs/cli/label) để vẽ bounding box và lưu nhãn,
[`libreyolo monitor`](/docs/cli/monitor) để theo dõi các run huấn luyện. Cả hai
đều là web server cục bộ với cùng cách xử lý cổng và trình duyệt.
