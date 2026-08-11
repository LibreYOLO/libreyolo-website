---
title: libreyolo doctor
seo_title: Tham chiếu lệnh libreyolo doctor
description: >-
  Kiểm tra một tập dữ liệu (dataset) phát hiện đối tượng trước khi huấn luyện:
  các tham số cùng giá trị mặc định, các nhóm kiểm tra có thể bỏ qua hoặc chọn
  riêng, và các mã thoát mà CI có thể dựa vào để chặn.
lead: >-
  Chạy một loạt kiểm tra sức khỏe trên một dataset phát hiện đối tượng và báo
  cáo những gì sẽ gây hại cho một lần huấn luyện: tệp bị thiếu, nhãn hỏng, ảnh
  lỗi, rò rỉ giữa các split và mất cân bằng lớp đối tượng.
keywords:
  - libreyolo doctor cli
  - kiểm tra sức khỏe dataset yolo
  - kiểm tra dataset yolo trước khi train
  - kiểm tra rò rỉ dữ liệu giữa các split
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Lệnh
    value: libreyolo doctor
    mono: true
  - label: Bắt buộc
    value: data
    mono: true
  - label: Đầu ra
    value: Báo cáo các phát hiện trên stdout. Thoát với mã 1 khi tìm thấy lỗi
snippets:
  examples:
    - label: Cơ bản
      language: bash
      code: |
        # download=true cho phép coco8.yaml đi kèm tải ảnh về nếu thiếu
        libreyolo doctor coco8.yaml download=true
    - label: 'Chạy nhanh, không giải mã ảnh'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Chặn CI theo các kiểm tra đã chọn
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Cú pháp

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

Dataset là tham số vị trí, và `data=<path>` cũng được chấp nhận như một cách
viết thay thế. Đưa cả hai với giá trị khác nhau sẽ thoát với `config_conflict`.
Mọi thứ còn lại là các cặp `key=value`, và dạng POSIX cũng dùng được, nên
`imgsz=1024` và `--imgsz 1024` là cùng một tham số.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `data` | | Tham số vị trí. Tệp YAML mô tả dataset theo định dạng phát hiện đối tượng của YOLO, ví dụ `coco8.yaml`. Bắt buộc |
| `imgsz` | `640` | Kích thước ảnh huấn luyện, dùng cho các kiểm tra dựa trên pixel như vật thể quá nhỏ |
| `fast` | `false` | Bỏ qua việc giải mã ảnh, kéo theo bỏ luôn các kiểm tra ảnh lỗi, trùng lặp và rò rỉ |
| `skip` | | Danh sách id kiểm tra hoặc nhóm cần bỏ qua, phân tách bằng dấu phẩy, ví dụ `images,labels.tiny_object` |
| `only` | | Danh sách id kiểm tra hoặc nhóm chỉ chạy riêng những mục đó, phân tách bằng dấu phẩy |
| `strict` | `false` | Cảnh báo cũng làm mã thoát báo lỗi, dùng cho các cổng chặn CI |
| `download` | `false` | Cho phép tải dataset qua URL nếu chưa có. Không bao giờ chạy script |
| `json` | `false` | Xuất JSON ra stdout |
| `quiet` | `false` | Tắt stderr |
| `help_json` | `false` | In schema của lệnh dưới dạng JSON rồi thoát |

### Các nhóm kiểm tra

`skip` và `only` nhận cả id kiểm tra đầy đủ lẫn tiền tố nhóm, nên `images` sẽ
chọn mọi kiểm tra `images.*`.

| Nhóm | Bao gồm |
|---|---|
| `config` | Bản thân tệp YAML của dataset: thiếu `names`, `nc` so với `names`, thiếu split, `path` không phân giải được, tên lớp đối tượng trùng nhau |
| `files` | Việc ghép cặp ảnh và nhãn: thiếu nhãn, thiếu ảnh, nhãn mồ côi, phần mở rộng không được hỗ trợ, xung đột chữ hoa chữ thường |
| `labels` | Nội dung nhãn: cú pháp, dòng polygon, id lớp đối tượng ngoài phạm vi, tọa độ ngoài phạm vi, box suy biến, vật thể quá nhỏ, box quá lớn, tỷ lệ cạnh cực đoan, box trùng lặp, ảnh quá dày đặc đối tượng, tệp giống hệt nhau |
| `images` | Dữ liệu pixel: tệp hỏng, hướng EXIF, chế độ màu bất thường, kích thước quá nhỏ hoặc cực đoan, ảnh đồng nhất một màu, trùng lặp chính xác và gần giống |
| `splits` | Rò rỉ giữa các split, chính xác và gần giống |
| `balance` | Phân bố lớp đối tượng: lớp đối tượng không có hoặc có rất ít thực thể, mất cân bằng, độ phủ theo split, tỷ lệ ảnh nền, lệch giữa các split |

## Ví dụ

<code-tabs name="examples" />

## Ghi chú

### Mã thoát

`0` khi không tìm thấy lỗi nào, `1` khi có bất kỳ phát hiện nào là lỗi. Với
`strict=true`, cảnh báo cũng đẩy mã thoát lên `1`, và đó là thiết lập mà một
cổng chặn CI cần.

Các lỗi khi dùng lệnh có mã riêng: `2` cho id kiểm tra hoặc nhóm không tồn tại
trong `skip` hay `only`, `3` khi không tìm thấy dataset, và `3` khi dataset
không có dạng phát hiện đối tượng.

### Việc chọn kiểm tra được phân giải trước khi quét

`skip` và `only` được phân giải dựa trên registry các kiểm tra trước khi đọc bất
cứ thứ gì từ đĩa, nên một lỗi gõ sai sẽ báo hỏng ngay lập tức thay vì sau một
lượt quét ảnh kéo dài. Một bộ chọn không khớp với gì cả là lỗi, và thông báo sẽ
liệt kê các nhóm đã biết.

Nếu tổ hợp `skip`, `only` và `fast` khiến không còn kiểm tra nào để chạy, đó
cũng là lỗi chứ không phải một lần chạy qua im lặng.

### Tải về

Dataset sẽ không được tải về trừ khi `download=true`, và chỉ tải qua URL. Script
tải về bằng Python nhúng trong tệp YAML của dataset không bao giờ được lệnh này
thực thi, dù cờ được đặt thế nào.

### Phạm vi

Các kiểm tra được viết cho dataset phát hiện đối tượng. Dataset có nhãn ở dạng
ước lượng tư thế, phân đoạn hoặc hộp xoay sẽ bị nhận diện và từ chối với
`data_invalid` thay vì bị chấm theo bộ quy tắc sai.

### Đầu ra

Báo cáo cho người đọc được in ra stdout, còn `json=true` thay nó bằng một đối
tượng có cấu trúc chứa các số liệu tổng hợp, thống kê của dataset, mọi phát
hiện, và danh sách các kiểm tra đã bị bỏ qua.

Liên quan: [`libreyolo train`](/docs/cli/train), lần chạy mà lệnh này được thiết
kế để chạy trước.
