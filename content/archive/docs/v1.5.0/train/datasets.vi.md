---
title: Tập dữ liệu
seo_title: Tập dữ liệu huấn luyện trong LibreYOLO
description: >-
  YAML dataset mà LibreYOLO đọc, bố cục thư mục cần có, cách tự động tải xuống
  hoạt động và lệnh doctor kiểm tra dataset trước khi huấn luyện.
lead: >-
  Một tập dữ liệu LibreYOLO là tệp YAML đặt tên thư mục gốc, các split và tên
  lớp đối tượng. Mọi nội dung khác, gồm cả vị trí tệp nhãn, được suy ra từ tệp
  đó theo quy ước.
keywords:
  - định dạng dataset yolo
  - data.yaml
  - huấn luyện dataset riêng
  - định dạng nhãn yolo
  - dataset coco json
  - tự động tải dataset
  - libreyolo doctor
  - kiểm tra mất cân bằng lớp
  - rò rỉ dữ liệu train val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Có thể dùng tên đi kèm, đường dẫn tương đối hoặc đường dẫn tuyệt đối.
        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Kiểm tra dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Cho job CI thất bại cả khi có cảnh báo
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Bỏ qua lượt decode ảnh
      language: bash
      code: |
        # Chỉ đọc nhãn và YAML. Kiểm tra hỏng dữ liệu, trùng lặp và rò rỉ giữa
        # các split đều cần pixel, vì vậy chúng bị bỏ qua.
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Trỏ train tới một dataset

`data=` nhận đường dẫn YAML hoặc tên cấu hình đi kèm package.

<code-tabs name="train" />

Tên được phân giải theo thứ tự cố định: đường dẫn tuyệt đối có tồn tại, sau đó
tên như đã cung cấp tương đối với thư mục làm việc, rồi cùng tên đó có thêm
`.yaml`, cuối cùng là thư mục cấu hình đi kèm. Khi không có gì khớp, lỗi nêu mọi
thư mục đã tìm kiếm và liệt kê các cấu hình đi kèm.

## Cấu hình đi kèm

Có 13 cấu hình dataset đi kèm bên trong package, tại
`libreyolo/config/datasets/`.

| Cấu hình | Tác vụ | Ghi chú |
|---|---|---|
| `coco8.yaml` | detect | 8 ảnh, tải từ URL thuần |
| `coco128.yaml` | detect | 128 ảnh |
| `coco1000.yaml` | detect | 800 train, 200 val |
| `coco5000.yaml` | detect | 4000 train, 1000 val |
| `coco.yaml` | detect | COCO 2017 đầy đủ |
| `coco-val-only.yaml` | detect | chỉ val2017 |
| `coco8-pose.yaml` | pose | 8 ảnh, keypoint COCO-17 |
| `coco-pose.yaml` | pose | keypoint COCO 2017 |
| `ade20k.yaml` | semantic | 150 lớp đối tượng |
| `cityscapes.yaml` | semantic | 19 lớp đối tượng, tải thủ công |
| `cocostuff.yaml` | semantic | 182 lớp đối tượng, tải thủ công |
| `gopro.yaml` | restore | cặp ảnh khử nhòe |
| `sr8.yaml` | restore | cặp ảnh super-resolution |

Chỉ `coco8.yaml` và `coco128.yaml` có URL tải xuống thuần. Các cấu hình còn lại
hoặc chứa khối tải xuống bằng Python, cần cơ chế đồng ý được mô tả bên dưới,
hoặc yêu cầu dữ liệu đã có trên ổ đĩa.

## Vị trí của dataset trên ổ đĩa

Key YAML `path` đặt tên thư mục gốc của dataset. `path` tuyệt đối được dùng
nguyên trạng. `path` tương đối được tìm trước trong thư mục dataset, sau đó bên
cạnh chính tệp YAML, còn dataset sắp được tải xuống sẽ nằm trong thư mục dataset.

Thư mục đó là `~/datasets`, có thể ghi đè bằng biến môi trường
`LIBREYOLO_DATASETS_DIR`. Không có tệp cài đặt dành cho nó.

## Các key YAML

```yaml
path: my-dataset        # thư mục gốc dataset
train: images/train     # bắt buộc để huấn luyện
val: images/val         # bắt buộc để xác thực
test: images/test       # tùy chọn
nc: 3                   # tùy chọn; phải khớp với names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # tùy chọn
```

Mỗi key `train`, `val` và `test` chấp nhận một thư mục ảnh, tệp `.txt` liệt kê
mỗi dòng một đường dẫn ảnh, hoặc danh sách trộn cả hai. Các dòng trong danh sách
`.txt` có thể là đường dẫn tương đối; khi đó chúng được phân giải theo thư mục
của chính tệp danh sách. Các dòng bắt đầu bằng `#` bị bỏ qua.

`names` có thể là danh sách hoặc mapping có key số nguyên. `nc` là tùy chọn; khi
cả hai cùng có mặt nhưng không khớp, doctor báo cáo lỗi.

## Bố cục thư mục và tệp nhãn

Phát hiện, phân đoạn, tư thế và hộp xoay đều dùng chung một bố cục. Đường dẫn
nhãn được suy ra từ đường dẫn ảnh bằng cách đổi component thư mục `images` thành
`labels` và đổi phần mở rộng thành `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Chỉ component đường dẫn mang tên đầy đủ `images` được thay đổi, vì vậy thư mục
có tên `images_old` được giữ nguyên.

Một dòng phát hiện có năm trường, tất cả đều được chuẩn hóa về `[0, 1]` theo
chiều rộng và chiều cao ảnh gốc:

```text
<class_id> <cx> <cy> <w> <h>
```

Tệp nhãn bị thiếu hoặc rỗng nghĩa là ảnh không có vật thể và được huấn luyện như
background thay vì phát sinh lỗi. Dòng có nhiều hơn năm trường được đọc như
polygon và hộp của nó trở thành phạm vi bao của polygon, vì vậy dữ liệu xuất từ
phân đoạn dùng cho huấn luyện phát hiện sẽ được nạp mà không bị phản đối. Doctor
báo cáo số lượng dòng đi qua đường dẫn đó.

## Các tác vụ khác

Phân đoạn giữ nguyên bố cục với các dòng polygon,
`<class_id> <x1> <y1> ... <xN> <yN>`, có ít nhất ba điểm. Dòng phát hiện năm
trường được chấp nhận và biểu thị một thực thể hình chữ nhật.

Tư thế thêm `kpt_shape: [K, D]` và permutation `flip_idx` tùy chọn vào YAML. Mỗi
dòng có chính xác `5 + K * D` trường: hộp, sau đó là `K` keypoint dạng `x y` hoặc
`x y v`, với độ hiển thị `0`, `1` hoặc `2`.

Hộp xoay dùng chính xác chín trường, gồm lớp đối tượng rồi bốn điểm góc ở tọa độ
chuẩn hóa. Tệp không lưu góc.

Phân đoạn ngữ nghĩa ghép mỗi ảnh với một mặt nạ một kênh cùng độ phân giải, được
phân giải bằng cách thay `images` bằng `masks_dir` (mặc định là `masks`). Giá trị
pixel `255` nghĩa là bỏ qua. `label_mapping` ánh xạ lại id nguồn sang id huấn
luyện tại thời điểm nạp.

Phân loại dùng cây ImageFolder thay vì tệp nhãn, với mỗi thư mục `train/` và
`val/` chứa một thư mục con cho từng lớp đối tượng. Mapping từ lớp đối tượng
sang index là thứ tự tên thư mục đã sắp xếp.

Phục hồi ghép một đầu vào suy giảm với target sạch có cùng độ phân giải thông
qua `input_dir` và `target_dir`. Độ sâu, pháp tuyến bề mặt và cạnh đều ghép ảnh
với dense map thông qua key thư mục riêng.

Hợp đồng đầy đủ theo tác vụ, gồm quy ước tỷ lệ độ sâu và cách mã hóa PNG segment
id toàn cảnh, nằm trong `docs/dataset_schema.md` ở repo thư viện.

## COCO JSON gốc

Có thể dùng trực tiếp tệp nhãn COCO JSON. Thêm một mapping `annotations`, và
đường dẫn split trở thành thư mục gốc của ảnh:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Khi có `names`, tên category trong JSON phải khớp và `names` định nghĩa các id
nhãn mà mô hình dự đoán. Khi không có `names`, category id COCO được sắp xếp và
ánh xạ dày đặc vào `0..N-1`.

Đường dẫn này yêu cầu một thư mục ảnh cho mỗi split. Danh sách đường dẫn hoặc
danh sách ảnh `.txt` sẽ phát sinh lỗi thay vì âm thầm nạp một tập khác.

## Tự động tải xuống

Dataset được xem là đã có khi đường dẫn `train` hoặc `val` phân giải thành thư
mục không rỗng hoặc tệp có tồn tại. Khi không có dữ liệu và YAML có key
`download`, giá trị này quyết định điều xảy ra tiếp theo.

URL `http` hoặc `https` được tải về và nếu là zip thì được giải nén vào thư mục
gốc của dataset. Mọi giá trị khác được xử lý như script Python nhúng và chỉ chạy
khi `allow_download_scripts=True`. Nếu không bật tùy chọn đó, script bị bỏ qua
kèm cảnh báo và quá trình huấn luyện tiếp tục với dữ liệu có trên ổ đĩa.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

Flag này là cổng thực thi mã, không phải cổng mạng. URL luôn được tải; chỉ các
khối `download: |` mới cần flag. CLI in cảnh báo khi flag được bật, còn doctor
không bao giờ bật nó.

## Kiểm tra dataset trước khi huấn luyện

`libreyolo doctor` đọc dataset phát hiện và báo cáo vấn đề trước khi GPU tham
gia. Lệnh thoát với mã 1 khi tìm thấy lỗi, vì vậy có thể dùng làm cổng CI.

<code-tabs name="doctor" />

Các kiểm tra thuộc sáu family:

| Family | Nội dung cần tìm |
|---|---|
| `config` | thiếu `names`, `nc` không khớp `names`, split thiếu hoặc rỗng, tên lớp đối tượng trùng lặp |
| `files` | ảnh không có tệp nhãn, nhãn không có ảnh, ảnh được liệt kê nhưng bị thiếu trong split, stem trùng nhau |
| `labels` | dòng sai định dạng, id lớp ngoài `[0, nc)`, tọa độ ngoài `[0, 1]`, hộp có diện tích bằng 0, hộp quá nhỏ hoặc quá lớn, hộp trùng lặp, tệp nhãn giống hệt từng byte |
| `balance` | lớp có 0 hoặc ít thực thể, tỷ lệ mất cân bằng lớp, lớp chỉ có trong một split, tỷ lệ ảnh background |
| `images` | tệp không decode được, xoay EXIF, bố cục kênh lạ, ảnh đồng nhất, ảnh trùng chính xác và gần trùng |
| `splits` | cùng một ảnh xuất hiện trong hai split, giống chính xác hoặc gần giống |

`--only` và `--skip` nhận check id hoặc tiền tố family, vì vậy
`skip=images,labels.tiny_object` là hợp lệ. `--fast` loại mọi kiểm tra cần decode
pixel, tức các family `images` và `splits`.

Cần biết hai hành vi. `--strict` khiến cảnh báo cũng làm exit code thất bại như
lỗi. Đồng thời, doctor chỉ hỗ trợ dataset phát hiện: dataset tư thế, phân đoạn
hoặc hộp xoay bị từ chối với thông báo nêu loại đã phát hiện, thay vì được kiểm
tra theo hợp đồng sai.

## Nội dung liên quan

- Xem [Siêu tham số](/docs/train/hyperparameters) để biết các đối số `train()`
  nhận sau khi dữ liệu đã sẵn sàng.
- Xem [Xác thực và metric](/docs/train/validation) để đánh giá trên split `val`
  hoặc `test`.
