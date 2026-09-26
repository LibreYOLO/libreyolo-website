---
title: Phân đoạn toàn cảnh
seo_title: Phân đoạn toàn cảnh trong LibreYOLO
description: >-
  Gán mỗi pixel vào một segment trong LibreYOLO: các family phục vụ tác vụ, định
  dạng dataset COCO-panoptic và lời gọi dự đoán cùng xác thực.
lead: >-
  Phân đoạn toàn cảnh gán mỗi pixel vào đúng một segment không chồng lấn, hợp
  nhất các thực thể đối tượng đếm được với vùng background vô định hình. Key tác
  vụ là panoptic.
keywords:
  - phân đoạn toàn cảnh python
  - panoptic quality
  - phân đoạn things và stuff
  - định dạng COCO panoptic
  - segment id map
  - metric PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -panoptic trong tên tệp chọn tác vụ, vì vậy không cần
        # đối số task.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # segment id (H, W)
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Mỗi lần một segment
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # boolean (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Checkpoint nhỏ hơn
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() trả về dict thuần, không phải đối tượng.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Định nghĩa

Phân đoạn toàn cảnh là sự hợp nhất của hai tác vụ phân đoạn còn lại. Mỗi pixel
nhận đúng một segment, các segment không bao giờ chồng lấn, và một segment là
thing, tức thực thể đối tượng đếm được, hoặc stuff, tức vùng vô định hình như
bầu trời hay đường. Điều đó khiến tác vụ nghiêm ngặt hơn [phân đoạn thực
thể](/docs/tasks/instance-segmentation), vốn để pixel background không được gán
và cho phép mặt nạ chồng lấn, đồng thời nghiêm ngặt hơn [phân đoạn ngữ
nghĩa](/docs/tasks/semantic-segmentation), vốn gán nhãn mọi pixel nhưng merge
các thực thể chạm nhau thuộc cùng một lớp.

`panoptic` là key tác vụ chuẩn, còn hậu tố `-panoptic` trong tên tệp checkpoint
chọn tác vụ này, vì vậy không cần `task=` khi nạp trọng số đã công bố.

`predict()` điền `result.panoptic`. `.data` là segment-id map số nguyên `(H, W)`
trên canvas ảnh gốc. `.segments_info` là danh sách dict, mỗi segment một dict,
mang ít nhất `{"id", "category_id"}`, trong đó `id` khớp một giá trị trong map
và `category_id` lập index `result.names`. `.segment_ids` liệt kê các id hiện có
theo thứ tự đã sắp xếp, còn `.segment_mask(id)` trả về vùng chọn boolean
`(H, W)` cho một segment. Segment id `0` là giá trị void: pixel không được gán
nhãn, bị loại khỏi metric và không nằm trong `.segment_ids`.

Thing hay stuff là thuộc tính của category, không phải của segment riêng lẻ. Nó
nằm trong metadata category của tập nhãn, và payload dự đoán có thể sao chép vào
từng segment dưới dạng `"isthing"` để tiện sử dụng, nhưng metadata category vẫn
là nguồn có thẩm quyền.

## Mô hình

[EoMT](/docs/models/eomt) là family phục vụ tác vụ này qua `LibreYOLO()`. Nó chạy
trên package cơ sở và cung cấp checkpoint panoptic ba kích thước s, b và l, được
huấn luyện trên COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) cũng phát ra panoptic map. Đây
là mô hình sinh theo prompt có factory riêng `LibreVLM` và thành phần bổ sung
riêng; khi không đặt từ vựng, nó quay về các category COCO panoptic đã dùng để
tinh chỉnh. Trọng số chỉ dùng cho mục đích phi thương mại. Độ trễ trên mỗi ảnh
cao hơn nhiều so với segmenter chuyên dụng vì mỗi dự đoán là một lượt diffusion
decode.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

`conf` lọc quá trình chọn query. Xem [dự đoán](/docs/predict) để biết về nguồn,
stream và cách xử lý kết quả.

## Định dạng dataset

LibreYOLO áp dụng nguyên vẹn định dạng COCO-panoptic của Kirillov và cộng sự,
CVPR 2019. Không có bố cục panoptic riêng cho LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Mỗi ảnh được ghép với một PNG RGB cùng độ phân giải, trong đó màu của mỗi pixel
mã hóa id segment mà nó thuộc về:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Segment id `0`, màu đen RGB, là void: các pixel không được gán nhãn không thưởng
cũng không phạt dự đoán. Mọi pixel khác thuộc về đúng một segment.

JSON liệt kê PNG segment-id và các segment bên trong theo từng ảnh:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` đặt tên PNG trong thư mục panoptic, còn
`segments_info[].id` khớp một giá trị trong PNG đó. `iscrowd` đánh dấu vùng
nhóm: chúng không bao giờ được tính là false negative, còn dự đoán bao phủ phần
lớn một vùng nhóm không phải false positive. `isthing` nằm trên `categories` và
không bao giờ nằm trên segment riêng lẻ.

YAML trỏ tới cả hai:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

Mỗi key `annotations` và `panoptic_dir` nhận một đường dẫn hoặc mapping theo
split. Category id COCO thô thường không liền nhau, trong khi mô hình dự đoán
dãy liền `0..nc-1`, vì vậy id được ánh xạ lại qua `names` theo tên category.
Category JSON không có trong `names` là lỗi thay vì bị âm thầm loại bỏ, vì loại
nó sẽ khiến category luôn bị tính là false negative.

Loader chuẩn là `libreyolo.data.PanopticDataset`.

## Huấn luyện

Hiện không family nào huấn luyện phân đoạn toàn cảnh trong LibreYOLO:
`train()` của EoMT phát sinh `NotImplementedError`, vì vậy checkpoint panoptic
được sử dụng như đã công bố.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`, được tính ở độ phân giải
ground truth trên split do `val` trong YAML dataset đặt tên. Segment dự đoán và
segment thật thuộc cùng category khớp khi IoU vượt quá 0.5, và phép khớp đó là
duy nhất.

<code-tabs name="val" />

`metrics/PQ` là Panoptic Quality, con số chính. Trong một category, nó là tích
của hai yếu tố. Segmentation quality là IoU trung bình trên các segment khớp và
cho biết các shape khớp nhau tốt đến đâu. Recognition quality là
`TP / (TP + 0.5 FP + 0.5 FN)`, điểm F1 của chính phép khớp, và cho biết bao nhiêu
segment được tìm thấy. Sau đó cả ba số liệu được lấy trung bình trên các
category đã xuất hiện và báo cáo dưới dạng `metrics/PQ`, `metrics/SQ` và
`metrics/RQ`, vì vậy PQ báo cáo là trung bình của tích theo từng category thay
vì tích của hai giá trị trung bình được báo cáo.

`metrics/PQ_things` và `metrics/PQ_stuff` lấy trung bình cùng PQ theo category
trên các category thing và stuff riêng, còn `metrics/categories` đếm số
category đã xuất hiện và do đó được đưa vào trung bình. Dictionary cũng có
`fitness`, một bản sao giá trị PQ.

## Xuất

Checkpoint panoptic không xuất được. `export()` phát sinh
`NotImplementedError` cho tác vụ này vì đầu ra query-mask chưa có hợp đồng xuất
runtime. Tác vụ semantic của EoMT có thể xuất; xem [phân đoạn ngữ
nghĩa](/docs/tasks/semantic-segmentation) và [xuất cùng triển
khai](/docs/export).
