---
title: Phân đoạn ngữ nghĩa
seo_title: Phân đoạn ngữ nghĩa trong LibreYOLO
description: >-
  Gán lớp đối tượng cho mọi pixel trong LibreYOLO: các family phục vụ tác vụ,
  định dạng mặt nạ dày đặc và các lời gọi dự đoán, huấn luyện, xác thực cùng
  xuất.
lead: >-
  Phân đoạn ngữ nghĩa gán một lớp đối tượng cho mỗi pixel của ảnh và không phân
  biệt các thực thể thuộc cùng một lớp. Key tác vụ là semantic.
keywords:
  - phân đoạn ngữ nghĩa python
  - phân loại pixel
  - dự đoán dày đặc
  - huấn luyện mô hình segmentation
  - mIoU
  - thư viện segmentation MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -sem trong tên tệp chọn tác vụ, vì vậy không cần
        # đối số task.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # id lớp (H, W) trên canvas gốc
        print(mask.classes)      # id lớp hiện có đã sắp xếp, bỏ qua 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Mỗi lần một lớp đối tượng
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # boolean (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Family khác, cùng lời gọi'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Trên ADE20K
      language: bash
      code: |
        # ade20k.yaml chứa script tải nhúng cho archive khoảng 1 GB,
        # vì vậy cần quyền tường minh trừ khi dữ liệu có sẵn cục bộ.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() trả về dict thuần, không phải đối tượng.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như checkpoint và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Định nghĩa

Phân đoạn ngữ nghĩa gán nhãn cho pixel, không phải đối tượng. Mỗi pixel nhận một
id lớp đối tượng, và hai chiếc xe chạm nhau trong ảnh trở thành một vùng của lớp
xe mà không có đường biên giữa chúng. Đếm thực thể là [phân đoạn thực
thể](/docs/tasks/instance-segmentation); gán nhãn mọi pixel đồng thời tách thực
thể là [phân đoạn toàn cảnh](/docs/tasks/panoptic-segmentation).

`semantic` là key tác vụ chuẩn, còn hậu tố `-sem` trong tên tệp checkpoint chọn
tác vụ, vì vậy không cần `task=` khi nạp trọng số đã công bố.

`predict()` điền `result.semantic_mask`. `.data` là class map số nguyên
`(H, W)` trên canvas ảnh gốc, `.classes` liệt kê các id hiện có theo thứ tự đã
sắp xếp, còn `.class_mask(id)` trả về vùng chọn boolean `(H, W)` cho một lớp.
Giá trị `255` là nhãn ignore: nó không bao giờ là một lớp, bị loại khỏi loss và
metric, còn `.classes` không liệt kê nó.

## Mô hình

Ba family vừa huấn luyện vừa dự đoán: [SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) và
[DINOv2](/docs/models/dinov2). SegFormer và LingBot-Vision chạy trên package cơ
sở và cung cấp trọng số đã công bố. DINOv2 cần
`pip install "libreyolo[rfdetr]"` và không có checkpoint được LibreYOLO lưu trữ:
nó nạp backbone upstream, còn dense head bắt đầu từ khởi tạo ngẫu nhiên, vì vậy
đây là điểm bắt đầu huấn luyện thay vì mô hình dự đoán sẵn sàng.

Bốn family khác dự đoán, xác thực và xuất, nhưng `train()` phát sinh
`NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) và
[EoMT](/docs/models/eomt).

Tập lớp đối tượng khác nhau theo checkpoint, không phải theo family. Trọng số đã
công bố đến từ các dataset có không gian nhãn rất khác nhau, trong đó có 150
lớp của ADE20K so với 19 lớp của Cityscapes, vì vậy `names` của checkpoint cho
biết nó có thể gán nhãn gì, còn hai checkpoint chỉ so sánh được khi huấn luyện
trên cùng tập lớp.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

Map là argmax trên mỗi pixel, vì vậy không có bước NMS và `iou` không bao giờ có
tác dụng. `conf` và `max_det` được chấp nhận để giữ tương thích API và không làm
gì trên SegFormer, PIDNet cùng các dense predictor khác; EoMT là ngoại lệ, nơi
`conf` lọc quá trình chọn query. Xem [dự đoán](/docs/predict) để biết về nguồn,
stream và cách xử lý kết quả.

## Định dạng dataset

Mỗi ảnh được ghép với mặt nạ một kênh dày đặc thay vì tệp nhãn `.txt`, được tìm
bằng cách thay `images` bằng thư mục mặt nạ trong đường dẫn ảnh.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Mặt nạ là ảnh một kênh không mất dữ liệu, thường là PNG, còn PNG palette mode
được đọc dưới dạng index palette. Mỗi giá trị pixel là id lớp trong `0..nc-1`,
giá trị `255` nghĩa là ignore, và độ phân giải mặt nạ phải bằng độ phân giải ảnh
ghép cặp.

YAML nhận hai key ngoài hợp đồng dùng chung:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` là tên thư mục được thay cho `images`, mặc định là `masks`.
`label_mapping` là mapping lại `{source_id: train_id}` tùy chọn được áp dụng cho
giá trị pixel mặt nạ khi nạp, đây là cách dataset được đánh số từ 1 đến 150 trở
thành 0 đến 149; mọi giá trị nguồn không được ánh xạ trở thành ignore, còn mọi
train id phải nằm trong `0..nc-1`.

Bỏ `masks_dir` sẽ chuyển loader sang fallback: mặt nạ được rasterize tại thời
điểm nạp từ nhãn polygon, được phân giải theo quy ước `images` sang `labels`
thông thường, và lớp `background` được thêm sau các lớp đối tượng, vì vậy `nc`
tăng thêm một.

Loader chuẩn là `libreyolo.data.SemanticDataset`.

## Huấn luyện

<code-tabs name="train" />

`imgsz` bị ràng buộc ở đây theo cách không có trên detector. Mỗi family khai
báo số chia mà đầu vào phải là bội số của nó, được đặt theo patch grid hoặc
output stride, còn cả huấn luyện lẫn xác thực đều phát sinh `ValueError` trước
khi lượt chạy bắt đầu nếu `imgsz` không chia hết. Số chia là 32 cho SegFormer,
16 cho LingBot-Vision và EoMT, 14 cho DINOv2, 8 cho FCN cùng PIDNet. Xem [huấn
luyện](/docs/train) để biết về dataset, augmentation, multi-GPU và logger.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`, được tính trên split do
`val` trong YAML dataset đặt tên.

<code-tabs name="val" />

`metrics/mIoU` là mean intersection over union: trên mỗi lớp, độ chồng lấn giữa
pixel dự đoán và pixel thật chia cho hợp của chúng, rồi lấy trung bình trên các
lớp. Đây là con số chính và được dùng để chọn epoch tốt nhất trong khi huấn
luyện. `metrics/pixel_accuracy` là tỷ lệ pixel được gán đúng lớp, có thể bị lớp
background lớn làm tăng giả tạo, vì vậy mIoU là số liệu nên so sánh. Pixel được
đánh dấu `255` không được tính vào phần nào. Dictionary còn có `fitness`, bản
sao của giá trị mIoU.

## Xuất

<code-tabs name="export" />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Phạm
vi định dạng khác nhau theo family; ma trận trên từng trang mô hình được tạo từ
tập đã xác thực thay vì nhập thủ công. Xem [xuất và triển khai](/docs/export) để
biết các định dạng, thành phần bổ sung và ràng buộc.
