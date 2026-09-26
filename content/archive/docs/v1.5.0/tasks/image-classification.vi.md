---
title: Phân loại ảnh
seo_title: Phân loại ảnh trong LibreYOLO
description: >-
  Gán nhãn cho toàn ảnh trong LibreYOLO: các family phục vụ tác vụ, bố cục
  dataset ImageFolder và các lời gọi dự đoán, huấn luyện, xác thực cùng xuất.
lead: >-
  Phân loại ảnh gán một phân phối nhãn cho toàn ảnh và không định vị gì bên
  trong. Key tác vụ là classify.
keywords:
  - phân loại ảnh python
  - huấn luyện image classifier
  - dataset ImageFolder
  - độ chính xác top-1
  - phân loại zero-shot
  - thư viện phân loại MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -cls trong tên tệp chọn tác vụ, vì vậy không cần
        # đối số task.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Toàn bộ phân phối
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data là toàn bộ vector (C,); top5/top5conf là các view đã sắp xếp.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, không huấn luyện'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP tính điểm ảnh so với text prompt, vì vậy tập nhãn được đặt

        # tại thời điểm gọi thay vì ghi cố định trong checkpoint.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # imagenette160 là tên dataset đã biết và được tải trong lần sử dụng đầu
        tiên.

        # Truyền thư mục có split train/ cho dữ liệu riêng của bạn.

        model = LibreYOLO("LibreResNet50-cls.pt")

        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() trả về dict thuần, không phải đối tượng.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như checkpoint và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Định nghĩa

Phân loại ảnh tạo một điểm số trên mỗi lớp đối tượng cho toàn ảnh và hoàn toàn
không có tọa độ. Nó trả lời trong ảnh có gì, không bao giờ trả lời ở đâu, đây là
điểm phân biệt với [phát hiện đối tượng](/docs/tasks/object-detection).

`classify` là key tác vụ chuẩn, còn hậu tố `-cls` trong tên tệp checkpoint chọn
tác vụ. Hậu tố này là bắt buộc thay vì tùy chọn trên classification family, vì
vậy `LibreResNet50.pt` không được đọc như classifier, chỉ
`LibreResNet50-cls.pt` mới được đọc như vậy.

`predict()` điền `result.probs` và để `boxes` rỗng. `.data` là toàn bộ vector
điểm số, `.top1` là index của điểm cao nhất và `.top1conf` là giá trị của nó,
`.top5` là năm index cao nhất theo thứ tự giảm dần và `.top5conf` là các điểm
tương ứng. Index trỏ vào `result.names`. Slicing đối tượng `Results` không bao
giờ cắt `probs` vì vector thuộc về ảnh, không phải một dòng.

## Mô hình

Năm family vừa huấn luyện vừa dự đoán: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) và
[DINOv2](/docs/models/dinov2). Bốn family đầu chạy trên package cơ sở và cung
cấp trọng số đã công bố. DINOv2 cần `pip install "libreyolo[rfdetr]"` và không
có checkpoint được LibreYOLO lưu trữ: nó nạp backbone upstream với linear head
được khởi tạo ngẫu nhiên, vì vậy đây là điểm bắt đầu tinh chỉnh thay vì mô hình
dự đoán sẵn sàng.

Năm family khác dự đoán, xác thực và xuất, nhưng `train()` phát sinh
`NotImplementedError`: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) và
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) và [SigLIP2](/docs/models/siglip2) phân loại mà không
có tập nhãn cố định. Chúng tính điểm ảnh so với text prompt, vì vậy
`set_classes()` định nghĩa lớp đối tượng tại thời điểm gọi và hoàn toàn không có
bước huấn luyện cho tập nhãn mới. Cả hai cũng phục vụ tác vụ `embed`.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

`conf`, `iou` và `max_det` không có tác dụng ở đây: không có candidate để đặt
ngưỡng hoặc suppression, chỉ có một phân phối. Xem [dự đoán](/docs/predict) để
biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Phân loại dùng cây thư mục, không dùng tệp nhãn hoặc YAML. `data` là thư mục gốc
dataset.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` là bắt buộc và định nghĩa mapping từ lớp đối tượng tới index bằng tên
thư mục đã sắp xếp, vì vậy thư mục đầu tiên theo bảng chữ cái trở thành lớp 0.
`val/` là bắt buộc để xác thực. Split `test/` có thể tồn tại nhưng lệnh huấn
luyện và xác thực mặc định không dùng. Mọi split ngoài `train` phải chứa cùng
tên thư mục lớp như tập lớp dự kiến, nhờ vậy khi không khớp, thao tác thất bại
rõ ràng thay vì tính điểm như dự đoán sai. Các phần mở rộng ảnh được chấp nhận
là `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` và `.tiff`.

`data` chấp nhận ba loại: đường dẫn tới thư mục chứa split `train/`, URL `.zip`
hoặc một trong các tên dataset đã biết `imagenette160` và `smoke10`, được tải và
cache trong lần sử dụng đầu tiên.

Loader chuẩn là `libreyolo.data.classify_dataset`.

## Huấn luyện

<code-tabs name="train" />

Không có `nc` để khai báo: số lượng lớp đối tượng lấy từ tên thư mục dưới
`train/`, còn linear layer cuối được dựng lại cho khớp trong khi backbone chuyển
nguyên trạng. Xem [huấn luyện](/docs/train) để biết về dataset, augmentation,
multi-GPU và logger.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`, được tính trên split
`val/` của thư mục gốc dataset.

<code-tabs name="val" />

`metrics/accuracy_top1` là tỷ lệ ảnh có lớp đạt điểm cao nhất đúng với lớp thật,
và là con số chính mà quá trình huấn luyện dùng để chọn epoch tốt nhất.
`metrics/accuracy_top5` là tỷ lệ ảnh có lớp thật xuất hiện ở bất kỳ vị trí nào
trong năm lớp đạt điểm cao nhất, số liệu này cung cấp ít thông tin hơn khi
dataset có ít lớp hơn. Dictionary còn có `fitness`, bản sao của giá trị top-1.

## Xuất

<code-tabs name="export" />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Phạm
vi định dạng khác nhau theo family; ma trận trên từng trang mô hình được tạo từ
tập đã xác thực thay vì nhập thủ công. Xem [xuất và triển khai](/docs/export) để
biết các định dạng, thành phần bổ sung và ràng buộc.
