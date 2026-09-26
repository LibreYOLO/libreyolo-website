---
title: Phát hiện đối tượng
seo_title: Phát hiện đối tượng trong LibreYOLO
description: >-
  Phát hiện đối tượng dưới dạng hộp thẳng trục trong LibreYOLO: các family phục
  vụ tác vụ, định dạng nhãn và các lời gọi dự đoán, huấn luyện, xác thực cùng
  xuất.
lead: >-
  Phát hiện đối tượng định vị từng thực thể trong ảnh và trả về hình chữ nhật
  thẳng trục, nhãn lớp cùng điểm số cho mỗi thực thể. Key tác vụ là detect.
keywords:
  - phát hiện đối tượng python
  - nhận diện vật thể trong ảnh
  - phát hiện bounding box
  - thư viện object detection MIT
  - lựa chọn thay YOLO
  - huấn luyện object detector
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Family khác, cùng lời gọi'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo checkpoint, còn mọi detector trả về cùng
        # đối tượng Results, vì vậy đổi family chỉ cần thay một dòng.
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: Video và luồng
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Mọi nguồn thư viện chấp nhận: tệp, thư mục, URL, index webcam,
        # luồng RTSP hoặc danh sách .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml tải sample 128 ảnh trong lần sử dụng đầu tiên. Trỏ data
        # tới YAML dataset riêng cho lượt chạy thực.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() trả về dict thuần, không phải đối tượng.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như checkpoint và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Định nghĩa

Phát hiện đối tượng trả lời mỗi đối tượng ở đâu và là gì. Một ảnh đi vào, mỗi
thực thể một dòng đi ra: bốn số cho hình chữ nhật, một index lớp và một điểm số.
Không có thông tin về hình dạng pixel, hướng hoặc bộ phận, đây là điểm phân biệt
với [phân đoạn thực thể](/docs/tasks/instance-segmentation), [hộp
xoay](/docs/tasks/oriented-detection) và [tư thế](/docs/tasks/pose-estimation).

`detect` là key tác vụ chuẩn và là mặc định: checkpoint có tên tệp không mang
hậu tố tác vụ sẽ được nạp như detector.

`predict()` điền `result.boxes`. `.xyxy` cho các góc theo pixel trên canvas ảnh
gốc, `.conf` là điểm số, còn `.cls` là index lớp trong `result.names`. `.xywh`,
`.xyxyn` và `.xywhn` là các view suy ra từ cùng các dòng, còn `.id` mang track id
khi gắn tracker. Lặp qua đối tượng `Boxes` yield các slice một dòng, vì vậy
`box.cls`, `box.conf` và `box.xyxy` đều hoạt động theo từng kết quả phát hiện.

## Mô hình

Mười hai family vừa huấn luyện vừa dự đoán: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr),
[YOLO-NAS](/docs/models/yolo-nas), [YOLOX](/docs/models/yolox),
[YOLOv7](/docs/models/yolov7), [RTMDet](/docs/models/rtmdet) và
[PicoDet](/docs/models/picodet). YOLOv9 và RF-DETR là hai family chủ lực, các
tính năng được đưa vào chúng trước. RF-DETR cần thành phần bổ sung riêng,
`pip install "libreyolo[rfdetr]"`; các family còn lại chạy trên package cơ sở.

Mười một family khác dự đoán, xác thực và xuất, nhưng `train()` phát sinh
`NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) và
[EfficientDet](/docs/models/efficientdet).

Dòng Darknet gồm [YOLOv1](/docs/models/yolov1), [YOLOv2](/docs/models/yolov2),
[YOLOv3](/docs/models/yolov3) và [YOLOv4](/docs/models/yolov4) được giữ như hiện
vật đóng băng: dự đoán, xác thực và xuất hoạt động, huấn luyện thì không.

Một nhóm riêng nhận danh sách lớp tại runtime thay vì từ checkpoint, vì vậy có
thể phát hiện tên chưa từng thấy trong huấn luyện:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) và [OV-DEIM](/docs/models/ov-deim),
cùng các vision-language family [Florence-2](/docs/models/florence-2),
[Kosmos-2](/docs/models/kosmos-2), [Qwen3-VL](/docs/models/qwen3-vl),
[SmolVLM2](/docs/models/smolvlm2), [InternVL3](/docs/models/internvl3),
[LFM2-VL](/docs/models/lfm2-vl), [LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) và
[LibreMODUS](/docs/models/libremodus). Chúng được nạp qua factory và thành phần
bổ sung riêng; từng trang mô hình chứa lời gọi chính xác.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

`conf` đặt ngưỡng độ tin cậy, còn `max_det` giới hạn số dòng. `iou` là ngưỡng
NMS, vì vậy chỉ có tác dụng trên family chạy NMS; RF-DETR và head YOLOv9 đầu
cuối decode một tập dự đoán cố định và bỏ qua nó. Xem [dự đoán](/docs/predict)
để biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Mỗi ảnh có một tệp nhãn `.txt`, được tìm bằng cách thay `images` thành `labels`
trong đường dẫn ảnh và đổi phần mở rộng.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Mỗi dòng có đúng năm trường, một index lớp rồi tới hộp tâm và kích thước đã
chuẩn hóa:

```text
<class_id> <cx> <cy> <w> <h>
```

Tọa độ là số float trong `[0, 1]`, tương đối với chiều rộng và chiều cao ảnh
gốc. `w` và `h` phải dương. Tệp nhãn bị thiếu hoặc rỗng nghĩa là ảnh không có
đối tượng. Các dòng không mang độ tin cậy hoặc track id.

YAML đặt tên các split và lớp đối tượng:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` và `val` có thể là thư mục ảnh, tệp `.txt` liệt kê ảnh hoặc danh sách
trộn một trong hai. `nc` là tùy chọn và phải khớp với `names` khi có mặt. COCO
JSON gốc cũng hoạt động: thêm mapping `annotations` từ tên split tới tệp JSON,
còn đường dẫn split khi đó cho biết thư mục gốc của ảnh. Khi có `names`, nó định
nghĩa id nhãn, vì vậy tên category JSON phải khớp.

## Huấn luyện

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` và `lr0` là những đối số đầu tiên cần thay đổi. `lr0`
là đối số không thể chuyển giữa các family: learning rate mà detector tích chập
chịu được sẽ làm transformer diverge, vì vậy hãy lấy giá trị từ trang mô hình
thay vì ví dụ của family khác. Family cũng có thể hoàn toàn bỏ qua một đối số,
và trang của nó liệt kê cụ thể. Xem [huấn luyện](/docs/train) để biết về dataset,
augmentation, multi-GPU và logger.

## Xác thực

`val()` trả về dictionary thuần gồm các key `metrics/`, được tính bằng đánh giá
COCO trên split do `val` trong YAML dataset đặt tên.

<code-tabs name="val" />

`metrics/mAP50-95` là mean average precision lấy trung bình trên các ngưỡng IoU
từ 0.50 đến 0.95 và là con số chính. `metrics/mAP50` và `metrics/mAP75` là các
phiên bản một ngưỡng. `metrics/mAP_small`, `metrics/mAP_medium` và
`metrics/mAP_large` chia cùng giá trị trung bình theo diện tích đối tượng, còn
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium` và `metrics/AR_large` là các số liệu average recall tương
ứng. `metrics/AR_max_det` và `metrics/max_det` ghi giới hạn phát hiện lượt chạy
đã dùng.

Hãy đọc kỹ `metrics/precision` và `metrics/recall` trên tác vụ này. Chúng được
giữ để tương thích ngược và là alias, không phải operating point:
`metrics/precision` giữ cùng giá trị với `metrics/mAP50-95`, còn
`metrics/recall` giữ cùng giá trị với `metrics/AR100`. Vẽ chúng như cặp
precision-recall sẽ báo cáo cùng một số hai lần. Bốn key còn được lặp lại dưới
hậu tố `(B)` cho hộp, để key phát hiện đọc giống nhau trên mô hình cũng dự đoán
mặt nạ: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`, `metrics/precision(B)` và
`metrics/recall(B)`.

## Xuất

<code-tabs name="export" />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Phạm
vi định dạng khác nhau theo family; ma trận trên từng trang mô hình được tạo từ
tập đã xác thực thay vì nhập thủ công. Xem [xuất và triển khai](/docs/export) để
biết các định dạng, thành phần bổ sung và ràng buộc.
