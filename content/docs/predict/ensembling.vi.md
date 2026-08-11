---
title: Kết hợp nhiều detector
seo_title: Kết hợp nhiều detector trong LibreYOLO
description: >-
  Chạy nhiều detector trên một ảnh và hợp nhất các box bằng weighted boxes
  fusion hoặc NMS, kể cả các mô hình có danh sách lớp khác nhau.
lead: >-
  LibreEnsemble chạy hai detector trở lên trên cùng ảnh đã giải mã và hợp nhất
  các box vào một đối tượng Results. Mỗi thành viên giữ trọng số, ngưỡng, thiết
  bị và danh sách lớp riêng.
keywords:
  - ensemble mô hình phát hiện đối tượng
  - weighted boxes fusion
  - wbf python
  - kết hợp hai detector
  - hợp nhất bounding box
  - LibreEnsemble
  - ensemble phát hiện python
  - min_votes
last_verified: 1.5.0
verification: >-
  Constructor và chữ ký lời gọi, giá trị mặc định, lỗi đánh giá, việc hợp nhất
  không gian lớp, đếm phiếu cùng Results trả về được đọc từ
  libreyolo/ensemble/model.py. Thuật toán hợp nhất và đối số lấy từ
  libreyolo/ops/fusion.py. Mục đích thiết kế lấy từ
  docs/adr/0004-model-ensembling.md. Mẫu sử dụng được đối chiếu với
  tests/unit/test_ensemble.py và tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: Hợp nhất hai detector
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # Thành viên có thể là đường dẫn checkpoint hoặc mô hình đã tải.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Trọng số và yêu cầu số phiếu
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # theo quy ước, tỉ lệ với mAP đánh giá
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # chỉ giữ box mà cả hai thành viên tìm thấy
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Ngưỡng theo thành viên
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])


        # Giá trị vô hướng áp dụng cho mọi thành viên; danh sách được đọc theo
        thành viên.

        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)

        print(len(result.boxes))
  external:
    - label: Đưa vào detector không do LibreYOLO tải
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Trả về (boxes, scores, labels): xyxy theo pixel của ảnh gốc.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Cùng loại nguồn như một mô hình đơn
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Thay clip.mp4 bằng tệp video trên đĩa.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 4f4c54c52b295795
---

## Ensemble là gì

`LibreEnsemble` nhận hai detector trở lên, chạy từng detector trên cùng một ảnh
và hợp nhất các box vào một `Results`. Đây là cấu trúc dùng khi dự đoán: không có
gì để huấn luyện, còn các thành viên vẫn là mô hình độc lập có thể được đánh giá
và xuất riêng.

Phát hiện là tác vụ duy nhất được hỗ trợ. Thành viên có tác vụ khác sẽ phát sinh
`ValueError` khi khởi tạo, nêu chỉ mục thành viên và tác vụ của nó.

Cả hai tên đều được import lười, nên không tốn chi phí cho đến khi dùng:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Tạo ensemble

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` là chuỗi gồm hai phần tử trở lên. Mục `str` hoặc `Path` được tải qua
`LibreYOLO()`; mọi dạng khác phải là callable và cung cấp dict `names`. Ít hơn
hai phần tử sẽ phát sinh `ValueError`, còn truyền chuỗi đơn sẽ phát sinh
`TypeError` thay vì duyệt các ký tự.

`weights` mặc định là `None`, tức trọng số đồng đều. Trọng số được cung cấp phải
có một giá trị cho mỗi thành viên và hoàn toàn dương, nên trọng số bằng 0 sẽ phát
sinh lỗi thay vì âm thầm loại thành viên. Quy ước trong tài liệu là đặt chúng tỉ
lệ với mAP đánh giá của từng thành viên.

`fusion_iou` mặc định là `0.55`, là IoU tại đó các box từ thành viên khác nhau
được gom cụm. Đây là ngưỡng khác với `iou` theo lời gọi, vốn là thiết lập NMS
riêng của từng thành viên.

`min_votes` mặc định là `1`, nghĩa là một thành viên bất kỳ có thể đưa ra box.
Tăng giá trị sẽ chỉ giữ các cụm được số thành viên riêng biệt tương ứng xác nhận.
Nó phải là số nguyên dương không lớn hơn số thành viên và được giới hạn theo từng
lớp bằng số thành viên thực sự biết lớp đó, nên lớp chỉ một thành viên được huấn
luyện sẽ không bị âm thầm xóa.

## Phương pháp hợp nhất

Ba phương pháp được chấp nhận theo tên, đồng thời cũng chấp nhận callable.

| `fusion` | Hành vi |
|---|---|
| `"wbf"` | Weighted boxes fusion tuần tự, trung thành với bài báo. Mặc định |
| `"wbf_seeded"` | Weighted boxes fusion một lượt; NMS nhận biết lớp chọn tâm cụm |
| `"nms"` | Nối box của mọi thành viên, sau đó chạy NMS nhận biết lớp |

Weighted boxes fusion lấy trung bình tọa độ của cụm theo trọng số độ tin cậy,
tạo ra box mà không thành viên đơn lẻ nào đề xuất. Hai biến thể có trọng số đồng
nhất khi cụm không nhập nhằng và có thể khác nhẹ trên chuỗi cụm chồng lấn.
`"nms"` chọn box sống sót thay vì lấy trung bình, nên box giữ điểm gốc còn trọng
số chỉ ảnh hưởng box nào thắng. Vì phương pháp này chọn thay vì gom cụm, nó không
thể đếm phiếu: kết hợp `fusion="nms"` với `min_votes` lớn hơn `1` sẽ phát sinh
`ValueError`.

Weighted boxes fusion scale lại điểm của cụm theo tỉ trọng trọng số thành viên
ủng hộ cụm đó. Với hai thành viên có trọng số bằng nhau, box chỉ một thành viên
tìm thấy giữ một nửa điểm: `0.9` trở thành `0.45`. Vì vậy, độ tin cậy sau hợp
nhất có thể thấp hơn `conf` dùng để chạy từng thành viên, nên hãy lọc theo điểm
đã hợp nhất thay vì giả định ngưỡng thành viên vẫn còn hiệu lực.

## Thành viên có danh sách lớp khác nhau

Các thành viên không cần dùng chung danh sách lớp. Không gian nhãn được hợp theo
tên, và mỗi thành viên nhận bảng tra cứu ánh xạ lại class id riêng vào tập hợp.
`ensemble.names` là tập hợp đó và được đưa vào `Results` trả về.

Box chỉ hợp nhất trong cùng tên lớp. Lớp chỉ một thành viên biết sẽ đi qua mà
không hợp nhất và không bị phạt: phép scale lại điểm dùng mẫu số theo lớp, nên
lớp chỉ một thành viên biết vẫn giữ điểm.

Khi chỉ chồng lấn một phần, hệ thống ghi cảnh báo nêu các lớp không được mọi
thành viên dùng chung. Cần đọc kỹ cảnh báo này vì checkpoint có tên lớp giữ chỗ
như `class_0` sẽ tạo tập hợp tách biệt với mọi thành viên khác và hoàn toàn không
có hợp nhất xuyên thành viên.

Thành viên trả về class id nằm ngoài `names` riêng sẽ phát sinh `RuntimeError`.

## Detector bên ngoài

<code-tabs name="external" />

`ExternalDetector(fn, names)` bọc bất kỳ callable nào nhận ảnh PIL và trả về
`(boxes, scores, labels)`, trong đó box là xyxy theo pixel của ảnh gốc. Nó kiểm
tra arity, shape box, độ dài khớp nhau và mọi class id đều xuất hiện trong
`names`, đồng thời tự áp dụng ngưỡng `conf`.

Đây là cách detector không do LibreYOLO tải tham gia vào phép hợp nhất.

## Gọi ensemble

<code-tabs name="sources" />

Chữ ký lời gọi phản ánh mô hình đơn và chấp nhận cùng các nguồn: ảnh, thư mục,
danh sách, video, chụp màn hình, webcam và luồng mạng. Nguồn trực tiếp cần
`stream=True` vì cùng lý do như ở nơi khác.

| Đối số | Mặc định | Ghi chú |
|---|---|---|
| `conf` | `0.25` | Theo thành viên; giá trị vô hướng phát rộng hoặc một giá trị cho mỗi thành viên |
| `iou` | `0.45` | Ngưỡng NMS riêng của từng thành viên, không phải ngưỡng hợp nhất |
| `imgsz` | `None` | `list` được đọc theo thành viên; `int` hoặc tuple được phát rộng |
| `device` | `None` | Giá trị vô hướng hoặc một giá trị cho mỗi thành viên, nên thành viên có thể ở thiết bị khác nhau |
| `classes` | `None` | Lọc kết quả đã hợp nhất theo class id của tập hợp |
| `max_det` | `300` | Áp dụng cho kết quả đã hợp nhất |

Vì `list` có nghĩa là theo thành viên đối với `imgsz`, `imgsz=[480, 640]` là 480
cho thành viên đầu và 640 cho thành viên thứ hai, còn `imgsz=(480, 640)` là một
kích thước hình chữ nhật cho mọi thành viên. Rất dễ nhầm sự khác biệt này.

Các thành viên được gọi với `max_det` ít nhất 300 bất kể bạn yêu cầu gì, nên mỗi
thành viên chạy rộng rãi và ensemble cắt bớt một lần ở cuối.

Ảnh được giải mã một lần và cùng đối tượng được chuyển cho mọi thành viên.
`batch` được chấp nhận để đồng nhất rồi bị bỏ qua; các ảnh được xử lý tuần tự.

## Giá trị trả về

Một `Results` thông thường, cùng loại mà mô hình đơn trả về, với `names` được đặt
thành không gian lớp hợp. Mọi nội dung trong
[Làm việc với kết quả](/docs/predict/results) áp dụng không thay đổi.

Một khác biệt là `result.speed`, trường mà ensemble có điền. Các khóa gồm
`member_0`, `member_1` và tiếp tục như vậy, cộng với `fusion`, theo mili giây.
Đây là nơi duy nhất trong thư viện mà `speed` được điền.

Các hàng chứa box hoặc điểm không hữu hạn bị loại trước khi hợp nhất. Khi thành
viên nằm trên thiết bị khác nhau, phép hợp nhất chạy trên thiết bị của thành viên
đầu tiên trả về dữ liệu.

## Những gì ensemble không thể làm

`val()` và `export()` đều phát sinh `NotImplementedError` và hướng bạn đến các
thành viên: hãy đánh giá và xuất từng mô hình riêng. Hoàn toàn không có phương
thức `train`, nên gọi nó sẽ phát sinh `AttributeError`.

Half precision không được xử lý ở cấp ensemble. `half=True` đi vào cùng luồng
no-op có cảnh báo như ở nơi khác; hãy cấu hình độ chính xác số trên từng thành viên.

Không có giao diện dòng lệnh cho ensembling. Đây là Python API.

