---
title: Ước lượng ánh nhìn
seo_title: Ước lượng ánh nhìn trong LibreYOLO
description: >-
  Ước lượng pitch và yaw của ánh nhìn cho từng khuôn mặt trong LibreYOLO. Dự
  đoán từ Python hoặc CLI, đọc góc theo radian và xuất gaze head sang ONNX.
lead: >-
  Ước lượng ánh nhìn trả về hướng nhìn cho mọi khuôn mặt trong ảnh. LibreYOLO
  biểu diễn đây là tác vụ hai giai đoạn: face detector chạy trước, sau đó gaze
  head đọc pitch và yaw từ từng vùng khuôn mặt được trả về.
keywords:
  - ước lượng ánh nhìn python
  - theo dõi mắt
  - pitch yaw ánh nhìn
  - L2CS-Net
  - hướng nhìn
  - head pose
  - tác vụ gaze libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Khi không cung cấp face_detector, dự đoán quay về detector đi kèm

        # OpenCV, vì vậy ngoài checkpoint sẽ không tải thêm gì.

        model = LibreYOLO("LibreL2CSr50.pt")

        result = model(SAMPLE_IMAGE)


        gaze = result.gaze

        print(gaze.pitch, gaze.yaw)              # radian, mỗi khuôn mặt một
        dòng

        print(gaze.pitch_deg, gaze.yaw_deg)      # cùng các góc đó theo độ

        print(gaze.direction_3d)                 # vector đơn vị (N, 3)
    - label: CLI
      language: bash
      code: >
        # Khác với đường dẫn Python, CLI không có fallback tự động: mô hình gaze

        # yêu cầu face detector tường minh, và đó phải là detector LibreYOLO

        # có các hộp biểu thị khuôn mặt.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Chọn nguồn khuôn mặt
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Truyền cho gaze head các hộp từ detector bạn đã chạy.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Hoặc nêu tên một trong các detector đi kèm.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Định nghĩa

Ước lượng ánh nhìn trả về hai góc cho mỗi khuôn mặt. `result.gaze` là payload
`Gaze` có shape `(N, 2)`, cột 0 là pitch và cột 1 là yaw theo radian, được căn
chỉnh từng dòng với `result.boxes`, tức các hộp khuôn mặt đã phát hiện. Quy ước
giống với L2CS-Net: yaw dương xoay ánh nhìn về bên trái của chủ thể, pitch dương
xoay xuống dưới.

Cùng payload cung cấp `pitch_deg` và `yaw_deg` cho đơn vị độ, cùng
`direction_3d`, một vector đơn vị `(N, 3)` trong hệ tọa độ camera với các cột
`(x, y, z)`.

Vì đây là tác vụ hai giai đoạn, một dự đoán phụ thuộc vào hai mô hình. Khuôn mặt
mà detector bỏ sót không có dòng ánh nhìn, còn hộp bị đặt sai tạo góc từ vùng
khuôn mặt được crop sai. Key tác vụ chuẩn là `gaze`; `gaze-estimation` được chuẩn
hóa về key đó.

## Mô hình

[L2CS-Net](/docs/models/l2cs) là family duy nhất phục vụ tác vụ này. Nó ghép
ResNet trunk với hai classification head song song theo bin góc, một cho pitch
và một cho yaw, trên vùng khuôn mặt 448x448. Năm độ sâu backbone được hỗ trợ về
mặt kiến trúc, và một trong số đó, ResNet-50, có checkpoint đã công bố.

Trọng số có hạn chế về giấy phép. Chúng được huấn luyện trên Gaze360, có giấy
phép chỉ cho phép sử dụng nghiên cứu và phi thương mại, đồng thời cấm phân phối
lại, vì vậy LibreYOLO không mirror gì cho family này. Checkpoint duy nhất thư
viện có thể tự động tải đến thẳng từ bản phân phối Google Drive riêng của tác
giả qua `gdown`, sau khi in điều khoản giấy phép. Hãy đọc
[L2CS-Net](/docs/models/l2cs) trước khi triển khai.

Đường dẫn tải đó cần thành phần bổ sung `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Nếu thiếu, thư viện in hướng dẫn tải thủ công thay vì thử truyền dữ liệu. Dự
đoán bằng checkpoint bạn đã có và xuất checkpoint đó không cần thành phần bổ
sung nào.

## Dự đoán

<code-tabs name="predict" />

Nguồn khuôn mặt được chọn theo một trong ba cách. `face_boxes` truyền các hộp
bạn đã tính và bỏ qua phát hiện. `face_detector` nhận `"auto"`, `"haar"`,
`"yunet"`, mô hình phát hiện LibreYOLO hoặc callable thuần, và có thể được đặt
trên constructor hoặc theo từng lời gọi. Khi không đặt trong Python, dự đoán
quay về detector đi kèm OpenCV, vì vậy lời gọi rút gọn hoạt động mà không cần
nối thêm gì. Trên OpenCV 4, đó là Haar cascade nằm trong wheel và hoàn toàn
không cần tải; trên OpenCV 5, nơi API Haar đã bị xóa, đó là YuNet, công cụ tải
một tệp mô hình nhỏ từ OpenCV zoo một lần.

CLI không dùng chung fallback đó. `libreyolo predict` từ chối mô hình gaze không
có `face_detector=`, còn giá trị nhận được là tên detector LibreYOLO hoặc đường
dẫn checkpoint. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và cách xử
lý kết quả.

## Huấn luyện

Không family nào trong tác vụ này huấn luyện bên trong LibreYOLO.
`LibreL2CS.train()` phát sinh lỗi: hãy huấn luyện tại dự án L2CS-Net upstream và
nạp state dict kết quả tại đây.

## Xác thực

Việc xác thực trên dataset có ground truth ánh nhìn nằm ngoài phạm vi, và
`val()` phát sinh lỗi thay vì trả về các metric không được tính. Tác vụ này
không có dictionary `metrics/`. Hãy đánh giá ở upstream trên dataset dùng để
huấn luyện checkpoint.

## Xuất

<code-tabs name="export" />

Hợp đồng xuất gaze bao gồm ONNX, TorchScript, ExecuTorch, TensorRT và OpenVINO.
Phần rời khỏi thư viện chỉ là ResNet trunk và hai head bin góc: graph nhận vùng
khuôn mặt 448x448 đã tiền xử lý và trả về logit yaw cùng pitch thô. Phát hiện
khuôn mặt, crop, softmax, kỳ vọng bin và chuyển đổi thành góc đều ở lại trong
Python, tại `libreyolo.models.l2cs.utils`. Xem [xuất](/docs/export) để biết các
định dạng và đối số.
