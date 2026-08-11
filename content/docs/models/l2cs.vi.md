---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: ước lượng hướng nhìn trong LibreYOLO'
description: >-
  Dùng L2CS-Net trong LibreYOLO để ước lượng pitch/yaw hướng nhìn hai giai đoạn.
  Cài đặt, dự đoán và xuất; checkpoint Gaze360 chỉ dành cho nghiên cứu.
lead: >-
  L2CS-Net là mô hình ước lượng hướng nhìn hai giai đoạn: detector khuôn mặt
  định vị khuôn mặt, còn trunk ResNet với hai classification head theo bin góc
  dự đoán pitch và yaw cho từng khuôn mặt. LibreYOLO chỉ bọc mô hình để
  inference.
keywords:
  - L2CS-Net
  - ước lượng hướng nhìn
  - theo dõi mắt
  - pitch yaw
  - Gaze360
  - phát hiện khuôn mặt
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Khi không cung cấp face_detector: dùng detector khuôn mặt đi kèm
        # OpenCV (Haar trên OpenCV 4, YuNet trên OpenCV 5), vì vậy không cần
        # tải thêm gì ngoài chính checkpoint L2CS.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Nguồn khuôn mặt
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Đưa cho L2CS các box từ detector bạn đã chạy.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Hoặc chỉ định một detector khuôn mặt đi kèm.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Graph đã xuất chỉ gồm trunk ResNet và hai angle-bin head: nhận crop

        # khuôn mặt 448x448 đã tiền xử lý và trả về (yaw_logits, pitch_logits)

        # thô, không phải góc đã giải mã. Softmax, kỳ vọng bin và phép chuyển
        đổi

        # sang độ vẫn ở Python; xem
        libreyolo.models.l2cs.utils.bin_logits_to_angles.

        session = ort.InferenceSession("LibreL2CSr50.onnx")

        name = session.get_inputs()[0].name

        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Cài đặt

L2CS-Net không cần extra để khởi tạo, dự đoán hoặc xuất mô hình mà bạn đã có checkpoint.

```bash
pip install libreyolo
```

Checkpoint duy nhất LibreYOLO có thể tự động tìm nạp là ResNet-50 được huấn luyện trên Gaze360, được tải qua `gdown` thay vì mirror HTTP thuần túy vì nằm trên Google Drive của tác giả thay vì tổ chức LibreYOLO. Tuyến đó cần extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Khi không có extra này, LibreYOLO in hướng dẫn tải thủ công thay vì âm thầm thất bại.

## Dự đoán

<code-tabs name="predict" />

L2CS-Net là mô hình ước lượng hai giai đoạn: detector khuôn mặt chạy trước, rồi gaze head đọc pitch và yaw từ từng crop khuôn mặt trả về. Khi giữ nguyên thiết lập, dự đoán dùng detector đi kèm OpenCV, vì vậy lệnh gọi thuần túy hoạt động mà không cần tải thêm sau khi đã có checkpoint L2CS. `face_boxes` nhận box từ detector bạn đã chạy; `face_detector` nhận `"auto"`, `"haar"`, `"yunet"`, mô hình phát hiện LibreYOLO hoặc callable thuần túy. `result.gaze` chứa pitch và yaw theo radian, được căn theo từng hàng với `result.boxes`, tức các box khuôn mặt đã phát hiện. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Năm độ sâu backbone dùng chung một độ phân giải đầu vào và nhận cùng đối số. Gaze360, tập dữ liệu (dataset) đứng sau checkpoint duy nhất đã phát hành, đã huấn luyện ResNet-50; bốn độ sâu còn lại được hỗ trợ về kiến trúc nhưng không có trọng số đã phát hành để tải.

## Xuất

<export-matrix />

<code-tabs name="export" />

## Giấy phép

<provenance-box>

LibreYOLO không host hoặc mirror checkpoint L2CS nào: không có nội dung nào cho họ mô hình này trong tổ chức LibreYOLO trên Hugging Face, khác với hầu hết họ khác trên trang. Checkpoint duy nhất thư viện có thể tự động tìm nạp đến thẳng từ bản phân phối Google Drive của tác giả, được kiểm soát bằng thông báo giấy phép Gaze360 in trước khi bắt đầu truyền, và không phải bản sao "được phát hành lại tại huggingface.co/LibreYOLO" như phần tóm tắt bên trên ngụ ý.

</provenance-box>


