---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: nhận dạng và xác minh khuôn mặt'
description: >-
  Dùng LibreFaceRec trong LibreYOLO để phát hiện, tạo embedding và xác minh
  khuôn mặt. Cài đặt và dự đoán; trọng số embedding dùng Apache-2.0.
lead: >-
  LibreFaceRec là tác vụ embedding khuôn mặt của LibreYOLO: detector khuôn mặt
  định vị và căn chỉnh khuôn mặt, còn recognition head tạo embedding danh tính
  được chuẩn hóa L2 để xác minh hoặc tìm kiếm.
keywords:
  - LibreFaceRec
  - nhận dạng khuôn mặt
  - embedding khuôn mặt
  - xác minh khuôn mặt
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tên librefacerec-* định tuyến đến họ này bất kể hậu tố tệp và tải
        # từ tổ chức LibreYOLO trên Hugging Face trong lần sử dụng đầu tiên,
        # cùng với detector khuôn mặt mặc định.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), được chuẩn hóa L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Xác minh
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # So sánh khuôn mặt nổi bật nhất trong mỗi ảnh bằng độ tương đồng
        # cosine của các embedding được chuẩn hóa L2.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Tìm kiếm trong gallery
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        query = model("query.jpg").embeddings          # các khuôn mặt trong ảnh
        này

        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)


        # Độ tương đồng cosine (query_faces, N_total).

        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Cài đặt

Recognition head của LibreFaceRec chạy qua `onnxruntime`, package này không thuộc bản cài đặt cơ sở.

```bash
pip install "libreyolo[onnx]"
```

## Dự đoán

<code-tabs name="predict" />

Phát hiện và nhận dạng là hai graph ONNX riêng biệt phía sau một lệnh gọi: detector khuôn mặt định vị rồi căn chỉnh từng khuôn mặt thành crop chuẩn, còn recognition head trả về một embedding được chuẩn hóa L2 cho mỗi khuôn mặt. Khi giữ nguyên thiết lập, `predict()` tự động tải và ghép detector mặc định đi kèm. `face_detector` nhận callable, mô hình phát hiện LibreYOLO hoặc thực thể `FaceDetector`; `face_boxes` bỏ qua hoàn toàn phát hiện bằng các box bạn đã có. `result.embeddings` chứa một hàng cho mỗi khuôn mặt đã phát hiện, căn với `result.boxes`; phương thức `.similarity()` tính độ tương đồng cosine với embedding khác hoặc toàn bộ gallery trong một lệnh gọi. Để so sánh trực tiếp hai ảnh thay vì hai embedding đã tính, `model.verify(image_a, image_b)` chạy phát hiện và embedding trên cả hai rồi so sánh khuôn mặt có độ tin cậy cao nhất. Có thể thay thế bằng mọi mô hình nhận dạng ONNX theo quy ước ArcFace khác (crop đã căn chỉnh ở đầu vào, embedding `(N, D)` ở đầu ra) bằng cách truyền đường dẫn tệp thay cho tên `librefacerec-*`. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Xuất

<export-matrix />

LibreFaceRec đã bọc graph ONNX được xuất sẵn; việc xuất lại sang định dạng khác chưa được triển khai.

## Giấy phép

<provenance-box>

Detector khuôn mặt mặc định đi kèm là artifact thứ hai theo giấy phép thứ hai: YuNet của OpenCV Zoo, MIT, bản quyền Shiqi Yu. Không có mã kiến trúc nào được port từ cả hai dự án; cả hai graph được sử dụng như hộp đen qua `onnxruntime`, vì vậy wrapper riêng của LibreYOLO không chứa mã bên thứ ba và hoàn toàn dùng MIT.

</provenance-box>

## Trích dẫn

<citation-block />


