---
title: Theo dõi đối tượng
seo_title: Theo dõi đối tượng trong LibreYOLO
description: >-
  Theo dõi đối tượng qua các frame video trong LibreYOLO bằng ByteTrack,
  BoT-SORT, OC-SORT hoặc Deep OC-SORT, trên mọi mô hình phát hiện, phân đoạn
  hoặc tư thế.
lead: >-
  Theo dõi gán định danh ổn định cho từng kết quả phát hiện qua các frame video.
  LibreYOLO không biểu diễn đây là tác vụ có trọng số riêng: đó là chế độ dự
  đoán model.track(), chạy tracker đã chọn trên đầu ra theo từng frame của mô
  hình phát hiện, phân đoạn hoặc tư thế.
keywords:
  - theo dõi đối tượng python
  - multi object tracking
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id
  - reid tracking
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() là generator: mỗi frame đã xử lý có một Results.
        for result in model.track("video.mp4"):
            print(result.track_id)        # tensor int (N,), căn theo các hộp
            print(result.boxes.xyxy)
    - label: Chọn tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (mặc định), "botsort", "ocsort" hoặc "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Lưu video đã chú thích
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Nếu không có output_path, tệp nằm tại runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Điều chỉnh tracker
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Kiểu cấu hình chọn tracker, vì vậy tracker= là dư thừa ở đây.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Hoặc truyền cùng các trường dưới dạng đối số keyword để track() tự
        dựng.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Định nghĩa

Theo dõi không phải một trong các key tác vụ của LibreYOLO và không có
checkpoint theo dõi để tải. Đây là phương thức trên mô hình,
`model.track(source)`, chạy phát hiện trên từng frame và liên kết kết quả theo
thời gian. Phương thức là generator: nó yield một `Results` trên mỗi frame đã xử
lý, với `result.track_id` được đặt thành tensor số nguyên `(N,)` căn theo
`result.boxes`. Cùng các ID đó cũng nằm trên `result.boxes.id`.

Chỉ các đối tượng đã xác nhận và hiện đang được theo dõi mới được yield. Track
bị mất liên kết vẫn tồn tại trong số frame đã cấu hình trước khi bị loại,
`track_buffer` cho ByteTrack và BoT-SORT, `max_age` cho hai biến thể OC-SORT, vì
vậy đối tượng được tìm lại trong cửa sổ đó giữ ID ban đầu.

Vì việc liên kết diễn ra sau phát hiện, các payload khác của frame vẫn còn:
`Results` được theo dõi là `Results` phát hiện được cắt theo các dòng khớp, vì
vậy mặt nạ và keypoint đi cùng hộp.

## Mô hình

Một lượt theo dõi có hai lựa chọn độc lập: mô hình tạo hộp trên mỗi frame và
tracker liên kết chúng.

Mọi mô hình LibreYOLO gốc có tác vụ phát hiện, phân đoạn hoặc tư thế đều cung cấp
`track()`, nên việc chọn detector vẫn như thông thường. Xem [chỉ mục mô
hình](/docs/models) để biết danh sách đầy đủ, hoặc bắt đầu với
[YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) hay [RTMDet](/docs/models/rtmdet). Tác vụ có kết
quả không chứa hộp để liên kết sẽ từ chối lời gọi thay vì trả về ID vô nghĩa:
phân loại, hộp xoay, điểm, độ sâu, pháp tuyến bề mặt, cạnh, phân đoạn ngữ nghĩa
và toàn cảnh, phục hồi, OCR cùng lưới cơ thể đều phát sinh lỗi từ `track()`.

Hai tầng mô hình LibreYOLO cũng từ chối. Mô hình được nạp qua `LibreSAM` là image
segmenter, còn mô hình được nạp qua `LibreOpenVocab` là detector theo từng frame;
cả hai phát sinh lỗi từ `track()` và được dùng với `predict()` trên mỗi frame.

Theo dõi chạy trên mô hình PyTorch gốc. Artifact đã xuất được nạp qua
`LibreYOLO("model.onnx")` trả về đối tượng backend runtime, có `predict()` nhưng
không có `track()`.

Bốn tracker đi kèm thư viện, được chọn bằng đối số `tracker`:

`"bytetrack"` là mặc định. Nó chỉ dùng chuyển động, với Kalman filter và liên
kết ba giai đoạn: kết quả phát hiện có độ tin cậy cao trước, sau đó lượt thứ hai
cho kết quả có độ tin cậy thấp cơ hội khớp với track hiện có trước khi bị loại,
cuối cùng là track chưa xác nhận. Cấu hình bằng `TrackConfig`.

`"botsort"` giữ vòng đời ba giai đoạn của ByteTrack nhưng dùng trạng thái Kalman
theo tâm, chiều rộng và chiều cao, đồng thời bù chuyển động camera cho track dự
đoán trước khi khớp. Đây là biến thể BoT-SORT chỉ dùng chuyển động; nó không chạy
mô hình appearance. Cấu hình bằng `BoTSortConfig`, bổ sung `enable_cmc`,
`cmc_method` và `cmc_downscale`.

`"ocsort"` cũng chỉ dùng chuyển động, bổ sung thành phần hướng vận tốc vào chi
phí liên kết, lượt liên kết thứ hai với quan sát thực cuối cùng của mỗi track và
làm mượt trạng thái Kalman theo quỹ đạo ảo khi tìm lại track. Cấu hình bằng
`OCSortConfig`.

`"deepocsort"` mở rộng OC-SORT bằng appearance. Mỗi track giữ moving average có
trọng số độ tin cậy của embedding tái định danh, còn thành phần cosine similarity
tham gia chi phí liên kết, giúp định danh tồn tại qua che khuất dài và target đi
cắt nhau. Nó tốn một forward pass của mạng embedding nhỏ trên mỗi frame, và
trọng số OSNet được tải trong lần sử dụng đầu tiên. Cấu hình bằng
`DeepOCSortConfig`.

## Dự đoán

<code-tabs name="predict" />

`track_conf` đặt ngưỡng cho giai đoạn liên kết đầu tiên: `track_high_thresh` cho
ByteTrack và BoT-SORT, `det_thresh` cho OC-SORT và Deep OC-SORT. Đây không phải
`conf` của `predict()`, và với ByteTrack, BoT-SORT cùng OC-SORT, detector chạy
nội bộ ở ngưỡng thấp hơn để kết quả yếu vẫn có sẵn cho lượt khôi phục. Deep
OC-SORT chạy detector ở chính `det_thresh`. Với ByteTrack và BoT-SORT,
`track_conf` phải lớn hơn hoặc bằng `track_low_thresh`, mặc định là 0.1.

Cài đặt tracker được truyền theo một trong hai cách. Truyền instance cấu hình
cho `tracker_config=`, kiểu của nó sẽ chọn tracker và khiến `tracker=` dư thừa.
Hoặc truyền các trường dưới dạng đối số keyword và để `track()` dựng cấu hình
cho tracker đã nêu; key không xác định sẽ cảnh báo thay vì được âm thầm áp dụng.
Theo cả hai cách, `track_conf` bị bỏ qua khi key tương ứng được đặt tường minh.

Các đối số còn lại giống dự đoán: `iou`, `imgsz`, `classes`, `max_det`,
`vid_stride`, `show` và `save` với `output_path`. Nguồn là đường dẫn tệp video.
Xem [dự đoán](/docs/predict) để biết cách xử lý kết quả.

## Huấn luyện

Tracker không được huấn luyện. Ba trong bốn loại là mô hình chuyển động thuần
không có tham số học được, còn mạng appearance của Deep OC-SORT là checkpoint
tái định danh đã công bố, được tải trong lần sử dụng đầu tiên. Cải thiện chất
lượng theo dõi nghĩa là cải thiện detector hoặc điều chỉnh các ngưỡng liên kết
ở trên.
