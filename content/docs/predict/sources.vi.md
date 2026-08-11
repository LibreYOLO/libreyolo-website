---
title: Nguồn dự đoán
seo_title: Nguồn dự đoán trong LibreYOLO
description: >-
  Mọi nguồn mà predict chấp nhận: ảnh, thư mục, URL, tệp video, webcam, RTSP,
  YouTube, chụp màn hình, danh sách ảnh và tệp .streams.
lead: >-
  Đối số source được phân loại trước khi mở bất kỳ thứ gì, nên một lời gọi xử lý
  được JPEG, thư mục, MP4, chỉ số webcam, URL RTSP, vùng màn hình hoặc danh sách
  camera.
keywords:
  - suy luận video yolo python
  - rtsp
  - phát hiện đối tượng webcam python
  - dự đoán trên thư mục ảnh
  - phát hiện đối tượng chụp màn hình
  - nhiều luồng rtsp
  - tệp streams
  - suy luận youtube
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Cách phân loại nguồn được đọc từ libreyolo/utils/source.py (classify_source,
  SourceKind, StreamSource, MultiStreamSource). Loại ảnh và phần mở rộng thư mục
  được chấp nhận lấy từ libreyolo/utils/image_loader.py. Phần mở rộng video và
  đường dẫn lưu lấy từ libreyolo/utils/video.py. Cú pháp màn hình lấy từ
  libreyolo/utils/screen.py. Shape trả về và giá trị mặc định đối số lấy từ
  InferenceRunner.__call__ trong libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Một ảnh
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Nguồn một ảnh trả về một Results, không phải danh sách.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Ảnh trong bộ nhớ
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Một thư mục
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("sample_folder")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # Thư mục trả về danh sách, mỗi ảnh một Results, sắp xếp theo đường dẫn.
        results = model(str(folder))
        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Tệp video (cung cấp clip của bạn)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Thay clip.mp4 bằng tệp video trên đĩa.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 'Mỗi frame thứ ba, ghi ra đĩa'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (cần camera được kết nối)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Webcam chỉ số 0. Nguồn trực tiếp không kết thúc, nên hãy giới hạn vòng
        lặp.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (cần URL camera có thể truy cập)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Tệp .streams (cung cấp camera của bạn)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: Danh sách camera
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Một ảnh chụp màn hình (cần mss và phiên desktop)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Khi không có stream=True, thao tác này lấy một frame.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 'Một vùng màn hình, liên tục'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <màn_hình> <trái> <trên> <chiều_rộng> <chiều_cao>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## Cách phân loại nguồn

`classify_source` kiểm tra giá trị trước khi mở hoặc tải bất kỳ thứ gì theo thứ tự
sau. Quy tắc khớp đầu tiên được áp dụng.

| Nguồn | Được đọc là |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Chụp màn hình |
| `int` không âm hoặc chuỗi chữ số không có tệp cùng tên | Webcam |
| URL `rtsp://`, `rtmp://`, `tcp://` hoặc `udp://` | Luồng mạng |
| URL `http(s)://` có đường dẫn kết thúc bằng `.m3u8` | Luồng mạng |
| URL trang YouTube | Luồng mạng |
| Danh sách hoặc tuple có tất cả mục là nguồn trực tiếp hoặc video | Nhiều luồng trực tiếp |
| Mọi danh sách hoặc tuple khác | Batch ảnh |
| Đường dẫn kết thúc bằng `.streams` | Nhiều luồng trực tiếp |
| Đường dẫn có phần mở rộng video | Tệp video |
| Thư mục hiện có | Thư mục ảnh |
| Mọi dạng khác | Một ảnh |

Danh sách trộn nguồn trực tiếp với ảnh sẽ phát sinh `TypeError`. Chỉ số webcam
âm sẽ phát sinh `ValueError`.

Bộ phân loại không bao giờ truy cập mạng, nên URL gõ sai chỉ lộ ra khi mở capture,
không phải khi gọi `predict`.

## Ảnh

<code-tabs name="images" />

Nguồn một ảnh chấp nhận bảy loại.

| Loại | Được đọc là |
|---|---|
| `str` hoặc `pathlib.Path` | Tệp cục bộ, `http(s)://`, `s3://` hoặc `gs://` |
| `PIL.Image.Image` | Chuyển sang RGB |
| `numpy.ndarray` | Ảnh xám 2D hoặc HWC/CHW 3D; mảng 4D dùng ảnh đầu tiên |
| `torch.Tensor` | CHW hoặc NCHW, đọc dưới dạng RGB; tensor theo batch dùng ảnh đầu tiên |
| `bytes` | Dữ liệu ảnh đã mã hóa |
| `io.BytesIO` | Dữ liệu ảnh đã mã hóa |

Mọi thứ được chuyển sang RGB trước khi tiền xử lý. Mảng NumPy là trường hợp duy
nhất có thứ tự kênh nhập nhằng, nên `color_format` điều khiển thứ tự: `"auto"`
(mặc định) giữ nguyên mảng, còn `"bgr"` đảo các kênh, đúng với yêu cầu của frame
đọc bằng OpenCV.

Mảng số thực được scale lại theo phạm vi riêng: giá trị bằng hoặc dưới `1.0` được
nhân với 255, còn giá trị cao hơn bị cắt vào `[0, 255]`. Mảng RGBA bỏ kênh alpha.

Mỗi loại đường dẫn từ xa cần một gói riêng và không gói nào được cài mặc định:
`requests` cho `http(s)://`, `boto3` cho `s3://` và `gcsfs` cho `gs://`.

## Thư mục

Thư mục được quét đệ quy và sắp xếp; mọi tệp có một trong các hậu tố sau trở thành
ảnh: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`. Mọi dạng
khác trong thư mục bị bỏ qua. Thư mục rỗng trả về danh sách rỗng thay vì phát sinh lỗi.

Thư mục và danh sách là hai nguồn chấp nhận `batch`, chạy một forward pass xếp
chồng cho mỗi nhóm trên các họ hỗ trợ. Xem
[Hiệu năng suy luận](/docs/predict/performance).

## Tệp video

<code-tabs name="video" />

Đường dẫn được tính là video khi hậu tố thuộc một trong các dạng `.asf`, `.avi`,
`.gif`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` xuất hiện trong cả hai danh sách. Đường dẫn `.gif` truyền trực tiếp vào
`predict` được mở dưới dạng video vì bước kiểm tra video chạy trước; tệp `.gif`
trong thư mục được quét được tải dưới dạng ảnh tĩnh.

`vid_stride` xử lý mỗi frame thứ N và mặc định là `1`. Khi không có `stream=True`,
toàn bộ video được giải mã vào danh sách; nếu có trên 500 frame sau khi lấy bước,
hệ thống phát cảnh báo đề xuất `stream=True`.

Mỗi `Results` từ video chứa `frame_idx`.

## Webcam, luồng mạng và YouTube

<code-tabs name="live" />

Nguồn trực tiếp không có giới hạn, nên cần `stream=True`. Nếu không có,
`predict` phát sinh `ValueError` thay vì cố thu thập danh sách vô tận.

Frame được đọc trên thread nền, mỗi capture một thread. Theo mặc định, hàng đợi
chỉ giữ frame mới nhất, nên mô hình chậm hơn camera sẽ bỏ qua frame thay vì tụt
lại. `stream_buffer=True` giữ mọi frame đã capture, bảo toàn chúng nhưng làm độ
trễ tăng dần.

Chỉ số webcam là `int` hoặc chuỗi chữ số. Trên Windows, capture được mở qua
backend DirectShow trước rồi quay về backend mặc định nếu thất bại.

URL trang YouTube được phân giải thành URL media trực tiếp mà không tải video,
việc này cần `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Nhãn luồng được che thông tin trước khi ghi log hoặc dùng làm tên tệp. URL chứa
thông tin xác thực xuất hiện dưới dạng `user:***@host`, còn chuỗi query bị loại
khỏi nhãn luồng trực tiếp vì URL đã ký và bearer token nằm tại đó. ID video
YouTube được giữ vì không phải thông tin xác thực.

## Nhiều camera cùng lúc

<code-tabs name="streams" />

Tệp `.streams` có một nguồn trên mỗi dòng. Dòng trống và dòng bắt đầu bằng `#`
bị bỏ qua. Mọi dòng còn lại phải là chỉ số webcam, luồng mạng, URL YouTube hoặc
đường dẫn tệp video; mọi dạng khác phát sinh `ValueError` nêu số dòng. Tệp rỗng
phát sinh lỗi thay vì khởi chạy mà không có camera.

Danh sách hoặc tuple nguồn trực tiếp làm điều tương tự mà không cần tệp.

Mỗi capture có thread riêng và frame từ tất cả capture được ghép vào một
generator. Mỗi lượt thăm dò từng luồng đang hoạt động và yield mọi dữ liệu đã
sẵn sàng, nên camera chậm không cản camera nhanh, còn frame từ các camera khác
nhau xen kẽ. Luồng kết thúc rời khỏi vòng quay trong khi các luồng khác tiếp tục.

## Chụp màn hình

<code-tabs name="screen" />

Nguồn màn hình là từ `screen` theo sau bởi không, một, bốn hoặc năm số nguyên.
Mọi số lượng khác phát sinh `ValueError`.

| Dạng | Nội dung capture |
|---|---|
| `"screen"` | Mọi màn hình, được hợp nhất |
| `"screen 1"` | Màn hình 1 |
| `"screen 100 200 512 256"` | Một box trên desktop đã hợp nhất |
| `"screen 1 100 200 512 256"` | Một box trên màn hình 1 |

Tọa độ box là `left top width height`, tương đối với góc trên-trái của màn hình
được chọn. Nguồn màn hình báo tốc độ frame bằng 30 chia cho `vid_stride`, cũng
là tốc độ ghi video đã lưu. Capture cần gói `mss`:

```bash
pip install mss
```

Khi không có `stream=True`, nguồn màn hình lấy một frame và trả về một `Results`,
tương đương việc dự đoán trên tệp ảnh đối với ảnh chụp màn hình. Với `stream=True`,
nguồn capture đến khi vòng lặp bị dừng.

## Giá trị predict trả về

Shape của giá trị trả về phụ thuộc vào nguồn và `stream`.

| Nguồn | `stream=False` | `stream=True` |
|---|---|---|
| Một ảnh | Một `Results` | Generator của một `Results` |
| Danh sách ảnh | Danh sách `Results` | Generator |
| Thư mục | Danh sách `Results` | Generator |
| Tệp video | Danh sách `Results` | Generator |
| Màn hình | Một `Results` | Generator không giới hạn |
| Webcam, luồng mạng, `.streams` | `ValueError` | Generator không giới hạn |

Một ảnh trả về chính đối tượng `Results`. Việc lập chỉ mục chọn một phát hiện,
không phải ảnh, nên `result[0]` trên dự đoán một ảnh là box đầu tiên thay vì ảnh
đầu tiên. Để biết các đối tượng đó chứa gì, hãy xem
[Làm việc với kết quả](/docs/predict/results).

## Nơi save ghi tệp

`save=True` ghi đầu ra có chú thích vào thư mục lượt chạy thay vì trả về dữ liệu đó.

Ảnh được đưa vào `runs/detect/predict`, `runs/detect/predict2` và tiếp tục tự tăng,
đồng thời giữ tên tệp nguồn. Mọi ảnh trong một tiến trình nằm trong cùng thư mục,
nên hai thư mục đầu vào có cùng tên tệp sẽ ghi đè lẫn nhau. Ảnh trong bộ nhớ không
có tên tệp để dùng lại và được đánh số `image0`, `image1` rồi tiếp tục.

Video và nguồn trực tiếp được ghi thành một tệp `.mp4` duy nhất đặt theo tên nguồn.

`output_path` ghi đè thư mục. Đường dẫn có hậu tố được coi là tệp, còn đường dẫn
không có hậu tố được coi là thư mục. `output_file_format` chọn kiểu mã hóa ảnh
tĩnh và chấp nhận `jpg`, `png` hoặc `webp`.

Sau khi lưu, đường dẫn đã ghi cũng được gắn vào kết quả dưới dạng `result.saved_path`.
