---
title: Phân đoạn theo prompt
seo_title: Phân đoạn theo prompt trong LibreYOLO
description: >-
  Chuyển một điểm, hộp hoặc khái niệm văn bản thành mặt nạ vật thể trong
  LibreYOLO. Nạp SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM hoặc PicoSAM3 qua
  LibreSAM.
lead: >-
  Phân đoạn theo prompt biến một cú nhấp thành mặt nạ: bạn chỉ vào vật thể hoặc
  vẽ hộp quanh nó, rồi mô hình trả về đường viền. Trong LibreYOLO, đây không
  phải key tác vụ riêng mà là một tầng mô hình được nạp qua factory LibreSAM,
  với kết quả là Results phân đoạn thông thường.
keywords:
  - phân đoạn theo prompt
  - phân đoạn tương tác
  - segment anything python
  - point prompt
  - box prompt
  - SAM python
  - tạo mask từ click
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt điểm và hộp
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Điểm là [x, y] theo pixel; nhãn 1 là dương, 0 là âm.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polygon
        print(result.boxes.xyxy)    # hộp khít được suy ra từ mặt nạ

        # Prompt hộp tạo một mặt nạ cho mỗi hộp.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Encode một lần, prompt nhiều lần'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_image chạy image encoder nặng một lần và cache kết quả.
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: Phân đoạn mọi thứ
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Không có prompt nghĩa là một grid điểm trên toàn ảnh. Grid mặc định

        # gồm 32 điểm mỗi cạnh tương đương khoảng 1024 decoder pass, rất chậm
        trên CPU.

        result = model.predict(SAMPLE_IMAGE, points_per_side=8)

        print(len(result.masks))
    - label: Mặt nạ cho trường hợp mơ hồ
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Một điểm có thể biểu thị tay áo, áo hoặc một người. multimask=True
        # trả về cả ba mặt nạ toàn thể và bộ phận thay vì chỉ mặt nạ tốt nhất.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Định nghĩa

Phân đoạn theo prompt nhận ảnh cùng prompt không gian và trả về mặt nạ của bất
cứ thứ gì prompt trỏ tới. Không có nội dung nào được phân loại: không có danh
sách lớp đối tượng, còn `result.boxes` chứa hộp khít được suy ra từ mặt nạ thay
vì bản thân là kết quả phát hiện. `result.masks` mang dữ liệu mặt nạ và
`result.masks.xy` mang các polygon.

Prompt là interface. `points` là tọa độ pixel `[x, y]`, mỗi vật thể một tập, với
`labels` đánh dấu từng điểm là dương (1, bao gồm phần này) hoặc âm (0, loại phần
này). `bboxes` là `[x1, y1, x2, y2]`, mỗi hộp một mặt nạ. Có thể kết hợp điểm và
hộp; khi đó chúng ghép cặp theo vật thể và phải có cùng độ dài. Bỏ mọi prompt sẽ
chạy đường dẫn phân đoạn mọi thứ, tức một grid điểm trên ảnh.

Một điểm vốn mang tính mơ hồ. Nhấp vào tay áo có thể biểu thị tay áo, chiếc áo
hoặc người đó, vì vậy `multimask=True` trả về cả ba mặt nạ toàn thể và bộ phận
trên mỗi prompt thay vì chỉ mặt nạ tốt nhất. `conf` lọc theo IoU dự đoán của mô
hình, tức điểm chất lượng mặt nạ, không phải độ tin cậy phát hiện.

LibreYOLO không có key tác vụ `promptable`. Tầng này đăng ký dưới dạng `segment`,
cùng key mà phân đoạn thực thể sử dụng. Điểm khác biệt là shape của lời gọi, vì
vậy nó có factory riêng `LibreSAM()`, ngang hàng với `LibreYOLO()`,
`LibreOpenVocab()` và `LibreVLM()`. Một signature `predict(image)` duy nhất
không thể biểu diễn vòng lặp mà các mô hình này được thiết kế cho: `set_image()`
chạy image encoder một lần và cache embedding, mỗi lời gọi `predict()` sau đó
với `source=None` chỉ trả chi phí decode prompt, còn `reset_image()` xóa cache.
Image encoder là phần tốn kém nhất và chỉ chạy một lần trên mỗi ảnh, vì vậy
prompt thứ hai trên cùng ảnh bỏ qua hoàn toàn phần này.

## Mô hình

Sáu family được nạp qua `LibreSAM` bằng alias.

[SAM](/docs/models/sam) là mặc định, có các kích thước `base`, `large` và `huge`,
cũng được viết là `b`, `l` và `h`.

[SAM 2](/docs/models/sam-2) có các alias `sam2-tiny`, `sam2-small`,
`sam2-base-plus` và `sam2-large`. LibreYOLO hỗ trợ đường dẫn ảnh của nó.

[SAM 3](/docs/models/sam-3), với alias `sam3`, là family duy nhất nhận prompt
khái niệm bằng văn bản: `text="yellow school bus"` trả về mọi thực thể khớp.
Truyền `text=` cho family khác sẽ phát sinh lỗi có thông báo nêu tên SAM 3.
Trọng số dùng SAM License tùy chỉnh của Meta thay vì giấy phép MIT của
LibreYOLO, và repo được kiểm soát truy cập: hãy chấp nhận điều khoản trên trang
mô hình và xác thực bằng `hf auth login` trước lần tải đầu tiên. Đọc [SAM
3](/docs/models/sam-3) trước khi triển khai.

[EdgeTAM](/docs/models/edgetam), với alias `edgetam`, là biến thể SAM 2 dành cho
thiết bị. LibreYOLO hỗ trợ đường dẫn ảnh của nó.

[MobileSAM](/docs/models/mobilesam), với alias `mobilesam`, thay encoder ViT-H
của SAM bằng TinyViT đã chưng cất.

[PicoSAM3](/docs/models/picosam3), với alias `picosam3`, là CNN nhỏ gọn cho vùng
được prompt bằng hộp trên cảm biến biên. Prompt hộp là toàn bộ hợp đồng ở đây:
điểm, văn bản, mặt nạ, multimask và phân đoạn mọi thứ đều phát sinh lỗi với
thông báo trỏ tới SAM 2 hoặc SAM 3.

Thành phần bổ sung của tầng này bao gồm bốn family được nạp qua `transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM và PicoSAM3 là các bản port LibreYOLO gốc và không cần cài
`transformers` để chạy.

## Dự đoán

<code-tabs name="predict" />

`source` và `set_image()` là hai cách thay thế nhau, không phải một trình tự:
truyền ảnh cho `predict()` để gọi một lần, hoặc gọi `set_image()` trước rồi
`predict(source=None)` cho từng prompt. Truyền `device=` cho `predict()` sẽ di
chuyển mô hình cho lời gọi đó và mọi lời gọi sau, đồng thời vô hiệu hóa embedding
đã cache.

Phân đoạn mọi thứ là chế độ tốn kém. `points_per_side` mặc định là 32, tương
đương khoảng 1024 decoder pass trên ảnh; hãy giảm giá trị cho mọi tác vụ tương
tác trên CPU. Trong chế độ đó, `conf` áp dụng ngưỡng grid của family khi không
được đặt, còn trên đường dẫn có prompt, `conf` không được đặt sẽ giữ mọi mặt nạ.
Truyền `conf=0.0` để tắt lọc ở cả hai chế độ và `max_det` để giới hạn số mặt nạ
trả về.

Phiên bản này không hỗ trợ prompt mặt nạ, và `masks=` phát sinh lỗi thay vì bị
bỏ qua. `track()` cũng phát sinh lỗi trên toàn tầng: đây là các image segmenter,
vì vậy hãy chạy `predict()` trên mỗi frame. Xem [dự đoán](/docs/predict) để biết
về nguồn và cách xử lý kết quả.

## Huấn luyện

Không family nào trong tầng này huấn luyện bên trong LibreYOLO. `train()` phát
sinh lỗi: hãy tinh chỉnh ở upstream rồi nạp trọng số kết quả.

## Xác thực

Tầng này không có validator và `val()` phát sinh lỗi. Mặt nạ theo prompt không
có tập lớp đối tượng cố định để tính điểm, nên metric phát hiện và phân đoạn
thông thường không có key để dựa vào. Tính điểm mặt nạ theo prompt nghĩa là so
sánh nó với mặt nạ tham chiếu do bạn tự cung cấp, theo các prompt bạn quan tâm.

## Xuất

Xuất nằm ngoài phạm vi của toàn tầng và `export()` phát sinh lỗi, với một ngoại
lệ. [PicoSAM3](/docs/models/picosam3) xuất CNN vùng 96x96 thô sang ONNX dưới dạng
`roi_image -> mask_logits`; thao tác crop hộp và đổi kích thước mặt nạ về tọa độ
ảnh vẫn ở trong Python. Mọi family khác chạy qua `predict()` trong PyTorch. Xem
[xuất](/docs/export) để biết các định dạng khả dụng ở nơi khác trong thư viện.
