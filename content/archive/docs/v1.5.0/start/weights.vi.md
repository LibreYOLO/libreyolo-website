---
title: Checkpoint và trọng số
seo_title: Checkpoint và trọng số LibreYOLO
description: >-
  Cách LibreYOLO tìm, tải và xác minh trọng số mô hình, nơi lưu trữ, cách chạy
  không cần mạng và yếu tố giúp checkpoint được nạp an toàn.
lead: >-
  Checkpoint LibreYOLO là dictionary torch.save chứa state dict cùng metadata
  cần để nhận diện. Trang này trình bày nguồn gốc, vị trí lưu và cách các file
  đó được nạp.
keywords:
  - trọng số libreyolo
  - checkpoint libreyolo
  - tải trọng số libreyolo
  - dùng libreyolo offline
  - libreyolo hugging face
  - metadata checkpoint
last_verified: 1.5.0
meta:
  - label: Lưu trữ tại
    value: 'Mỗi checkpoint có một repo Hugging Face:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Cache cục bộ
    value: weights/ dưới thư mục làm việc
    mono: true
  - label: Metadata schema
    value: v1.0
snippets:
  load:
    - label: Tự động tải
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tên file thuần được phân giải thành weights/LibreYOLO9t.pt và
        # tải xuống đó nếu chưa có.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Đường dẫn tường minh
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Đường dẫn có thành phần thư mục được dùng đúng như đã viết và
        # không bao giờ được lấy từ mạng.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Đọc metadata mà không dựng mô hình và báo liệu nó
        # có đáp ứng schema hay không.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Trả về danh sách vấn đề. Danh sách rỗng nghĩa là file đáp ứng v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Nơi tìm checkpoint

Tham chiếu mô hình không có thành phần thư mục, như `LibreYOLO9t.pt`, được phân
giải theo `weights/` tương đối với thư mục làm việc hiện tại. Nếu
`weights/LibreYOLO9t.pt` tồn tại thì file đó được dùng; nếu file cùng tên tồn
tại ngay trong thư mục làm việc thì nó được dùng thay thế; nếu không,
`weights/LibreYOLO9t.pt` trở thành đích tải xuống.

Tham chiếu có thư mục, tuyệt đối hoặc tương đối, được hiểu theo đúng nghĩa đen.
Đây là dạng nên dùng khi trọng số nằm ở vị trí trung tâm và không được tải gì.

<code-tabs name="load" />

## Tự động tải

Khi đường dẫn đã phân giải không tồn tại, LibreYOLO phân tích tên file để lấy
họ, kích thước và tác vụ, rồi yêu cầu họ tương ứng cung cấp URL tải. Phần lớn
các họ tạo URL từ tổ chức LibreYOLO trên Hugging Face, nơi mỗi checkpoint có
repo riêng mang tên file:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Hậu tố biến thể dataset vẫn là một phần tên repo, vì vậy checkpoint huấn luyện
trên dữ liệu khác mặc định sẽ phân giải về repo riêng thay vì ghi đè bản mặc
định.

Quá trình truyền có cơ chế phòng vệ vì file trọng số bị cắt cụt sẽ gây lỗi khó
hiểu về sau. Nội dung tải xuống được stream vào file `.part` và chỉ được di
chuyển nguyên tử vào vị trí cuối khi hoàn tất, nên tiến trình gián đoạn không
thể để lại checkpoint chỉ ghi một phần tại đường dẫn cuối. Lần truyền bị gián
đoạn tiếp tục từ byte offset bằng HTTP validator và bắt đầu lại từ đầu nếu máy
chủ báo object đã thay đổi. Lỗi được thử lại ba lần với exponential backoff.
Các tiến trình đồng thời nhắm cùng đường dẫn dùng lock file, nên hai lần huấn
luyện khởi động cùng lúc chỉ tải một lần. Khi một họ lấy file từ host bên thứ
ba thay vì tổ chức LibreYOLO, họ có thể cố định checksum và từ chối file không
khớp.

Nếu `HF_TOKEN` được đặt hoặc token đã cache tại `~/.cache/huggingface/token`,
nó được đính kèm dưới dạng bearer token. Token chỉ được gửi đến URL
`huggingface.co`, nên họ tải từ host khác không bao giờ nhận token.

Không phải họ nào cũng tự động tải. Một số cố ý không trả về URL vì trọng số
phát hành không được phép phân phối lại, và lỗi sẽ giải thích cần cung cấp gì.
Một số khác in thông báo giấy phép trước khi truyền. Đây là tín hiệu runtime
cho biết điều khoản checkpoint hẹp hơn điều khoản mã nguồn, và bạn nên đọc
thay vì lướt qua.

## Tổ chức Hugging Face

Trọng số công bố nằm tại
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), mỗi checkpoint một
repo. Mỗi repo có giấy phép và giấy phép không đồng nhất trong một họ: họ có mã
MIT vẫn có thể có trọng số không phải MIT. Repo là nguồn có thẩm quyền. Mỗi
trang mô hình liệt kê checkpoint đã công bố của họ và giấy phép trong phần
Checkpoint và Giấy phép.

## Làm việc offline

Không phần nào của thư viện cần truy cập mạng khi các file đã có cục bộ. Có hai
cách:

Chuẩn bị trước thư mục `weights/` cạnh nơi job chạy. Chỉ cần tải checkpoint một
lần trên máy có mạng rồi sao chép thư mục; bước phân giải ở trên sẽ tìm thấy và
không truy cập mạng.

Hoặc truyền đường dẫn tuyệt đối đến vị trí dùng chung. Tham chiếu có thư mục
được dùng như đã cho, nên mount chỉ đọc chứa trọng số đã tuyển chọn là cấu hình
hợp lệ. Nếu tiến trình không thể ghi cạnh checkpoint cần chuyển đổi, quá trình
chuyển đổi quay về thư mục tạm riêng thay vì thất bại.

Dataset theo quy tắc riêng: phân giải dưới `~/datasets` hoặc thư mục do
`LIBREYOLO_DATASETS_DIR` chỉ định khi biến này được đặt.

## An toàn khi nạp

Checkpoint là pickle, và pickle có thể thực thi mã tùy ý khi mở. LibreYOLO coi
mọi file trọng số là không đáng tin cậy và nạp bằng pipeline
`weights_only=True` của PyTorch, giới hạn unpickler ở tensor và một tập nhỏ kiểu
an toàn. Điều này áp dụng cho file bạn truyền chứ không chỉ file LibreYOLO tải.
Trên bản build PyTorch quá cũ không hỗ trợ đối số này, quá trình nạp bị từ chối
thay vì thực hiện thiếu an toàn.

Một số checkpoint huấn luyện upstream nhúng object mà unpickler hạn chế từ
chối, chẳng hạn object cấu hình từ framework dùng để huấn luyện. LibreYOLO
không cần metadata đó, nên trong lúc chuyển đổi, mỗi lớp bị chặn được thay bằng
stand-in trơ đáp ứng unpickler mà không thực thi gì, và chỉ tensor tồn tại trong
file đã chuyển đổi. Tên module nhạy cảm bị từ chối hoàn toàn thay vì stub, còn
vòng lặp thử lại bị giới hạn để file được thiết kế nhằm đưa vào chuỗi lớp bị
chặn vô hạn sẽ thất bại an toàn. Xem [nhập trọng số có sẵn](/docs/migrate) để
biết phần còn lại của pipeline.

## Metadata checkpoint

Checkpoint LibreYOLO là dictionary có khóa `model` chứa state dict PyTorch.
Schema v1.0 yêu cầu chín khóa; kết hợp lại, chúng giúp factory nhận diện file
mà không phân tích tên hoặc đoán từ hình dạng tensor.

| Khóa | Ý nghĩa |
|---|---|
| `model` | State dict PyTorch |
| `schema_version` | Phiên bản contract metadata. v1.0 dùng chuỗi `1.0` |
| `libreyolo_version` | Phiên bản LibreYOLO tạo file |
| `model_family` | Định danh họ đã đăng ký, chẳng hạn `yolo9` |
| `size` | Biến thể trong họ đó, chẳng hạn `t` hoặc `r18` |
| `task` | Một tên tác vụ chuẩn |
| `nc` | Số lớp dương |
| `names` | Ánh xạ chỉ mục lớp sang nhãn, phủ từ `0` đến `nc - 1` |
| `imgsz` | Độ phân giải đầu vào dương |

Tác vụ có cấu trúc bổ sung ghi cấu trúc đó cạnh các khóa. Checkpoint pose thêm
`num_keypoints` và `keypoint_dim`, và có thể thêm sigma OKS theo keypoint.
Checkpoint OCR nhúng toàn bộ charset CTC để file tự chứa. Checkpoint restore có
thể ghi loại suy giảm và hệ số upscale. Checkpoint trainer thêm trạng thái tiếp
tục như `epoch`, trạng thái optimizer và trọng số EMA; trọng số inference công
bố không nên chứa chúng.

File đáp ứng cả chín khóa được nạp qua pipeline metadata. File không đáp ứng sẽ
được chuyển đổi nếu một họ nhận diện layout, hoặc được nạp qua pipeline tương
thích với cảnh báo nêu những gì còn thiếu.

## Kiểm tra checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` không bao giờ dựng mô hình, nên hoạt động trên file có họ
chưa được cài đặt và trên file bạn chưa chắc chắn.
