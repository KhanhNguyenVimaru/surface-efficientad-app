# System Pipeline

Tài liệu này mô tả pipeline xử lý của hệ thống EfficientAD Web App.

## 1. Luồng tổng quan

```mermaid
flowchart TD
    A[User mở Nuxt Frontend] --> B[Upload ảnh + chọn threshold]
    B --> C[POST /predict multipart form]
    C --> D[FastAPI validate input]
    D --> E[Load ảnh PIL RGB]
    E --> F[Run EfficientAD inference từ checkpoints/capsule.ckpt]
    F --> G[Trích xuất pred_score / pred_label / anomaly_map / pred_mask]
    G --> H[Phân loại defect type]
    H --> I[Tạo ảnh trực quan hóa \n anomaly map / overlay / mask overlay]
    I --> J[Classical CV hậu xử lý \n đếm lỗi, contour, largest area]
    J --> K[Trả JSON response chứa metadata + ảnh base64]
    K --> L[Frontend hiển thị kết quả]
```

## 2. Chi tiết backend `/predict`

```mermaid
flowchart TD
    A[Nhận UploadFile image, threshold] --> B{anomalib import OK?}
    B -- No --> B1[HTTP 500]
    B -- Yes --> C{checkpoint capsule.ckpt tồn tại?}
    C -- No --> C1[HTTP 400]
    C -- Yes --> D{threshold >= 0?}
    D -- No --> D1[HTTP 400]
    D -- Yes --> E[Đọc bytes + parse ảnh]
    E --> F{Ảnh hợp lệ?}
    F -- No --> F1[HTTP 400]
    F -- Yes --> G[Engine.predict]
    G --> H{Inference thành công?}
    H -- No --> H1[HTTP 500]
    H -- Yes --> I[Chuẩn hóa output model]
    I --> J[Classify GOOD/DEFECT + defect_type]
    J --> K[Generate base64 visual outputs]
    K --> L[Classical CV postprocess]
    L --> M[Build PredictResponse]
```

## 3. Thành phần chính trong pipeline

- Frontend: Nuxt (thư mục `web/`) gửi request và render kết quả.
- API layer: FastAPI trong `app.py`.
- Model inference: `anomalib` + `EfficientAdModelSize.M`, cố định model `capsule`.
- Defect type classification:
- `hash-based lookup` nếu ảnh trùng tập mẫu `test_img/`.
- fallback `feature similarity` nếu không trùng hash.
- Classical CV: `postprocess_anomaly_map`, `draw_contours_on_image` trong `classical_cv_utils.py`.
- Output: JSON gồm nhãn, score, số lỗi, diện tích lỗi lớn nhất và các ảnh base64.

## 4. I/O contract của endpoint chính

- Input: `POST /predict` (multipart form)
- `image`: file ảnh
- `threshold`: số thực (>= 0)
- Output: `PredictResponse`
- `label`, `defect_type`, `defect_confidence`, `score`, `pred_label`
- `anomaly_map_base64`, `anomaly_overlay_base64`, `pred_mask_overlay_base64`
- `defect_count`, `largest_defect_area`, `classical_overlay_base64`

## 5. Ghi chú vận hành

- Hệ thống đang khóa model về `capsule` trong logic backend.
- Checkpoint bắt buộc: `checkpoints/capsule.ckpt`.
- API phụ trợ:
- `GET /health` kiểm tra trạng thái import model stack.
- `GET /models` trả thông tin model/checkpoint khả dụng.
