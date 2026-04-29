# Báo cáo Đánh giá Mô hình EfficientAD — Tập Capsule

> **Ngày thực hiện:** 29/04/2026  
> **Mô hình:** EfficientAD (model_size = **M**)  
> **Checkpoint:** `checkpoints/capsule.ckpt`  
> **Dataset kiểm thử:** MVTec AD — `capsule/test/` (132 ảnh: 23 Good, 109 Defect)  
> **Config huấn luyện:** 100 epochs, image_size = 256, batch_size = 1

---

## 1. Tổng quan kết quả

Sau khi tăng số epoch từ 30 lên **100** và chuyển sang kiến trúc **model_size = M**, hiệu suất mô hình cải thiện vượt trội so với phiên bản trước.

| Chỉ số | Model cũ (S, 30 epochs) | **Model mới (M, 100 epochs)** | Cải thiện |
|--------|------------------------|------------------------------|-----------|
| **image_AUROC** | 0.7587 | **0.9557** | +25.96% |
| **image_F1Score** | 0.9076 | **0.9474** | +4.39% |
| **pixel_AUROC** | 0.9151 | **0.9788** | +6.96% |
| **pixel_F1Score** | 0.4097 | **0.4949** | +20.79% |
| **Accuracy (Youden)** | 0.606 | **0.841** | +38.78% |
| **Precision** | 0.967 | **1.000** | +3.41% |
| **Recall** | 0.541 | **0.807** | +49.17% |
| **F1-Score** | 0.694 | **0.893** | +28.67% |

**Nhận xét:** Mọi chỉ số đều tăng, đặc biệt là **ROC-AUC** (từ 0.76 lên 0.96) và **Recall** (từ 54% lên 81%). Điều này chứng tỏ việc tăng số epoch và dùng kiến trúc lớn hơn giúp mô hình học được đặc trưng phân biệt rõ ràng hơn giữa ảnh bình thường và ảnh bất thường.

---

## 2. Phân tích Evaluation Charts

### 2.1. ROC Curve (Image-level)

![ROC Curve](outputs/evaluation_report.png)

- **AUC = 0.9557**
- Đường cong nằm rất gần góc trên-trái, cho thấy khả năng phân biệt **Good/Defect xuất sắc**.
- Ở ngưỡng **FPR ≈ 0.1**, model đã đạt **TPR ≈ 0.85**.
- So với đường chéo ngẫu nhiên (AUC = 0.5), model vượt trội hoàn toàn.

**Ý nghĩa:** Mô hình có thể phát hiện đa số defect mà chỉ tốn rất ít báo động giả.

---

### 2.2. Precision-Recall Curve (Image-level)

- **PR-AUC = 0.9906**
- Precision duy trì **gần 1.0** trong phần lớn dải Recall (0 → 0.85).
- Baseline = 0.826 (tỷ lệ defect trong tập test).

**Ý nghĩa:** Dù tập test bị mất cân bằng (82.6% là defect), mô hình vẫn giữ được độ chính xác cao. Khi báo cáo "DEFECT", gần như luôn đúng.

---

### 2.3. Anomaly Score Distribution

Phân phối điểm anomaly của hai lớp:

| Nhóm | Khoảng điểm chính | Nhận xét |
|------|-------------------|----------|
| **Good** | 0.45 – 0.50 | Tập trung thấp, phân tách rõ |
| **Defect** | 0.50 – 0.60 + đuôi dài đến ~0.9 | Phần lớn cao hơn Good |

- Hai histogram **tách biệt tốt** hơn hẳn model cũ (cũ bị chồng lấn nhiều).
- Có một số ít defect nhẹ vẫn nằm gần vùng Good (khoảng 0.48–0.52), đây là nguyên nhân chính của **21 False Negative**.

---

### 2.4. Confusion Matrix — Youden Index (thr = 0.5151)

| | Pred: Good | Pred: Defect |
|--|-----------|-------------|
| **True Good** (23) | **23** (TN) | 0 (FP) |
| **True Defect** (109) | 21 (FN) | **88** (TP) |

**Metrics:**
- **Accuracy:** 0.841
- **Precision:** 1.000 (không có báo động giả)
- **Recall:** 0.807 (bắt được 80.7% defect)
- **F1-Score:** 0.893

**Nhận xét:** Youden Index tối ưu hóa (TPR − FPR) cho ra ngưỡng cân bằng tốt nhất. Không có FP nào → hệ thống không báo nhầm Good thành Defect. Tuy nhiên 21 defect bị bỏ sót (FN), chủ yếu là các defect rất nhẹ hoặc nền phức tạp.

---

### 2.5. Confusion Matrix — Percentile 95 (thr = 0.5142)

| | Pred: Good | Pred: Defect |
|--|-----------|-------------|
| **True Good** (23) | 21 (TN) | 2 (FP) |
| **True Defect** (109) | 20 (FN) | **89** (TP) |

**Metrics:**
- **Accuracy:** 0.833
- **Precision:** 0.978
- **Recall:** 0.817
- **F1-Score:** 0.890

**Nhận xét:** Khi nâng ngưỡng lên P95 (chấp nhận 5% Good bị báo nhầm), Recall tăng thêm 1% (bắt thêm 1 defect), nhưng Precision giảm nhẹ xuống 97.8%. F1 gần như tương đương Youden.

---

### 2.6. So sánh Metrics giữa hai ngưỡng

| Metric | Youden (0.515) | Percentile 95 (0.514) | Khuyến nghị |
|--------|---------------|----------------------|-------------|
| Accuracy | **0.841** | 0.833 | Youden |
| Precision | **1.000** | 0.978 | Youden |
| Recall | 0.807 | **0.817** | P95 |
| F1 | **0.893** | 0.890 | Youden |

**→ Khuyến nghị sử dụng ngưỡng Youden = 0.515** cho production, vì F1 cao nhất và Precision 100% (không false alarm).

---

## 3. Phân tích Visualization

![Sample Visualizations](outputs/sample_visualizations.png)

Các ảnh trong `sample_visualizations.png` được chia thành 4 cột:

| Cột | Nội dung | Mô tả |
|-----|---------|-------|
| 1 | **Original** | Ảnh gốc từ tập test |
| 2 | **Anomaly Map** | Heatmap mức độ bất thường (jet colormap) |
| 3 | **Overlay** | Ảnh gốc pha trộn với heatmap |
| 4 | **Prediction** | GT (Ground Truth) / Pred (Dự đoán) / Score |

### Nhận xét từ visualization:

- **Anomaly Map** tập trung chính xác vào vùng defect (vết nứt, xước, lõm) thay vì nền.
- **Overlay** giúp người dùng dễ dàng xác định vị trí lỗi mà không cần chuyên môn deep learning.
- Các ảnh **Good** có score thấp (~0.45–0.50), map gần như đen (không bất thường).
- Các ảnh **Defect** có score cao hơn ngưỡng 0.515, map sáng rõ tại vùng bị lỗi.
- Một số defect nhẹ (ví dụ: scratch mờ) có score gần ngưỡng, map mờ nhạt → dễ bị FN.

---

## 4. Kết luận & Đề xuất

### Điểm mạnh
1. **ROC-AUC = 0.956** đạt mức xuất sắc cho bài toán anomaly detection.
2. **Precision = 100%** ở ngưỡng Youden: không có báo động giả, phù hợp cho dây chuyền sản xuất (tránh dừng máy oan).
3. **Recall = 80.7%**: bắt được đa số defect, chỉ bỏ sót các lỗi rất nhẹ.
4. Phân phối score tách biệt rõ ràng giữa Good và Defect.

### Hạn chế
1. **21/109 defect bị bỏ sót** (FN = 21). Các defect này thường là:
   - Lỗi kích thước rất nhỏ
   - Tương đồng màu sắc/vân nền với ảnh Good
   - Góc chụp hoặc ánh sáng khác biệt nhẹ
2. **Pixel-F1 chỉ đạt 0.495**: phân đoạn pixel-level chưa sắc nét, heatmap còn lan ra ngoài vùng defect.

### Đề xuất cải thiện
| Hướng | Cách làm | Kỳ vọng |
|-------|----------|---------|
| **Tăng image_size** | Từ 256 lên **512** | Phát hiện defect nhỏ hơn, pixel-F1 tăng |
| **Tăng epochs** | 100 → 150–200 | Student học sâu hơn, giảm FN |
| **Data augmentation** | Thêm xoay, lật, điều chỉnh độ sáng cho tập train | Tăng tính tổng quát |
| **Thử model_size = L** | Nếu GPU cho phép | Độ phân giải feature cao hơn |
| **Fine-tune threshold** | Theo từng loại defect riêng | Tối ưu recall cho defect nguy hiểm nhất |

---

## 5. Thông số kỹ thuật sử dụng

| Thành phần | Giá trị |
|-----------|---------|
| Framework | anomalib ≥ 1.2.0, PyTorch Lightning |
| Model | EfficientAD (model_size = M) |
| Teacher | Pre-trained on ImageNet (frozen) |
| Student | Trainable, học tái tạo Teacher output |
| Input size | 256 × 256 |
| Loss | L2 distance between Teacher & Student feature maps |
| Optimizer | Adam (mặc định anomalib) |
| Hardware train | Google Colab (GPU) |
| Hardware infer | CPU (local) |

---

*Report generated automatically from evaluation outputs.*
