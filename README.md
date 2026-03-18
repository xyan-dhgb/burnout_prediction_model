# Dự đoán Burnout (Burnout Prediction Model)

## Tổng quan

- Dự án này ứng dụng học máy (Machine Learning - ML) để phân loại và dự đoán mức độ "Burnout" (kiệt sức) dựa trên dữ liệu khảo sát và các chỉ số sức khỏe tinh thần cho môn học Tâm lý học đại cương

- Dự án sử dụng mô hình **Random Forest Classifier** kết hợp kỹ thuật **SMOTE** (xử lý mất cân bằng lớp), **VarianceThreshold** (lọc dữ liệu thưa), và tối ưu siêu tham số bằng **GridSearchCV**.

- Mô hình cũng sử dụng **SHAP** (SHapley Additive exPlanations) để giải thích mức độ ảnh hưởng của từng đặc trưng lên kết quả.

## Tổ chức mã nguồn (Directory Structure)

```text
burnout-model/
├── data/                    # Thư mục chứa dữ liệu thô (raw) và dữ liệu đã qua xử lý (processed)
├── models/                  # Nơi lưu trữ mô hình đã huấn luyện (.pkl) và các config (.json)
├── notebook/                # Các file Jupyter Notebook (phục vụ nghiên cứu & chạy thử nghiệm)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── reports/                 # Chứa các báo cáo đầu ra
│   └── figures/             # Các biểu đồ trực quan hóa (EDA, Model Metrics, SHAP values...)
├── src/                     # Mã nguồn Python độc lập để chạy tự động (tương tự như notebook nhưng dạng script)
│   ├── data_loader.py       # Load và phân tích dữ liệu cơ bản
│   ├── preprocess.py        # Xử lý dữ liệu (Mã hóa One-Hot, Handling Imbalanced Classes)
│   ├── train.py             # Huấn luyện mô hình, Feature Selection và GridSearchCV
│   └── evaluate.py          # Đánh giá mô hình và trích xuất SHAP values
├── utility/                 # Các tiện ích kịch bản phụ trợ
│   └── convert_to_csv.py    # Chuyển đổi file Excel sang CSV để thuận tiện cho việc pre-process
├── requirements.txt         # Các thư viện Python cần thiết
└── README.md
```

## Hướng dẫn cài đặt (Installation)

1. Cài đặt Python (phiên bản 3.8+ được khuyến nghị).
2. Tạo môi trường ảo (tùy chọn nhưng nên làm):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Trên Windows
   # source .venv/bin/activate # Trên macOS/Linux
   ```
3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Cách sử dụng

- Dự án cung cấp 2 cách để chạy và thử nghiệm: Dùng **Jupyter Notebook** (Khuyên dùng để theo dõi visual/report) hoặc chạy các **File Script (.py)** (Dùng cho tự động hóa terminal workflow). Trình tự thực thi như sau:

### Cách 1: Chạy bằng Terminal

- Mở terminal tại thư mục gốc của project (nơi chứa file README) và chạy lần lượt:

1. **Khám phá dữ liệu (EDA)**

   ```bash
   python src/data_loader.py
   ```

2. **Tiền xử lý (Preprocessing)** (Encode dữ liệu, Train/Test split...)

   ```bash
   python src/preprocess.py
   ```

3. **Huấn luyện mô hình (Training & Tuning)** (Random Forest, áp dụng SMOTE, Hyperparameter tuning...)

   ```bash
   python src/train.py
   ```

4. **Đánh giá và Giải thích (Evaluation & SHAP)** (Ra biểu đồ Metrics summary ROC-AUC, F1-score, Beeswarm SHAP...)
   ```bash
   python src/evaluate.py
   ```

_(Lưu ý: Mọi output đầu ra như data sau tiền xử lý, model objects, biểu đồ phân tích... sẽ được ghi tự động vào các thư mục `data/processed/`, `models/` và `reports/figures/`)_

### Cách 2: Dùng Jupyter Notebook

- Chúng ta có thể mở và chạy lần lượt các file trong thư mục `notebook/` theo thứ tự từ `01_eda`, `02_preprocessing`, `03_modeling` đến `04_evaluation`.

- Nội dung đã được thiết kế sẵn dưới dạng các block cell kèm theo các dòng comment giải thích ý nghĩa của từng bước, rất thuận tiện cho quá trình debug và xuất report tĩnh phục vụ làm tài liệu nghiên cứu.

## Các kỹ thuật nổi bật được ứng dụng

- **Xử lý mất cân bằng dữ liệu (Imbalanced Data)**:
  Sử dụng kỹ thuật over-sampling **SMOTE** (Synthetic Minority Over-sampling Technique) cho tập train nhằm nâng số lượng mẫu lớp thiểu số (minority labels). Kết hợp tham số `class_weight='balanced'` trong kiến trúc RF.

- **Lựa chọn đặc trưng (Feature Selection)**:
  Sử dụng `VarianceThreshold` để tự động loại bỏ các feature phân bố quá nghèo nàn / có phương sai gần tịt về 0 (đóng góp rất ít thông tin dự đoán).

- **Tránh Mẫu khớp (Overfitting Prevention)**:
  Dùng `GridSearchCV` trên 5-fold cross-validation kèm theo custom parameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) giúp model mang tính khái quát cao (generalized) hơn.

- **Explainable AI (Interpretable ML)**:
  Tích hợp cấu trúc **SHAP** TreeExplainer để đo lường mức độ quan trọng biểu diễn toàn cục (global) và cả hướng vận động, cách giải thích xác suất theo từng mẫu dữ liệu đơn lẻ dự đoán (waterfall, beeswarm plot).
