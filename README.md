# Dự đoán giá vàng
Đồ án cuối kỳ môn học MTH10605 - Python cho Khoa học dữ liệu

Giảng viên: ThS. Hà Minh Tuấn

## Thông tin thành viên 
* Huỳnh Nguyễn Bảo Nguyên - 23280001
* Nguyễn Triều Dương - 23280015
* Nguyễn Hoàng Trọng Nghĩa - 23280030

## Giới thiệu bài toán
**Dự đoán giá vàng** là một bài toán quan trọng trong phân tích tài chính, giúp hỗ trợ nhà đầu tư đưa ra quyết định giao dịch và quản trị rủi ro. Giá vàng thường biến động mạnh bởi nhiều yếu tố kinh tế vĩ mô như lạm phát, lãi suất, tỷ giá và thị trường năng lượng. Do đó, xây dựng mô hình học máy để dự báo giá vàng có ý nghĩa thực tiễn và mang tính thách thức cao.

Bộ dữ liệu được sử dụng trong dự án được lấy từ Kaggle: *Gold Price Prediction Dataset*. Đây là tập dữ liệu dạng chuỗi thời gian bao gồm thông tin giá vàng theo ngày cùng các thuộc tính liên quan đến thị trường tài chính. Các trường dữ liệu chính bao gồm giá mở cửa (Open), giá đóng cửa (Close), giá cao nhất (High), giá thấp nhất (Low) và khối lượng giao dịch (Volume). Để đánh giá hoạt động của lớp Tiền xử lý dữ liệu, nhóm đã chủ động điều chỉnh bộ dữ liệu gốc bằng cách bổ sung một số yếu tố gây nhiễu và sai lệch. 

Nguồn của bộ dữ liệu: [https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset/data](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset/data)

Trong phạm vi môn học, nhóm xây dựng một quy trình tổng thể bao gồm các bước: tiền xử lý dữ liệu – lựa chọn mô hình học máy – huấn luyện – đánh giá – trực quan hoá kết quả. Đồng thời nhóm so sánh hiệu quả giữa các mô hình để rút ra những nhận định về ưu điểm và hạn chế của từng phương pháp đối với bài toán dự đoán giá vàng.

## Cấu trúc thư mục
```
project/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   └── processed/
│       └── data_clean.csv
│
├── preprocessing/
│   ├── loader.py
│   │   └── DataLoader
│   ├── analyzer.py
│   │   └── StatisticsAnalyzer
│   ├── transformers.py
│   │   ├── DataPreprocessor
│   │   └── DataScaler
│   └── visualization.py
│       ├── BasicVisualizer
│       └── DataFrameVisualizer
│
├── training/
│   ├── config.py
│   │   └── ConfigManager
│   ├── datamodule.py
│   │   ├── DataModule
│   │   └── TimeSeriesSplitter
│   ├── trainer.py
│   │   ├── ModelTrainer
│   │   └── HyperParameterOptimizer
│   └── model_summary.py
│       └── ModelVisualizer
│
├── utils/
│   └── logger.py
│       └── Logger    
│
├── results/
│   ├── models/
│   │   ├── catboost_final.joblib
│   │   ├── xgboost_final.joblib
│   │   ├── lightgbm_final.joblib
│   │   ├── random_forest_final.joblib
│   │   └── final_scaler.joblib/
│   ├── plots/
│   │   ├── leaderboard.png
│   │   ├── forecast_catboost.png
│   │   ├── residuals_catboost.png
│   │   └── shap_catboost.png
│   ├── metrics/
│   │   ├── best_hyperparameters.json
│   │   └── final_scores.json
│   └── logs/
│       ├── preprocessing.log
│       └── training.log   
│
├── requirements.txt
├── config.yaml
├── main.py                           
└── README.md
```
## Hướng dẫn cài đặt

Vì đây là gói mã nguồn được nén (file .zip), vui lòng làm theo các bước sau để thiết lập môi trường chạy.

### Bước 1: Giải nén và mở Terminal
1.  Giải nén file `.zip` vào một thư mục trên máy tính.
2.  Mở **Terminal** (macOS/Linux) hoặc **Command Prompt / PowerShell** (Windows).
3.  Di chuyển đường dẫn vào thư mục vừa giải nén:
    ```bash
    cd Duong/Dan/Toi/Thu/Muc/Gold-Price-Prediction
    ```

### Bước 2: Tạo môi trường ảo (Virtual Environment)
Khuyên dùng môi trường ảo để tránh xung đột thư viện.

* **Đối với Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

* **Đối với macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### Bước 3: Cài đặt thư viện
Cài đặt các gói cần thiết được liệt kê trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy chương trình
Trước khi chạy, hãy kiểm tra thư mục data/raw/ xem đã có file data.csv chưa (nếu chưa, vui lòng tải từ Kaggle và đặt vào đó).

Chạy lệnh sau để bắt đầu huấn luyện và dự đoán:
```bash
python main.py
```
