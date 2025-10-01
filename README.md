# Vietnamese Abstractive Summarization

Một project nghiên cứu về tự động tóm tắt văn bản tiếng Việt sử dụng mô hình Transformer.

## 📋 Tổng quan

Project này xây dựng một hệ thống tóm tắt trừu tượng (abstractive summarization) cho văn bản tiếng Việt, sử dụng:
- Dataset VietNews-Abs-Sum từ HuggingFace
- Tokenizer BPE (Byte-Pair Encoding) tùy chỉnh
- Mô hình Transformer cho text summarization
- Các metrics đánh giá ROUGE, F1, Accuracy

## 🗂️ Cấu trúc Project

```
Vietnamese-Abstractive-Summarization/
├── data/
│   └── raw/                    # Dữ liệu thô từ dataset
│       ├── dataset_info.json   # Thông tin dataset
│       ├── train.csv          # Dữ liệu huấn luyện
│       ├── validation.csv     # Dữ liệu validation
│       └── test.csv           # Dữ liệu test
├── models/
│   └── transformer_summarizer.py  # Mô hình Transformer
├── src/
│   └── train.py               # Script huấn luyện
├── utils/
│   └── metrics.py             # Các metrics đánh giá
├── bpe_tokenizer/
│   └── bpe_merges.json        # BPE tokenizer đã train
├── datasetdownloader.ipynb    # Download dataset từ HuggingFace
├── tokenizer.ipynb            # Train BPE tokenizer
└── requirements.txt           # Dependencies
```

## 📊 Dataset

**VietNews-Abs-Sum** - Dataset tóm tắt tin tức tiếng Việt:
- **Train**: 303,686 mẫu
- **Validation**: 67,010 mẫu  
- **Test**: 67,640 mẫu
- **Tổng dung lượng**: ~2.9GB
- **Features**: guid, title, abstract, article

## 🚀 Cách chạy Project

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone <repository-url>
cd Vietnamese-Abstractive-Summarization

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Chạy notebook `datasetdownloader.ipynb` để tải dataset:

```bash
jupyter notebook datasetdownloader.ipynb
```

Hoặc chạy từng cell:
- Cell 1: Import thư viện
- Cell 2: Class HuggingFaceAPIDownloader để download dataset
- Cell 3: Khởi tạo downloader và download
- Cell 4: Kiểm tra dữ liệu đã download

### 3. Train BPE Tokenizer

Chạy notebook `tokenizer.ipynb`:

```bash
jupyter notebook tokenizer.ipynb
```

Notebook này sẽ:
- Tạo vocabulary từ corpus
- Train BPE tokenizer
- Lưu tokenizer vào `bpe_tokenizer/bpe_merges.json`

### 4. Train Model

```bash
python src/train.py
```

## 🛠️ Công nghệ sử dụng

- **Python**: Ngôn ngữ chính
- **PyTorch/TensorFlow**: Framework deep learning
- **HuggingFace**: Dataset và tools
- **BPE**: Byte-Pair Encoding tokenization
- **Transformer**: Kiến trúc mô hình
- **ROUGE**: Metrics đánh giá

## 📈 Metrics đánh giá

Project sử dụng các metrics trong `utils/metrics.py`:
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **F1 Score**: Macro F1
- **Accuracy**: Độ chính xác

## 🔧 Tiến độ hiện tại

### ✅ Đã hoàn thành:
- [x] Thiết lập cấu trúc project
- [x] Download dataset VietNews-Abs-Sum
- [x] Implement BPE tokenizer
- [x] Thiết lập metrics đánh giá

### 🚧 Đang thực hiện:
- [ ] Hoàn thiện mô hình Transformer
- [ ] Script training
- [ ] Preprocessing data

### 📝 Kế hoạch:
- [ ] Huấn luyện mô hình
- [ ] Đánh giá performance
- [ ] Fine-tuning hyperparameters
- [ ] Demo ứng dụng

## 💡 Ghi chú

1. **Dataset**: Đã download thành công với 438,336 mẫu tổng cộng
2. **Tokenizer**: BPE tokenizer đã được implement với preprocessing tiếng Việt
3. **Model**: Transformer summarizer đang trong giai đoạn phát triển
4. **Metrics**: Sẵn sàng cho việc đánh giá model

## 📚 Tài liệu tham khảo

- [VietNews-Abs-Sum Dataset](https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [BPE Tokenization](https://arxiv.org/abs/1508.07909)
