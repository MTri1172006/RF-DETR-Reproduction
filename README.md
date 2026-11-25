# RF-DETR Weight-Sharing NAS Reproduction

Project Python được chuẩn hóa từ notebook `RFDETR_Reproduction_Full_NAS_L4.ipynb`, loại bỏ hoàn toàn đường dẫn cố định của Google Colab/Google Drive, áp dụng truyền tham số tường minh (Dependency Injection), lưu checkpoint an toàn (atomic save) và có sẵn bộ unit test kiểm định search space & dữ liệu.

## Cài đặt

```bash
cd RF-DETR-Reproduction
pip install -r requirements.txt
pip install -e .
```

> `requirements.txt` chốt cứng `roboflow/rf-detr` tại đúng commit `382607f3d7ecbf513b991a8ee9f1a3abe3a340f0` để đảm bảo khả năng tái lập mà không cần `git clone` động hay chỉnh sửa `sys.path`.

## Cấu trúc thư mục

- `configs/` – Toàn bộ siêu tham số và cấu hình thí nghiệm (`paper_config.py`, `nas_config.py`, `experiment_config.py`, `baseline_config.py`)
- `src/rfdetr_reproduction/` – Mã nguồn lõi chia theo trách nhiệm (`environment`, `data`, `nas`, `training`, `diagnostics`, `search`, `deployment`, `baselines`, `segmentation`, `utils`)
- `scripts/` – Các entry point dòng lệnh mỏng (chỉ parse tham số `argparse`, nạp config/paths rồi gọi hàm trong `src/`)
- `data/`, `checkpoints/`, `outputs/` – Thư mục lưu dữ liệu, trọng số và kết quả chạy (đã loại trừ khỏi Git trong `.gitignore`)
- `tests/` – Bộ unit test kiểm định validator, số lượng cấu hình search space và tính toàn vẹn dataset

## Hướng dẫn chạy (3 giai đoạn)

Mọi script trong `scripts/` đều hỗ trợ tùy chỉnh đường dẫn qua tham số `--data-dir`, `--ckpt-dir`, `--output-dir` (hoặc biến môi trường `RFDETR_DATA_DIR`, `RFDETR_CKPT_DIR`, `RFDETR_OUTPUT_DIR`) và tùy chọn sao lưu ngoài (`--backup-backend {local,hf,wandb,s3} --backup-dest ...`).

### Giai đoạn 1 – Tái lập lõi (Core Reproduction)
```bash
python scripts/setup_environment.py
python scripts/prepare_dataset.py
python scripts/train_nas.py --epochs 20
```

### Giai đoạn 2 – Thí nghiệm chẩn đoán & Grid Search
```bash
python scripts/run_gradient_conflict.py --tag ep20
python scripts/run_hypothesis1.py --epochs 20 --eval-every 2
python scripts/run_hypothesis2.py --extra-epochs 5
python scripts/run_grid_search.py --mode representative
```

### Giai đoạn 3 – Triển khai, Baselines & Mở rộng
```bash
python scripts/run_deployment.py
python scripts/run_baselines.py --epochs 20
python scripts/run_segmentation.py --enable
```

## Kiểm thử & Nghiệm thu

```bash
pytest -v
# Hoặc chạy trực tiếp bằng unittest:
python -m unittest discover -s tests -v
```

### Checklist nghiệm thu
- [x] **Provenance:** Lưu đầy đủ thông tin GPU, CUDA, PyTorch, phiên bản thư viện, Git SHA và `SEED=42` vào `outputs/provenance.json`.
- [x] **Search Space Validator:** Khớp tuyệt đối số lượng cấu hình hợp lệ so với notebook gốc:
  - Training (giới hạn ngân sách token L4 ≤ 5,776): **111** cấu hình hợp lệ (từ 231 cấu hình thô)
  - Inference (8 patch sizes liệt kê): **3,948** cấu hình hợp lệ (từ 7,392 cấu hình thô)
  - Inference (7 train patch sizes): **3,780** cấu hình hợp lệ (từ 6,468 cấu hình thô)
- [x] **Dataset Integrity:** Đã kiểm thử zero-overlap giữa `train` / `valid` / `test` và lọc đúng nhãn `difficult=1` trong `tests/test_dataset.py`.
- [x] **Checkpoint & Backup:** Dùng atomic save (`.pth.tmp` → `os.replace`), tách riêng `checkpoints/nas/nas_supernet_latest.pth`, `checkpoints/nas/nas_supernet_ema_only.pth`, `checkpoints/pretrained/rf-detr-base.pth` và `checkpoints/diagnostics/hypo1_state.pth`, không phụ thuộc Google Drive.
- [x] **Headless & Không phụ thuộc Colab:** Toàn bộ biểu đồ xuất qua `matplotlib` backend `Agg` (`savefig`), không còn `google.colab`, `display()` hay `!nvidia-smi`.
