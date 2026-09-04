from configs.configs import Configs


def main():
    # Khởi tạo config
    config = Configs()

    print("=" * 50)
    print("🚀 Vi-OCR-Handwritten Project")
    print("=" * 50)
    print(f"📁 Project Root: {config.PROJECT_ROOT}")
    print(f"📂 Data Dir: {config.DATA_DIR}")
    print(f"🤗 HF Token: {'✅' if config.HF_TOKEN else '❌ (Chưa set)'}")
    print(f" Dataset: {config.DATASET_NAME}")
    print(f"🧠 Model: {config.MODEL_NAME}")
    print("=" * 50)

    # Kiểm tra token
    if not config.HF_TOKEN:
        print("⚠️  Cảnh báo: HF_TOKEN chưa được set!")
        print("   Hãy thêm token vào file .env.dev")
        print("   Lấy token tại: https://huggingface.co/settings/tokens")
        return

    # Tiếp tục với code load dataset...


if __name__ == "__main__":
    main()