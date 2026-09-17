"""Kiểm tra HF_TOKEN đang dùng: tài khoản nào + có đọc được dataset gated không."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.configs import Configs


def _mask_token(token):
    """Che token, chỉ chừa đầu/đuôi để nhận diện."""
    if len(token) <= 7:
        return "***"
    return f"{token[:3]}***{token[-2:]}"


def main():
    """Parse CLI và kiểm tra token + quyền đọc dataset."""
    parser = argparse.ArgumentParser(description="Kiểm tra HF_TOKEN (tài khoản + quyền dataset)")
    parser.add_argument("--token", type=str, default=None, help="Token cần check (mặc định: từ Configs)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset cần check (mặc định: từ config)")
    parser.add_argument("--hub-repo", type=str, default=None, help="Repo adapter (check tồn tại, vd owner/repo)")
    args = parser.parse_args()

    config = Configs()
    token = args.token or config.HF_TOKEN
    if not token:
        print("Chưa có HF_TOKEN (env, .env.dev, --token đều trống).")
        return
    print(f"Token: {_mask_token(token)} (dài {len(token)} ký tự)")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Thiếu huggingface_hub — chạy: uv pip install huggingface_hub")
        return
    api = HfApi()
    try:
        info = api.whoami(token=token)
    except Exception as exc:  # token sai/hết hạn/mạng
        print(f"Token KHÔNG hợp lệ: {exc}")
        return
    print(f"Tài khoản: {info.get('name')} (type={info.get('type')})")

    dataset = args.dataset or config.DATASET_NAME
    try:
        api.dataset_info(dataset, token=token)
        print(f"Dataset {dataset}: ĐỌC ĐƯỢC")
    except Exception as exc:
        print(f"Dataset {dataset}: KHÔNG đọc được: {exc}")
        print("Mở trang dataset bằng đúng account trên để accept điều khoản.")

    if args.hub_repo:
        try:
            api.model_info(args.hub_repo, token=token)
            print(f"Repo {args.hub_repo}: ĐÃ tồn tại")
        except Exception as exc:
            print(f"Repo {args.hub_repo}: chưa thấy ({exc}) — push lần đầu sẽ tự tạo.")


if __name__ == "__main__":
    main()
