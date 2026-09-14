"""Collator Qwen2.5-VL: chỉ tính loss trên phần assistant."""


class DataCollatorForQwenVL:
    """Collator cho Qwen2.5-VL: tokenize chat template + chỉ tính loss trên phần assistant.

    max_length: giới hạn độ dài chuỗi (truncate) để kiểm soát VRAM.

    Lưu ý: KHÔNG tính prompt_len bằng cách tokenize prompt tách rời — Qwen2.5-VL
    mở rộng `<|image_pad|>` thành nhiều vision-token, nên độ dài đó nhỏ hơn thực
    tế và sẽ để lọt token ảnh vào loss. Ở đây lấy đúng ranh giới từ ``attention_mask``
    (độ dài thực) rồi mask mọi thứ trừ ``assistant content + <|im_end|>`` (phần đuôi).
    """

    def __init__(self, processor, max_length=None, min_pixels=None, max_pixels=None):
        self.processor = processor
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _assistant_text(self, example):
        """Lấy text của lượt assistant cuối cùng (phần cần tính loss)."""
        content = example["messages"][-1]["content"]
        if isinstance(content, str):
            return content
        for block in content:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""

    def _labels(self, batch, examples):
        """Gán -100 cho prompt + padding, giữ nguyên phần assistant."""
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        attention = batch.get("attention_mask")
        tokenizer = self.processor.tokenizer
        for i, example in enumerate(examples):
            real_len = int(attention[i].sum()) if attention is not None else labels.shape[1]
            answer = self._assistant_text(example)
            n_answer = len(tokenizer(answer, add_special_tokens=False)["input_ids"]) + 1  # + <|im_end|>
            labels[i, : max(0, real_len - n_answer)] = -100
        return labels

    def __call__(self, examples):
        texts = [
            self.processor.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
            for e in examples
        ]
        images = [e["images"] for e in examples]
        kwargs = {}
        if self.min_pixels is not None:
            kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            kwargs["max_pixels"] = self.max_pixels
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            **kwargs,
        )
        batch["labels"] = self._labels(batch, examples)
        return batch
