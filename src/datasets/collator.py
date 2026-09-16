"""Collator Qwen2.5-VL: chỉ tính loss trên phần assistant."""
_PATCH = 28  # lưới patch của Qwen2.5-VL
# Token ảnh của Qwen2.5-VL — mask dự phòng trong labels (xem _mask_prompt).
_IMAGE_TOKENS = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


class DataCollatorForQwenVL:
    """Collator cho Qwen2.5-VL: tokenize chat template + chỉ tính loss trên phần assistant.

    ``max_length`` là budget dành cho TEXT (image-token được cộng thêm tự động,
    xem ``_effective_max_length``) — vì với VLM, chuỗi cuối cùng = text token +
    image token, và truncation cắt image-token sẽ gây lỗi "Mismatch in image
    token count".
    """

    def __init__(self, processor, max_length=None, min_pixels=None, max_pixels=None):
        self.processor = processor
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _effective_max_length(self):
        """Trả max_length thật sự truyền cho processor = text budget + image budget.

        Image-token ≤ max_pixels / (28*28) (smart_resize giữ diện tích ≤ max_pixels),
        nên cộng thêm phần này để truncation không bao giờ cắt image-token.
        """
        if self.max_length is None:
            return None
        image_budget = (self.max_pixels // (_PATCH * _PATCH)) if self.max_pixels else 0
        return self.max_length + image_budget

    def __call__(self, examples):
        texts = [
            self.processor.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
            for e in examples
        ]
        images = [e["images"] for e in examples]
        try:
            batch = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._effective_max_length(),
                add_special_tokens=False,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        except ValueError as exc:
            _log_batch_diagnostic(texts, images, self.processor, self._effective_max_length())
            raise exc

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        self._mask_prompt(examples, images, labels)

        batch["labels"] = labels
        return batch

    def _mask_prompt(self, examples, images, labels):
        """Mask toàn bộ prompt, chỉ giữ answer trong labels.

        BẮT BUỘC đo độ dài prompt trong KHÔNG GIAN MULTIMODAL (processor kèm
        ảnh): placeholder ``<|image_pad|>`` nở ra hàng trăm image-token khi qua
        processor, nên đo bằng tokenizer text-only sẽ ngắn hơn hàng trăm token
        so với ``input_ids`` thật — mask sót token ảnh của prompt vào labels,
        loss ≈ 0 và model không học gì (đã gặp thực tế: loss 0.0 suốt 800 step,
        decode labels toàn ``<|image_pad|>``).
        """
        prompt_texts = [
            self.processor.apply_chat_template(
                e["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            for e in examples
        ]
        prompt_ids = self.processor(
            text=prompt_texts,
            images=images,
            padding=False,
            truncation=True,
            max_length=self._effective_max_length(),
            add_special_tokens=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )["input_ids"]
        for i, ids in enumerate(prompt_ids):
            labels[i, : len(ids)] = -100
        # Dự phòng: mask mọi image-token còn sót (vd mẫu bị truncate).
        for tok in _IMAGE_TOKENS:
            try:
                tid = self.processor.tokenizer.convert_tokens_to_ids(tok)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(tid, int) and tid >= 0:
                labels[labels == tid] = -100


def _log_batch_diagnostic(texts, images, processor, max_length):
    """In chi tiết từng mẫu khi processor ném lỗi (vd Mismatch image token) để biết
    ảnh nào quá to / text quá dài so với max_length."""
    print("[collator] === LOI PROCESSOR — chi tiet batch ===")
    for i, (text, imgs) in enumerate(zip(texts, images)):
        text_tokens = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
        for j, img in enumerate(imgs):
            w, h = img.size
            tiles = ((w + 27) // 28) * ((h + 27) // 28)
            print(
                f"[collator]  sample {i} anh {j}: {w}x{h} ~{tiles} tile"
                f" | text ~{text_tokens} token"
                f" | uoc tinh tong ~{tiles + text_tokens} token (max_length={max_length})"
            )
    print("[collator] ================================")
