class DataCollatorForQwenVL:
    """Collator cho Qwen2.5-VL: tokenize chat template + chỉ tính loss trên phần assistant.

    max_length: giới hạn độ dài chuỗi (truncate) để kiểm soát VRAM.
    """

    def __init__(self, processor, max_length=None):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples):
        texts = [
            self.processor.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
            for e in examples
        ]
        images = [e["images"] for e in examples]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, e in enumerate(examples):
            prompt = self.processor.apply_chat_template(
                e["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(
                self.processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch