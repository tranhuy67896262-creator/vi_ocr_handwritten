import torch
import torch.nn.functional as F
from transformers import Trainer


def _kl_divergence(logits, ref_logits, mask):
    """KL(ref || active) trung bình trên các token có mask=True."""
    log_probs = F.log_softmax(logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    kl = (ref_probs * (ref_probs.log() - log_probs)).sum(dim=-1)
    if mask is not None:
        kl = (kl * mask).sum() / mask.sum().clamp(min=1)
    else:
        kl = kl.mean()
    return kl


class KLLoRATrainer(Trainer):
    """Trainer chống mất kiến thức gốc (chỉ thêm kiến thức, không phá kiến thức cũ).

    Mỗi batch forward 2 lần:
      - Có LoRA  (train mode, có grad): logits của model đang fine-tune.
      - Tắt LoRA (`disable_adapter` + no_grad + eval): logits của model gốc = tham chiếu.

    Cộng `KL(ref || active) * kl_coef` vào CE loss → model học kiến thức mới nhưng
    phân phối không drift xa model gốc (chống catastrophic forgetting).
    Không cần copy model gốc thứ 2 nên không tốn thêm VRAM.
    """

    def __init__(self, *args, kl_coef=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        ce_loss = outputs.loss

        kl_loss = torch.zeros((), device=logits.device)
        if self.kl_coef > 0 and self.model is not None:
            with torch.no_grad():
                self.model.eval()
                with self.model.disable_adapter():
                    ref_outputs = self.model(**inputs)
                self.model.train()
            mask = labels != -100
            kl_loss = _kl_divergence(logits, ref_outputs.logits, mask)

        loss = ce_loss + self.kl_coef * kl_loss
        return (loss, outputs) if return_outputs else loss