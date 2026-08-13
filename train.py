'''train.py is just a replica of train.ipynb'''

import math
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import PAD_IDX, SOS_IDX, EOS_IDX
import sacrebleu
from fix_punctuation import clean_text

class InverseSigmoidDecay:
    def __init__(self, k=8.0):
        self.k = k
    def step(self, epoch):
        return self.k / (self.k + math.exp(epoch / self.k))


def _decode_ids_to_strings(pred_ids, label_ids, tgt_tokenizer):
    hyps, refs = [], []
    for pred, ref in zip(pred_ids, label_ids):
        pred = pred.tolist()
        ref = ref.tolist()
        if EOS_IDX in pred:
            pred = pred[:pred.index(EOS_IDX)]
        if EOS_IDX in ref:
            ref = ref[:ref.index(EOS_IDX)]
        pred = [t for t in pred if t not in (PAD_IDX, SOS_IDX)]
        ref = [t for t in ref if t not in (PAD_IDX, SOS_IDX)]
        
        hyp_str = tgt_tokenizer.decode(pred, skip_special_tokens=True).strip()
        ref_str = tgt_tokenizer.decode(ref, skip_special_tokens=True).strip()
        
        hyps.append(clean_text(hyp_str))
        refs.append(clean_text(ref_str))
        
    return hyps, refs


def _run_epoch(model, dataloader, criterion, optimizer, device, tf_prob: float,
               is_training: bool, tgt_tokenizer=None):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    all_hyps, all_refs = [], []

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for batch in tqdm(dataloader, leave=False, desc="Train" if is_training else "Val"):
            src_ids = batch["src_ids"].to(device)
            src_mask = batch["src_attention_mask"].to(device)
            tgt_inputs = batch["tgt_inputs"].to(device)
            tgt_labels = batch["tgt_labels"].to(device)
            tgt_mask = batch["tgt_attention_mask"].to(device)

            B, T = tgt_inputs.shape

            encoder_out = model.encode(src_ids, src_mask)
            hidden = model.init_decoder_hidden(encoder_out)
            token = tgt_inputs[:, 0]
            logit_slices = []

            # Generate a single random number before the sequence loop starts
            # Create a boolean variable representing whether teacher forcing is active for this entire batch
            use_teacher_forcing = is_training and random.random() < tf_prob

            for t in range(T):
                # Forward pass through the decoder for the current time step
                step_logit, hidden = model.decode_step(token, hidden, encoder_out, src_mask)
                logit_slices.append(step_logit.unsqueeze(1))
                
                # Prepare the input token for the next time step naturally avoiding off-by-one errors
                if t + 1 < T:
                    if use_teacher_forcing:
                        # If true, the entire batch uses the true ground truth words for the next step.
                        # E.g., at t=0, we assign the actual first word `tgt_inputs[:, 1]` as the next input.
                        token = tgt_inputs[:, t + 1]
                    else:
                        # If false, the entire batch uses the model's highest probability predictions.
                        token = step_logit.argmax(dim=-1)

            logits = torch.cat(logit_slices, dim=1)

            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            n_tokens = tgt_mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            if not is_training and tgt_tokenizer is not None:
                preds = logits.argmax(dim=-1).cpu()
                labels = tgt_labels.cpu()
                hyps, refs = _decode_ids_to_strings(preds, labels, tgt_tokenizer)
                all_hyps.extend(hyps)
                all_refs.extend(refs)

    avg_loss = total_loss / max(total_tokens, 1)

    if not is_training and tgt_tokenizer is not None and len(all_hyps) > 0:
        bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score
        chrf = sacrebleu.corpus_chrf(all_hyps, [all_refs]).score
        ter = sacrebleu.corpus_ter(all_hyps, [all_refs]).score
        return avg_loss, {"bleu": bleu, "chrf": chrf, "ter": ter}

    return avg_loss, None

def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs: int = 30, save_path: str = "best_model.pth",
                scheduler=None, tf_mode: str = "sigmoid", fixed_tf_ratio: float = 0.6,
                tgt_tokenizer=None):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    model.to(device)

    best_val_bleu = -1.0
    epochs_without_improvement = 0
    scheduler_patience = 3
    early_stopping_patience = 6

    tf_scheduler = InverseSigmoidDecay(k=8.0)
    
    history = {
        "train_losses": [], "val_losses": [], "tf_probs": [],
        "val_bleu": [], "val_chrf": [], "val_ter": []
    }

    for epoch in range(num_epochs):
        if tf_mode == "sigmoid":
            p_t = tf_scheduler.step(epoch)
        else:
            p_t = fixed_tf_ratio

        train_loss, _ = _run_epoch(model, train_loader, criterion, optimizer, device,
                                   tf_prob=p_t, is_training=True)
        val_loss, val_metrics = _run_epoch(model, val_loader, criterion, optimizer, device,
                                           tf_prob=0.0, is_training=False, tgt_tokenizer=tgt_tokenizer)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["tf_probs"].append(p_t)

        metric_str = ""
        current_bleu = 0.0
        if val_metrics:
            current_bleu = val_metrics["bleu"]
            history["val_bleu"].append(current_bleu)
            history["val_chrf"].append(val_metrics["chrf"])
            history["val_ter"].append(val_metrics["ter"])
            metric_str = f" │ BLEU: {val_metrics['bleu']:.2f} │ chrF: {val_metrics['chrf']:.2f} │ TER: {val_metrics['ter']:.2f}"

        print(f"Epoch {epoch+1:3d}/{num_epochs} │ TF: {p_t:.4f} │ Train: {train_loss:.4f} │ Val: {val_loss:.4f}{metric_str}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_bleu)
            else:
                scheduler.step()

        if current_bleu > best_val_bleu:
            best_val_bleu = current_bleu
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Saved new best model (val_bleu={current_bleu:.2f})")
        else:
            epochs_without_improvement += 1
            print(f"  ↳ No improvement for {epochs_without_improvement} epochs.")
            
            if epochs_without_improvement >= early_stopping_patience:
                print("  ↳ Early stopping limit reached. Halting training.")
                break

    print(f"\nTraining complete. Best val BLEU: {best_val_bleu:.2f}")
    return history