import torch
import os
import sys
import sentencepiece as spm

# Add project root to path so trained_model.* is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trained_model.constants import (
    modes, mode, batch_size, block_size, eval_interval, eval_iters,
    learning_rate, training_iteration, base_training_file_name,
    qa_training_file_name, train_file_name, text_training_steps,
    qa_training_steps, checkpoint_start,
)


match mode:
    case 1:
        learning_rate = 1e-4
        train_file_name = base_training_file_name
        max_iters = text_training_steps
    case 2:
        learning_rate = 3e-4
        train_file_name = qa_training_file_name
        checkpoint_start = text_training_steps
        max_iters = text_training_steps + qa_training_steps


spm.SentencePieceTrainer.train(
    input=f'{train_file_name}.txt',
    model_prefix='trained_model/spm',
    vocab_size=8000,
    model_type='bpe',
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    pad_id=3
)

from trained_model.bigram import (
    BigramLanguageModel, encode, decode, device,
    sp, EOS_ID, vocab_size, log_seperator
)

log_seperator()
with open(f"{train_file_name}.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"Number of lines : {len(lines)}")
    print("Sample lines : ")
    print(lines[:10])
log_seperator()


tokens = []
for line in lines:
    ids = sp.encode(line.strip(), out_type=int)
    tokens.extend(ids + [sp.eos_id()])


data = torch.tensor(tokens, dtype=torch.long)
print("Data loaded to tensor")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


log_seperator()
print("Data spilt")
print("Vocab size:", vocab_size)
print("Data length:", len(data))
print("Sample tokens:", data[:20])
print("Decoded sample:", decode(data[:50].tolist()))
log_seperator()


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # say len of data is 1000, then this gives randint from : 0 to 992-1 with the size of 4
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

save_path = f"{base_training_file_name}/checkpoints/{training_iteration}"
if not os.path.exists(save_path):
	os.makedirs(save_path)

print("Model started training")

log_seperator()
if mode == 2:
    ckpt_path = f"{save_path}/ckpt_{text_training_steps}.pt"
    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_iter = checkpoint["step"] + 1
        print(f"Resuming from step {start_iter}")
    else:
        print("No checkpoint found, starting fresh")



print(f"mode : {mode}")
print(f"batch_size : {batch_size}")
print(f"block_size : {block_size}")
print(f"eval_interval : {eval_interval}")
print(f"n_embd : {384}")
print(f"n_head : {6}")
print(f"n_layer : {6}")
print(f"dropout : {0.1}")
print(f"eval_iters : {eval_iters}")
print(f"learning_rate : {learning_rate}")
print(f"training_iteration : {training_iteration}")
print(f"base_training_file_name : {base_training_file_name}")
print(f"qa_training_file_name : {qa_training_file_name}")
print(f"train_file_name : {train_file_name}")
print(f"text_training_steps : {text_training_steps}")
print(f"qa_training_steps : {qa_training_steps}")
print(f"checkpoint_start : {checkpoint_start}")
print(f"Save path : {save_path}")
print(f"Start checkpoint : {checkpoint_start}")
print(f"Max iterations : {max_iters}")

log_seperator()


with open(f"{save_path}/loss_eval.txt", "a") as loss_eval_file:
    for iter in range(checkpoint_start+1, max_iters + 1):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print("="*50)
            loss_str = f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            print(loss_str)
            loss_eval_file.write(loss_str + "\n")
            loss_eval_file.flush()

            torch.save(
                {
                    "step": iter,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": losses["train"],
                    "val_loss": losses["val"]
                },
                f"{save_path}/ckpt_{iter}.pt"
            )
            if mode == 1:
                context = torch.zeros((1,1), dtype=torch.long, device=device)
                print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
            elif mode == 2:
                prompt = "<bos>Q: What is AI?\nA:"
                context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
                print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
            print("="*50)

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
print("Model training completed")




# context = torch.zeros((1,1), dtype=torch.long, device=device)
# text_length = 10000
# generated_text = decode(m.generate(context, max_new_tokens=text_length)[0].tolist())
# print(generated_text)
# with open(f"{text_file_name}_trained_output.txt", "w") as output_file:
#     output_file.write(generated_text)




# Generation :

# ckpt = torch.load(f"{save_path}/ckpt_93000.pt", map_location=device)
# print("checkpoint loaded")
# model.load_state_dict(ckpt["model_state"])
# model.eval()

# print("model loaded")
# optimizer.load_state_dict(ckpt["optimizer_state"])


# prompt = "<bos>Q: Where is America?\nA:"
# context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

# print("context found")
# with torch.no_grad():
#     tokens = model.generate(context, max_new_tokens=1000)
# print("tokens generated")
# generated_text = decode(tokens[0].tolist())
# print(generated_text)
