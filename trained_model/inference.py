import os
import torch
from .bigram import BigramLanguageModel, encode, decode, device, EOS_ID, block_size

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_model():
    model = BigramLanguageModel()
    ckpt_path = os.path.join(_BASE_DIR, "plain_english", "checkpoints", "1", "ckpt_93000.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


_model = _load_model()


def generate_answer(query: str, max_new_tokens: int = 200) -> str:
    prompt = f"<bos>Q: {query}\nA:"
    prompt_tokens = encode(prompt)
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = _model.generate(context, max_new_tokens=max_new_tokens)

    new_tokens = output[0][len(prompt_tokens):].tolist()
    response = decode(new_tokens)
    response = response.rstrip("<eos>")
    return response
