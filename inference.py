import argparse
import torch
import tiktoken
from utils.training_utils import generate, text_to_token_ids, token_ids_to_text
from model import GPTModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_CONFIG_124M = {
      "vocab_size": 50257,   # Vocabulary size
      "context_length": 256, # Shortened context length (orig: 1024)
      "emb_dim": 768,        # Embedding dimension
      "n_heads": 12,         # Number of attention heads
      "n_layers": 12,        # Number of layers
      "drop_rate": 0.1,      # Dropout rate
      "qkv_bias": False      # Query-key-value bias
  }
  
tokenizer = tiktoken.get_encoding("gpt2")

def gpt2_openai_weight_inference(input_text: str):
  from utils.gpt_download import download_and_load_gpt2
  from utils.gpt_load import load_weights_into_gpt

  settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

  # Define model configurations in a dictionary for compactness
  model_configs = {
      "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
      "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
      "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
      "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
  }

  # Copy the base configuration and update with specific model settings
  model_name = "gpt2-small (124M)"  # Example model name
  NEW_CONFIG = GPT_CONFIG_124M.copy()
  NEW_CONFIG.update(model_configs[model_name])
  NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

  gpt = GPTModel(NEW_CONFIG)
  gpt.eval()

  load_weights_into_gpt(gpt, params)
  gpt.to(device)

  token_ids = generate(
      model=gpt,
      idx=text_to_token_ids(input_text, tokenizer).to(device),
      max_new_tokens=25,
      context_size=NEW_CONFIG["context_length"],
      top_k=50,
      temperature=1.5
  )

  print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

def gpt2_scratch_weight_inference(input_text: str):
  #checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
  checkpoint = torch.load("model.pth", weights_only=True)

  model = GPTModel(GPT_CONFIG_124M)
  model.load_state_dict(checkpoint)
  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total number of parameters: {total_params:,}")

  total_size_bytes_ = total_params * 4
  total_size_mb_ = total_size_bytes_ / (1024 * 1024)

  print(f"Total size of the model: {total_size_mb_:.2f} MB")

  #total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
  #print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

  # # Calculate parameter size
  # param_size = 0
  # for param in model.parameters():
  #     param_size += param.nelement() * param.element_size()
  # print(f"""parameter size: {param_size}
  # param.nelement() {param.nelement()}
  # param.element_size() {param.element_size()}""")
  # print("\n")

  # # Calculate buffer size
  # buffer_size = 0
  # for buffer in model.buffers():
  #     buffer_size += buffer.nelement() * buffer.element_size()
  # print(f"""biffer size: {buffer_size}
  # buffer.nelement() {buffer.nelement()}
  # buffer.element_size() {buffer.element_size()}""")

  # total_size_bytes = param_size + buffer_size
  # total_size_mb = total_size_bytes / (1024**2)

  # print(f"Total model parameters byte size: {total_size_bytes} bytes")
  # print(f"Total model parameters size: {total_size_mb:.2f} MB")

  model.eval()
  model.to(device)

  token_ids = generate(
      model=model,
      idx=text_to_token_ids(input_text, tokenizer).to(device),
      max_new_tokens=25,
      context_size=GPT_CONFIG_124M["context_length"],
      top_k=50,
      temperature=1.5
  )

  print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="GPT-2 Inference")

  parser.add_argument("user_input", type=str, help="Starting prompt.")
  parser.add_argument("--load_openaigpt2_weight", 
  type=str, default="no", help="Set whether to load openai gpt2 weights or not.")

  args = parser.parse_args()
  
  if args.load_openaigpt2_weight=="yes":
    print("Inferencing on OpenaAI GPT2 pretrained weights.")
    gpt2_openai_weight_inference(args.user_input)
  else:
    print("Inferencing on scratch pretrained weights.")
    gpt2_scratch_weight_inference(args.user_input)
