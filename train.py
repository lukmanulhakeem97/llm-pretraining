import os
import time
import urllib.request
import tiktoken
import torch

from load_data import create_dataloader_v1
from model import GPTModel
from utils.training_utils import train_model_simple, plot_losses


def main():
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
  

  ######## Preparing Data ########

  file_path = "./data/the-verdict.txt"
  url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

  print("Data Loaded...")
  if not os.path.exists(file_path):
      with urllib.request.urlopen(url) as response:
          text_data = response.read().decode('utf-8')
      with open(file_path, "w", encoding="utf-8") as file:
          file.write(text_data)
  else:
      with open(file_path, "r", encoding="utf-8") as file:
          text_data = file.read()
  
  tokenizer = tiktoken.get_encoding("gpt2") #  use Byte Pair Encoding (BPE)

  total_characters = len(text_data)
  total_tokens = len(tokenizer.encode(text_data))

  print("Total Characters:", total_characters)
  print("Total Tokens:", total_tokens)

  # Train/validation ratio
  train_ratio = 0.90
  split_idx = int(train_ratio * len(text_data))
  train_data = text_data[:split_idx]
  val_data = text_data[split_idx:]


  ####### Creating data loaders ##########

  train_loader = create_dataloader_v1(
      train_data,
      batch_size=2,
      max_length=GPT_CONFIG_124M["context_length"],
      stride=GPT_CONFIG_124M["context_length"],
      drop_last=True,
      shuffle=True,
      num_workers=0
  )

  val_loader = create_dataloader_v1(
      val_data,
      batch_size=2,
      max_length=GPT_CONFIG_124M["context_length"],
      stride=GPT_CONFIG_124M["context_length"],
      drop_last=False,
      shuffle=False,
      num_workers=0
  )

  train_tokens = 0
  for input_batch, target_batch in train_loader:
      train_tokens += input_batch.numel()

  val_tokens = 0
  for input_batch, target_batch in val_loader:
      val_tokens += input_batch.numel()

  print("Training tokens:", train_tokens)
  print("Validation tokens:", val_tokens)
  print("All tokens:", train_tokens + val_tokens)


  ######### Initializing Model and training ###########

  model = GPTModel(GPT_CONFIG_124M)
  model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

  print("Training started...")
  start_time = time.time()

  num_epochs = 10
  train_losses, val_losses, tokens_seen = train_model_simple(
      model, train_loader, val_loader, optimizer, device,
      num_epochs=num_epochs, eval_freq=5, eval_iter=5,
      start_context="Every effort moves you", tokenizer=tokenizer
  )

  end_time = time.time()
  execution_time_minutes = (end_time - start_time) / 60
  print(f"Training completed in {execution_time_minutes:.2f} minutes.")

  epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
  plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

  #torch.save({
  #    "model_state_dict": model.state_dict(),
  #    "optimizer_state_dict": optimizer.state_dict(),
  #    },
  #    "model_and_optimizer.pth"
  #)
  torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
  main()
  