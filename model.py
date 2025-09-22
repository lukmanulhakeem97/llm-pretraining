{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/lukmanulhakeem97/LLM-Pretraining/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qqLsL40Wjffb",
        "outputId": "b71756bc-bcb0-44b9-f3f5-9d6253bafacf"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/llm-from-scratch\n"
          ]
        }
      ],
      "source": [
        "%cd /content/drive/MyDrive/llm-from-scratch"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YyEnquxwj5ye",
        "outputId": "3fb5d4ac-452a-487c-c271-f05b1e53d5c7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/llm-from-scratch/work\n"
          ]
        }
      ],
      "source": [
        "%cd work/"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!python train.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TflkPYHnf3DF",
        "outputId": "7c600e58-cefe-4843-96ac-6fb10eb3d5d0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Data Loaded...\n",
            "Total Characters: 20479\n",
            "Total Tokens: 5145\n",
            "Training tokens: 4608\n",
            "Validation tokens: 512\n",
            "All tokens: 5120\n",
            "Training started...\n",
            "Ep 1 (Step 000000): Train loss 9.683, Val loss 10.049\n",
            "Ep 1 (Step 000005): Train loss 8.122, Val loss 8.253\n",
            "Every effort moves you,, the, the,, the, the,, the, the, the, the,,,, the, the, the, the, the, the,,,, the, the, the, the,,,, the\n",
            "Ep 2 (Step 000010): Train loss 6.717, Val loss 7.042\n",
            "Ep 2 (Step 000015): Train loss 6.093, Val loss 6.597\n",
            "Every effort moves you of the of the of the\" of the of the\" of the of the the\" of the\" of the\" of the\"\" of the\"\" the\" of the of the of the\" of the\" of the\" of the\"\n",
            "Ep 3 (Step 000020): Train loss 5.697, Val loss 6.509\n",
            "Ep 3 (Step 000025): Train loss 5.561, Val loss 6.475\n",
            "Every effort moves you of the--and of theisburn, I had Gisburn, I had I had Gisburn, and, and I had a I had I had I had the-- to the I had the--, the I had I had the\n",
            "Ep 4 (Step 000030): Train loss 5.215, Val loss 6.432\n",
            "Ep 4 (Step 000035): Train loss 4.758, Val loss 6.388\n",
            "Every effort moves you of the of the a of the--I of the fact of the--I. Gisburn's of the \" of the--I--I--I that he had a of the--the-- of the--I of the I had the\n",
            "Ep 5 (Step 000040): Train loss 4.466, Val loss 6.279\n",
            "Every effort moves you I felt to have a little to have to have the fact a little. \"Oh, and I had a little.   \"--and of the house.\" \"--and he had been to me, and he had been his\n",
            "Ep 6 (Step 000045): Train loss 3.873, Val loss 6.223\n",
            "Ep 6 (Step 000050): Train loss 3.472, Val loss 6.176\n",
            "Every effort moves you of Jack's--I must to have to have been--as, and in the house.\" \"Oh, and I had been to me--as, the picture, and up. Gisburn's modesty, and in the picture, and\n",
            "Ep 7 (Step 000055): Train loss 3.209, Val loss 6.222\n",
            "Ep 7 (Step 000060): Train loss 2.755, Val loss 6.196\n",
            "Every effort moves you know.\" \"--and, I was not the fact with a one of the house.\" \"Oh, and I had made him, and he had the same quality as his pictures--the quality of the fact--because he was, I\n",
            "Ep 8 (Step 000065): Train loss 2.384, Val loss 6.188\n",
            "Ep 8 (Step 000070): Train loss 1.879, Val loss 6.184\n",
            "Every effort moves you know,\" was not that, and he was the the fact with equanimity. \"--had, in the women had made him, and he had the same quality as his pictures--the quality of the end of the room, I had\n",
            "Ep 9 (Step 000075): Train loss 1.483, Val loss 6.276\n",
            "Ep 9 (Step 000080): Train loss 1.225, Val loss 6.306\n",
            "Every effort moves you?\" \"I.\" \"--I felt nervous and a single one in the house.\" \"Oh, and I felt to see a degree he had the same quality as his pictures--the quality of his ease--because he was, and\n",
            "Ep 10 (Step 000085): Train loss 0.900, Val loss 6.307\n",
            "Every effort moves you?\"  \"--forming, and it were, and uncertain. Stroud so when she began to me!\"  \"Oh, a smile behind his close grayish beard--as if he had the donkey. \"There were days when I\n",
            "Training completed in 23.78 minutes.\n",
            "Figure(500x300)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!python inference.py \"what is the real motivation\" --load-openaigpt2-weight=True"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HA_z2V1txooG",
        "outputId": "f1696e63-28f9-472e-c0d9-3ff868a20bec"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2025-09-22 04:59:52.387616: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "E0000 00:00:1758517192.420860    4701 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "E0000 00:00:1758517192.433175    4701 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "W0000 00:00:1758517192.458084    4701 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1758517192.458126    4701 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1758517192.458132    4701 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1758517192.458136    4701 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "2025-09-22 04:59:52.463839: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "Inferencing on OpenaAI GPT2 pretrained weights.\n",
            "File already exists and is up-to-date: gpt2/124M/checkpoint\n",
            "File already exists and is up-to-date: gpt2/124M/encoder.json\n",
            "File already exists and is up-to-date: gpt2/124M/hparams.json\n",
            "File already exists and is up-to-date: gpt2/124M/model.ckpt.data-00000-of-00001\n",
            "File already exists and is up-to-date: gpt2/124M/model.ckpt.index\n",
            "File already exists and is up-to-date: gpt2/124M/model.ckpt.meta\n",
            "File already exists and is up-to-date: gpt2/124M/vocab.bpe\n",
            "Output text:\n",
            " what is the real motivation behind the \"lifestyle movement?\" A real goal has to be the elimination of waste, create value and ensure one does not\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!python inference.py \"what is the real motivation\" --load_openaigpt2_weight \"yes\""
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yb84V5i6xodt",
        "outputId": "c9a80a64-833e-4a49-cd9a-ecd0286b75d5"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Inferencing on OpenaAI GPT2 pretrained weights.\n",
            "2025-09-22 05:30:19.853930: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "E0000 00:00:1758519019.898926   12165 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "E0000 00:00:1758519019.908616   12165 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "W0000 00:00:1758519019.929988   12165 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1758519019.930033   12165 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1758519019.930037   12165 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "W0000 00:00:1758519019.930044   12165 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
            "2025-09-22 05:30:19.936629: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "File already exists and is up-to-date: gpt2/124M/checkpoint\n",
            "File already exists and is up-to-date: gpt2/124M/encoder.json\n",
            "File already exists and is up-to-date: gpt2/124M/hparams.json\n",
            "File already exists and is up-to-date: gpt2/124M/model.ckpt.data-00000-of-00001\n",
            "File already exists and is up-to-date: gpt2/124M/model.ckpt.index\n",
            "File already exists and is up-to-date: gpt2/124M/model.ckpt.meta\n",
            "File already exists and is up-to-date: gpt2/124M/vocab.bpe\n",
            "Output text:\n",
            " what is the real motivation for a new government that has this level of ambition, and with that ambition comes a deep dissatisfaction over the level of resources,\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1f_bDV5GG8lw"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import time\n",
        "import urllib.request\n",
        "import tiktoken\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import Dataset, DataLoader"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TPmISaf0G6M6"
      },
      "outputs": [],
      "source": [
        "GPT_CONFIG_124M = {\n",
        "    \"vocab_size\": 50257,   # Vocabulary size\n",
        "    \"context_length\": 256, # Shortened context length (orig: 1024)\n",
        "    \"emb_dim\": 768,        # Embedding dimension\n",
        "    \"n_heads\": 12,         # Number of attention heads\n",
        "    \"n_layers\": 12,        # Number of layers\n",
        "    \"drop_rate\": 0.1,      # Dropout rate\n",
        "    \"qkv_bias\": False      # Query-key-value bias\n",
        "}"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QpQgldAKCz2C"
      },
      "source": [
        "# 1. Loading Preparation\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pXwK3vz2_i3s",
        "outputId": "2fdb404c-fe44-4e6c-fa31-6d8379988ed4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loading Data...\n"
          ]
        }
      ],
      "source": [
        "file_path = \"./data/the-verdict.txt\"\n",
        "url = \"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt\"\n",
        "\n",
        "print(\"Loading Data...\")\n",
        "if not os.path.exists(file_path):\n",
        "    with urllib.request.urlopen(url) as response:\n",
        "        text_data = response.read().decode('utf-8')\n",
        "    with open(file_path, \"w\", encoding=\"utf-8\") as file:\n",
        "        file.write(text_data)\n",
        "else:\n",
        "    with open(file_path, \"r\", encoding=\"utf-8\") as file:\n",
        "        text_data = file.read()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9JEBWzPW_6ox",
        "outputId": "7544cd81-d27d-4e3b-d50c-7524b004fb2a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total Characters: 20479\n",
            "Total Tokens: 5145\n"
          ]
        }
      ],
      "source": [
        "tokenizer = tiktoken.get_encoding(\"gpt2\")\n",
        "\n",
        "total_characters = len(text_data)\n",
        "total_tokens = len(tokenizer.encode(text_data))\n",
        "\n",
        "print(\"Total Characters:\", total_characters)\n",
        "print(\"Total Tokens:\", total_tokens)\n",
        "\n",
        "# Train/validation ratio\n",
        "train_ratio = 0.90\n",
        "split_idx = int(train_ratio * len(text_data))\n",
        "train_data = text_data[:split_idx]\n",
        "val_data = text_data[split_idx:]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "stcZ16GKHN55"
      },
      "outputs": [],
      "source": [
        "class GPTDatasetV1(Dataset):\n",
        "    def __init__(self, txt, tokenizer, max_length, stride):\n",
        "        self.input_ids = []\n",
        "        self.target_ids = []\n",
        "\n",
        "        # Tokenize the entire text\n",
        "        token_ids = tokenizer.encode(txt, allowed_special={\"<|endoftext|>\"})\n",
        "\n",
        "        # Use a sliding window to chunk the book into overlapping sequences of max_length\n",
        "        for i in range(0, len(token_ids) - max_length, stride):\n",
        "            input_chunk = token_ids[i:i + max_length]\n",
        "            target_chunk = token_ids[i + 1: i + max_length + 1]\n",
        "            self.input_ids.append(torch.tensor(input_chunk))\n",
        "            self.target_ids.append(torch.tensor(target_chunk))\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.input_ids)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        return self.input_ids[idx], self.target_ids[idx]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PvqDFU-gHF-i"
      },
      "outputs": [],
      "source": [
        "def create_dataloader_v1(txt, batch_size=4, max_length=256,\n",
        "                         stride=128, shuffle=True, drop_last=True, num_workers=0):\n",
        "    # Initialize the tokenizer\n",
        "    tokenizer = tiktoken.get_encoding(\"gpt2\")\n",
        "\n",
        "    # Create dataset\n",
        "    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)\n",
        "\n",
        "    # Create dataloader\n",
        "    dataloader = DataLoader(\n",
        "        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)\n",
        "\n",
        "    return dataloader"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "X5LSvQ-JA4Su"
      },
      "outputs": [],
      "source": [
        "train_loader = create_dataloader_v1(\n",
        "    train_data,\n",
        "    batch_size=2,\n",
        "    max_length=GPT_CONFIG_124M[\"context_length\"],\n",
        "    stride=GPT_CONFIG_124M[\"context_length\"],\n",
        "    drop_last=True,\n",
        "    shuffle=True,\n",
        "    num_workers=0\n",
        ")\n",
        "\n",
        "val_loader = create_dataloader_v1(\n",
        "    val_data,\n",
        "    batch_size=2,\n",
        "    max_length=GPT_CONFIG_124M[\"context_length\"],\n",
        "    stride=GPT_CONFIG_124M[\"context_length\"],\n",
        "    drop_last=False,\n",
        "    shuffle=False,\n",
        "    num_workers=0\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AmkkPp7oBGlr",
        "outputId": "ae9239a0-603b-421e-91c4-e02d75d64680"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training tokens: 4608\n",
            "Validation tokens: 512\n",
            "All tokens: 5120\n"
          ]
        }
      ],
      "source": [
        "train_tokens = 0\n",
        "for input_batch, target_batch in train_loader:\n",
        "    train_tokens += input_batch.numel()\n",
        "\n",
        "val_tokens = 0\n",
        "for input_batch, target_batch in val_loader:\n",
        "    val_tokens += input_batch.numel()\n",
        "\n",
        "print(\"Training tokens:\", train_tokens)\n",
        "print(\"Validation tokens:\", val_tokens)\n",
        "print(\"All tokens:\", train_tokens + val_tokens)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5XXWpQwpDVXn"
      },
      "source": [
        "# 2. Defining Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kXN0d5fKELqM"
      },
      "outputs": [],
      "source": [
        "class MultiHeadAttention(nn.Module):\n",
        "    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):\n",
        "        super().__init__()\n",
        "        assert d_out % num_heads == 0, \"d_out must be divisible by n_heads\"\n",
        "\n",
        "        self.d_out = d_out\n",
        "        self.num_heads = num_heads\n",
        "        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim\n",
        "\n",
        "        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)\n",
        "        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)\n",
        "        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)\n",
        "        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs\n",
        "        self.dropout = nn.Dropout(dropout)\n",
        "        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))\n",
        "\n",
        "    def forward(self, x):\n",
        "        b, num_tokens, d_in = x.shape\n",
        "\n",
        "        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)\n",
        "        queries = self.W_query(x)\n",
        "        values = self.W_value(x)\n",
        "\n",
        "        # We implicitly split the matrix by adding a `num_heads` dimension\n",
        "        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)\n",
        "        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)\n",
        "        values = values.view(b, num_tokens, self.num_heads, self.head_dim)\n",
        "        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)\n",
        "\n",
        "        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)\n",
        "        keys = keys.transpose(1, 2)\n",
        "        queries = queries.transpose(1, 2)\n",
        "        values = values.transpose(1, 2)\n",
        "\n",
        "        # Compute scaled dot-product attention (aka self-attention) with a causal mask\n",
        "        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head\n",
        "\n",
        "        # Original mask truncated to the number of tokens and converted to boolean\n",
        "        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]\n",
        "\n",
        "        # Use the mask to fill attention scores\n",
        "        attn_scores.masked_fill_(mask_bool, -torch.inf)\n",
        "\n",
        "        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)\n",
        "        attn_weights = self.dropout(attn_weights)\n",
        "\n",
        "        # Shape: (b, num_tokens, num_heads, head_dim)\n",
        "        context_vec = (attn_weights @ values).transpose(1, 2)\n",
        "\n",
        "        # Combine heads, where self.d_out = self.num_heads * self.head_dim\n",
        "        context_vec = context_vec.reshape(b, num_tokens, self.d_out)\n",
        "        context_vec = self.out_proj(context_vec)  # optional projection\n",
        "\n",
        "        return context_vec"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pOgYhP2kECVe"
      },
      "outputs": [],
      "source": [
        "class LayerNorm(nn.Module):\n",
        "    def __init__(self, emb_dim):\n",
        "        super().__init__()\n",
        "        self.eps = 1e-5\n",
        "        self.scale = nn.Parameter(torch.ones(emb_dim))\n",
        "        self.shift = nn.Parameter(torch.zeros(emb_dim))\n",
        "\n",
        "    def forward(self, x):\n",
        "        mean = x.mean(dim=-1, keepdim=True)\n",
        "        var = x.var(dim=-1, keepdim=True, unbiased=False)\n",
        "        norm_x = (x - mean) / torch.sqrt(var + self.eps)\n",
        "        return self.scale * norm_x + self.shift"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aWjJc-TJD_gE"
      },
      "outputs": [],
      "source": [
        "class GELU(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "    def forward(self, x):\n",
        "        return 0.5 * x * (1 + torch.tanh(\n",
        "            torch.sqrt(torch.tensor(2.0 / torch.pi)) *\n",
        "            (x + 0.044715 * torch.pow(x, 3))\n",
        "        ))\n",
        "\n",
        "\n",
        "class FeedForward(nn.Module):\n",
        "    def __init__(self, cfg):\n",
        "        super().__init__()\n",
        "        self.layers = nn.Sequential(\n",
        "            nn.Linear(cfg[\"emb_dim\"], 4 * cfg[\"emb_dim\"]),\n",
        "            GELU(),\n",
        "            nn.Linear(4 * cfg[\"emb_dim\"], cfg[\"emb_dim\"]),\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.layers(x)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "11XLWUHmD7oq"
      },
      "outputs": [],
      "source": [
        "class TransformerBlock(nn.Module):\n",
        "    def __init__(self, cfg):\n",
        "        super().__init__()\n",
        "        self.att = MultiHeadAttention(\n",
        "            d_in=cfg[\"emb_dim\"],\n",
        "            d_out=cfg[\"emb_dim\"],\n",
        "            context_length=cfg[\"context_length\"],\n",
        "            num_heads=cfg[\"n_heads\"],\n",
        "            dropout=cfg[\"drop_rate\"],\n",
        "            qkv_bias=cfg[\"qkv_bias\"])\n",
        "        self.ff = FeedForward(cfg)\n",
        "        self.norm1 = LayerNorm(cfg[\"emb_dim\"])\n",
        "        self.norm2 = LayerNorm(cfg[\"emb_dim\"])\n",
        "        self.drop_shortcut = nn.Dropout(cfg[\"drop_rate\"])\n",
        "\n",
        "    def forward(self, x):\n",
        "        # Shortcut connection for attention block\n",
        "        shortcut = x\n",
        "        x = self.norm1(x)\n",
        "        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]\n",
        "        x = self.drop_shortcut(x)\n",
        "        x = x + shortcut  # Add the original input back\n",
        "\n",
        "        # Shortcut connection for feed-forward block\n",
        "        shortcut = x\n",
        "        x = self.norm2(x)\n",
        "        x = self.ff(x)\n",
        "        x = self.drop_shortcut(x)\n",
        "        x = x + shortcut  # Add the original input back\n",
        "\n",
        "        return x"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ECNPOekyD2go"
      },
      "outputs": [],
      "source": [
        "class GPTModel(nn.Module):\n",
        "    def __init__(self, cfg):\n",
        "        super().__init__()\n",
        "        self.tok_emb = nn.Embedding(cfg[\"vocab_size\"], cfg[\"emb_dim\"])\n",
        "        self.pos_emb = nn.Embedding(cfg[\"context_length\"], cfg[\"emb_dim\"])\n",
        "        self.drop_emb = nn.Dropout(cfg[\"drop_rate\"])\n",
        "\n",
        "        self.trf_blocks = nn.Sequential(\n",
        "            *[TransformerBlock(cfg) for _ in range(cfg[\"n_layers\"])])\n",
        "\n",
        "        self.final_norm = LayerNorm(cfg[\"emb_dim\"])\n",
        "        self.out_head = nn.Linear(cfg[\"emb_dim\"], cfg[\"vocab_size\"], bias=False)\n",
        "\n",
        "    def forward(self, in_idx):\n",
        "        batch_size, seq_len = in_idx.shape\n",
        "        tok_embeds = self.tok_emb(in_idx)\n",
        "        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))\n",
        "        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]\n",
        "        x = self.drop_emb(x)\n",
        "        x = self.trf_blocks(x)\n",
        "        x = self.final_norm(x)\n",
        "        logits = self.out_head(x)\n",
        "        return logits"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dPpNyNFHBG70"
      },
      "outputs": [],
      "source": [
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "model = GPTModel(GPT_CONFIG_124M)\n",
        "model.to(device)\n",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xTPsRtBzERtZ"
      },
      "source": [
        "# 3. Training"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7NQoND5RFlJJ"
      },
      "outputs": [],
      "source": [
        "def generate_text_simple(model, idx, max_new_tokens, context_size):\n",
        "    # idx is (B, T) array of indices in the current context\n",
        "    for _ in range(max_new_tokens):\n",
        "\n",
        "        # Crop current context if it exceeds the supported context size\n",
        "        # E.g., if LLM supports only 5 tokens, and the context size is 10\n",
        "        # then only the last 5 tokens are used as context\n",
        "        idx_cond = idx[:, -context_size:]\n",
        "\n",
        "        # Get the predictions\n",
        "        with torch.no_grad():\n",
        "            logits = model(idx_cond)\n",
        "\n",
        "        # Focus only on the last time step\n",
        "        # (batch, n_token, vocab_size) becomes (batch, vocab_size)\n",
        "        logits = logits[:, -1, :]\n",
        "\n",
        "        # Get the idx of the vocab entry with the highest logits value\n",
        "        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)\n",
        "\n",
        "        # Append sampled index to the running sequence\n",
        "        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)\n",
        "\n",
        "    return idx"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Qfmj8V0KF0tN"
      },
      "outputs": [],
      "source": [
        "def text_to_token_ids(text, tokenizer):\n",
        "    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})\n",
        "    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension\n",
        "    return encoded_tensor\n",
        "\n",
        "def token_ids_to_text(token_ids, tokenizer):\n",
        "    flat = token_ids.squeeze(0) # remove batch dimension\n",
        "    return tokenizer.decode(flat.tolist())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2gQ09QjXEx_w"
      },
      "outputs": [],
      "source": [
        "def calc_loss_batch(input_batch, target_batch, model, device):\n",
        "    input_batch, target_batch = input_batch.to(device), target_batch.to(device)\n",
        "    logits = model(input_batch)\n",
        "    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())\n",
        "    return loss\n",
        "\n",
        "def calc_loss_loader(data_loader, model, device, num_batches=None):\n",
        "    total_loss = 0.\n",
        "    if len(data_loader) == 0:\n",
        "        return float(\"nan\")\n",
        "    elif num_batches is None:\n",
        "        num_batches = len(data_loader)\n",
        "    else:\n",
        "        # Reduce the number of batches to match the total number of batches in the data loader\n",
        "        # if num_batches exceeds the number of batches in the data loader\n",
        "        num_batches = min(num_batches, len(data_loader))\n",
        "    for i, (input_batch, target_batch) in enumerate(data_loader):\n",
        "        if i < num_batches:\n",
        "            loss = calc_loss_batch(input_batch, target_batch, model, device)\n",
        "            total_loss += loss.item()\n",
        "        else:\n",
        "            break\n",
        "    return total_loss / num_batches"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sb9X-LE3Erag"
      },
      "outputs": [],
      "source": [
        "def evaluate_model(model, train_loader, val_loader, device, eval_iter):\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)\n",
        "        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)\n",
        "    model.train()\n",
        "    return train_loss, val_loss"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hZ8ngD3PFRz1"
      },
      "outputs": [],
      "source": [
        "def generate_and_print_sample(model, tokenizer, device, start_context):\n",
        "    model.eval()\n",
        "    context_size = model.pos_emb.weight.shape[0]\n",
        "    encoded = text_to_token_ids(start_context, tokenizer).to(device)\n",
        "    with torch.no_grad():\n",
        "        token_ids = generate_text_simple(\n",
        "            model=model, idx=encoded,\n",
        "            max_new_tokens=50, context_size=context_size\n",
        "        )\n",
        "    decoded_text = token_ids_to_text(token_ids, tokenizer)\n",
        "    print(decoded_text.replace(\"\\n\", \" \"))  # Compact print format\n",
        "    model.train()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):\n",
        "\n",
        "    # For-loop is the same as before: Get logits, and only focus on last time step\n",
        "    for _ in range(max_new_tokens):\n",
        "        idx_cond = idx[:, -context_size:]\n",
        "        with torch.no_grad():\n",
        "            logits = model(idx_cond)\n",
        "        logits = logits[:, -1, :]\n",
        "\n",
        "        # New: Filter logits with top_k sampling\n",
        "        if top_k is not None:\n",
        "            # Keep only top_k values\n",
        "            top_logits, _ = torch.topk(logits, top_k)\n",
        "            min_val = top_logits[:, -1]\n",
        "            logits = torch.where(logits < min_val, torch.tensor(float(\"-inf\")).to(logits.device), logits)\n",
        "\n",
        "        # New: Apply temperature scaling\n",
        "        if temperature > 0.0:\n",
        "            logits = logits / temperature\n",
        "\n",
        "            # Apply softmax to get probabilities\n",
        "            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)\n",
        "\n",
        "            # Sample from the distribution\n",
        "            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)\n",
        "\n",
        "        # Otherwise same as before: get idx of the vocab entry with the highest logits value\n",
        "        else:\n",
        "            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)\n",
        "\n",
        "        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified\n",
        "            break\n",
        "\n",
        "        # Same as before: append sampled index to the running sequence\n",
        "        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)\n",
        "    return idx"
      ],
      "metadata": {
        "id": "uvWPbu_XUTpx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bzdZa_0zEdqB"
      },
      "outputs": [],
      "source": [
        "def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,\n",
        "                       eval_freq, eval_iter, start_context, tokenizer):\n",
        "    # Initialize lists to track losses and tokens seen\n",
        "    train_losses, val_losses, track_tokens_seen = [], [], []\n",
        "    tokens_seen, global_step = 0, -1\n",
        "\n",
        "    # Main training loop\n",
        "    for epoch in range(num_epochs):\n",
        "        model.train()  # Set model to training mode\n",
        "\n",
        "        for input_batch, target_batch in train_loader:\n",
        "            optimizer.zero_grad() # Reset loss gradients from previous batch iteration\n",
        "            loss = calc_loss_batch(input_batch, target_batch, model, device)\n",
        "            loss.backward() # Calculate loss gradients\n",
        "            optimizer.step() # Update model weights using loss gradients\n",
        "            tokens_seen += input_batch.numel()\n",
        "            global_step += 1\n",
        "\n",
        "            # Optional evaluation step\n",
        "            if global_step % eval_freq == 0:\n",
        "                train_loss, val_loss = evaluate_model(\n",
        "                    model, train_loader, val_loader, device, eval_iter)\n",
        "                train_losses.append(train_loss)\n",
        "                val_losses.append(val_loss)\n",
        "                track_tokens_seen.append(tokens_seen)\n",
        "                print(f\"Ep {epoch+1} (Step {global_step:06d}): \"\n",
        "                      f\"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}\")\n",
        "\n",
        "        # Print a sample text after each epoch\n",
        "        generate_and_print_sample(\n",
        "            model, tokenizer, device, start_context\n",
        "        )\n",
        "\n",
        "    return train_losses, val_losses, track_tokens_seen"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZZ08q3ZMEWKk",
        "outputId": "5a6899a5-2cdc-4403-c02a-c75cdab7bc37"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ep 1 (Step 000000): Train loss 9.846, Val loss 9.951\n",
            "Ep 1 (Step 000005): Train loss 8.112, Val loss 8.324\n",
            "Every effort moves you, the, the,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n",
            "Ep 2 (Step 000010): Train loss 6.663, Val loss 7.082\n",
            "Ep 2 (Step 000015): Train loss 6.133, Val loss 6.658\n",
            "Every effort moves you, and he, and he, and he, and he, and he, and he, and he, and, and, and he, and he, and.                \n",
            "Ep 3 (Step 000020): Train loss 9.089, Val loss 10.345\n",
            "Ep 3 (Step 000025): Train loss 5.526, Val loss 6.459\n",
            "Every effort moves you--                                                 \n",
            "Ep 4 (Step 000030): Train loss 4.944, Val loss 6.441\n",
            "Ep 4 (Step 000035): Train loss 4.398, Val loss 6.370\n",
            "Every effort moves you to the that to have the picture--as--I of the a of the a him, I was, in the his of the man of the his of the a of the a a his of the a of the man of the a of the\n",
            "Ep 5 (Step 000040): Train loss 3.744, Val loss 6.317\n",
            "Every effort moves you know I was not that, and I was not the fact the his pictures--I turned, and I had the fact, and I had been the end of the fact, and I had the end of the, and I had to see it.\n",
            "Ep 6 (Step 000045): Train loss 3.219, Val loss 6.183\n",
            "Ep 6 (Step 000050): Train loss 2.759, Val loss 6.216\n",
            "Every effort moves you know to the fact-century a. \"I told Mrs. \"I turned--and I had been his pictures--and I was just was his that, I had been his eyes I had the donkey, and I had been his pictures\n",
            "Ep 7 (Step 000055): Train loss 2.526, Val loss 6.227\n",
            "Ep 7 (Step 000060): Train loss 2.009, Val loss 6.278\n",
            "Every effort moves you in the inevitable gar--as, as it--I told Mrs.  \"I looked about me, in fact, the her poverty. \"--as Jack himself, as he was his own _mine_--because he had been his\n",
            "Ep 8 (Step 000065): Train loss 1.409, Val loss 6.275\n",
            "Ep 8 (Step 000070): Train loss 1.124, Val loss 6.314\n",
            "Every effort moves you in the inevitable garlanded frame.  I told me the last word.          I moved away, and I looked at the donkey again. I had the donkey.  I didn't want\n",
            "Ep 9 (Step 000075): Train loss 0.861, Val loss 6.363\n",
            "Ep 9 (Step 000080): Train loss 0.633, Val loss 6.383\n",
            "Every effort moves you?\"  \"Yes--quite insensible to the irony.   \"Once, when I looked up, I felt to see a smile behind his close grayish beard--as if he had the donkey. \"There were days when I\n",
            "Ep 10 (Step 000085): Train loss 0.472, Val loss 6.475\n",
            "Every effort moves you?\"  \"Yes--quite insensible to the irony. She wanted him vindicated--and by me!\"  I made him--it was fitting that they should mourn him. \"I turned, my eye fell on a small picture\n",
            "Training completed in 20.85 minutes.\n"
          ]
        }
      ],
      "source": [
        "start_time = time.time()\n",
        "\n",
        "num_epochs = 10\n",
        "train_losses, val_losses, tokens_seen = train_model_simple(\n",
        "    model, train_loader, val_loader, optimizer, device,\n",
        "    num_epochs=num_epochs, eval_freq=5, eval_iter=5,\n",
        "    start_context=\"Every effort moves you\", tokenizer=tokenizer\n",
        ")\n",
        "\n",
        "end_time = time.time()\n",
        "execution_time_minutes = (end_time - start_time) / 60\n",
        "print(f\"Training completed in {execution_time_minutes:.2f} minutes.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EB8OipUyHzSv",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 307
        },
        "outputId": "242f1f2a-0f0d-48f6-e249-edef99bd2c65"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 500x300 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "from matplotlib.ticker import MaxNLocator\n",
        "\n",
        "\n",
        "def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):\n",
        "    fig, ax1 = plt.subplots(figsize=(5, 3))\n",
        "\n",
        "    # Plot training and validation loss against epochs\n",
        "    ax1.plot(epochs_seen, train_losses, label=\"Training loss\")\n",
        "    ax1.plot(epochs_seen, val_losses, linestyle=\"-.\", label=\"Validation loss\")\n",
        "    ax1.set_xlabel(\"Epochs\")\n",
        "    ax1.set_ylabel(\"Loss\")\n",
        "    ax1.legend(loc=\"upper right\")\n",
        "    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis\n",
        "\n",
        "    # Create a second x-axis for tokens seen\n",
        "    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis\n",
        "    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks\n",
        "    ax2.set_xlabel(\"Tokens seen\")\n",
        "\n",
        "    fig.tight_layout()  # Adjust layout to make room\n",
        "    plt.savefig(\"loss-plot.pdf\")\n",
        "    plt.show()\n",
        "\n",
        "epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))\n",
        "plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zezW-ehIFjA-"
      },
      "outputs": [],
      "source": [
        "torch.save({\n",
        "    \"model_state_dict\": model.state_dict(),\n",
        "    \"optimizer_state_dict\": optimizer.state_dict(),\n",
        "    },\n",
        "    \"model_and_optimizer.pth\"\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 4. Inferencing"
      ],
      "metadata": {
        "id": "tjpyFy7SU3IS"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CqyI5M58IEUf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5501e59c-0c14-4636-e556-ed15a9da0ad8"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "probs: tensor([[0., 0., 0.,  ..., 0., 0., 0.]])\n",
            "tensor([[1701]])\n",
            "tensor([[760]])\n",
            "tensor([[588]])\n",
            "tensor([[287]])\n",
            "tensor([[276]])\n",
            "tensor([[1549]])\n",
            "tensor([[1701]])\n",
            "tensor([[1701]])\n",
            "tensor([[287]])\n",
            "tensor([[760]])\n",
            "Output text:\n",
            " Every effort moves you know\n"
          ]
        }
      ],
      "source": [
        "token_ids = generate(\n",
        "    model=model,\n",
        "    idx=text_to_token_ids(\"Every effort moves you\", tokenizer),\n",
        "    max_new_tokens=15,\n",
        "    context_size=GPT_CONFIG_124M[\"context_length\"],\n",
        "    top_k=25,\n",
        "    temperature=1.4\n",
        ")\n",
        "\n",
        "print(\"Output text:\\n\", token_ids_to_text(token_ids, tokenizer))"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "3s5if6EbYjPa"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1pY3pxfyqvZLrg5Ue0J_gcrdoZJTcVnfC",
      "authorship_tag": "ABX9TyNSvR+BRpcUSAmUScC3vRFZ",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}