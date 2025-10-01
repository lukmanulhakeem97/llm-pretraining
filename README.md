# Pretraining of GPT2
This is a llm pretraining of GPT2 architecture on a small dataset from scratch. A re-implementation of pretraining done in [LLM-from-sctach](https://github.com/rasbt/LLMs-from-scratch). Trained model can generate text for a given token size.

Here gpt2 architecture pretrained with 124M parameters on a very small short-story book text data `./data/the-verdict.txt`. Alternatively, can use OpenAI gpt2 pretrained weights mentioned in below section. Model architecture configuration is given `model_info.txt`.

# Run the code
### Setup
Pre-requisites are `python<=3.13` and `uv` package manger, instructions to set up can be found [here](https://docs.astral.sh/uv/getting-started/).
1. **Clone this repository**
   
   Either by download as zip option or by `git clone https://github.com/lukmanulhakeem97/llm-pretraining.git` command in CLI tool.
2. **Create an python environment and install dependencies**

   create environment: `uv venv [name]`, name is optional.
   
   Navigate to cloned repo directory and install dependency given in `pyproject.toml` file:
      > `cd llm-pretraining`,
      
      > `uv sync`.
4. Activate `venv` by `.\.venv\Scripts\activate`

### Generate text
- Download pretrained `model.pth` from my [huggingfaceHub]() and place it on cloned `llm-pretraining` path.
- Run `inference.py` with any starting prompt
     > by using `model.pth`: `uv run inference.py "Be now, then will be "`.

     > by using OpenAI gpt2 pretrained weights: `uv run inference.py "Be now, then will be " --load_openaigpt2_weight="yes"`.





