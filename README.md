# Pretraining of GPT2
This is a llm pretraining of GPT2 architecture on a small dataset from scratch. A re-implementation of pretraining done in [LLM-from-sctach](https://github.com/rasbt/LLMs-from-scratch). Trained model can generate text for a given token size.

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
- Download pretrained model.pth from my huggingfaceHub and place it on cloned `llm-pretraining` path.
- 





