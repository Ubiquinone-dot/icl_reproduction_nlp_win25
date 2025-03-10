# Downloads llama models

from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os, sys, glob, json

from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_TOKEN"])


MODEL_SAVE_DIR = Path("/net/scratch/jbutch/nlp/models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_and_save(model_name, save_ext):

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    print("Saving model...", model_name)
    model.save_pretrained(str(MODEL_SAVE_DIR / save_ext))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR / save_ext))
    print("Done!")
    print("Saving LLAMA-13B model...")

## LLAMA
load_and_save("meta-llama/Llama-2-7b", "huggingface/7B")
load_and_save("meta-llama/Llama-2-13b", "huggingface/13")
load_and_save("meta-llama/Llama-2-30b", "huggingface/30")


# load_and_save("meta-llama/Llama-2-7b")
# load_and_save("meta-llama/Llama-2-13b")


# "7B": llama_local_path("huggingface", "7B"),
# "13B": llama_local_path("huggingface", "13B"),
# "30B": llama_local_path("huggingface", "30B"),
# "65B": llama_local_path("huggingface", "65B"),

print("Use export LLAMA_DIR={}".format(MODEL_SAVE_DIR))