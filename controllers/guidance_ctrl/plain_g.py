from guidance import select, gen
import guidance.models.transformers
import transformers
import torch


def main():
    torch.set_default_device("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/Orca-2-13b", revision="pr/23", use_fast=True
    )
    llama = guidance.models.transformers.Llama(
        model="microsoft/Orca-2-13b", revision="pr/22", tokenizer=tokenizer
    )
    grm = (
        "Here's a " + select(["joke", "poem"]) + " about cats: " + gen(regex=r"[A-Z]+")
    )
    print(llama + grm)


main()
