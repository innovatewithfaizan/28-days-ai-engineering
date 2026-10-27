# Week 1: LLM foundations

**Goal:** understand how a language model is built, end to end, by writing one.

## Resources
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (lectures 1 and 7) · [code](https://github.com/karpathy/nn-zero-to-hero)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch), chapters 1–3
- [nanochat](https://github.com/karpathy/nanochat), to read, not train

## Deliverables
- [ ] `01-micrograd/` — autograd engine and a tiny MLP, written from scratch (Zero to Hero lecture 1)
- [ ] `02-gpt-from-scratch/` — character-level GPT trained on Tiny Shakespeare on `mps` (lecture 7)
- [ ] `03-tokenizer-and-dataloader/` — BPE tokenization with `tiktoken` and a sliding-window data loader (ch. 2)
- [ ] `04-attention/` — self-attention, causal masking, multi-head attention (ch. 3)
- [ ] `notes/attention-in-ste.md` — attention explained in my own words, in Simplified Technical English
- [ ] `notes/nanochat-walkthrough.md` — what each stage of nanochat does (tokenizer, pretrain, midtrain, SFT), from reading the code

## Notes
What surprised me, what I'd explain differently, open questions.
