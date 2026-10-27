# 28 Days of AI Engineering

A four-week, build-everything-yourself course I'm running on myself: from writing a GPT by hand, to evaluating LLM apps, to working with coding agents, to designing systems that scale. About an hour a day, every exercise committed here.

**Started:** 2026-10-26 · **Hardware:** MacBook (Apple Silicon, no discrete GPU) · **Daily log:** [LOG.md](LOG.md)

## Progress

<!-- progress:start -->
| Week | Theme | Done |
|---|---|---|
| 1 | [LLM foundations](week-01-llm-foundations) | ░░░░░░░░░░ 0/6 |
| 2 | [Building with models and agents](week-02-models-and-agents) | ░░░░░░░░░░ 0/6 |
| 3 | [Working with coding agents](week-03-claude-workflow) | ░░░░░░░░░░ 0/5 |
| 4 | [System design and backend](week-04-system-design) | ░░░░░░░░░░ 0/6 |
| | **Total** | **0/23** |
<!-- progress:end -->

The table updates itself from the checklists in each week's README (`scripts/progress.py`, run by GitHub Actions on every push).

## The rules

1. Code along, don't just watch. Every lecture or chapter ends with something runnable in this repo.
2. Log every session in [LOG.md](LOG.md): what I did, how long, one thing I learned.
3. Write it in my own words. Notes explain things as if to a colleague, not copied from the source.
4. Missed a day? Don't double up. Pick up where I left off.

## Curriculum

| Week | What I build | Main sources |
|---|---|---|
| 1 | micrograd, a character-level GPT, a BPE tokenizer and data loader, multi-head attention | Karpathy's *Zero to Hero*, Raschka's *LLMs from Scratch* ch. 1–3 |
| 2 | A GPT-2-sized model, a small pretraining run, best-of-N sampling with a verifier, a RAG eval harness | *LLMs from Scratch* ch. 4–5, Stanford CS329A, Chip Huyen's *AI Engineering* |
| 3 | A CLAUDE.md that learns from mistakes, a prompt A/B experiment, a token-cost measurement | Boris Cherny's and Felix Rieseberg's workflow advice, Matt Pocock's skills |
| 4 | A rate limiter, a URL shortener service, three design docs | *System Design Interview* notes, Hussein Nasser, engineering postmortems |

Each week folder has its own README with goals, resources, deliverables and notes.

## Setup

```bash
git clone https://github.com/innovatewithfaizan/28-days-ai-engineering
cd 28-days-ai-engineering
uv sync            # or: python -m venv .venv && pip install -e .
uv run jupyter lab
```

PyTorch uses Apple's `mps` backend automatically where the code supports it. Anything that needs real GPUs (full nanochat training) is run at toy scale and noted as such.

## Resources

Full list with links: [RESOURCES.md](RESOURCES.md).

## License

Code is MIT. Notes are my own; linked material belongs to its authors.
