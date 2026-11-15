# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/mihawnk-nft/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/mihawnk-nft/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

The best live free-model aggregator: one server that hosts an OpenAI-compatible
and an Anthropic-compatible API and dials the highest-quality free OpenRouter
models it can reach, with a `--list` of the catalog it serves.

## Run it

```sh
node main.ts --list
```

prints the catalog document the server registers, byte for byte, and exits 0.
With a trial env, the server itself runs as:

```sh
MAKEFUN_ENV_URL=<env url> MAKEFUN_PROMPT_FILE=<brief path> node main.ts
```

It listens on `0.0.0.0` at a free port, registers with the env, serves
`GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`, polls
`GET /work`, and exits 0 when the workload is done. Every prompt is answered
with a real completion from a free model, dialled through the grader's gateway
on Chat Completions; an Anthropic-shaped request is translated there and back.

## What it does

For each prompt it ranks every free model that meets the request's
requirements by the brief's published rating, best first, and spreads the load
over the models with the fewest calls so far. A refusal is counted and the
request moves on to the next-best model; a model that refuses three calls in
a row is treated as spent for the trial. The refusals a served answer cost
are reported on it as `x-makefun-upstream-failures`. Nothing here reads the
wall clock or an unseeded random source, so a live pass and its replay make
the same calls.

## Key

The models are OpenRouter's free models. The gateway swaps the trial's dummy
key for the entrant's own `OPENROUTER_API_KEY`, pushed to the competition as
an entry secret — no key ever lives in this repository.
