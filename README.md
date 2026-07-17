# Poker44 Miner v2

Bot-detection miner for [Poker44](https://poker44.net) — Bittensor subnet 126.

The validator sends `DetectionSynapse(chunks=...)`; each chunk is a run of poker
hands from one player. The miner returns one bot-risk score in `[0, 1]` per
chunk, from betting behaviour alone — no cards, board, or identities.

## Model

A supervised classifier over behavioural features derived from betting actions.
Trained only on the public Poker44 benchmark.

## Layout

```
neurons/miner.py   entrypoint
detector/          model — features, inference, weights
poker44/           vendored subnet package (miner subset)
scripts/           run + pm2 config
```

## Run

```bash
cp .env.example .env        # set wallet, hotkey, port, repo url
pip install -e .
pm2 start scripts/ecosystem.config.js
```

Trained only on the public Poker44 benchmark; features use miner-visible
behaviour only.

## License

MIT
