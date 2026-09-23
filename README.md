# PRISM — Phone-to-Phone Mesh Network Simulator

PRISM is a decentralized mesh network for smartphones: messages hop phone-to-phone
over Bluetooth / WiFi Direct with **no cell towers, routers, or internet**. Target
use cases are dead zones, disaster response, and privacy-sensitive communication.

This repository holds the **Python validation simulator**. It builds a synthetic
county (cities, towns, roads, buildings, thousands of moving phones and cars),
boots every device in random order, and measures whether real bytes actually
get delivered. A production implementation in Rust is planned once the protocol
is validated here.

## Core constraint

No node ever gets a god's-eye view. Every phone knows only the neighbors it has
recently heard over the radio, and routing must work from that local knowledge
alone. An `oracle` routing mode exists purely as a comparison baseline.

## What's simulated

- **Self-organizing mesh** — random boot order, self-generated 256-bit node IDs, HELLO
  beacons with neighbor caches and TTL expiry, adaptive beacon rates by mobility class
- **Privacy spheres** — opaque random zone labels mapped to ~50 m areas; routing
  targets a zone, never coordinates
- **Cache-based line-of-sight routing** — greedy zone forwarding with link-lifetime
  prediction, rescans, backtracking, and dead-end detection
- **Road network** — highways between towns plus branching backroads; vehicles
  drive the roads and act as relay supernodes
- **Reliable transport** — hop-by-hop store-and-forward with per-link ACKs
  (`--transport hop`) vs legacy end-to-end (`--transport e2e`)
- **Byte-level integrity** — real frames, per-hop CRC-24 (BLE-style) plus an
  end-to-end SHA-256 backstop; tracks corruption that slips each layer
- **DTN custody transfer** — when no route exists, a sealed message is handed to a
  vehicle that carries it (and can chain it to other vehicles) until a route appears
- **WiFi Direct fallback** — isolated nodes burst at extended range during windows
  synchronized to Eastern Time on every device
- **Battery model** — idle / low / medium / high usage states, movement drain,
  radio burst cost
- **Realistic traffic** — a generated corpus of messages, reports, and photo-sized
  payloads with per-class delivery telemetry
- **Concurrency & scale** — simultaneous flows with shared-medium contention;
  multiprocess Monte Carlo runs with aggregated telemetry

## Install

```bash
git clone https://github.com/<you>/prism.git
cd prism
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The payload corpus (`prism_payloads/`) is generated automatically on first run.
It is deterministic, so every machine produces the same test data.

## Usage

```bash
# Quick sanity run
python prism.py --nodes 1500 --miles 3 --tests 3

# Watch it live (keep node counts modest on laptops)
python prism.py --nodes 2000 --miles 4 --tests 5 --visualize --realtime --seed 42

# Desktop benchmark: 12 independent worlds, aggregated statistics
python prism.py --nodes 7000 --miles 10 --max_hops 150 --tests 15 --parallel 12 --seed 600
```

### Controlled experiments

Use matched `--seed` values so both runs test identical worlds:

```bash
python prism.py ... --seed 600 --transport e2e     # vs --transport hop
python prism.py ... --seed 600 --no_mules          # DTN custody off
python prism.py ... --seed 600 --route_mode oracle # god-view baseline
```

Seeded results are only comparable **within the same code version** — changes to
RNG usage (e.g. vectorization) intentionally break byte-for-byte baselines.

### Flags

| Flag | Purpose |
|---|---|
| `--nodes`, `--miles` | Population and map size |
| `--tests`, `--retries`, `--max_hops` | Test count, route attempts, TTL |
| `--seed` | Reproducible runs |
| `--parallel N` | N independent worlds across CPU cores |
| `--concurrent K` | K simultaneous flows per batch |
| `--route_mode cache\|oracle` | Local-knowledge routing vs god-view baseline |
| `--transport hop\|e2e` | Hop-by-hop ACKs vs end-to-end retransmit |
| `--traffic mix\|message\|report\|photo` | Payload selection |
| `--radio_los` | Buildings block radio signal |
| `--cache_ttl`, `--boot_window`, `--settle` | Discovery tuning |
| `--mule_ttl`, `--no_mules` | DTN custody tuning / disable |
| `--no_e2e_integrity` | Drop the SHA-256 backstop (low-power protocol modeling) |
| `--visualize`, `--realtime`, `--verbose` | Live view, 1:1 time, per-hop logs |

Run `python prism.py --help` for defaults.

## Known limitations

- Radio model is a range disc (optionally with building line-of-sight); no fading or
  partial wall attenuation.
- Simulated link rates (1–15 Mbps) are optimistic versus real BLE throughput, and iOS
  background BLE restrictions are not modeled yet.
- Battery drain constants are compressed for simulation timescales, not calibrated to
  a specific handset.

## Roadmap

- [ ] Calibrate radio and battery models against real devices
- [ ] Model iOS background BLE constraints
- [ ] Adversarial nodes / cache poisoning (signed HELLOs)
- [ ] Rust implementation of the protocol core

## License

**Source-available, not open source.** PRISM is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md).

- **Free** for personal use, research, education, and noncommercial organizations
  (charities, schools, public research, public safety, government).
- **Commercial use requires a paid license.** See [COMMERCIAL.md](COMMERCIAL.md).

Contributions require a Contributor License Agreement. See [CONTRIBUTING.md](CONTRIBUTING.md).
