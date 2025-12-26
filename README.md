# Ticket to Ride RL

A reinforcement learning system to discover optimal strategies for playing Ticket to Ride (USA map).

## Overview

This project uses Proximal Policy Optimization (PPO) with self-play to train AI agents that compete against each other in Ticket to Ride. The system explores three key research questions:

1. **Destination Card Importance** - How many destination cards should you keep?
2. **Hand Tracking Value** - Does tracking opponent hands improve win rates?
3. **Timing Strategies** - When should you build routes vs gather cards?

## Features

- Complete Ticket to Ride game engine (USA map, 36 cities, 100 routes)
- 5 distinct AI archetypes with different play styles
- PPO implementation with Actor-Critic architecture
- Bayesian hand tracking for opponent modeling
- Parallel game simulation (scales to 30+ CPU cores)
- Strategy evolution tracking over training
- Rich CLI output with progress visualization

## AI Archetypes

| Archetype | Strategy |
|-----------|----------|
| **Architect** | Focuses on long routes and completing destination tickets |
| **Instant Gratification** | Claims routes aggressively, prioritizes immediate points |
| **Hoarder** | Accumulates large hand before claiming routes |
| **Blocker** | Watches opponents and blocks strategic routes |
| **Wildcard** | Prioritizes collecting wild cards for flexibility |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ticket-to-ride-claude.git
cd ticket-to-ride-claude

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy >= 1.21.0
- torch >= 1.9.0
- tqdm >= 4.60.0
- rich >= 13.0.0
- matplotlib >= 3.4.0 (optional, for visualizations)

## Usage

### Quick Validation Test

Verify the system works correctly:

```bash
python ticket_to_ride_rl/main.py --quick-test
```

### Training

**Sequential training (single core):**
```bash
python ticket_to_ride_rl/main.py --games 50000 --output results/
```

**Parallel training (recommended):**
```bash
# Auto-detect CPU cores
python ticket_to_ride_rl/main.py --games 50000 --parallel --output results/

# Specify worker count
python ticket_to_ride_rl/main.py --games 50000 --parallel --workers 16
```

### Analyzing Results

```bash
python ticket_to_ride_rl/main.py --analyze results/ttr_parallel
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--games` | Number of games to train | 50000 |
| `--players` | Players per game (2-5) | 4 |
| `--output` | Output directory | results |
| `--parallel` | Enable parallel simulation | False |
| `--workers` | CPU workers for parallel mode | auto |
| `--analyze` | Path to results to analyze | - |
| `--no-plots` | Skip generating plots | False |
| `--quick-test` | Run validation test | False |

## Output Files

Training produces these files in the output directory:

```
results/ttr_parallel/
├── final_results.json      # Win rates, tracking comparison, config
├── current_stats.json      # Latest statistics
├── win_rate_history.json   # Win rates over training
├── checkpoints/            # Model checkpoints
└── strategy_snapshots/     # Strategy evolution data
```

## Performance

| CPU Cores | Games/Second | 50k Games |
|-----------|--------------|-----------|
| 1 (sequential) | ~1.5 | ~9 hours |
| 8 cores | ~10 | ~1.4 hours |
| 16 cores | ~20 | ~42 min |
| 32 cores | ~35 | ~24 min |

## Example Results

After training on 1,000 games:

**Win Rates by Archetype:**
| Archetype | Win Rate |
|-----------|----------|
| Hoarder | 35.5% |
| Instant Gratification | 30.7% |
| Blocker | 26.3% |
| Architect | 16.0% |
| Wildcard | 15.7% |

**Tracking vs Blind Performance:**
| Archetype | Blind | Tracking | Advantage |
|-----------|-------|----------|-----------|
| Architect | 14.7% | 17.2% | +2.6% |
| Instant Gratification | 29.8% | 31.5% | +1.7% |
| Hoarder | 35.5% | 35.6% | +0.2% |
| Blocker | 27.8% | 24.9% | -2.9% |
| Wildcard | 15.8% | 15.7% | -0.0% |

## Project Structure

```
ticket_to_ride_rl/
├── main.py                 # CLI entry point
├── game/                   # Game engine
│   ├── board.py           # USA map, routes, cities
│   ├── cards.py           # Resource and destination cards
│   ├── game_state.py      # Game logic and rules
│   └── player.py          # Player state management
├── agents/                 # AI agents
│   ├── archetypes.py      # 5 archetype configurations
│   ├── networks.py        # Actor-Critic neural networks
│   ├── ppo_agent.py       # PPO implementation
│   └── hand_tracker.py    # Bayesian opponent modeling
├── training/               # Training infrastructure
│   ├── trainer.py         # Sequential trainer
│   ├── parallel.py        # Parallel game runner
│   ├── parallel_trainer.py # Parallel training loop
│   ├── self_play.py       # Self-play game management
│   └── metrics.py         # Metrics collection
└── analysis/               # Results analysis
    ├── analyze.py         # Statistical analysis
    └── visualize.py       # Plot generation
```

## License

MIT License
