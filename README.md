# SIH 25050 (MVP)

> Multi-Agent Traffic Management with Transformers and SUMO

## Overview

This project implements a **multi-agent reinforcement learning (RL) system** for controlling traffic lights in urban networks using **Proximal Policy Optimization (PPO)** with **Transformer-based policy and value networks**. The system interfaces with **SUMO (Simulation of Urban Mobility)** via **TraCI** to learn optimal traffic signal strategies based on real-time traffic states.

The goal is to **minimize congestion and waiting times** across multiple junctions while handling dynamic and stochastic traffic flows.

---

## Features

- **Multi-Agent Transformer Architecture**
  - Parameter sharing across junctions.
  - Positional encoding to capture junction order.
  - Separate policy and value heads for PPO.
  
- **PPO-Based Training**
  - Clipped surrogate loss with advantage normalization.
  - Generalized Advantage Estimation (GAE).
  - Entropy bonus for exploration.
  
- **SUMO Integration**
  - Realistic traffic simulation.
  - Traffic light control via TraCI.
  - State representation: waiting times, queue lengths, signal phases, mean speed, vehicle counts.
  - Rewards tied to average waiting time per junction.
  
- **Visualization**
  - Plot training progress (episode rewards and lengths).
  - Test trained agent with deterministic actions.

---

## Project Structure
```
sw-SIH25050-MVP/
├── data/
│   ├── maps/              # OSM map files (.osm)
│   ├── nets/              # Generated SUMO network files (.net.xml)
│   ├── trips/             # Generated trips files (.trips.xml)
│   ├── routes/            # Generated routes files (.rou.xml)
│   └── sumocfgs/          # SUMO configuration files (.sumocfg)
├── scripts/
│   └── osmToSumo.sh       # Script to convert OSM → SUMO network and generate routes
├── sumo_env.py            # SUMO-based environment class using TraCI
├── transformer_agent.py   # Transformer-based PPO agent
├── ppo_trainer.py         # PPO training loop
├── run_sumo.py            # Main script to train/test the agent
└── README.md
```
