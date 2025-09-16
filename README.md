# SIH 25050 (MVP)

Multi-Agent Traffic Management with Transformers and SUMO

> [!IMPORTANT]
> This is NOT a complete project and only meant to showcase a proposed techstack for SIH25050. It contains scripts meant to demonstrate a pipeline integrating CV & IoT information with Open Source Maps to simulate and train a multi-agentic reinforcement learning model.

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
│   └── osmToSumo.sh       # Script to convert OSM → SUMO network and 
├── model
│   ├── computer_vision/
│   ├── reinforcement_learning/
│   │   └── RL_traffic.py
│   ├── simulator/
│   │   └── sim1.py
└── README.md
```
