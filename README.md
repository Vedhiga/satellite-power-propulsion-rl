# AI-Based Intelligent Power and Propulsion Management for Satellites

⚠️ **Note:** This project presents a **conceptual simulation model** exploring how Artificial Intelligence can support decision-making in spacecraft power and propulsion management. The model is intended for academic exploration and does not represent real spacecraft dynamics.

---

## Overview

Satellites operate under strict power constraints where onboard energy is generated primarily through solar panels and stored in batteries. Critical subsystems such as communication, onboard processing, and propulsion all compete for this limited power.

This project proposes a **simulation-based AI decision system** that learns to intelligently allocate power between propulsion maneuvers and other operational tasks, improving overall mission efficiency.

The aim is to demonstrate how **Artificial Intelligence could assist autonomous spacecraft operations** in energy-constrained environments.

---

## Problem Statement

Spacecraft propulsion maneuvers required for orbit maintenance consume significant energy. Inefficient power allocation may lead to:

- Reduced satellite lifetime
- Unstable orbital conditions
- Unsafe battery levels
- Reduced mission performance

Traditional control systems rely on predefined rules. However, dynamic space environments may benefit from **adaptive decision-making systems powered by AI**.

---

## Concept Model

The proposed system simulates a simplified satellite environment where an AI agent evaluates the spacecraft's operational state and decides when to:

- Perform propulsion maneuvers for orbit correction
- Conserve energy when battery levels are low
- Maintain safe operational conditions

The AI agent observes variables such as:

- Battery level
- Orbit deviation
- Power consumption

Using **reinforcement learning (Q-learning)**, the agent learns optimal actions that improve long-term system stability.

---

## Optimization Equation

The decision process is modeled as a **multi-objective optimization problem**.

The operational cost function is defined as:

f = α(PowerUsage) + β(OrbitError) + γ(BatteryRisk)

Where:

- **PowerUsage** represents energy consumed by propulsion or operations  
- **OrbitError** represents deviation from the desired orbital position  
- **BatteryRisk** represents the penalty for unsafe battery levels  

The AI agent learns to **minimize this cost function over time**.

In reinforcement learning form:

Reward = -f

This allows the system to learn policies that balance:

- Energy efficiency
- Orbital stability
- Battery safety

---

## Future Work

This conceptual model can be extended in several ways:

- Incorporating realistic orbital mechanics simulations
- Modeling solar power generation and eclipse cycles
- Expanding to multi-satellite coordination systems
- Applying deep reinforcement learning for complex decision spaces
- Integrating spacecraft telemetry data for real-time learning

---

## Results

The trained RL agent successfully learned policies that balance:

- Power efficiency
- Orbit maintenance
- Battery safety

Simulation results demonstrated stable operation with low orbit error and safe battery levels throughout training.

---

**Author:**  
Vedhiga V B  
Final year B.E. Artificial Intelligence and Data Science  
Avinashilingam University, Coimbatore
