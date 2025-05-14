# Project 4: Ghostbusters Inference

## Overview
Implement probabilistic inference to track hidden ghosts using noisy distance sensors:

- **ExactInference**: Computes full belief distributions via bayesian updates.
- **ParticleFilter**: Approximates beliefs with a finite set of particles.

## Structure
- `inference.py`: Includes both ExactInference and ParticleFilter classes.
- `bustersAgents.py`: Agents that leverage inference to chase ghosts.

## Usage Examples
```bash
# Run Ghostbusters with exact inference
python busters.py -p BustersAgent -a inferenceType=Exact

# Run Ghostbusters with particle filtering (300 particles)
python busters.py -p BustersAgent -a inferenceType=Particle,numParticles=300
```
