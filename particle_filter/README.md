# Particle filter libary
Matlab OOP implementation of particle filters.

## File list
### Particle filters
- PfAux: Auxiliary Particle Filter
- PfBs: boot strap particle filter
- PfGau: Gaussian Particle Filter
- PfUnc: Uncented Particle Filter
- **ParticleFilter**: base class of all above

### Multi-model particle filter
- PfBsMM: Multi transition model boot strap particle filter
    - modelParticles: class for particles of one model

### Examples
- EXAMPLE_Pf: example illustrating single model filters
- GenSigHMM: signal generating class used in examples
    - GenSigHMM_test: test code of signal gnerating class