# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExoJAX is a **differentiable spectral modeling package** for exoplanets, brown dwarfs, and M dwarfs built on JAX. The entire computational pipeline is auto-differentiable, enabling gradient-based optimizations, HMC-NUTS sampling, and stochastic variational inference (SVI).

## Essential Commands

### Testing
```bash
# Run unit tests (fast, frequent use during development)
python -m pytest tests/unittests/

# Run integration tests (longer computation, uses real molecular databases)  
python -m pytest tests/integration/

# Run end-to-end tests (much longer computation time)
python -m pytest tests/endtoend/

# Run specific test file
python -m pytest tests/unittests/test_filename.py

# Run tests with specific pattern
python -m pytest -k "test_pattern"
```

### Installation & Setup
```bash
# Install in development mode
python setup.py install

# Alternative: Install with pip in development mode
pip install -e .

# Check installed version
python -c "import exojax; print(exojax.__version__)"

# Update documentation
./update_doc.sh
```

### Documentation
```bash
# Build documentation (requires sphinx)
cd documents && make clean && make html
```

## Architecture Overview

### Core Modules and Data Flow

```
Database Layer → Opacity Calculation → Atmospheric RT → Post-Processing
     ↓                    ↓                   ↓              ↓
 Line Lists         Cross-Sections      Flux/Transmission  Observed Spectrum
```

### Key Module Responsibilities

- **`database/`**: Manages spectroscopic databases (HITRAN, ExoMol, VALD, CIA)
  - `api.py`: Common API for molecular databases (MdbExomol, MdbHitran, MdbHitemp)
  - `moldb.py`/`atomll.py`: Atomic line lists 
  - `contdb.py`: Continuum opacity (CIA)
  - `pardb.py`: Particle/cloud databases

- **`opacity/`**: Line-by-line opacity calculations with performance trade-offs
  - `OpaPremodit`: **Fastest** - Pre-computed Modified Discrete Integral Transform
  - `OpaModit`: **Medium** - On-the-fly Modified Discrete Integral Transform  
  - `OpaDirect` (LPF): **Flexible** - Direct Line Profile Function
  - `OpaCKD`: Correlated-K Distribution method
  - Continuum: `OpaCIA`, `OpaRayleigh`, `OpaHminus`, `OpaMie`

- **`rt/`**: Radiative transfer solvers
  - `ArtEmisPure`/`ArtEmisScat`: Emission with/without scattering
  - `ArtTransPure`: Transmission spectra
  - `ArtReflectPure`: Reflection spectra
  - `opart.py`: Optimized layer-by-layer RT computations

- **`atm/`**: Atmospheric physics models
  - `atmprof.py`: Temperature-pressure profiles (Guillot, power-law)
  - `atmphys.py`: Cloud microphysics (Ackerman & Marley model)
  - `idealgas.py`: Gas physics calculations

- **`postproc/`**: Observational effects
  - `specop.py`: Spectral operators (convolution, instrumental response)
  - `spin_rotation.py`: Planetary rotation effects
  - `response.py`: Instrumental response functions

### Key Architectural Patterns

1. **JAX-First Design**: All computations use JAX arrays for auto-differentiation and JIT compilation
2. **Lazy Loading**: Heavy modules use lazy imports to minimize memory footprint
3. **Database Abstraction**: Common API across different spectroscopic databases
4. **Memory Optimization**: Overlap-and-Add (OLA) convolution for large spectral ranges

## Development Guidelines

### Test Organization
- **unittests/**: Fast tests for frequent use during development (no real molecular databases)
- **integration/**: Longer tests that may use real molecular databases
- **endtoend/**: Full pipeline tests with significant computation time

### Code Conventions
- All calculations should maintain JAX compatibility for auto-differentiation
- Use the existing database abstraction layer when adding new data sources
- Follow the established opacity calculator interface when implementing new methods
- Maintain lazy loading patterns for computational modules

### Performance Considerations
- Choose appropriate opacity method based on speed vs. flexibility trade-offs
- Use OLA convolution for memory-constrained calculations
- Leverage JAX JIT compilation for performance-critical functions
- Consider GPU acceleration for matrix-heavy operations

## Common Workflows

### Forward Modeling Pipeline
1. Load molecular databases → `MdbExomol`, `MdbHitran`, etc.
2. Configure opacity calculator → `OpaPremodit`, `OpaModit`, `OpaDirect`
3. Set atmospheric structure → temperature, pressure, mixing ratios
4. Compute radiative transfer → `ArtEmisPure`, `ArtTransPure`, etc.
5. Apply observational effects → instrumental response, rotation

### Adding New Databases
- Inherit from appropriate base classes in `database/api.py`
- Implement required methods for data loading and JAX array conversion
- Add database manager in `database/dbmanager.py`

### Implementing New Opacity Methods
- Inherit from `OpaCalc` base class in `opacity/opacalc.py`
- Implement required interface methods
- Consider memory and computational trade-offs
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.