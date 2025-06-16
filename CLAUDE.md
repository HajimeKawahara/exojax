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

## CKD Implementation Status (2025-06-16)

### ✅ Completed:
- **core.py**: All CKD core functions implemented and tested
  - `compute_g_ordinates()`: Sorts cross-sections, computes g-ordinates
  - `gauss_legendre_grid()`: Generates [0,1] quadrature points 
  - `safe_log_k()`: **FIXED** precision-aware defaults (1e-100 for float64, 1e-30 for float32)
  - `interpolate_log_k_to_g_grid()`: JAX interpolation to g-grid
  - `compute_ckd_tp_grid()`: Full T,P grid CKD computation using vmap
- **Tests**: All 7 unit tests passing
- **Git**: Changes committed and pushed to `correlatedk` branch

### 🔄 Next: Implement `precompute_tables()` in `api.py`
**Key Issue Identified**: 
- `core.py` returns shape `(nT, nP, Ng)` 
- `api.py` expects shape `(nT, nP, Ng, nnu_bands)`

**Recommended Implementation**:
```python
def precompute_tables(self, T_grid, P_grid):
    from .core import compute_ckd_tp_grid
    log_kggrid, ggrid, weights = compute_ckd_tp_grid(T_grid, P_grid, self.base_opa, self.Ng)
    
    # Add band dimension for compatibility: (nT, nP, Ng) -> (nT, nP, Ng, 1)
    self.ckd_info = CKDTableInfo(
        log_kggrid=log_kggrid[..., jnp.newaxis],
        ggrid=ggrid, weights=weights,
        T_grid=jnp.asarray(T_grid), P_grid=jnp.asarray(P_grid),
        nu_bands=jnp.array([jnp.mean(self.base_opa.nu_grid)])
    )
    self.ready = True
```

### 🎯 Office Quick Start:
```bash
git pull origin correlatedk
# Tell Claude: "Continue CKD implementation. Implement precompute_tables() in api.py"
```