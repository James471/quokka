# Python Problem Hooks Plan

## Motivation

We want to let users (and future Python bindings) customize Quokka scenarios without recompiling every host-side specialization. To do that safely we need to know which customization points must remain compile-time template specializations (because they are inlined into GPU kernels) and which can be swapped out at runtime through `std::function` callbacks exposed to pybind11.

## Classification Table

| Category | Examples | Requirement |
| --- | --- | --- |
| Boundary fills | `AMRSimulation::setCustomBoundaryConditions`, `setCustomBoundaryConditionsFaceVar` and their problem specializations | Must stay as template specializations. These functions are marked `AMREX_GPU_DEVICE` and are invoked inside fill-patch kernels, so the device compiler needs compile-time knowledge of their implementations. |
| Microphysics traits & inline physics helpers | `Physics_Traits`, `quokka::EOS_Traits`, `RadSystem_Traits`, opacity/closure functions, EOS conversions, limiter helpers | Must stay as template specializations. Flux/closure kernels call these functions from inside `amrex::ParallelFor` lambdas, so they have to be available as inline device code. |
| Reconstruction and flux helpers | `HydroSystem::*ComputeFluxes*`, `RadSystem::*ComputeFluxes*`, linear advection fluxes/reconstruction, limiter logic | Must stay as template specializations. Reconstruction/flux routines run inside device kernels and depend on compile-time problem traits. |
| Host-only hooks | Grid initialization (`setInitialConditionsOnGrid`, `setInitialConditionsOnGridFaceVars`), AMR tagging orchestration (`refineGrid`, `ErrorEst` entrypoints), particle seeding (`createInitial*Particles`), timestep lifecycle (`computeBeforeTimestep`, `computeAfterTimestep`, `computeAfterLevelAdvance`, `computeAfterEvolve`), diagnostics/statistics (`ComputeDerivedVar`, `ComputeProjections`, `ComputeStatistics`, `WriteSingleLevelPlotfileSimplified`), source wrappers (`addStrangSplitSources`, `fillPoissonRhsAtLevel`, `applyPoissonGravityAtLevel`), metadata updates | Can be converted to `std::function` callbacks. These methods execute on the host and only launch precompiled kernels, so they can pull logic from dynamic callbacks registered via C++ or Python. |

## Implementation Plan

1. **Hook Registry:** Introduce a `ProblemHooks` struct inside `AMRSimulation` to store `std::function` members for every host-only extension point (initial conditions, timestep callbacks, diagnostics, particle seeding, etc.). Provide setter APIs so both C++ problems and future pybind11 bindings can register callbacks.
2. **Bridge Existing Problems:** Update each problem specialization to populate the relevant hook rather than overriding the virtual method. This keeps current behavior while enabling runtime overrides (orderly migration: start with grid initialization and lifecycle hooks, then diagnostics).
3. **pybind11 Exposure:** When building the upcoming `pyquokka` module, expose hook setters/getters so Python scripts can register functions. Python users will select one of the precompiled problem types (to satisfy the compile-time requirements above) and then attach host-only behavior dynamically.
4. **Documentation & Examples:** Once the hook infrastructure lands, add user-facing docs/examples that demonstrate registering initialization logic and diagnostics from Python while relying on precompiled device kernels for performance-sensitive code.

## pyquokka pybind11 Module Structure

- **Single extension:** Build one `pyquokka` module (`PYBIND11_MODULE(pyquokka, m)`) that depends only on pybind11 plus the Quokka/AMReX headers already in tree. Keep nanobind out so any types shared with pyAMReX continue to use the same pybind11 registries.
- **File layout:** Organize bindings under `bindings/` (e.g., `simulation.cpp`, `hooks.cpp`, `layout.cpp`) but export everything through the single module to avoid duplicate registries. A shared `bindings/common.hpp` should include pybind11 headers and any `NB_MAKE_OPAQUE` macros once needed.
- **Submodules/namespaces:**
  - `pyquokka.simulation`: wraps the non-templated `QuokkaSimulation`/`AMRSimulation` facade. Expose constructors that take a descriptor enum plus runtime `PhysicsLayout` and inputs path, along with methods such as `initialize()`, `evolve(nsteps)`, and `write_plotfile()`. Any pyAMReX types required here should be passed via capsules or not exposed until pyAMReX provides casters.
  - `pyquokka.hooks`: mirrors the `ProblemHooks` struct, providing setters like `set_initial_conditions(func)`, `set_before_timestep(func)`, etc. Each setter stores a `std::function` that captures the Python callable with a GIL-safe trampoline.
  - `pyquokka.layout`: binds the runtime `PhysicsLayout` struct and descriptor enums so Python can introspect offsets and feature flags when constructing hooks. All properties should be read-only to keep invariants enforced on the C++ side.
  - `pyquokka.state`: utility helpers that expose read-only views of `MultiFab`/`FArrayBox` data. For now this can return capsules that pyAMReX understands, or lightweight NumPy arrays built via `py::array` referencing host data.
- **Binding patterns:** use `py::class_` for POD structs and shared-pointer-aware `py::class_` for owning simulation objects, wrap hook callbacks with helper lambdas that acquire the GIL before invoking Python, and lean on pyAMReX casters where they already exist.
- **Incremental plan:** (1) land the `ProblemHooks` and runtime descriptor/layout refactor, (2) add the pybind11 target and stub module exporting the descriptor/layout APIs, (3) wire in hook setters plus a smoke-test script that registers a Python callable and runs a short evolve step.

## Runtime Physics Configuration Without Kernel Overhead

We need three properties simultaneously:

1. Kernels should keep using `constexpr` indices wherever possible to avoid extra register pressure.
2. Users must be able to switch physics modules (hydro / radiation / MHD / dust / chemistry) and scalar counts at runtime without recompiling.
3. The number of template instantiations must stay modest.

### Descriptor + Runtime Layout Hybrid

*Compile-time descriptors:*  
Maintain a small set of `PhysicsDescriptor` templates that capture only the “big” compile-time switches, e.g.

- `PhysicsDescriptor<HydroEnabled=true, RadiationEnabled=false, MHD=false>`
- `PhysicsDescriptor<HydroEnabled=true, RadiationEnabled=true, MHD=false>`
- `PhysicsDescriptor<HydroEnabled=true, RadiationEnabled=true, MHD=true>`

Each descriptor defines `constexpr` offsets and strides for the fixed parts of the state layout (hydro conserved variables, radiation group stride structure, face-centered arrays, etc.), as well as flags indicating whether optional blocks exist at all. Reconstruction, flux, and closure kernels include the descriptor so the compiler can inline the relevant code paths.

*Runtime layout struct:*  
Only the *length* of the variable-sized regions (radiation groups, mass scalars, passive scalars) is supplied at runtime. We compute the final offsets once during initialization and store them in a `PhysicsLayout` struct that mirrors:

```cpp
struct PhysicsLayout {
  int num_rad_groups;
  int rad_offset;        // first Erad component
  int rad_group_stride;  // constexpr: e.g., 4 (Erad + Fx + Fy + Fz)

  int num_mass_scalars;
  int mass_scalar_offset; // start of mass scalars
  int scalar_stride;      // constexpr: 1

  int num_passive_scalars;
  int passive_scalar_offset;

  int total_cc_components;
  // Additional flags/offsets for face-centered data, dust fluids, etc.
};
```

We place a single instance of `PhysicsLayout` into device constant memory (e.g., `amrex::Gpu::DeviceScalar<PhysicsLayout>`). After parsing runtime inputs we fill the host copy, compute the derived offsets using the descriptor’s known ordering, and copy it to the device once before any kernels run.

### Kernel Access Patterns

Hydro-only sections continue to use `constexpr` indices from the descriptor:

```cpp
constexpr int rho_idx = Descriptor::gasDensity_index;
const Real rho = cons(i, j, k, rho_idx);
```

Radiation loops combine the descriptor stride with the runtime count:

```cpp
if constexpr (Descriptor::has_radiation) {
  const auto layout = getDevicePhysicsLayout();
  for (int g = 0; g < layout.num_rad_groups; ++g) {
    const int base = layout.rad_offset + g * layout.rad_group_stride;
    const Real Erad = cons(i, j, k, base + 0);
    const Real Fx   = cons(i, j, k, base + 1);
    // ...
  }
}
```

Mass scalars and passive scalars follow the same pattern:

```cpp
const auto layout = getDevicePhysicsLayout();
for (int s = 0; s < layout.num_mass_scalars; ++s) {
  const int idx = layout.mass_scalar_offset + s * layout.scalar_stride;
  // operate on scalar `s`
}
```

Because `rad_group_stride` and `scalar_stride` are still `constexpr`, the compiler can strength-reduce the indexing math: only the loop bounds are runtime values, which are necessary regardless. Optional features remain compiled out entirely when the descriptor disables them (e.g., `if constexpr (!Descriptor::has_radiation) return;`).

### Benefits

- **No explosion of template instantiations:** only one descriptor per major physics combination is required, not per scalar/group count.
- **Runtime flexibility:** the layout struct encodes the requested number of radiation groups and scalars on a per-run basis.
- **Kernel efficiency:** hot loops retain `constexpr` indices where possible, and the loops over groups/scalars add only the minimal induction math already needed to iterate through those arrays.
- **Python compatibility:** Python-driven runs can choose physics options at runtime, and the hook infrastructure can query the same `PhysicsLayout` object to understand the state vector shape.

This design keeps boundary fills, microphysics traits, and reconstruction/flux helpers fully templated while meeting the runtime configurability requirements for future Python bindings.

## Class Architecture Implications

Adopting the descriptor + runtime layout approach implies the following class refactor:

1. **`AMRSimulation` / `QuokkaSimulation` become non-templated.**  
   - They own an instance of a `PhysicsDescriptor` (selected at runtime from the small catalog described above) and a `PhysicsLayout` structure populated after parsing inputs.  
   - Host-only hooks, Python bindings, and runtime configuration logic all live here without needing separate template instantiations.

2. **Physics systems remain templated.**  
   - `HydroSystem`, `RadSystem`, `LinearAdvectionSystem`, etc. stay templated on the descriptor type (or a thin wrapper that exposes both descriptor constants and accessors to the shared `PhysicsLayout`).  
   - This lets reconstruction/flux kernels keep their `constexpr` indices while grabbing runtime counts from the layout struct when they iterate over variable-length blocks (radiation groups, scalars).

3. **Problem-specific traits/boundaries still specialize.**  
   - Boundary fills and microphysics trait specializations are now keyed on the descriptor type rather than a monolithic `problem_t`, so they continue to compile inline into the relevant kernels.

In short, the driver classes handle runtime polymorphism and hook registration, while the performance-critical systems stay templated with just enough compile-time information (the descriptor) to keep GPU kernels efficient. This split matches the requirements for the Python hook plan and for dynamic physics configuration.

### Templates That Must Remain

After the refactor only the physics systems that execute GPU kernels remain templated:

| Template class | Location | Role |
| --- | --- | --- |
| `HyperbolicSystem<DescriptorT>` | `src/hyperbolic_system.hpp` | Base class implementing reconstruction (PLM/PPM/WENO), predictor/corrector loops, and limiter logic in device kernels. |
| `HydroSystem<DescriptorT>` | `src/hydro/hydro_system.hpp` | Hydrodynamics solver (indices, Riemann solvers, flux assembly, limit enforcement). Needs descriptor to keep component indices `constexpr`. |
| `MHDSystem<DescriptorT>` | `src/hydro/mhd_system.hpp` | Face-centered magnetic evolution, EMF reconstruction/averaging, and induction updates. |
| `RadSystem<DescriptorT>` | `src/radiation/radiation_system.hpp` | Radiation moment system (multi-group fluxes, opacities, source coupling). |
| `LinearAdvectionSystem<DescriptorT>` | `src/linear_advection/linear_advection.hpp` | Scalar advection testbed sharing the same reconstruction/flux framework. |

Each of these classes will switch from the old `problem_t` template parameter to a descriptor type. Boundary conditions and microphysics traits continue to specialize on the descriptor as well so their inline device code is preserved.

### Descriptor Type Sketch

A descriptor captures compile-time facts about one mix of physics features. An example definition:

```cpp
struct HydroModule {};
struct RadiationModule {};
struct MHDModule {};

template <typename... Modules>
struct PhysicsDescriptor {
  static constexpr bool has_hydro     = (std::is_same_v<Modules, HydroModule> || ...);
  static constexpr bool has_radiation = (std::is_same_v<Modules, RadiationModule> || ...);
  static constexpr bool has_mhd       = (std::is_same_v<Modules, MHDModule> || ...);

// Example alias:
using ActiveDescriptor = PhysicsDescriptor<HydroModule, RadiationModule>; // hydro + radiation, no MHD

// Strides (compile-time)
  static constexpr int hydro_cc_size       = Hydro ? Physics_NumVars::numHydroVars : 0;
  static constexpr int rad_group_stride    = Radiation ? Physics_NumVars::numRadVarsPerGroup : 0;
  static constexpr int face_mhd_stride     = MHD ? Physics_NumVars::numMHDVars_per_dim : 0;
  static constexpr int scalar_stride       = 1;  // single component per scalar

  // Index helpers used by kernels
  static constexpr int gasDensity_index    = 0;
  static constexpr int x1GasMomentum_index = gasDensity_index + 1;
  // ...

  // Access to runtime layout
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE
  static const PhysicsLayout& layout() {
    return quokka::devicePhysicsLayout.data();
  }
};
```

Key points:

- The descriptor encodes *only* the information that never changes at runtime (which blocks exist, their ordering, and constant strides). We instantiate one descriptor per supported combination of `Hydro/Radiation/MHD` flags (and any other binary features that truly change the ordering). That keeps the number of template instantiations manageable.
- Kernels can query `Descriptor::has_radiation` in `if constexpr` blocks to drop unused code entirely. When `has_radiation` is `true`, they read runtime counts/offsets from `Descriptor::layout()` as described earlier.
- Boundary condition and trait specializations are written against `PhysicsDescriptor<...>` so they remain available to the templated systems.

At runtime the driver picks a descriptor (e.g., `using ActiveDescriptor = PhysicsDescriptor<HydroModule, RadiationModule>;`) based on the parsed physics options, instantiates the templated systems with that descriptor, fills the `PhysicsLayout`, and launches the simulation.
Using module tags keeps the syntax readable while still giving the compiler compile-time booleans. Additional modules (chemistry, dust, cosmic rays, etc.) can add their own tag structs without changing call sites: `using ActiveDescriptor = PhysicsDescriptor<HydroModule, RadiationModule, DustModule>;`.

### Prototype Implementation

The first code prototype for this architecture lives under `src/experimental/DescriptorPrototype.{hpp,cpp}`:

- `AMRSimulationPrototype` is a non-templated driver that stores the runtime `PhysicsLayout` and a set of host-only hooks (`pre_timestep`, `post_timestep`, `diagnostics`). It demonstrates how runtime configuration and callback registration can happen once, independent of the physics modules that the run will use.
- `AdvectionSimulationPrototype` derives from the driver and hard-wires `using Descriptor = PhysicsDescriptor<HydroModule>`. It exposes a simple runtime interface (`setAdvectionVelocity`, `estimateMaxSignalSpeed`, `step()`) while keeping the descriptor alias available for templated physics systems.

These prototype classes do not affect the existing templated `AMRSimulation` / `AdvectionSimulation` hierarchy or `QuokkaSimulation`; they are meant to exercise the descriptor concepts in isolation so we can iterate on the API before integrating it into the production solvers.
