//==============================================================================
// MHD Rotor test (Balsara & Spicer 1999) — minimal Quokka setup
// Periodic BCs (you can change to reflective later).
//==============================================================================
// Copyright 2025 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================

#include <array>
#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Gpu.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct MHDRotor {};

template <> struct quokka::EOS_Traits<MHDRotor> {
  // Classical rotor tests commonly use gamma = 1.4
  static constexpr double gamma = 1.4;
  // Units aren’t essential here; Quokka uses Heaviside-Lorentz (no 4π in energy).
  static constexpr double mean_molecular_weight = C::m_u;
  static constexpr double boltzmann_constant   = C::k_B;
};

template <> struct Physics_Traits<MHDRotor> {
  static constexpr bool is_hydro_enabled        = true;
  static constexpr int  numMassScalars          = 0;
  static constexpr int  numPassiveScalars       = numMassScalars + 0;
  static constexpr bool is_self_gravity_enabled = false;
  static constexpr bool is_radiation_enabled    = false;
  static constexpr bool is_mhd_enabled          = true;
  static constexpr int  nGroups                 = 1;
  static constexpr UnitSystem unit_system       = UnitSystem::CGS;
};

// ---- Canonical rotor parameters (FLASH/PLUTO-style) -------------------------
// Domain: [0,1]×[0,1], rotor centered at (0.5, 0.5)
constexpr double gamma_gas      = quokka::EOS_Traits<MHDRotor>::gamma;
constexpr double p0             = 1.0;     // uniform pressure
constexpr double rho_out        = 1.0;
constexpr double rho_in         = 10.0;
constexpr double vtheta_core    = 2.0;     // max tangential speed inside core
constexpr double B0x            = 5.0;     // HL units (i.e., B^2/2 is magnetic energy density)
constexpr double r0             = 0.1;     // solid-body rotation radius
constexpr double r1             = 0.115;   // linear taper ends; outside this, v=0, density=rho_out
constexpr double xc             = 0.5;
constexpr double yc             = 0.5;

// Helper to get rotor profile multipliers (velocity taper and density)
AMREX_GPU_HOST_DEVICE
inline void rotor_profile(double x, double y, double& rho, double& vx, double& vy)
{
  const double dx = x - xc;
  const double dy = y - yc;
  const double r  = std::sqrt(dx*dx + dy*dy);

  // Solid-body rotation inside r0; linear taper r0..r1; zero outside.
  double f_vel = 0.0;
  if (r <= r0) {
    f_vel = 1.0;
  } else if (r < r1) {
    f_vel = (r1 - r) / (r1 - r0);
  } else {
    f_vel = 0.0;
  }

  // Tangential direction: (-dy, +dx) / r; avoid 0/0 at center.
  const double inv_r = (r > 0.0) ? (1.0 / r) : 0.0;
  const double vtheta = vtheta_core * f_vel;
  vx = -vtheta * dy * inv_r;
  vy =  vtheta * dx * inv_r;

  // Density: rotor core dense, linear blend through the taper, ambient outside
  if (r <= r0) {
    rho = rho_in;
  } else if (r < r1) {
    const double w = (r1 - r) / (r1 - r0);
    rho = rho_out + (rho_in - rho_out) * w;
  } else {
    rho = rho_out;
  }
}

// Cell-centered initial condition kernel
AMREX_GPU_DEVICE
inline void set_cc_state(int i, int j, int k,
                         amrex::Array4<amrex::Real> const& U,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& prob_lo)
{
  // cell centers
  const amrex::Real x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
  (void)k; // 2D problem; leave z unused

  double rho, vx, vy;
  rotor_profile(x, y, rho, vx, vy);
  const double vz = 0.0;

  // Uniform magnetic field B = (B0x, 0, 0)
  const double Bx = B0x, By = 0.0, Bz = 0.0;

  const double v2   = vx*vx + vy*vy + vz*vz;
  const double Ekin = 0.5 * rho * v2;
  const double Emag = 0.5 * (Bx*Bx + By*By + Bz*Bz); // Heaviside-Lorentz
  const double Eint = p0 / (gamma_gas - 1.0);
  const double Etot = Eint + Ekin + Emag;

  U(i,j,k, HydroSystem<MHDRotor>::density_index)        = rho;
  U(i,j,k, HydroSystem<MHDRotor>::x1Momentum_index)     = rho * vx;
  U(i,j,k, HydroSystem<MHDRotor>::x2Momentum_index)     = rho * vy;
  U(i,j,k, HydroSystem<MHDRotor>::x3Momentum_index)     = rho * 0.0;
  U(i,j,k, HydroSystem<MHDRotor>::energy_index)         = Etot;
  U(i,j,k, HydroSystem<MHDRotor>::internalEnergy_index) = Eint;
}

// Face-centered B initialization: uniform Bx = B0x, By=Bz=0
AMREX_GPU_DEVICE
inline void set_fc_b(int i, int j, int k,
                     amrex::Array4<amrex::Real> const& Bface,
                     quokka::direction dir)
{
  double val = 0.0;
  if (dir == quokka::direction::x) {
    val = B0x;
  } else {
    val = 0.0;
  }
  Bface(i,j,k, MHDSystem<MHDRotor>::bfield_index) = val;
}

template <>
void QuokkaSimulation<MHDRotor>::setInitialConditionsOnGrid(quokka::grid const& grid_elem)
{
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx      = grid_elem.dx_;
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
  const amrex::Array4<amrex::Real>& U                        = grid_elem.array_;
  const amrex::Box& indexRange                               = grid_elem.indexRange_;

  const int ncomp_cc = Physics_Indices<MHDRotor>::nvarTotal_cc;

  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    for (int n = 0; n < ncomp_cc; ++n) {
      U(i,j,k,n) = 0.0;
    }
    set_cc_state(i, j, k, U, dx, prob_lo);
  });
}

template <>
void QuokkaSimulation<MHDRotor>::setInitialConditionsOnGridFaceVars(quokka::grid const& grid_elem)
{
  const amrex::Array4<amrex::Real>& Bf   = grid_elem.array_;
  const amrex::Box& indexRange           = grid_elem.indexRange_;
  const quokka::direction dir            = grid_elem.dir_;

  const int ncomp_fc = Physics_Indices<MHDRotor>::nvarPerDim_fc;

  amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    for (int n = 0; n < ncomp_fc; ++n) {
      Bf(i,j,k,n) = 0.0;
    }
    set_fc_b(i, j, k, Bf, dir);
  });
}

auto problem_main() -> int
{
  // Periodic on everything (you’ll swap to reflective walls as needed)
  auto BCs_cc = quokka::BC<MHDRotor>(quokka::BCType::int_dir);

  const int nvars_fc = Physics_Indices<MHDRotor>::nvarTotal_fc;
  amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
  for (int icomp = 0; icomp < nvars_fc; ++icomp) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BCs_fc[icomp].setLo(d, amrex::BCType::int_dir);
      BCs_fc[icomp].setHi(d, amrex::BCType::int_dir);
    }
  }

  QuokkaSimulation<MHDRotor> sim(BCs_cc, BCs_fc);
  // Reference solution not defined; run the evolution
  sim.computeReferenceSolution_ = false;

  sim.setInitialConditions();
  sim.evolve();

  // No pass/fail criterion here; you can add diagnostics if desired.
  return 0;
}
