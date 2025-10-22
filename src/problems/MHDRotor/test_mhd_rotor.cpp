//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_mhd_rotor.cpp
/// \brief TBD
///

#include <array>
#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_Gpu.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct MHDRotor {};

template <> struct quokka::EOS_Traits<MHDRotor> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<MHDRotor> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double gamma_gas = quokka::EOS_Traits<MHDRotor>::gamma;
constexpr double bg_pressure = 0.5;
constexpr double density_inside = 10.0;
constexpr double density_outside = 1.0;
constexpr double rotor_speed = 1.0; // v_0
constexpr double b_x1 = 2.5 / (4.0 * M_PI);
constexpr double b_x2 = 0.0;
constexpr double b_x3 = 0.0;
constexpr double rotor_radius = 0.1;  // r_0
constexpr double taper_scale_height = 0.115;  // r_1
constexpr double rotor_center_x1 = 0.5;
constexpr double rotor_center_x2 = 0.5;

AMREX_GPU_DEVICE
inline void setICs_cc(
	int i, int j, int k,
	amrex::Array4<amrex::Real> const& state_cc,
	amrex::GpuArray<amrex::Real,3> const& cell_width,
	amrex::GpuArray<amrex::Real,3> const& prob_lo)
{
	const amrex::Real x1_C = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * cell_width[0];
	const amrex::Real x2_C = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * cell_width[1];

	const double x1_rel = static_cast<double>(x1_C) - rotor_center_x1;
	const double x2_rel = static_cast<double>(x2_C) - rotor_center_x2;
	const double radius_from_center = std::sqrt(x1_rel * x1_rel + x2_rel * x2_rel);

	double radial_taper = 0.0;
	if (radius_from_center <= rotor_radius) {
		radial_taper = 1.0;
	} else if (radius_from_center < taper_scale_height) {
		const double delta_radius = taper_scale_height - rotor_radius;
		radial_taper = (taper_scale_height - radius_from_center) / delta_radius;
	}
	const double density = density_outside + (density_inside - density_outside) * radial_taper;

	double vel_tangent = 0.0;
	if (radius_from_center <= rotor_radius) {
		vel_tangent = rotor_speed * (radius_from_center / rotor_radius);
	} else if (radius_from_center < taper_scale_height) {
		vel_tangent = rotor_speed * radial_taper;
	}

	const double safe_inv_radius = (radius_from_center == 0.0) ? 0.0 : 1.0 / radius_from_center;
	const double vel_x1 = -vel_tangent * x2_rel * safe_inv_radius;
	const double vel_x2 = vel_tangent * x1_rel * safe_inv_radius;
	const double vel_x3 = 0.0;

	const double vel_magn_sq = vel_x1 * vel_x1 + vel_x2 * vel_x2 + vel_x3 * vel_x3;
	const double Ekin = 0.5 * density * vel_magn_sq;
	const double Emag = 0.5 * (b_x1 * b_x1 + b_x2 * b_x2 + b_x3 * b_x3);
	const double Eint = bg_pressure / (gamma_gas - 1.0);
	const double Etot = Eint + Ekin + Emag;

	const int ncomp_cc = Physics_Indices<MHDRotor>::nvarTotal_cc;
	for (int icomp = 0; icomp < ncomp_cc; ++icomp) {
		state_cc(i, j, k, icomp) = 0.0;
	}
	state_cc(i, j, k, HydroSystem<MHDRotor>::density_index) = density;
	state_cc(i, j, k, HydroSystem<MHDRotor>::x1Momentum_index) = density * vel_x1;
	state_cc(i, j, k, HydroSystem<MHDRotor>::x2Momentum_index) = density * vel_x2;
	state_cc(i, j, k, HydroSystem<MHDRotor>::x3Momentum_index) = density * vel_x3;
	state_cc(i, j, k, HydroSystem<MHDRotor>::energy_index) = Etot;
	state_cc(i, j, k, HydroSystem<MHDRotor>::internalEnergy_index) = Eint;
}


template <quokka::direction dir>
AMREX_GPU_DEVICE
inline void setICs_fc(
	int i, int j, int k,
	amrex::Array4<amrex::Real> const& state_fc)
{
	const int ncomp_fc = Physics_Indices<MHDRotor>::nvarPerDim_fc;
	for (int icomp = 0; icomp < ncomp_fc; ++icomp) {
		state_fc(i,j,k,icomp) = 0.0;
	}
	if constexpr (dir == quokka::direction::x) {
		state_fc(i, j, k, MHDSystem<MHDRotor>::bfield_index) = b_x1;
	} else if constexpr (dir == quokka::direction::y) {
		state_fc(i, j, k, MHDSystem<MHDRotor>::bfield_index) = b_x2;
	} else if constexpr (dir == quokka::direction::z) {
		state_fc(i, j, k, MHDSystem<MHDRotor>::bfield_index) = b_x3;
	}
}

template <>
void QuokkaSimulation<MHDRotor>::setInitialConditionsOnGrid(quokka::grid const& grid_elem)
{
	const amrex::GpuArray<amrex::Real,3> cell_width = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real,3> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real>& state_cc = grid_elem.array_;
	const amrex::Box& indexRange = grid_elem.indexRange_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		setICs_cc(i, j, k, state_cc, cell_width, prob_lo);
	});
}

template <>
void QuokkaSimulation<MHDRotor>::setInitialConditionsOnGridFaceVars(quokka::grid const& grid_elem)
{
	const amrex::Array4<amrex::Real>& state_fc = grid_elem.array_;
	const amrex::Box& indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			setICs_fc<quokka::direction::x>(i, j, k, state_fc);
		} else if (dir == quokka::direction::y) {
			setICs_fc<quokka::direction::y>(i, j, k, state_fc);
		} else {
			setICs_fc<quokka::direction::z>(i, j, k, state_fc);
		}
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDRotor>::setCustomBoundaryConditions(
		const amrex::IntVect &iv,
		amrex::Array4<amrex::Real> const &consVar,
		int /*dcomp*/, int /*numcomp*/,
		amrex::GeometryData const &geom,
		const amrex::Real /*time*/,
		const amrex::BCRec * /*bcr*/,
		int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.toArray();
	amrex::GpuArray<amrex::Real, 3> prob_lo = { geom.ProbLo()[0], geom.ProbLo()[1], geom.ProbLo()[2] };
	amrex::GpuArray<amrex::Real, 3> cell_width = { geom.CellSize()[0], geom.CellSize()[1], geom.CellSize()[2] };
	const amrex::Real x1_C = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * cell_width[0];
	const amrex::Real x2_C = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * cell_width[1];
	const bool is_outside_x1 = (x1_C <= 0.0 || 1.0 <= x1_C);
	const bool is_outside_x2 = (x2_C <= 0.0 || 1.0 <= x2_C);
	if (is_outside_x1 || is_outside_x2) {
		setICs_cc(i, j, k, consVar, cell_width, prob_lo);
	}
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDRotor>::setCustomBoundaryConditionsFaceVar(
		const amrex::IntVect &iv,
		amrex::Array4<amrex::Real> const &dest,
		int /*dcomp*/, int /*numcomp*/,
		amrex::GeometryData const &geom,
		const amrex::Real /*time*/,
		const amrex::BCRec * /*bcr*/,
		int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.toArray();
	// this casting is not necessary
	amrex::GpuArray<amrex::Real, 3> prob_lo = { geom.ProbLo()[0], geom.ProbLo()[1], geom.ProbLo()[2] };
	amrex::GpuArray<amrex::Real, 3> cell_width = { geom.CellSize()[0], geom.CellSize()[1], geom.CellSize()[2] };
	const amrex::Real x1_L = prob_lo[0] + i * cell_width[0];
	const amrex::Real x2_L = prob_lo[1] + j * cell_width[1];
	if constexpr (dir == quokka::direction::x) {
		if (x1_L <= 0.0 || 1.0 <= x1_L) {
			setICs_fc<quokka::direction::x>(i, j, k, dest);
		}
	} else if constexpr (dir == quokka::direction::y) {
		if (x2_L <= 0.0 || 1.0 <= x2_L) {
			setICs_fc<quokka::direction::y>(i, j, k, dest);
		}
	}
}

auto problem_main() -> int
{
	const int ncomp_cc = Physics_Indices<MHDRotor>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int icomp = 0; icomp < ncomp_cc; ++icomp) {
		for (int idim = 0; idim < 2; ++idim) {
			BCs_cc[icomp].setLo(idim, amrex::BCType::ext_dir); // Dirichlet
			BCs_cc[icomp].setHi(idim, amrex::BCType::ext_dir);
		}
		BCs_cc[icomp].setLo(2, amrex::BCType::int_dir); // periodic
		BCs_cc[icomp].setHi(2, amrex::BCType::int_dir);
	}
	const int nvars_fc = Physics_Indices<MHDRotor>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < 2; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::ext_dir); // Dirichlet
			BCs_fc[icomp].setHi(idim, amrex::BCType::ext_dir);
		}
		BCs_fc[icomp].setLo(2, amrex::BCType::int_dir); // periodic
		BCs_fc[icomp].setHi(2, amrex::BCType::int_dir);
	}

	QuokkaSimulation<MHDRotor> sim(BCs_cc, BCs_fc);
	sim.computeReferenceSolution_ = false;
	sim.setInitialConditions();
	sim.evolve();

	return 0;
}
