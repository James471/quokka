//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shear_shocktube.cpp
/// \brief Defines a shear shock tube test with strong transverse velocity
///        discontinuity.

#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"
#include <cmath>
#include <fmt/format.h>
#include <string>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_PODVector.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct ShearShocktubeProblem {
};

template <> struct quokka::EOS_Traits<ShearShocktubeProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ShearShocktubeProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

namespace {
constexpr amrex::Real rho_L = 1.0;
constexpr amrex::Real P_L = 10.0;
constexpr amrex::Real rho_R = 0.01;
constexpr amrex::Real P_R = 0.01;
constexpr amrex::Real vx_L = -5.0;
constexpr amrex::Real vx_R = 5.0;
constexpr amrex::Real vy_L = -10.0;
constexpr amrex::Real vy_R = 10.0;
constexpr amrex::Real interface_position = 2.0;
} // namespace

template <> void QuokkaSimulation<ShearShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const int ncomp_cc = Physics_Indices<ShearShocktubeProblem>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];

		double vx = NAN;
		double vy = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < interface_position) {
			rho = rho_L;
			P = P_L;
			vx = vx_L;
			vy = vy_L;
		} else {
			rho = rho_R;
			P = P_R;
			vx = vx_R;
			vy = vy_R;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto gamma = quokka::EOS_Traits<ShearShocktubeProblem>::gamma;
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<ShearShocktubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShearShocktubeProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<ShearShocktubeProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<ShearShocktubeProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShearShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx + vy * vy);
		state_cc(i, j, k, HydroSystem<ShearShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShearShocktubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv,
							     amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int numcomp,
							     amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							     int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();
	const auto gamma = quokka::EOS_Traits<ShearShocktubeProblem>::gamma;

	if (i < lo[0]) {
		for (int n = 0; n < numcomp; ++n) {
			consVar(i, j, k, n) = 0.;
		}

		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::gasEnergy_index) = P_L / (gamma - 1.) + 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L);
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::gasInternalEnergy_index) = P_L / (gamma - 1.);
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::gasDensity_index) = rho_L;
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::x1GasMomentum_index) = rho_L * vx_L;
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::x2GasMomentum_index) = rho_L * vy_L;
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::x3GasMomentum_index) = 0.;

	} else if (i >= hi[0]) {
		for (int n = 0; n < numcomp; ++n) {
			consVar(i, j, k, n) = 0.;
		}

		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::gasEnergy_index) = P_R / (gamma - 1.) + 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R);
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::gasInternalEnergy_index) = P_R / (gamma - 1.);
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::gasDensity_index) = rho_R;
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::x1GasMomentum_index) = rho_R * vx_R;
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::x2GasMomentum_index) = rho_R * vy_R;
		consVar(i, j, k, RadSystem<ShearShocktubeProblem>::x3GasMomentum_index) = 0.;
	}
}

template <> void QuokkaSimulation<ShearShocktubeProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	const Real eta_threshold = 0.1;
	const Real rho_min = 0.01;
	auto const &dx = geom[lev].CellSizeArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			int const n = 0;
			Real const rho = state(i, j, k, n);
			Real const del_x = (state(i + 1, j, k, n) - state(i - 1, j, k, n)) / (2.0 * dx[0]);
			Real const gradient_indicator = std::sqrt(del_x * del_x) / rho;

			if (gradient_indicator > eta_threshold && rho >= rho_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

namespace {
auto compute_region_mean_vy(const std::vector<double> &positions, const amrex::PODVector<double> &density,
				 const amrex::PODVector<double> &yMomentum, double x_lo, double x_hi) -> double
{
	double rho_sum = 0.0;
	double mom_sum = 0.0;
	for (std::size_t i = 0; i < positions.size(); ++i) {
		if (positions[i] >= x_lo && positions[i] <= x_hi) {
			rho_sum += density[i];
			mom_sum += yMomentum[i];
		}
	}
	AMREX_ALWAYS_ASSERT(rho_sum > 0.0);
	return mom_sum / rho_sum;
}
} // namespace

auto problem_main() -> int
{
	const double max_time = 0.4;
	const int max_timesteps = 8000;

	auto BCs_cc = quokka::BC<ShearShocktubeProblem>(quokka::BCType::ext_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);

	QuokkaSimulation<ShearShocktubeProblem> sim(BCs_cc);
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.computeReferenceSolution_ = false;

	sim.setInitialConditions();
	sim.evolve();

	auto [positions, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	const auto &density = values.at(HydroSystem<ShearShocktubeProblem>::density_index);
	const auto &y_momentum = values.at(HydroSystem<ShearShocktubeProblem>::x2Momentum_index);

	double meanVyLeft = compute_region_mean_vy(positions, density, y_momentum, 0.0, 0.5);
	double meanVyRight = compute_region_mean_vy(positions, density, y_momentum, 4.5, 5.0);

	const double rel_tolerance = 1.0e-3;
	const double shear_expected = vy_L - vy_R;

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const gamma = quokka::EOS_Traits<ShearShocktubeProblem>::gamma;
		std::vector<double> xs(positions.begin(), positions.end());
		const auto &x_momentum = values.at(HydroSystem<ShearShocktubeProblem>::x1Momentum_index);
		const auto &internal_energy = values.at(HydroSystem<ShearShocktubeProblem>::internalEnergy_index);
		std::vector<double> rho_profile(density.begin(), density.end());
		std::vector<double> vx_profile(xs.size());
		std::vector<double> vy_profile(xs.size());
		std::vector<double> pressure_profile(xs.size());
		for (std::size_t i = 0; i < xs.size(); ++i) {
			double const rho = rho_profile[i];
			double const mom_x = x_momentum[i];
			double const mom_y = y_momentum[i];
			double const eint = internal_energy[i];
			vx_profile[i] = mom_x / rho;
			vy_profile[i] = mom_y / rho;
			pressure_profile[i] = (gamma - 1.0) * eint;
		}
		namespace plt = matplotlibcpp;
		plt::figure();
		plt::subplot(2, 2, 1);
		plt::plot(xs, rho_profile);
		plt::title("density");
		plt::subplot(2, 2, 2);
		plt::plot(xs, vx_profile);
		plt::title("v_x");
		plt::subplot(2, 2, 3);
		plt::plot(xs, vy_profile);
		plt::title("v_y");
		plt::subplot(2, 2, 4);
		plt::plot(xs, pressure_profile);
		plt::title("pressure");
		plt::tight_layout();
		plt::save(fmt::format("./hydro_shear_shocktube_{:.4f}.png", sim.tNew_[0]));
	}
#endif

	const double shear_measured = meanVyLeft - meanVyRight;

	int status = 0;
	if (std::abs(meanVyLeft - vy_L) > rel_tolerance * std::abs(vy_L)) {
		amrex::Print() << fmt::format("Left-state transverse velocity drifted: {} vs {}\n", meanVyLeft, vy_L);
		status = 1;
	}
	if (std::abs(meanVyRight - vy_R) > rel_tolerance * std::abs(vy_R)) {
		amrex::Print() << fmt::format("Right-state transverse velocity drifted: {} vs {}\n", meanVyRight, vy_R);
		status = 1;
	}
	if (std::abs(shear_measured) < 0.9 * std::abs(shear_expected)) {
		amrex::Print() << fmt::format("Shear diminished: measured {} expected {}\n", shear_measured, shear_expected);
		status = 1;
	}

	return status;
}

