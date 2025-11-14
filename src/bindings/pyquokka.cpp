#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AMReX.H"
#include "experimental/DescriptorPrototype.hpp"
#include "grid.hpp"

namespace py = pybind11;
using quokka::experimental::AdvectionSimulationPrototype;

namespace
{

auto makeInitializeHook(py::function func)
{
	return [func = std::move(func)](quokka::grid const &grid) {
		py::gil_scoped_acquire gil;
		const auto dx = grid.dx_;
		const auto prob_lo = grid.prob_lo_;
		const amrex::Box &indexRange = grid.indexRange_;
		const amrex::Array4<double> &state = grid.array_;
		const int k_lo = (AMREX_SPACEDIM >= 3) ? indexRange.smallEnd(2) : 0;
		const int k_hi = (AMREX_SPACEDIM >= 3) ? indexRange.bigEnd(2) : 0;
		const int j_lo = (AMREX_SPACEDIM >= 2) ? indexRange.smallEnd(1) : 0;
		const int j_hi = (AMREX_SPACEDIM >= 2) ? indexRange.bigEnd(1) : 0;
		for (int k = k_lo; k <= k_hi; ++k) {
			for (int j = j_lo; j <= j_hi; ++j) {
				for (int i = indexRange.smallEnd(0); i <= indexRange.bigEnd(0); ++i) {
					amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
					auto value = func(x).cast<amrex::Real>();
					state(i, j, k, 0) = value;
				}
			}
		}
	};
}

auto makeExactHook(py::function func)
{
	return [func = std::move(func)](quokka::grid const &grid, amrex::Real time) {
		py::gil_scoped_acquire gil;
		const auto dx = grid.dx_;
		const auto prob_lo = grid.prob_lo_;
		const amrex::Box &indexRange = grid.indexRange_;
		const amrex::Array4<double> &state = grid.array_;
		const int k_lo = (AMREX_SPACEDIM >= 3) ? indexRange.smallEnd(2) : 0;
		const int k_hi = (AMREX_SPACEDIM >= 3) ? indexRange.bigEnd(2) : 0;
		const int j_lo = (AMREX_SPACEDIM >= 2) ? indexRange.smallEnd(1) : 0;
		const int j_hi = (AMREX_SPACEDIM >= 2) ? indexRange.bigEnd(1) : 0;
		for (int k = k_lo; k <= k_hi; ++k) {
			for (int j = j_lo; j <= j_hi; ++j) {
				for (int i = indexRange.smallEnd(0); i <= indexRange.bigEnd(0); ++i) {
					amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
					auto value = func(x, time).cast<amrex::Real>();
					state(i, j, k, 0) = value;
				}
			}
		}
	};
}

auto makeDiagnosticsHook(py::function func)
{
	return [func = std::move(func)](AdvectionSimulationPrototype &sim) {
		py::gil_scoped_acquire gil;
		func(sim.currentTime(), sim.errorNorm());
	};
}

} // namespace

PYBIND11_MODULE(pyquokka, m)
{
	m.def("initialize", []() {
		if (!amrex::Initialized()) {
			int argc = 1;
			char arg0[] = "pyquokka";
			char *argv[] = {arg0, nullptr};
			char **argv_ptr = argv;
			amrex::Initialize(argc, argv_ptr);
		}
	});
	m.def("finalize", []() {
		if (amrex::Initialized()) {
			amrex::Finalize();
		}
	});

	py::class_<AdvectionSimulationPrototype>(m, "AdvectionSimulation")
	    .def(py::init<>())
	    .def("configure_grid", &AdvectionSimulationPrototype::configureGrid, py::arg("nx"), py::arg("prob_lo"),
		 py::arg("prob_hi"))
	    .def("set_advection_velocity", &AdvectionSimulationPrototype::setAdvectionVelocity, py::arg("vx"), py::arg("vy"),
		 py::arg("vz"))
	    .def("set_cfl", &AdvectionSimulationPrototype::setCfl)
	    .def("set_max_time", &AdvectionSimulationPrototype::setMaxTime)
	    .def("set_max_timesteps", &AdvectionSimulationPrototype::setMaxTimesteps)
	    .def("set_python_hooks",
		 [](AdvectionSimulationPrototype &sim, py::object init, py::object exact, py::object diagnostics) {
			 AdvectionSimulationPrototype::Callbacks callbacks;
			 if (!init.is_none()) {
				 callbacks.hooks.initialize = makeInitializeHook(py::function(init));
			 }
			 if (!exact.is_none()) {
				 callbacks.hooks.exact_solution = makeExactHook(py::function(exact));
			 }
			 if (!diagnostics.is_none()) {
				 callbacks.hooks.diagnostics = makeDiagnosticsHook(py::function(diagnostics));
			 }
			 sim.setCallbacks(std::move(callbacks));
		 },
		 py::arg("initial") = py::none(), py::arg("exact") = py::none(), py::arg("diagnostics") = py::none())
	    .def("run", &AdvectionSimulationPrototype::run)
	    .def("error_norm", &AdvectionSimulationPrototype::errorNorm)
	    .def("current_time", &AdvectionSimulationPrototype::currentTime);
}
