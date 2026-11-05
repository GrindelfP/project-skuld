#include <cstdlib> // std::atof
#include <iostream> // std::cout
#include <numeric> // std::accumulate
#include <vector> // std::vector

#include <secdecutil/integrators/cuba.hpp> // secdecutil::cuba::Vegas, secdecutil::cuba::Suave, secdecutil::cuba::Cuhre, secdecutil::cuba::Divonne
#include <secdecutil/integrators/qmc.hpp> // secdecutil::integrators::Qmc
#include <secdecutil/series.hpp> // secdecutil::Series
#include <secdecutil/uncertainties.hpp> // secdecutil::UncorrelatedDeviation
#include <secdecutil/deep_apply.hpp> // secdecutil::deep_apply

#include "integral_4.hpp"

void print_integral_info()
{
    std::cout << "-- print_integral_info --" << std::endl;
    std::cout << "integral_4::number_of_sectors " << integral_4::number_of_sectors << std::endl;

    std::cout << "integral_4::number_of_regulators " << integral_4::number_of_regulators << std::endl;
    std::cout << "integral_4::names_of_regulators ";
    for ( const auto& name : integral_4::names_of_regulators )
        std::cout << " " << name;
    std::cout << std::endl;

    std::cout << "integral_4::number_of_real_parameters " << integral_4::number_of_real_parameters << std::endl;
    std::cout << "integral_4::names_of_real_parameters ";
    for ( const auto& name : integral_4::names_of_real_parameters )
        std::cout << " " << name;
    std::cout << std::endl;

    std::cout << "integral_4::number_of_complex_parameters " << integral_4::number_of_complex_parameters << std::endl;
    std::cout << "integral_4::names_of_complex_parameters ";
    for ( const auto& name : integral_4::names_of_complex_parameters )
        std::cout << " " << name;
    std::cout << std::endl;

    std::cout << "integral_4::lowest_orders";
    for ( const auto& lowest_order : integral_4::lowest_orders )
        std::cout << " " << lowest_order;
    std::cout << std::endl;

    std::cout << "integral_4::highest_orders";
    for ( const auto& highest_order : integral_4::highest_orders )
        std::cout << " " << highest_order;
    std::cout << std::endl;

    std::cout << "integral_4::lowest_prefactor_orders";
    for ( const auto& highest_order : integral_4::lowest_prefactor_orders )
        std::cout << " " << highest_order;
    std::cout << std::endl;

    std::cout << "integral_4::highest_prefactor_orders";
    for ( const auto& highest_order : integral_4::highest_prefactor_orders )
        std::cout << " " << highest_order;
    std::cout << std::endl;

    std::cout << "integral_4::requested_orders";
    for ( const auto& requested_order : integral_4::requested_orders )
        std::cout << " " << requested_order;
    std::cout << std::endl;
}

int main(int argc, const char *argv[])
{
    // Check the command line argument number
    if (argc != 1 + 8 + 2*0) {
        std::cout << "usage: " << argv[0];
        for ( const auto& name : integral_4::names_of_real_parameters )
            std::cout << " " << name;
        for ( const auto& name : integral_4::names_of_complex_parameters )
            std::cout << " re(" << name << ") im(" << name << ")";
        std::cout << std::endl;
        return 1;
    }

    std::vector<integral_4::real_t> real_parameters; // = { real parameter values ("PP","P1P1","P2P2","M1sq","M2sq","M3sq","R2MID","LL") go here };
    std::vector<integral_4::complex_t> complex_parameters; // = { complex parameter values () go here };

    // Load parameters from the command line arguments
    for (int i = 1; i < 1 + 8; i++)
        real_parameters.push_back(integral_4::real_t(std::atof(argv[i])));

    for (int i = 1 + 8; i < 1 + 8 + 2*0; i += 2) {
        integral_4::real_t re = std::atof(argv[i]);
        integral_4::real_t im = std::atof(argv[i+1]);
        complex_parameters.push_back(integral_4::complex_t(re, im));
    }

    // Generate the integrands (optimisation of the contour if applicable)
    std::cerr << "Generating integrands (optimising contour if required)" << std::endl;
    const std::vector<integral_4::nested_series_t<integral_4::cuda_integrand_t>> sector_integrands =
        integral_4::make_cuda_integrands(real_parameters, complex_parameters);

    // Add integrands of sectors (together flag)
    std::cerr << "Summing integrands" << std::endl;
    const integral_4::nested_series_t<integral_4::cuda_together_integrand_t> all_sectors =
        std::accumulate(++sector_integrands.begin(), sector_integrands.end(), integral_4::cuda_together_integrand_t()+*sector_integrands.begin());

    // Integrate
    std::cerr << "Integrating" << std::endl;
    secdecutil::integrators::Qmc<
                                    integral_4::integrand_return_t,
                                    integral_4::maximal_number_of_integration_variables,
                                    integrators::transforms::Korobov<3>::type,
                                    integral_4::cuda_together_integrand_t
                                > integrator;
    integrator.verbosity = 1;
    const integral_4::nested_series_t<secdecutil::UncorrelatedDeviation<integral_4::integrand_return_t>> result_all =
        secdecutil::deep_apply( all_sectors, integrator.integrate );

    std::cout << "------------" << std::endl << std::endl;

    std::cout << "-- integral info -- " << std::endl;
    print_integral_info();
    std::cout << std::endl;

    std::cout << "-- integral without prefactor -- " << std::endl;
    std::cout << result_all << std::endl << std::endl;

    std::cout << "-- prefactor -- " << std::endl;
    const integral_4::nested_series_t<integral_4::integrand_return_t> prefactor =
        integral_4::prefactor(real_parameters, complex_parameters);
    std::cout << prefactor << std::endl << std::endl;

    std::cout << "-- full result (prefactor*integral) -- " << std::endl;
    std::cout << prefactor*result_all << std::endl;

    return 0;
}
