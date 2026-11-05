#ifndef integral_4_codegen_sector_1_0_hpp_included
#define integral_4_codegen_sector_1_0_hpp_included
#include "integral_4.hpp"
#include "functions.hpp"
namespace integral_4
{
#ifdef SECDEC_WITH_CUDA
__host__ __device__
#endif
secdecutil::SectorContainerWithoutDeformation<real_t, complex_t, integrand_return_t>::IntegrandFunction sector_1_order_0_integrand;
#ifdef SECDEC_WITH_CUDA
secdecutil::SectorContainerWithoutDeformation<real_t, complex_t, integrand_return_t>::IntegrandFunction* get_device_sector_1_order_0_integrand();
#endif
}
#endif
