#include "sector_1_0.hpp"
namespace integral_4
{
#ifdef SECDEC_WITH_CUDA
__host__ __device__
#endif
integrand_return_t sector_1_order_0_integrand
(
    real_t const * restrict const integration_variables,
    real_t const * restrict const real_parameters,
    complex_t const * restrict const complex_parameters,
    secdecutil::ResultInfo * restrict const result_info
)
{
    const auto x1 = integration_variables[0]; (void)x1;
    const auto x2 = integration_variables[1]; (void)x2;
    const auto x3 = integration_variables[2]; (void)x3;
    const auto PP = real_parameters[0]; (void)PP;
    const auto P1P1 = real_parameters[1]; (void)P1P1;
    const auto P2P2 = real_parameters[2]; (void)P2P2;
    const auto M1sq = real_parameters[3]; (void)M1sq;
    const auto M2sq = real_parameters[4]; (void)M2sq;
    const auto M3sq = real_parameters[5]; (void)M3sq;
    const auto R2MID = real_parameters[6]; (void)R2MID;
    const auto LL = real_parameters[7]; (void)LL;
    auto tmp1_1 = LL*x1;
    auto tmp1_2 = P2P2-M3sq;
    auto tmp1_3 = 2*P1P1;
    auto tmp1_4 = -PP + M2sq + tmp1_2 + tmp1_3;
    auto tmp1_5 = R2MID-tmp1_4;
    auto tmp3_1 = x3*tmp1_5;
    auto tmp1_6 = -P1P1 + M3sq-M2sq;
    auto tmp3_2 = tmp3_1-tmp1_6;
    auto tmp3_3 = x2*tmp3_2;
    auto tmp3_4 = tmp1_2 + M1sq;
    auto tmp3_5 = tmp3_4*x3;
    auto tmp3_6 = tmp3_5 + M3sq;
    auto tmp3_7 = tmp3_3 + tmp3_6;
    auto tmp1_7 = P1P1 + P2P2-PP;
    auto tmp1_8 = -R2MID + tmp1_7;
    auto tmp3_8 = x2*tmp1_8;
    auto tmp3_9 = -2*P2P2 + tmp3_8;
    auto tmp1_9 = tmp1_3*x3;
    auto tmp3_10 = tmp1_9-P1P1;
    auto tmp1_10 = 2*tmp3_10;
    auto tmp1_11 = x2*R2MID;
    auto tmp1_12 = -x3*tmp1_11;
    auto tmp3_11 = P2P2 + tmp1_11;
    auto tmp3_12 = x3*tmp1_4;
    auto tmp3_13 = tmp3_12 + tmp1_6;
    auto tmp3_14 = x2*tmp3_13;
    auto tmp3_15 = tmp3_14-tmp3_6;
    auto tmp3_16 = -x2*tmp1_7;
    auto tmp3_17 = P2P2 + tmp3_16;
    auto tmp3_18 = -2*tmp1_1;
    auto tmp3_19 = x3-1;
    auto tmp3_20 = LL*tmp3_19;
    auto tmp1_13 = 2*tmp3_20;
    auto _expCall1 = exp(tmp3_18);
    auto _expCall3 = exp(tmp1_1);
    auto _expCall4 = exp(-tmp1_1);
    auto __PowCall1 = SecDecInternalSqr(x2);
    auto __PowCall2 = SecDecInternalSqr(x3);
    auto tmp2_10 = _expCall4*tmp3_11;
    auto tmp2_11 = _expCall3*tmp3_17;
    auto tmp2_12 = _expCall3 + _expCall4;
    auto tmp2_13 = P1P1*tmp2_12;
    auto tmp3_21 = -tmp1_3 + tmp2_13;
    auto tmp3_22 = __PowCall1*tmp3_21;
    auto tmp3_23 = tmp3_22 + tmp2_11 + tmp3_9 + tmp2_10;
    auto tmp3_24 = __PowCall2*tmp3_23;
    auto tmp3_25 = _expCall4*tmp1_12;
    auto tmp3_26 = _expCall3*tmp3_15;
    auto tmp3_27 = -tmp3_10*tmp2_12;
    auto tmp3_28 = tmp1_10 + tmp3_27;
    auto tmp3_29 = __PowCall1*tmp3_28;
    auto tmp3_30 = tmp3_24 + tmp3_29 + tmp3_26 + tmp3_7 + tmp3_25;
    auto _expCall2 = exp(tmp3_30);
    auto __PowCall3 = SecDecInternalSqr(_expCall3);
    auto tmp3_31 = -1-__PowCall3;
    auto tmp3_32 = tmp3_20*tmp3_31;
    auto tmp3_33 = _expCall3*tmp1_13;
    auto tmp3_34 = tmp3_33 + tmp3_32;
    auto tmp3_35 = tmp3_34*_expCall1*_expCall2;
    return(tmp3_35);
}
#ifdef SECDEC_WITH_CUDA
__device__ secdecutil::SectorContainerWithoutDeformation<real_t, complex_t, integrand_return_t>::IntegrandFunction* const device_sector_1_order_0_integrand = sector_1_order_0_integrand;
secdecutil::SectorContainerWithoutDeformation<real_t, complex_t, integrand_return_t>::IntegrandFunction* get_device_sector_1_order_0_integrand()
{
    using IntegrandFunction = secdecutil::SectorContainerWithoutDeformation<real_t, complex_t, integrand_return_t>::IntegrandFunction;
    IntegrandFunction* device_address_on_host;
    auto errcode = cudaMemcpyFromSymbol(&device_address_on_host,device_sector_1_order_0_integrand, sizeof(IntegrandFunction*));
    if (errcode != cudaSuccess) throw secdecutil::cuda_error( cudaGetErrorString(errcode) );
    return device_address_on_host;
}
#endif
}
