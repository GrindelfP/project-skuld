#include <secdecutil/series.hpp>

#include "sector_1_0.hpp"

namespace integral_6
{
nested_series_t<sector_container_t> get_integrand_of_sector_1()
{
return {0,0,{{1,{0},3,sector_1_order_0_integrand
#ifdef SECDEC_WITH_CUDA
,get_device_sector_1_order_0_integrand
#endif
}},true,"eps"};
}

}
