#include <secdecutil/series.hpp>
#include <secdecutil/uncertainties.hpp>
#include <vector>

#include "integral_2.hpp"
#include "functions.hpp"

namespace integral_2
{
    nested_series_t<integrand_return_t> prefactor(const std::vector<real_t>& real_parameters, const std::vector<complex_t>& complex_parameters)
    {
        #define PP real_parameters.at(0)
        #define P1P1 real_parameters.at(1)
        #define P2P2 real_parameters.at(2)
        #define M1sq real_parameters.at(3)
        #define M2sq real_parameters.at(4)
        #define M3sq real_parameters.at(5)
        #define R2MID real_parameters.at(6)
        #define LL real_parameters.at(7)
        return {0,0,{{1.0}},true,"eps"};
        #undef PP
        #undef P1P1
        #undef P2P2
        #undef M1sq
        #undef M2sq
        #undef M3sq
        #undef R2MID
        #undef LL
    }
};
