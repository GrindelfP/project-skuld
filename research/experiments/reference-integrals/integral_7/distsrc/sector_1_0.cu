#define SECDEC_RESULT_IS_COMPLEX 0
#include "common_cuda.h"

#define SecDecInternalSignCheckPositivePolynomial(cond, id) if (unlikely(cond)) {val = REAL_NAN; break;}
#define SecDecInternalSignCheckContourDeformation(cond, id) if (unlikely(cond)) {val = REAL_NAN; break;}

extern "C" __global__ void
integral_7__sector_1_order_0(
    result_t * __restrict__ result,
    const uint64_t lattice,
    const uint64_t index1,
    const uint64_t index2,
    const uint64_t * __restrict__ genvec,
    const real_t * __restrict__ shift,
    const real_t * __restrict__ realp,
    const complex_t * __restrict__ complexp,
    const real_t * __restrict__ deformp
)
{
    // assert(blockDim.x == 128);
    const uint64_t bid = blockIdx.x;
    const uint64_t tid = threadIdx.x;
    const real_t PP = realp[0]; (void)PP;
    const real_t P1P1 = realp[1]; (void)P1P1;
    const real_t P2P2 = realp[2]; (void)P2P2;
    const real_t M1sq = realp[3]; (void)M1sq;
    const real_t M2sq = realp[4]; (void)M2sq;
    const real_t M3sq = realp[5]; (void)M3sq;
    const real_t R2MID = realp[6]; (void)R2MID;
    const real_t LL = realp[7]; (void)LL;
    const real_t invlattice = SecDecInternalDenominator((real_t)(double)lattice);
    result_t val = (result_t)0;
    uint64_t index = index1 + (bid*128 + tid)*8;
    uint64_t li_x1 = mulmod(index, genvec[0], lattice);
    uint64_t li_x2 = mulmod(index, genvec[1], lattice);
    uint64_t li_x3 = mulmod(index, genvec[2], lattice);
    for (uint64_t i = 0; (i < 8) && (index < index2); i++, index++) {
        real_t x1 = warponce(invlattice*(double)li_x1 + shift[0], 1.0);
        li_x1 = warponce_i(li_x1 + genvec[0], lattice);
        real_t x2 = warponce(invlattice*(double)li_x2 + shift[1], 1.0);
        li_x2 = warponce_i(li_x2 + genvec[1], lattice);
        real_t x3 = warponce(invlattice*(double)li_x3 + shift[2], 1.0);
        li_x3 = warponce_i(li_x3 + genvec[2], lattice);
        real_t w_x1 = korobov3x3_w(x1);
        real_t w_x2 = korobov3x3_w(x2);
        real_t w_x3 = korobov3x3_w(x3);
        real_t w = w_x1*w_x2*w_x3;
        x1 = korobov3x3_f(x1);
        x2 = korobov3x3_f(x2);
        x3 = korobov3x3_f(x3);
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
        auto tmp3_18 = 2*LL;
        auto tmp3_19 = -x1*tmp3_18;
        auto tmp3_20 = x2*LL;
        auto tmp1_13 = tmp3_20*x3;
        auto tmp1_14 = 2*tmp3_20;
        auto tmp1_15 = 2*x3;
        auto tmp3_21 = -tmp3_20*tmp1_15;
        auto tmp1_16 = 4*tmp3_20;
        auto _expCall1 = exp(tmp3_19);
        auto _expCall3 = exp(tmp1_1);
        auto _expCall4 = exp(-tmp1_1);
        auto __PowCall1 = SecDecInternalSqr(x2);
        auto __PowCall2 = SecDecInternalSqr(x3);
        auto __PowCall3 = SecDecInternalSqr(x3)*x3;
        auto tmp2_10 = _expCall4*tmp3_11;
        auto tmp2_11 = _expCall3*tmp3_17;
        auto tmp2_12 = _expCall3 + _expCall4;
        auto tmp2_13 = P1P1*tmp2_12;
        auto tmp3_22 = -tmp1_3 + tmp2_13;
        auto tmp3_23 = __PowCall1*tmp3_22;
        auto tmp3_24 = tmp3_23 + tmp2_11 + tmp3_9 + tmp2_10;
        auto tmp3_25 = __PowCall2*tmp3_24;
        auto tmp3_26 = _expCall4*tmp1_12;
        auto tmp3_27 = _expCall3*tmp3_15;
        auto tmp3_28 = -tmp3_10*tmp2_12;
        auto tmp3_29 = tmp1_10 + tmp3_28;
        auto tmp3_30 = __PowCall1*tmp3_29;
        auto tmp3_31 = tmp3_25 + tmp3_30 + tmp3_27 + tmp3_7 + tmp3_26;
        auto _expCall2 = exp(tmp3_31);
        auto __PowCall4 = SecDecInternalSqr(_expCall3);
        auto tmp3_32 = -__PowCall2*tmp1_14;
        auto tmp3_33 = tmp3_20*__PowCall3;
        auto tmp3_34 = tmp3_32 + tmp1_13 + tmp3_33;
        auto tmp3_35 = __PowCall4 + 1;
        auto tmp3_36 = _expCall1*_expCall2;
        auto tmp3_37 = tmp3_34*tmp3_36*tmp3_35;
        auto tmp3_38 = -__PowCall3*tmp1_14;
        auto tmp3_39 = tmp1_16*__PowCall2;
        auto tmp3_40 = tmp3_39 + tmp3_21 + tmp3_38;
        auto tmp3_41 = tmp3_36*_expCall3*tmp3_40;
        auto tmp3_42 = tmp3_41 + tmp3_37;
        val = val + w*(tmp3_42);
    }
    // Sum up 128*8=1024 values across 4 warps.
    typedef cub::BlockReduce<result_t, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> Reduce;
    __shared__ typename Reduce::TempStorage shared;
    result_t sum = Reduce(shared).Sum(val);
    if (tid == 0) result[bid] = sum;
}
