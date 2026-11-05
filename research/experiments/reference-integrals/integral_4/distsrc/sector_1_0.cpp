#define SECDEC_RESULT_IS_COMPLEX 0
#include "common_cpu.h"

#define SecDecInternalSignCheckPositivePolynomial(cond, id) if (unlikely(cond)) {*presult = REAL_NAN; return 1; }
#define SecDecInternalSignCheckContourDeformation(cond, id) if (unlikely(cond)) {*presult = REAL_NAN; return 2; }

extern "C" int
integral_4__sector_1_order_0(
    result_t * restrict presult,
    const uint64_t lattice,
    const uint64_t index1,
    const uint64_t index2,
    const uint64_t * restrict genvec,
    const real_t * restrict shift,
    const real_t * restrict realp,
    const complex_t * restrict complexp,
    const real_t * restrict deformp
)
{
    const real_t PP = realp[0]; (void)PP;
    const real_t P1P1 = realp[1]; (void)P1P1;
    const real_t P2P2 = realp[2]; (void)P2P2;
    const real_t M1sq = realp[3]; (void)M1sq;
    const real_t M2sq = realp[4]; (void)M2sq;
    const real_t M3sq = realp[5]; (void)M3sq;
    const real_t R2MID = realp[6]; (void)R2MID;
    const real_t LL = realp[7]; (void)LL;
    const real_t invlattice = SecDecInternalDenominator((real_t)(double)lattice);
    resultvec_t acc = RESULTVEC_ZERO;
    uint64_t index = index1;
    int_t li_x1 = mulmod(genvec[0], index, lattice);
    int_t li_x2 = mulmod(genvec[1], index, lattice);
    int_t li_x3 = mulmod(genvec[2], index, lattice);
    for (; index < index2; index += 4) {
        int_t li_x1_0 = li_x1; li_x1 = warponce_i(li_x1 + genvec[0], lattice);
        int_t li_x1_1 = li_x1; li_x1 = warponce_i(li_x1 + genvec[0], lattice);
        int_t li_x1_2 = li_x1; li_x1 = warponce_i(li_x1 + genvec[0], lattice);
        int_t li_x1_3 = li_x1; li_x1 = warponce_i(li_x1 + genvec[0], lattice);
        realvec_t x1 = {{ li_x1_0*invlattice, li_x1_1*invlattice, li_x1_2*invlattice, li_x1_3*invlattice }};
        x1 = warponce(x1 + shift[0], 1);
        int_t li_x2_0 = li_x2; li_x2 = warponce_i(li_x2 + genvec[1], lattice);
        int_t li_x2_1 = li_x2; li_x2 = warponce_i(li_x2 + genvec[1], lattice);
        int_t li_x2_2 = li_x2; li_x2 = warponce_i(li_x2 + genvec[1], lattice);
        int_t li_x2_3 = li_x2; li_x2 = warponce_i(li_x2 + genvec[1], lattice);
        realvec_t x2 = {{ li_x2_0*invlattice, li_x2_1*invlattice, li_x2_2*invlattice, li_x2_3*invlattice }};
        x2 = warponce(x2 + shift[1], 1);
        int_t li_x3_0 = li_x3; li_x3 = warponce_i(li_x3 + genvec[2], lattice);
        int_t li_x3_1 = li_x3; li_x3 = warponce_i(li_x3 + genvec[2], lattice);
        int_t li_x3_2 = li_x3; li_x3 = warponce_i(li_x3 + genvec[2], lattice);
        int_t li_x3_3 = li_x3; li_x3 = warponce_i(li_x3 + genvec[2], lattice);
        realvec_t x3 = {{ li_x3_0*invlattice, li_x3_1*invlattice, li_x3_2*invlattice, li_x3_3*invlattice }};
        x3 = warponce(x3 + shift[2], 1);
        auto w_x1 = korobov3x3_w(x1);
        auto w_x2 = korobov3x3_w(x2);
        auto w_x3 = korobov3x3_w(x3);
        realvec_t w = w_x1*w_x2*w_x3;
        if (unlikely(index + 1 >= index2)) w.x[1] = 0;
        if (unlikely(index + 2 >= index2)) w.x[2] = 0;
        if (unlikely(index + 3 >= index2)) w.x[3] = 0;
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
        acc = acc + w*(tmp3_35);
    }
    *presult = componentsum(acc);
    return 0;
}

