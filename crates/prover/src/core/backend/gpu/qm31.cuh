
// qm31.cuh
#ifndef QM31_H
#define QM31_H

#include "m31.cuh"

__device__ void mul_qm31(unsigned int *lhs, unsigned int *rhs, unsigned int *out);
__device__ void mul_cm31(unsigned int *lhs, unsigned int *rhs, unsigned int *out);

#endif 

__device__  void mul_cm31(unsigned int *lhs,  unsigned int *rhs,  unsigned int *out) {
    unsigned int ac = mul_m31(lhs[0], rhs[0]);
    unsigned int bd = mul_m31(lhs[1], rhs[1]);

    unsigned int ab_t_cd = mul_m31(add_m31(lhs[0], lhs[1]), add_m31(rhs[0], rhs[1])); 
    out[0] = sub_m31(ac, bd); 
    out[1] = sub_m31(ab_t_cd, add_m31(ac, bd)); 
}

__device__  void mul_qm31(unsigned int *lhs, unsigned int *rhs, unsigned int *out) {
    unsigned int ac[2];
    unsigned int bd[2];
    unsigned int bd_times_1_plus_i[2];
    unsigned int ac_p_bd[2];
    unsigned int ad_p_bc[2];
    unsigned int l[2];

    mul_cm31(lhs, rhs, ac);
    mul_cm31(lhs + 2, rhs + 2, bd);

    bd_times_1_plus_i[0] = sub_m31(bd[0], bd[1]);
    bd_times_1_plus_i[1] = add_m31(bd[0], bd[1]);

    ac_p_bd[0] = add_m31(ac[0], bd[0]);
    ac_p_bd[1] = add_m31(ac[1], bd[1]);

    unsigned int lhs_a_plus_b[2];
    unsigned int rhs_a_plus_b[2];
    unsigned int res[2]; 

    lhs_a_plus_b[0] = add_m31(lhs[0], lhs[2]);
    lhs_a_plus_b[1] = add_m31(lhs[1], lhs[3]);

    rhs_a_plus_b[0] = add_m31(rhs[0], rhs[2]);
    rhs_a_plus_b[1] = add_m31(rhs[1], rhs[3]);

    mul_cm31(lhs_a_plus_b, rhs_a_plus_b, res);

    ad_p_bc[0] = sub_m31(res[0], ac_p_bd[0]);
    ad_p_bc[1] = sub_m31(res[1], ac_p_bd[1]);

    l[0] = add_m31(ac_p_bd[0], bd_times_1_plus_i[0]);
    l[1] = add_m31(ac_p_bd[1], bd_times_1_plus_i[1]);
    
    out[0] = l[0];
    out[1] = l[1]; 
    out[2] = ad_p_bc[0];
    out[3] = ad_p_bc[1];
}