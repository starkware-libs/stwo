// #include "m31.cuh"

// extern "C" __device__  void mul_cm31(unsigned int *lhs,  unsigned int *rhs,  unsigned int *out) {
//     unsigned int ac = mul_m31(lhs[0], rhs[0]);
//     unsigned int bd = mul_m31(lhs[1], rhs[1]);

//     unsigned int ab_t_cd = mul_m31(add_m31(lhs[0], lhs[1]), add_m31(rhs[0], rhs[1])); 
//     out[0] = sub_m31(ac, bd); 
//     out[1] = sub_m31(ab_t_cd, add_m31(ac, bd)); 
// }

// extern "C" __device__  void mul_qm31(unsigned int *lhs, unsigned int *rhs, unsigned int *out) {
//     unsigned int ac[2];
//     unsigned int bd[2];
//     unsigned int bd_times_1_plus_i[2];
//     unsigned int ac_p_bd[2];
//     unsigned int ad_p_bc[2];
//     unsigned int l[2];

//     mul_cm31(lhs, rhs, ac);
//     mul_cm31(&(lhs[2]), &(rhs[2]), bd);

//     bd_times_1_plus_i[0] = sub_m31(bd[0], bd[1]);
//     bd_times_1_plus_i[1] = add_m31(bd[0], bd[1]);

//     ac_p_bd[0] = add_m31(ac[0], bd[0]);
//     ac_p_bd[1] = add_m31(ac[1], bd[1]);

//     unsigned int temp1[2];
//     unsigned int temp2[2];
//     temp1[0] = add_m31(lhs[0], lhs[2]);
//     temp1[1] = add_m31(lhs[1], lhs[3]);
//     temp2[0] = add_m31(rhs[0], rhs[2]);
//     temp2[1] = add_m31(rhs[1], rhs[3]);
//     temp2[0] = sub_m31(temp2[0], ac_p_bd[0]);
//     temp2[1] = sub_m31(temp2[1], ac_p_bd[1]);
//     mul_cm31(temp1, temp2, ad_p_bc);

//     l[0] = add_m31(ac_p_bd[0], bd_times_1_plus_i[0]);
//     l[1] = add_m31(ac_p_bd[1], bd_times_1_plus_i[1]);
    
//     out[0] = l[0];
//     out[1] = l[1]; 
//     out[2] = ad_p_bc[0];
//     out[2] = ad_p_bc[1];
// }

extern "C" __global__ void mul(unsigned int *lhs, unsigned int *rhs, unsigned int *out, int size) {
    // unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if (tid < size) {
    //     unsigned int idx = tid * 4; 
    //     mul_qm31(lhs + idx, rhs + idx, out + idx);
    //}
}

// // Compute using Karatsuba.
//         // //   (a + ub) * (c + ud) =
//         // //   (ac + (2+i)bd) + (ad + bc)u =
//         // //   ac + 2bd + ibd + (ad + bc)u.
//         // let ac = self.a() * rhs.a();
//         // let bd = self.b() * rhs.b();
//         // let bd_times_1_plus_i = PackedCM31([bd.a() - bd.b(), bd.a() + bd.b()]);
//         // // Computes ac + bd.
//         // let ac_p_bd = ac + bd;
//         // // Computes ad + bc.
//         // let ad_p_bc = (self.a() + self.b()) * (rhs.a() + rhs.b()) - ac_p_bd;
//         // // ac + 2bd + ibd =
//         // // ac + bd + bd + ibd
//         // let l = PackedCM31([
//         //     ac_p_bd.a() + bd_times_1_plus_i.a(),
//         //     ac_p_bd.b() + bd_times_1_plus_i.b(),
//         // ]);
//         // Self([l, ad_p_bc])


// // Compute using Karatsuba.
// fn mul_cm31(
//     lhs_a: PackedM31,
//     lhs_b: PackedM31,
//     rhs_a: PackedM31,
//     rhs_b: PackedM31,
// ) -> (PackedM31, PackedM31) {
//     let ac = lhs_a * rhs_a;
//     let bd = lhs_b * rhs_b;
//     // Computes (a + b) * (c + d).
//     let ab_t_cd = (lhs_a + lhs_b) * (rhs_a + rhs_b);
//     // (ac - bd) + (ad + bc)i.
//     (ac - bd, ab_t_cd - ac - bd)
// }