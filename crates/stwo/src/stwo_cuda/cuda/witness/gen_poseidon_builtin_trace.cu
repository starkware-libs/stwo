// CUDA trace generation for poseidon_builtin component
// 341 trace columns, multiple lookups for Poseidon hash computation
//
// Lookups:
// - 6 memory_address_to_id (2 elements each)
// - 6 memory_id_to_big (29 elements each)
// - 8 poseidon_full_round_chain (32 elements each)
// - 2 range_check_felt_252_width_27 (10 elements each)
// - 2 cube_252 (20 elements each)
// - 2 range_check_3_3_3_3_3 (5 elements each)
// - 6 range_check_4_4_4_4 (4 elements each)
// - 3 range_check_4_4 (2 elements each)
// - 27 poseidon_3_partial_rounds_chain (42 elements each)

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "../constraints/relations.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
// Note: Not including poseidon252_constants.cuh to avoid link conflicts
// We define our own POSEIDON_WIDTH27_ROUND_KEYS below
#include <cstdint>
#include <cstdio>

// ============================================================================
// Constants
// ============================================================================

#define N_TRACE_COLUMNS 341
#define N_LOGUP_COLS 17  // Number of logup interaction columns (0-16)
#define BLOCK_SIZE 256
#define POSEIDON_BUILTIN_MEMORY_CELLS 6  // 3 input + 3 output state elements

// Poseidon constants
#define N_STATE 3
#define N_HALF_FULL_ROUNDS 4
#define N_PARTIAL_ROUNDS 83

// M31 constants for Poseidon computation
#define M31_0 {0}
#define M31_1 {1}
#define M31_2 {2}
#define M31_3 {3}
#define M31_4 {4}
#define M31_5 {5}
#define M31_6 {6}
#define M31_512 {512}
#define M31_262144 {262144}
#define M31_8192 {8192}
#define M31_16 {16}
#define M31_27 {27}
#define M31_28 {28}
#define M31_29 {29}
#define M31_30 {30}
#define M31_31 {31}
#define M31_32 {32}
#define M31_33 {33}
#define M31_34 {34}
#define M31_35 {35}

// M31 constant values for use in expressions
__device__ const m31 M31_0_val = {0};
__device__ const m31 M31_1_val = {1};
__device__ const m31 M31_3_val = {3};
__device__ const m31 M31_4_val = {4};
__device__ const m31 M31_31_val = {31};

// ============================================================================
// LinearCombination bias constants (Width27 format)
// These are the bias values used in LinearCombinationN2Coefs11 for columns 87-119
// NOT the same as Poseidon round keys!
// ============================================================================
__device__ __constant__ uint32_t LINEAR_COMB_BIAS_0[10] = {
    74972783, 117420501, 112795138, 91013252, 60709090,
    44848225, 108487870, 44781849, 102193642, 208
};
__device__ __constant__ uint32_t LINEAR_COMB_BIAS_1[10] = {
    41224388, 90391646, 36279186, 129717753, 94624323,
    75104388, 133303902, 48945103, 41320857, 112
};
__device__ __constant__ uint32_t LINEAR_COMB_BIAS_2[10] = {
    4883209, 28820206, 79012328, 49157069, 78826183,
    72285071, 33413160, 90842759, 60124463, 116
};

// ============================================================================
// Bias constants for partial round linear combinations
// ============================================================================

// LinearCombinationN4Coefs11M21 (cols 160-170): 1*state0 + 1*state1 - 2*cube + BIAS
// BIAS = Felt252_11041071929982523380_7503192613203831446_4943121247101232560_560497091765764140
// In Width27 format:
__device__ __constant__ uint32_t LC_BIAS_N4_COEFS_1_1_M2_1[10] = {
    103094260, 121146754, 95050340, 16173996, 50758155,
    54415179, 19292069, 45351266, 122233508, 248
};

// LinearCombinationN4Coefs42M21 (cols 181-191): 4*state0 + 2*cube0 - 2*cube1 + BIAS
// From poseidon_hades_permutation.rs lines 669-678:
// M31_121657377, M31_112479959, M31_130418270, M31_4974792, M31_59852719,
// M31_120369218, M31_62439890, M31_50468641, M31_86573645, M31_154
__device__ __constant__ uint32_t LC_BIAS_N4_COEFS_4_2_M2_1[10] = {
    121657377, 112479959, 130418270, 4974792, 59852719,
    120369218, 62439890, 50468641, 86573645, 154
};

// ============================================================================
// Device helper functions for Poseidon state manipulation
// ============================================================================

// Felt252 with 27-bit limbs (10 limbs)
struct Felt252Width27 {
    m31 limbs[10];
};

// Pack 28 9-bit limbs into 10 27-bit limbs
__device__ void pack_felt252_to_width27(const m31* src_limbs, Felt252Width27& dst) {
    // Each group of 3 9-bit limbs (27 bits total) becomes one 27-bit limb
    for (int i = 0; i < 9; i++) {
        m31 l0 = src_limbs[i * 3];
        m31 l1 = src_limbs[i * 3 + 1];
        m31 l2 = src_limbs[i * 3 + 2];
        // dst[i] = l0 + l1 * 512 + l2 * 262144
        dst.limbs[i] = add(add(l0, mul(l1, (m31){512})), mul(l2, (m31){262144}));
    }
    // Last limb is just the 28th 9-bit limb
    dst.limbs[9] = src_limbs[27];
}

// Unpack 10 27-bit limbs back to 28 9-bit limbs
__device__ void unpack_felt252_from_width27(const Felt252Width27& src, m31* dst_limbs) {
    for (int i = 0; i < 9; i++) {
        m31 val = src.limbs[i];
        // Extract 3 9-bit values (m31 is uint32_t)
        dst_limbs[i * 3] = val & 0x1FF;
        dst_limbs[i * 3 + 1] = (val >> 9) & 0x1FF;
        dst_limbs[i * 3 + 2] = (val >> 18) & 0x1FF;
    }
    // Last limb
    dst_limbs[27] = src.limbs[9];
}

// ============================================================================
// Felt252Field - 256-bit field element for Poseidon computation
// Note: Named Felt252Field to avoid conflict with Felt252 in gen_memory_id_to_big_trace.cuh
// ============================================================================

// Type alias for 256-bit field element (Starknet field)
typedef ff_storage<8> Felt252Field;

// Felt252Field field operations using Starknet config
__device__ __forceinline__ Felt252Field felt_add(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

__device__ __forceinline__ Felt252Field felt_sub(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

__device__ __forceinline__ Felt252Field felt_mul(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

__device__ __forceinline__ Felt252Field felt_to_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

__device__ __forceinline__ Felt252Field felt_from_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

// Montgomery cube factor: 2^768 % PRIME (compensates for Montgomery form cubing)
// These values must match the Rust FELT252_MONT_CUBE_FACTOR from poseidon.rs
// Rust u64 values: [14731687596718420366, 8450283861232831494, 17617383518939119640, 256247204371237485]
// Converting to hex:
//   u64[0] = 0xCC7177D1406DF18E -> limbs[0] = 0x406DF18E, limbs[1] = 0xCC7177D1
//   u64[1] = 0x7545706677FFCC06 -> limbs[2] = 0x77FFCC06, limbs[3] = 0x75457066
//   u64[2] = 0xF47D84F836300018 -> limbs[4] = 0x36300018, limbs[5] = 0xF47D84F8
//   u64[3] = 0x038E5F79873C0A6D -> limbs[6] = 0x873C0A6D, limbs[7] = 0x038E5F79
__device__ __constant__ Felt252Field FELT252_MONT_CUBE_FACTOR = {{
    0x406DF18E, 0xCC7177D1, 0x77FFCC06, 0x75457066,
    0x36300018, 0xF47D84F8, 0x873C0A6D, 0x038E5F79
}};

// Cube operation with Montgomery compensation
// Input x is already in Montgomery form (the internal representation from Width27)
// We want to compute [x]^3 where [x] is the raw internal storage
// Using Montgomery mult: [x*y] = [x] * [y] * R^(-1)
// So: x^2 = [x]^2 * R^(-1), x^3 = [x]^3 * R^(-2)
// x^3 * FACTOR = [x]^3 * R^(-2) * R^3 * R^(-1) = [x]^3 (when FACTOR = R^3)
__device__ Felt252Field felt_cube(const Felt252Field& x) {
    Felt252Field x2 = felt_mul(x, x);
    Felt252Field x3 = felt_mul(x2, x);
    // Use a local copy of CUBE_FACTOR to avoid potential issues with __constant__ memory access
    Felt252Field local_factor = FELT252_MONT_CUBE_FACTOR;
    return felt_mul(x3, local_factor);
}

// Convert Width27 (4 x u64 storage) to Felt252Field (8 x u32)
// The round keys in POSEIDON_WIDTH27_ROUND_KEYS are stored as raw 256-bit values
// (the u64 array IS the raw field element representation, not Width27 packed format)
__device__ Felt252Field width27_to_felt252field(const uint64_t* width27) {
    // Simple reinterpretation: 4 x u64 -> 8 x u32 (little-endian)
    Felt252Field result;
    result.limbs[0] = (uint32_t)(width27[0] & 0xFFFFFFFF);
    result.limbs[1] = (uint32_t)((width27[0] >> 32) & 0xFFFFFFFF);
    result.limbs[2] = (uint32_t)(width27[1] & 0xFFFFFFFF);
    result.limbs[3] = (uint32_t)((width27[1] >> 32) & 0xFFFFFFFF);
    result.limbs[4] = (uint32_t)(width27[2] & 0xFFFFFFFF);
    result.limbs[5] = (uint32_t)((width27[2] >> 32) & 0xFFFFFFFF);
    result.limbs[6] = (uint32_t)(width27[3] & 0xFFFFFFFF);
    result.limbs[7] = (uint32_t)((width27[3] >> 32) & 0xFFFFFFFF);
    return result;
}

// Convert Felt252Field (8 x u32) to Width27 format (stored in 10 M31 limbs)
__device__ void felt252field_to_width27_m31(const Felt252Field& felt, Felt252Width27& dst) {
    // Extract 27-bit chunks from the 256-bit value
    // The 252-bit value is stored in 10 x 27-bit limbs (270 bits total)

    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    // Extract 27-bit limbs
    // Limb 0-3 from val0-val1 (lower 108 bits)
    dst.limbs[0] = (m31){(uint32_t)(val0 & 0x7FFFFFF)};           // bits 0-26
    dst.limbs[1] = (m31){(uint32_t)((val0 >> 27) & 0x7FFFFFF)};   // bits 27-53

    uint64_t cross01 = (val0 >> 54) | (val1 << 10);
    dst.limbs[2] = (m31){(uint32_t)(cross01 & 0x7FFFFFF)};        // bits 54-80

    dst.limbs[3] = (m31){(uint32_t)((val1 >> 17) & 0x7FFFFFF)};   // bits 81-107

    uint64_t cross12 = (val1 >> 44) | (val2 << 20);
    dst.limbs[4] = (m31){(uint32_t)(cross12 & 0x7FFFFFF)};        // bits 108-134

    dst.limbs[5] = (m31){(uint32_t)((val2 >> 7) & 0x7FFFFFF)};    // bits 135-161

    dst.limbs[6] = (m31){(uint32_t)((val2 >> 34) & 0x7FFFFFF)};   // bits 162-188

    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    dst.limbs[7] = (m31){(uint32_t)(cross23 & 0x7FFFFFF)};        // bits 189-215

    dst.limbs[8] = (m31){(uint32_t)((val3 >> 24) & 0x7FFFFFF)};   // bits 216-242

    dst.limbs[9] = (m31){(uint32_t)((val3 >> 51) & 0x1FF)};       // bits 243-251 (9 bits for last limb)
}

// Convert 10 M31 Width27 limbs to Felt252Field
__device__ Felt252Field width27_m31_to_felt252field(const Felt252Width27& src) {
    // Reconstruct 256-bit value from 10 x 27-bit limbs
    // Note: Last limb is only 9 bits (total = 9*27 + 9 = 252 bits)

    uint64_t val0 = 0, val1 = 0, val2 = 0, val3 = 0;

    // Pack limbs into 64-bit values
    val0 = (uint64_t)src.limbs[0];
    val0 |= ((uint64_t)src.limbs[1]) << 27;
    val0 |= ((uint64_t)src.limbs[2]) << 54;  // 10 bits overflow to val1

    val1 = ((uint64_t)src.limbs[2]) >> 10;   // remaining 17 bits of limb2
    val1 |= ((uint64_t)src.limbs[3]) << 17;
    val1 |= ((uint64_t)src.limbs[4]) << 44;  // 20 bits overflow to val2

    val2 = ((uint64_t)src.limbs[4]) >> 20;   // remaining 7 bits of limb4
    val2 |= ((uint64_t)src.limbs[5]) << 7;
    val2 |= ((uint64_t)src.limbs[6]) << 34;
    val2 |= ((uint64_t)src.limbs[7]) << 61;  // 3 bits overflow to val3

    val3 = ((uint64_t)src.limbs[7]) >> 3;    // remaining 24 bits of limb7
    val3 |= ((uint64_t)src.limbs[8]) << 24;
    val3 |= ((uint64_t)src.limbs[9]) << 51;  // last 9-bit limb

    Felt252Field result;
    result.limbs[0] = (uint32_t)(val0 & 0xFFFFFFFF);
    result.limbs[1] = (uint32_t)((val0 >> 32) & 0xFFFFFFFF);
    result.limbs[2] = (uint32_t)(val1 & 0xFFFFFFFF);
    result.limbs[3] = (uint32_t)((val1 >> 32) & 0xFFFFFFFF);
    result.limbs[4] = (uint32_t)(val2 & 0xFFFFFFFF);
    result.limbs[5] = (uint32_t)((val2 >> 32) & 0xFFFFFFFF);
    result.limbs[6] = (uint32_t)(val3 & 0xFFFFFFFF);
    result.limbs[7] = (uint32_t)((val3 >> 32) & 0xFFFFFFFF);

    return result;
}

// Convert bias constants array to Felt252Field
__device__ Felt252Field bias_to_felt252field(const uint32_t* bias) {
    Felt252Width27 w27;
    for (int i = 0; i < 10; i++) {
        w27.limbs[i] = (m31){bias[i]};
    }
    return width27_m31_to_felt252field(w27);
}

// Compute linear combination: combination = input + bias (in Felt252)
// Also computes p_coef based on carry from first limb
// Input: input_width27 (10 Width27 limbs), bias_width27 (10 Width27 limbs from constants)
// Output: combination_width27 (10 Width27 limbs), p_coef (carry coefficient)
__device__ void compute_linear_combination_with_p_coef(
    const Felt252Width27& input_width27,
    const uint32_t* bias,
    Felt252Width27& combination_width27,
    m31& p_coef
) {
    // Convert input and bias to Felt252Field
    Felt252Field input_felt = width27_m31_to_felt252field(input_width27);
    Felt252Field bias_felt = bias_to_felt252field(bias);

    // Compute combination = input + bias in Felt252 field
    Felt252Field combination_felt = felt_add(input_felt, bias_felt);

    // Convert result back to Width27
    felt252field_to_width27_m31(combination_felt, combination_width27);

    // Compute p_coef from carry of first limb:
    // p_coef = ((input_limb_0 + bias_limb_0 - combination_limb_0 + 134217729) & 0xFFFF) - 1
    // Where 134217729 = 2^27 + 1 is an offset to ensure the value is always positive
    uint32_t input_0 = input_width27.limbs[0];
    uint32_t bias_0 = bias[0];
    uint32_t combo_0 = combination_width27.limbs[0];
    uint32_t biased = input_0 + bias_0 - combo_0 + 134217729;
    uint32_t low_16 = biased & 0xFFFF;
    p_coef = (m31){low_16 - 1};
}

// ============================================================================
// Poseidon round key constants in Width27 format (4 x u64 each)
// 35 rounds, 3 constants per round
// Generated from CPU poseidon_round_keys.rs
// ============================================================================
__device__ __constant__ uint64_t POSEIDON_WIDTH27_ROUND_KEYS[35][3][4] = {
    // Round 0 (full round)
    {{9808894619969057997ULL, 2962375666393338310ULL, 17382841788414994265ULL, 443257643709112289ULL},
     {12537484503666775718ULL, 3256805997184644908ULL, 6617722259049010207ULL, 543112534054733059ULL},
     {1454046077829682943ULL, 14331133962181073949ULL, 2327346812919484995ULL, 379005027604567203ULL}},
    // Round 1 (full round)
    {{316912300309518807ULL, 9546737057323600779ULL, 4990939959663297477ULL, 409555158710929193ULL},
     {14375050875883784322ULL, 3258765518491372314ULL, 6123276414968091301ULL, 564945574506516589ULL},
     {16399159056375946558ULL, 12401617009203210820ULL, 11251954111719545950ULL, 433429710746163456ULL}},
    // Round 2 (full round)
    {{7006359817426385501ULL, 17203056170488800107ULL, 10266463410669146573ULL, 302632258003414824ULL},
     {14549101708442237159ULL, 5447808788302094550ULL, 5985460154360671870ULL, 377474158904626280ULL},
     {2016536305309843132ULL, 7819086366821881070ULL, 6549900492473011498ULL, 375041666190951811ULL}},
    // Round 3 (full round)
    {{10958522786453251491ULL, 9697564334799485296ULL, 12061515884908749995ULL, 382992757552468126ULL},
     {3123128533353263104ULL, 1101320594306927398ULL, 12277506650622088974ULL, 151394834303922635ULL},
     {10558209943415701402ULL, 3761550961988184469ULL, 3770582263098070207ULL, 337917216919135628ULL}},
    // Round 4 (partial rounds chain)
    {{11059652905750625380ULL, 13475195141561210865ULL, 13294395003503408798ULL, 543485850395037306ULL},
     {2130930448912281502ULL, 11333634387982439184ULL, 8850610548639699306ULL, 457475955817445493ULL},
     {13832015370839427617ULL, 3536623570151876469ULL, 6528270901734940966ULL, 1727329127918258ULL}},
    // Round 5 (partial rounds chain)
    {{8115407971155582787ULL, 9978560128345434391ULL, 5056408649803520810ULL, 262112615523232333ULL},
     {15132830981034848104ULL, 221062278661831986ULL, 642558393488344280ULL, 294435853161867218ULL},
     {17720328485765873457ULL, 906259302840864515ULL, 9886887042042513701ULL, 91167054851387838ULL}},
    // Round 6 (partial rounds chain)
    {{17686069520976319126ULL, 357140690021361429ULL, 4698816318705416205ULL, 393981709058899502ULL},
     {11422699654326280778ULL, 16059236267229938280ULL, 14304086719964370791ULL, 408074902120160445ULL},
     {10098039853740591407ULL, 18346213706869023683ULL, 9856189649941491293ULL, 184406899276982606ULL}},
    // Round 7 (partial rounds chain)
    {{4111625807920905640ULL, 13925198121954558587ULL, 12310073145155562618ULL, 235927056615592132ULL},
     {6384587362686501122ULL, 8879249956632890389ULL, 16548116661510213280ULL, 336779148206005613ULL},
     {16031192111936359076ULL, 5992351855224207619ULL, 12627781605286612627ULL, 62096344547068532ULL}},
    // Round 8 (partial rounds chain)
    {{7132509749599922320ULL, 6162523224461213171ULL, 8812310904075333603ULL, 485641259025949528ULL},
     {9608543551775247566ULL, 5196481376567266799ULL, 4241060526105290574ULL, 127878632248644222ULL},
     {5678830138638110916ULL, 17150803417208083936ULL, 3818159621611526901ULL, 334934582699708306ULL}},
    // Round 9 (partial rounds chain)
    {{2411984091808734148ULL, 4676885686514770974ULL, 16024545775274701230ULL, 35144855618233700ULL},
     {3764367625527066390ULL, 16203185340231970163ULL, 3091249709657140605ULL, 246768161927392025ULL},
     {14145295330358598665ULL, 8189373517343951529ULL, 9185965492588609992ULL, 84513112494072013ULL}},
    // Round 10 (partial rounds chain)
    {{11207056279882910057ULL, 8343632314543039012ULL, 11825707279115803854ULL, 290028553136950168ULL},
     {8324430733814960957ULL, 1295757100659713237ULL, 13793768882092694703ULL, 170786252004278897ULL},
     {3470195965229650799ULL, 7014342387715054610ULL, 8068066979310954905ULL, 29052085287755578ULL}},
    // Round 11 (partial rounds chain)
    {{16182857524744847037ULL, 13508798581767021717ULL, 11609421482229546503ULL, 516546051298708222ULL},
     {15033308840249833218ULL, 2943419432621774691ULL, 12721848833217608341ULL, 495114643959326355ULL},
     {15900588831390415422ULL, 13235287741266617180ULL, 199968676028566291ULL, 31464852864192972ULL}},
    // Round 12 (partial rounds chain)
    {{1620930607287324279ULL, 5691881440575675201ULL, 14029655266804088527ULL, 85281494397439074ULL},
     {14358430905948328990ULL, 8174075507093900204ULL, 4259719420728355987ULL, 198633356278451635ULL},
     {9809328616759644792ULL, 9452995911559605327ULL, 14571337138054143278ULL, 30595350758349726ULL}},
    // Round 13 (partial rounds chain)
    {{5052814846190951527ULL, 2564319269931470445ULL, 11667947324380057882ULL, 381968969372291382ULL},
     {10270535945835093416ULL, 7013539859298536233ULL, 12880625276280589921ULL, 423512085516371736ULL},
     {9247593602616013019ULL, 9800080834126373385ULL, 15154714092547675637ULL, 85949681409890944ULL}},
    // Round 14 (partial rounds chain)
    {{16691816165224508898ULL, 7632563883998222450ULL, 7283702476287088318ULL, 122927053010244988ULL},
     {5281932702559195717ULL, 3912411525754476767ULL, 2751980448518808692ULL, 449246095011271218ULL},
     {2154533012881281233ULL, 12108066475824676498ULL, 3101185982842383519ULL, 23082295823839220ULL}},
    // Round 15 (partial rounds chain)
    {{7605923575436520679ULL, 17775553940505278137ULL, 12354955681295648587ULL, 509503557137574051ULL},
     {6859516056437622059ULL, 15185460371714151768ULL, 11379739398280558941ULL, 467020453759984910ULL},
     {16906035870412618946ULL, 2048289172670790831ULL, 3913835398798558993ULL, 247123888097172708ULL}},
    // Round 16 (partial rounds chain)
    {{14861675641163720274ULL, 13490102368184285242ULL, 7347027430097237399ULL, 12293023119986689ULL},
     {17015804524472763158ULL, 2030415039026408622ULL, 17809691364575575612ULL, 373903064080638948ULL},
     {8022416475434186219ULL, 17815483025186149958ULL, 17841645611508634712ULL, 214671237987350594ULL}},
    // Round 17 (partial rounds chain)
    {{12836252995927977384ULL, 12965348847163767059ULL, 16404178258733598267ULL, 90570121994582570ULL},
     {3307613700375182919ULL, 6181136657427428089ULL, 13131983874186228376ULL, 111501226499533359ULL},
     {17976156798761747365ULL, 5762323702974965097ULL, 2597451573586851781ULL, 505236697901381580ULL}},
    // Round 18 (partial rounds chain)
    {{10381335841935625777ULL, 14975760611930379479ULL, 14435427058050060920ULL, 398310795157470149ULL},
     {7159397558615159963ULL, 7188734421404393410ULL, 1719787959693584840ULL, 314760383409924924ULL},
     {2143638283124439842ULL, 13645456387251540292ULL, 13644498560249152495ULL, 40707004462247922ULL}},
    // Round 19 (partial rounds chain)
    {{4076286098075094248ULL, 6047377110922245529ULL, 10310252161423143831ULL, 4203916682452144ULL},
     {16974446701450911254ULL, 13817121135003214195ULL, 2110477587661043379ULL, 408404362623939579ULL},
     {2551943238291520253ULL, 7863747909853014700ULL, 10038172819555036178ULL, 498557445795487158ULL}},
    // Round 20 (partial rounds chain)
    {{2539457924220004741ULL, 6798810574668326903ULL, 734801439896130781ULL, 197318323104987578ULL},
     {5630865015115736926ULL, 6924395279250128121ULL, 6087898613499446423ULL, 97920604124542022ULL},
     {8238671520399847831ULL, 17200586436505834896ULL, 18050188643002787777ULL, 522418299476559161ULL}},
    // Round 21 (partial rounds chain)
    {{16671769425050870867ULL, 9818859487908268561ULL, 14628982416326270968ULL, 105495391240150891ULL},
     {15264270982597349937ULL, 3214172504508351054ULL, 96620451664846254ULL, 107082329324411929ULL},
     {14217429574808978483ULL, 128541115086728122ULL, 9630653827036234478ULL, 337586787343095831ULL}},
    // Round 22 (partial rounds chain)
    {{8212059728564478324ULL, 2347043101088709486ULL, 6567058597747348925ULL, 136303161555124818ULL},
     {9215571201366006957ULL, 18390624930749960250ULL, 12318590206736769157ULL, 94289926799047171ULL},
     {2681199449952507734ULL, 2490827916210922767ULL, 17337862272405306868ULL, 167761531076143152ULL}},
    // Round 23 (partial rounds chain)
    {{12798085655163047736ULL, 13696387070792973059ULL, 5787352356986496426ULL, 499982807322000917ULL},
     {11741293120501172298ULL, 2334843635281516844ULL, 168280946537445205ULL, 83199885793358504ULL},
     {12631247450401696760ULL, 1333500347809883553ULL, 7960218164236031817ULL, 545118190714259783ULL}},
    // Round 24 (partial rounds chain)
    {{10647924774016649377ULL, 7324763009135962770ULL, 16081867801897155361ULL, 428325268027940951ULL},
     {9916218426490841056ULL, 1030092679665695813ULL, 1263314736050787503ULL, 189382575143994932ULL},
     {13675857909985031082ULL, 12446306110735075056ULL, 5033197056310105549ULL, 437182141684164304ULL}},
    // Round 25 (partial rounds chain)
    {{16017091423340680337ULL, 16608552316205419099ULL, 8224269358142049653ULL, 421369926480120239ULL},
     {9595998208446022457ULL, 16134937223929573489ULL, 14473485045201252070ULL, 492654444776713988ULL},
     {5219139941060837063ULL, 15853778231351784921ULL, 17809532253019979818ULL, 417855011179639500ULL}},
    // Round 26 (partial rounds chain)
    {{10403233932030073012ULL, 16730279770404598818ULL, 5644362027553147371ULL, 71735369062479113ULL},
     {9631935352839043273ULL, 17054291492818491992ULL, 17392960852478338690ULL, 460451922811752815ULL},
     {8477235527901859940ULL, 9814894372396933310ULL, 4799990685272759189ULL, 400282443455003622ULL}},
    // Round 27 (partial rounds chain)
    {{1576843956219139158ULL, 702430080765383770ULL, 17965431827539789120ULL, 396822748194700979ULL},
     {6534711024274141327ULL, 7130143603302325050ULL, 16132773823416316657ULL, 291127484699867608ULL},
     {4689783592352337689ULL, 13240213583589885272ULL, 9538724043191226625ULL, 53501278744087361ULL}},
    // Round 28 (partial rounds chain)
    {{5303664022074446245ULL, 11845428113003974855ULL, 12846417839211241712ULL, 261206892901600457ULL},
     {7213274845438153874ULL, 3182604331988734929ULL, 3825798403644353861ULL, 498424027422114180ULL},
     {16374080072108357107ULL, 12602249521880983826ULL, 16789038177262469212ULL, 434844701481670312ULL}},
    // Round 29 (partial rounds chain)
    {{5424818379886973048ULL, 12246700522389374047ULL, 16105275706111921973ULL, 171092744151981801ULL},
     {6597251438734065845ULL, 2269299153703490182ULL, 1681550773894047556ULL, 497581200702741603ULL},
     {448123443688089620ULL, 2093069703231428951ULL, 10690354368775868807ULL, 47970718237487928ULL}},
    // Round 30 (partial rounds chain)
    {{5663590943682895948ULL, 1657980836047728369ULL, 14473564132866859515ULL, 63033233261729097ULL},
     {10253436943656002480ULL, 496762026162837186ULL, 4416882861358244138ULL, 410250456421992694ULL},
     {13092039789419326359ULL, 12701778245175598192ULL, 10053832990213334033ULL, 99263799527290004ULL}},
    // Round 31 (final full round)
    {{17612850421305358241ULL, 8270527177451683327ULL, 3824004781155827930ULL, 23420660055616514ULL},
     {14792590694225191985ULL, 10340526750113228103ULL, 13907692663639317222ULL, 419027522602786902ULL},
     {6537420399978421861ULL, 13000095247594509242ULL, 4689550804574686720ULL, 216508944422249833ULL}},
    // Round 32 (final full round)
    {{5284804903132760421ULL, 6021193890942533180ULL, 12919709475177128988ULL, 388552658857864838ULL},
     {14487504387541221402ULL, 8671715521049733970ULL, 11505630672145478718ULL, 340569955046273223ULL},
     {15605227016987613715ULL, 4467780628446399859ULL, 1916547247173479880ULL, 360408062797054196ULL}},
    // Round 33 (final full round)
    {{2318724554222443904ULL, 12462857660673509117ULL, 1043912215626002694ULL, 444903370426104614ULL},
     {13771333397279872933ULL, 8504629457688506196ULL, 17402104297977249580ULL, 365482958936297625ULL},
     {9663191740751744091ULL, 17588211649412389869ULL, 8849849772044756264ULL, 441288247586026977ULL}},
    // Round 34 (unused - zeros)
    {{0ULL, 0ULL, 0ULL, 0ULL},
     {0ULL, 0ULL, 0ULL, 0ULL},
     {0ULL, 0ULL, 0ULL, 0ULL}}
};

// MDS matrix multiplication: [[3,1,1], [1,-1,1], [1,1,-2]]
// Using 7 field adds/subs (more efficient than direct matrix multiply)
__device__ void poseidon_mds_mix(Felt252Field state[3]) {
    Felt252Field x = state[0];
    Felt252Field y = state[1];
    Felt252Field z = state[2];

    // y1_zm1 = y - z
    Felt252Field y1_zm1 = felt_sub(y, z);
    // x1_ym1_z1 = x - y1_zm1 = x - y + z
    Felt252Field x1_ym1_z1 = felt_sub(x, y1_zm1);
    // x1_y1_zm1 = x + y1_zm1 = x + y - z
    Felt252Field x1_y1_zm1 = felt_add(x, y1_zm1);
    // x1_y1 = x + y
    Felt252Field x1_y1 = felt_add(x, y);
    // x2_y2 = 2*(x + y)
    Felt252Field x2_y2 = felt_add(x1_y1, x1_y1);

    // new_x = x2_y2 + x1_ym1_z1 = 3x + y + z (note: no round key added here)
    state[0] = felt_add(x2_y2, x1_ym1_z1);
    // new_y = x1_ym1_z1 = x - y + z
    state[1] = x1_ym1_z1;
    // new_z = x1_y1_zm1 - z = x + y - 2z
    state[2] = felt_sub(x1_y1_zm1, z);
}

// Get round key from constant array and convert to Felt252Field
// The round keys are stored in Montgomery form (same as Rust POSEIDON_ROUND_KEYS)
// FieldElement::from_mont in Rust creates a FieldElement where the internal storage IS the Montgomery form
// So we just use the values directly without conversion
__device__ Felt252Field get_round_key(unsigned round, unsigned state_idx) {
    // The 4 x u64 values ARE the Montgomery representation, used directly
    return width27_to_felt252field((const uint64_t*)POSEIDON_WIDTH27_ROUND_KEYS[round][state_idx]);
}

// Full round: cube all elements, apply MDS, add round keys
__device__ void poseidon_full_round(Felt252Field state[3], unsigned round) {
    // Cube all state elements
    state[0] = felt_cube(state[0]);
    state[1] = felt_cube(state[1]);
    state[2] = felt_cube(state[2]);

    // Apply MDS matrix
    poseidon_mds_mix(state);

    // Add round keys
    state[0] = felt_add(state[0], get_round_key(round, 0));
    state[1] = felt_add(state[1], get_round_key(round, 1));
    state[2] = felt_add(state[2], get_round_key(round, 2));
}

// Partial round: cube only first element, apply MDS, add round keys
// Uses 4-element state representation [z0^3, z1, z1^3, z2] for efficient computation
__device__ void poseidon_partial_round(Felt252Field state[4], Felt252Field half_key) {
    // z23 = cube(z2)
    Felt252Field z23 = felt_cube(state[3]);

    // Compute z3 = 8*z03 + 4*z1 + 6*z13 + 2*z2 - 2*z23 + 2*half_key
    // Using efficient formula with adds/subs only
    Felt252Field z03 = state[0];
    Felt252Field z1 = state[1];
    Felt252Field z13 = state[2];
    Felt252Field z2 = state[3];

    // z03_z13 = z03 + z13
    Felt252Field z03_z13 = felt_add(z03, z13);
    // z03_z13_z1 = z03_z13 + z1
    Felt252Field z03_z13_z1 = felt_add(z03_z13, z1);
    // longsum = z03_z13_z1 + z2 - z23 + half_key
    Felt252Field longsum = felt_add(felt_sub(felt_add(z03_z13_z1, z2), z23), half_key);
    // half_z3 = longsum + z03_z13_z1 + z03_z13 + z03
    Felt252Field half_z3 = felt_add(felt_add(felt_add(longsum, z03_z13_z1), z03_z13), z03);
    // z3 = half_z3 + half_z3
    Felt252Field z3 = felt_add(half_z3, half_z3);

    // Update state: [z13, z2, z23, z3]
    state[0] = z13;
    state[1] = z2;
    state[2] = z23;
    state[3] = z3;
}

// Poseidon 3 partial rounds chain: processes 3 partial rounds at once
__device__ void poseidon_3_partial_rounds(Felt252Field state[4], unsigned round) {
    // Each round uses 3 half keys (one per state element position)
    for (int i = 0; i < 3; i++) {
        Felt252Field half_key = get_round_key(round, i);
        poseidon_partial_round(state, half_key);
    }
}

// Convert 28 9-bit M31 limbs to Felt252Field
__device__ Felt252Field m31_limbs_28_to_felt252field(const m31* limbs) {
    // Build 256-bit value from 28 * 9 = 252 bits
    uint64_t val[4] = {0, 0, 0, 0};
    int bit_pos = 0;

    for (int i = 0; i < 28; i++) {
        uint64_t limb_val = limbs[i];
        int word_idx = bit_pos / 64;
        int bit_offset = bit_pos % 64;

        if (word_idx < 4) {
            val[word_idx] |= (limb_val << bit_offset);
            // Handle overflow to next word
            if (bit_offset + 9 > 64 && word_idx + 1 < 4) {
                val[word_idx + 1] |= (limb_val >> (64 - bit_offset));
            }
        }
        bit_pos += 9;
    }

    Felt252Field result;
    result.limbs[0] = (uint32_t)(val[0] & 0xFFFFFFFF);
    result.limbs[1] = (uint32_t)((val[0] >> 32) & 0xFFFFFFFF);
    result.limbs[2] = (uint32_t)(val[1] & 0xFFFFFFFF);
    result.limbs[3] = (uint32_t)((val[1] >> 32) & 0xFFFFFFFF);
    result.limbs[4] = (uint32_t)(val[2] & 0xFFFFFFFF);
    result.limbs[5] = (uint32_t)((val[2] >> 32) & 0xFFFFFFFF);
    result.limbs[6] = (uint32_t)(val[3] & 0xFFFFFFFF);
    result.limbs[7] = (uint32_t)((val[3] >> 32) & 0xFFFFFFFF);

    return result;
}

// Convert Felt252Field to 10 27-bit M31 limbs (Width27 format)
__device__ void felt252field_to_m31_width27(const Felt252Field& felt, m31* limbs) {
    Felt252Width27 w27;
    felt252field_to_width27_m31(felt, w27);
    for (int i = 0; i < 10; i++) {
        limbs[i] = w27.limbs[i];
    }
}

// Capture intermediate full round chain state into a lookup array (32 elements)
// Format: [chain_id, round, state0[10], state1[10], state2[10]]
__device__ void capture_pfrc_state(m31** lookup, const Felt252Field state[3],
                                    m31 chain_id, int round, int row) {
    lookup[0][row] = chain_id;
    lookup[1][row] = (m31){(uint32_t)round};
    m31 w27[10];
    felt252field_to_m31_width27(state[0], w27);
    for (int i = 0; i < 10; i++) lookup[2 + i][row] = w27[i];
    felt252field_to_m31_width27(state[1], w27);
    for (int i = 0; i < 10; i++) lookup[12 + i][row] = w27[i];
    felt252field_to_m31_width27(state[2], w27);
    for (int i = 0; i < 10; i++) lookup[22 + i][row] = w27[i];
}

// ============================================================================
// Base trace generation kernel for poseidon_builtin
// ============================================================================
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void generate_poseidon_builtin_base_trace_kernel(
    m31 **traces,

    // Lookup data arrays - 6 MemoryAddressToId lookups (2 elements each)
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,
    m31 **lookup_memory_address_to_id_5,

    // Lookup data arrays - 6 MemoryIdToBig lookups (29 elements each)
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,

    // Lookup data arrays - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
    m31 **lookup_range_check_3_3_3_3_3_0,
    m31 **lookup_range_check_3_3_3_3_3_1,

    // Lookup data arrays - range_check_4_4_4_4 (6 lookups, 4 elements each)
    m31 **lookup_range_check_4_4_4_4_0,
    m31 **lookup_range_check_4_4_4_4_1,
    m31 **lookup_range_check_4_4_4_4_2,
    m31 **lookup_range_check_4_4_4_4_3,
    m31 **lookup_range_check_4_4_4_4_4,
    m31 **lookup_range_check_4_4_4_4_5,

    // Lookup data arrays - range_check_4_4 (3 lookups, 2 elements each)
    m31 **lookup_range_check_4_4_0,
    m31 **lookup_range_check_4_4_1,
    m31 **lookup_range_check_4_4_2,

    // Lookup data arrays - poseidon_full_round_chain (8 lookups, 32 elements each)
    // pfrc_0: input to round 0, pfrc_1: input to round 31
    // pfrc_2-4: intermediate states for rounds 1-3, pfrc_5-7: intermediate states for rounds 32-34
    m31 **lookup_poseidon_full_round_chain_0,
    m31 **lookup_poseidon_full_round_chain_1,
    m31 **lookup_poseidon_full_round_chain_2,
    m31 **lookup_poseidon_full_round_chain_3,
    m31 **lookup_poseidon_full_round_chain_4,
    m31 **lookup_poseidon_full_round_chain_5,
    m31 **lookup_poseidon_full_round_chain_6,
    m31 **lookup_poseidon_full_round_chain_7,

    // Lookup data arrays - poseidon_3_partial_rounds_chain (27 lookups, 42 elements each)
    // Each lookup: [chain_id, round, state[0..3] as Width27 (40 elements)]
    m31 **lookup_poseidon_3_partial_rounds_chain_0,
    m31 **lookup_poseidon_3_partial_rounds_chain_1,
    m31 **lookup_poseidon_3_partial_rounds_chain_2,
    m31 **lookup_poseidon_3_partial_rounds_chain_3,
    m31 **lookup_poseidon_3_partial_rounds_chain_4,
    m31 **lookup_poseidon_3_partial_rounds_chain_5,
    m31 **lookup_poseidon_3_partial_rounds_chain_6,
    m31 **lookup_poseidon_3_partial_rounds_chain_7,
    m31 **lookup_poseidon_3_partial_rounds_chain_8,
    m31 **lookup_poseidon_3_partial_rounds_chain_9,
    m31 **lookup_poseidon_3_partial_rounds_chain_10,
    m31 **lookup_poseidon_3_partial_rounds_chain_11,
    m31 **lookup_poseidon_3_partial_rounds_chain_12,
    m31 **lookup_poseidon_3_partial_rounds_chain_13,
    m31 **lookup_poseidon_3_partial_rounds_chain_14,
    m31 **lookup_poseidon_3_partial_rounds_chain_15,
    m31 **lookup_poseidon_3_partial_rounds_chain_16,
    m31 **lookup_poseidon_3_partial_rounds_chain_17,
    m31 **lookup_poseidon_3_partial_rounds_chain_18,
    m31 **lookup_poseidon_3_partial_rounds_chain_19,
    m31 **lookup_poseidon_3_partial_rounds_chain_20,
    m31 **lookup_poseidon_3_partial_rounds_chain_21,
    m31 **lookup_poseidon_3_partial_rounds_chain_22,
    m31 **lookup_poseidon_3_partial_rounds_chain_23,
    m31 **lookup_poseidon_3_partial_rounds_chain_24,
    m31 **lookup_poseidon_3_partial_rounds_chain_25,
    m31 **lookup_poseidon_3_partial_rounds_chain_26,

    // Base trace cols 120-283 (164 columns) for interaction kernels
    m31 **lookup_base_trace_cols,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0_val = {0};
    const m31 M31_1_val = {1};
    const m31 M31_2_val = {2};
    const m31 M31_3_val = {3};
    const m31 M31_4_val = {4};
    const m31 M31_5_val = {5};
    const m31 M31_6_val = {6};
    const m31 M31_512_val = {512};
    const m31 M31_262144_val = {262144};

    if (row < trace_size) {
        m31 seq = {row};
        m31 segment_start_m31 = {segment_start};

        // Poseidon instance address = segment_start + 6 * seq
        // Memory layout: [input0, input1, input2, output0, output1, output2]
        m31 instance_addr = add(segment_start_m31, mul(M31_6_val, seq));

        // ============ Read input state 0 ============
        m31 input_addr_0 = instance_addr;
        m31 input_state_0_id = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_addr_0,
            &input_state_0_id
        );
        traces[0][row] = input_state_0_id;
        lookup_memory_address_to_id_0[0][row] = input_addr_0;
        lookup_memory_address_to_id_0[1][row] = input_state_0_id;

        // Read 28 limbs for input state 0
        m31 input_state_0_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            input_state_0_id,
            input_state_0_limbs
        );

        // Store input state 0 limbs (columns 1-28)
        for (int i = 0; i < 28; i++) {
            traces[1 + i][row] = input_state_0_limbs[i];
        }

        lookup_memory_id_to_big_0[0][row] = input_state_0_id;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_0[1 + i][row] = input_state_0_limbs[i];
        }

        // ============ Read input state 1 ============
        m31 input_addr_1 = add(instance_addr, M31_1_val);
        m31 input_state_1_id = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_addr_1,
            &input_state_1_id
        );
        traces[29][row] = input_state_1_id;
        lookup_memory_address_to_id_1[0][row] = input_addr_1;
        lookup_memory_address_to_id_1[1][row] = input_state_1_id;

        // Read 28 limbs for input state 1
        m31 input_state_1_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            input_state_1_id,
            input_state_1_limbs
        );

        // Store input state 1 limbs (columns 30-57)
        for (int i = 0; i < 28; i++) {
            traces[30 + i][row] = input_state_1_limbs[i];
        }

        lookup_memory_id_to_big_1[0][row] = input_state_1_id;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_1[1 + i][row] = input_state_1_limbs[i];
        }

        // ============ Read input state 2 ============
        m31 input_addr_2 = add(instance_addr, M31_2_val);
        m31 input_state_2_id = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_addr_2,
            &input_state_2_id
        );
        traces[58][row] = input_state_2_id;
        lookup_memory_address_to_id_2[0][row] = input_addr_2;
        lookup_memory_address_to_id_2[1][row] = input_state_2_id;

        // Read 28 limbs for input state 2
        m31 input_state_2_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            input_state_2_id,
            input_state_2_limbs
        );

        // Store input state 2 limbs (columns 59-86)
        for (int i = 0; i < 28; i++) {
            traces[59 + i][row] = input_state_2_limbs[i];
        }

        lookup_memory_id_to_big_2[0][row] = input_state_2_id;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_2[1 + i][row] = input_state_2_limbs[i];
        }

        // Pack input states to width27 format
        Felt252Width27 packed_input_state_0, packed_input_state_1, packed_input_state_2;
        pack_felt252_to_width27(input_state_0_limbs, packed_input_state_0);
        pack_felt252_to_width27(input_state_1_limbs, packed_input_state_1);
        pack_felt252_to_width27(input_state_2_limbs, packed_input_state_2);

        // ============ Poseidon permutation computation ============
        // Step 1: Compute linear combinations using correct bias constants (columns 87-119)
        // LinearCombination computes: combination = input + bias (in Felt252)
        // and also computes p_coef based on carry propagation
        Felt252Width27 lc_result_0, lc_result_1, lc_result_2;
        m31 p_coef_0, p_coef_1, p_coef_2;

        compute_linear_combination_with_p_coef(
            packed_input_state_0, LINEAR_COMB_BIAS_0, lc_result_0, p_coef_0);
        compute_linear_combination_with_p_coef(
            packed_input_state_1, LINEAR_COMB_BIAS_1, lc_result_1, p_coef_1);
        compute_linear_combination_with_p_coef(
            packed_input_state_2, LINEAR_COMB_BIAS_2, lc_result_2, p_coef_2);

        // Columns 87-96: Linear combination for state 0
        for (int i = 0; i < 10; i++) {
            traces[87 + i][row] = lc_result_0.limbs[i];
        }
        // Column 97: p_coef for state 0
        traces[97][row] = p_coef_0;

        // Columns 98-107: Linear combination for state 1
        for (int i = 0; i < 10; i++) {
            traces[98 + i][row] = lc_result_1.limbs[i];
        }
        // Column 108: p_coef for state 1
        traces[108][row] = p_coef_1;

        // Columns 109-118: Linear combination for state 2
        for (int i = 0; i < 10; i++) {
            traces[109 + i][row] = lc_result_2.limbs[i];
        }
        // Column 119: p_coef for state 2
        traces[119][row] = p_coef_2;

        // Convert linear combination results to Felt252Field for full round computation
        // The 10 M31 limbs pack into the Montgomery form representation
        Felt252Field lc_state_0 = width27_m31_to_felt252field(lc_result_0);
        Felt252Field lc_state_1 = width27_m31_to_felt252field(lc_result_1);
        Felt252Field lc_state_2 = width27_m31_to_felt252field(lc_result_2);

        // Step 2: Run 4 full rounds (columns 120-149)
        // Unrolled to capture intermediate states for pfrc_2, pfrc_3, pfrc_4
        Felt252Field state[3] = {lc_state_0, lc_state_1, lc_state_2};
        m31 chain_id = {row * 2};  // chain_id = seq * 2 (used for pfrc captures)

        // Round 0 (input already captured as pfrc_0 above)
        poseidon_full_round(state, 0);
        // Capture input to round 1 → pfrc_2
        capture_pfrc_state(lookup_poseidon_full_round_chain_2, state, chain_id, 1, row);

        poseidon_full_round(state, 1);
        // Capture input to round 2 → pfrc_3
        capture_pfrc_state(lookup_poseidon_full_round_chain_3, state, chain_id, 2, row);

        poseidon_full_round(state, 2);
        // Capture input to round 3 → pfrc_4
        capture_pfrc_state(lookup_poseidon_full_round_chain_4, state, chain_id, 3, row);

        poseidon_full_round(state, 3);

        // Store full round chain output (30 Width27 limbs = 3 states × 10)
        m31 full_round_out_0[10], full_round_out_1[10], full_round_out_2[10];
        felt252field_to_m31_width27(state[0], full_round_out_0);
        felt252field_to_m31_width27(state[1], full_round_out_1);
        felt252field_to_m31_width27(state[2], full_round_out_2);

        // Columns 120-129: State 0 after 4 full rounds
        for (int i = 0; i < 10; i++) {
            traces[120 + i][row] = full_round_out_0[i];
        }
        // Columns 130-139: State 1 after 4 full rounds
        for (int i = 0; i < 10; i++) {
            traces[130 + i][row] = full_round_out_1[i];
        }
        // Columns 140-149: State 2 after 4 full rounds
        for (int i = 0; i < 10; i++) {
            traces[140 + i][row] = full_round_out_2[i];
        }

        // Populate poseidon_full_round_chain lookup data
        // poseidon_full_round_chain_0: [chain_id, round=0, cols 87-96, cols 98-107, cols 109-118]
        // chain_id already declared above as {row * 2}
        lookup_poseidon_full_round_chain_0[0][row] = chain_id;
        lookup_poseidon_full_round_chain_0[1][row] = M31_0_val;  // round = 0
        for (int i = 0; i < 10; i++) {
            lookup_poseidon_full_round_chain_0[2 + i][row] = lc_result_0.limbs[i];      // cols 87-96
            lookup_poseidon_full_round_chain_0[12 + i][row] = lc_result_1.limbs[i];     // cols 98-107
            lookup_poseidon_full_round_chain_0[22 + i][row] = lc_result_2.limbs[i];     // cols 109-118
        }

        // poseidon_full_round_chain_1 is populated AFTER partial rounds and linear combinations
        // (see below, after comb2_width27 is computed)

        // Step 3: Compute cube of full_round_out_2 (columns 150-159)
        // CPU: cube_252_output_tmp_51986_89 = Cube252::cube(poseidon_full_round_chain_output.2[2])
        // Convert full_round_out_2 (Width27 m31[10]) to Felt252Field for cubing
        Felt252Width27 fro2_w27;
        for (int i = 0; i < 10; i++) {
            fro2_w27.limbs[i] = full_round_out_2[i];
        }
        Felt252Field fro2_felt = width27_m31_to_felt252field(fro2_w27);
        Felt252Field cube_fro2 = felt_cube(fro2_felt);
        m31 cube_fro2_width27[10];
        felt252field_to_m31_width27(cube_fro2, cube_fro2_width27);
        for (int i = 0; i < 10; i++) {
            traces[150 + i][row] = cube_fro2_width27[i];
        }

        // Step 4: Linear combination for partial round setup (columns 160-169)
        // CPU formula: combination = 0 + 1*state0 + 1*state1 - 2*cube_output + BIAS
        // Where state0 = full_round_out_0, state1 = full_round_out_1, cube_output = cube_fro2

        // Convert full_round_out_0 and full_round_out_1 to Felt252Field
        Felt252Width27 fro0_w27, fro1_w27;
        for (int i = 0; i < 10; i++) {
            fro0_w27.limbs[i] = full_round_out_0[i];
            fro1_w27.limbs[i] = full_round_out_1[i];
        }
        Felt252Field fro0_felt = width27_m31_to_felt252field(fro0_w27);
        Felt252Field fro1_felt = width27_m31_to_felt252field(fro1_w27);

        // Get BIAS constant for LinearCombinationN4Coefs11M21
        Felt252Field bias_felt = bias_to_felt252field(LC_BIAS_N4_COEFS_1_1_M2_1);

        // Compute: fro0 + fro1 - 2*cube_fro2 + bias
        Felt252Field two_cube = felt_add(cube_fro2, cube_fro2);
        Felt252Field lc_partial = felt_add(felt_sub(felt_add(fro0_felt, fro1_felt), two_cube), bias_felt);
        m31 lc_partial_width27[10];
        felt252field_to_m31_width27(lc_partial, lc_partial_width27);
        for (int i = 0; i < 10; i++) {
            traces[160 + i][row] = lc_partial_width27[i];
        }

        // Column 170: p_coef for LinearCombinationN4Coefs11M21
        // Formula: p_coef = ((state0[0] + state1[0] - 2*cube[0] + bias[0] - combination[0] + 402653187) & 0xFFFF) - 3
        // Where 402653187 = 3 * 134217729 (bias for range [-3, 3])
        m31 p_coef_170;
        m31 carries_33333_0[9];  // carries for range_check_3_3_3_3_3_0 and _1 (M31 values!)
        {
            // Compute p_coef and carries for range_check_3_3_3_3_3
            // All operations in M31 field arithmetic
            int64_t biased_val = (int64_t)full_round_out_0[0] + (int64_t)full_round_out_1[0]
                        - 2 * (int64_t)cube_fro2_width27[0]
                        + (int64_t)LC_BIAS_N4_COEFS_1_1_M2_1[0]
                        - (int64_t)lc_partial_width27[0]
                        + 402653187LL;
            uint32_t low16 = (uint32_t)(biased_val & 0xFFFF);
            // Use M31 subtraction to avoid underflow when low16 < 3
            p_coef_170 = sub((m31)low16, M31_3_val);
            traces[170][row] = p_coef_170;

            // Compute carry_0 using M31 arithmetic:
            // carry_0 = (state0[0] + state1[0] - 2*cube[0] + bias[0] - combination[0] - p_coef) * 16
            // All using M31 field operations from fields.cuh
            m31 two_cube_0 = mul(2, cube_fro2_width27[0]);
            m31 sum_0 = add(add(add(sub(full_round_out_0[0], two_cube_0), full_round_out_1[0]),
                              LC_BIAS_N4_COEFS_1_1_M2_1[0]), neg(lc_partial_width27[0]));
            carries_33333_0[0] = mul(sub(sum_0, p_coef_170), 16);

            // Compute carries 1-8 using M31 arithmetic
            for (int i = 1; i < 9; i++) {
                m31 two_cube_i = mul(2, cube_fro2_width27[i]);
                m31 sum_i = add(add(add(add(sub(full_round_out_0[i], two_cube_i), full_round_out_1[i]),
                                       LC_BIAS_N4_COEFS_1_1_M2_1[i]), neg(lc_partial_width27[i])),
                               carries_33333_0[i-1]);
                // Special case for limb 7: subtract p_coef * 136
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_170, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_33333_0[i] = mul(sum_i, 16);
            }

            // Output range_check_3_3_3_3_3_0: [p_coef + 3, carry_0 + 3, ..., carry_3 + 3]
            lookup_range_check_3_3_3_3_3_0[0][row] = add(p_coef_170, 3);
            for (int i = 0; i < 4; i++) {
                lookup_range_check_3_3_3_3_3_0[1 + i][row] = add(carries_33333_0[i], 3);
            }

            // Output range_check_3_3_3_3_3_1: [carry_4 + 3, ..., carry_8 + 3]
            for (int i = 0; i < 5; i++) {
                lookup_range_check_3_3_3_3_3_1[i][row] = add(carries_33333_0[4 + i], 3);
            }
        }

        // Cube of linear combination (columns 171-180)
        // lc_partial is in raw form
        Felt252Field cube_lc_partial = felt_cube(lc_partial);
        m31 cube_lc_partial_width27[10];
        felt252field_to_m31_width27(cube_lc_partial, cube_lc_partial_width27);
        for (int i = 0; i < 10; i++) {
            traces[171 + i][row] = cube_lc_partial_width27[i];
        }

        // Get BIAS constant for LinearCombinationN4Coefs42M21
        Felt252Field bias2_felt = bias_to_felt252field(LC_BIAS_N4_COEFS_4_2_M2_1);

        // More linear combinations for partial rounds (columns 181-190)
        // CPU: combination = 0 + 4*state0 + 2*cube_fro2 - 2*cube_lc_partial + BIAS
        Felt252Field four_fro0 = felt_add(felt_add(fro0_felt, fro0_felt), felt_add(fro0_felt, fro0_felt));
        Felt252Field two_cube_fro2 = felt_add(cube_fro2, cube_fro2);
        Felt252Field two_cube_lc = felt_add(cube_lc_partial, cube_lc_partial);
        Felt252Field lc_partial_2 = felt_add(felt_sub(felt_add(four_fro0, two_cube_fro2), two_cube_lc), bias2_felt);
        m31 lc_partial_2_width27[10];
        felt252field_to_m31_width27(lc_partial_2, lc_partial_2_width27);
        for (int i = 0; i < 10; i++) {
            traces[181 + i][row] = lc_partial_2_width27[i];
        }

        // Column 191: p_coef for LinearCombinationN4Coefs42M21
        // Formula: p_coef = ((4*state0[0] + 2*cube[0] - 2*cube_lc[0] + bias[0] - combination[0] + 402653187) & 0xFFFF) - 3
        m31 p_coef_191;
        m31 carries_4444[9];  // carries for range_check_4_4_4_4_0..1 and range_check_4_4_0 (M31 values!)
        {
            // Compute p_coef and carries for range_check_4_4_4_4_0/1 and range_check_4_4_0
            // All operations in M31 field arithmetic
            int64_t biased_val = 4 * (int64_t)full_round_out_0[0]
                        + 2 * (int64_t)cube_fro2_width27[0]
                        - 2 * (int64_t)cube_lc_partial_width27[0]
                        + (int64_t)LC_BIAS_N4_COEFS_4_2_M2_1[0]
                        - (int64_t)lc_partial_2_width27[0]
                        + 402653187LL;
            uint32_t low16 = (uint32_t)(biased_val & 0xFFFF);
            // Use M31 subtraction to avoid underflow when low16 < 3
            p_coef_191 = sub((m31)low16, M31_3_val);
            traces[191][row] = p_coef_191;

            // Compute carry_0 using M31 arithmetic:
            // carry_0 = (4*state0[0] + 2*cube[0] - 2*cube_lc[0] + bias[0] - combination[0] - p_coef) * 16
            m31 four_state0_0 = mul(4, full_round_out_0[0]);
            m31 two_cube_fro2_0 = mul(2, cube_fro2_width27[0]);
            m31 two_cube_lc_0 = mul(2, cube_lc_partial_width27[0]);
            m31 sum_0 = add(add(add(sub(four_state0_0, two_cube_lc_0), two_cube_fro2_0),
                              LC_BIAS_N4_COEFS_4_2_M2_1[0]), neg(lc_partial_2_width27[0]));
            carries_4444[0] = mul(sub(sum_0, p_coef_191), 16);

            // Compute carries 1-8 using M31 arithmetic
            for (int i = 1; i < 9; i++) {
                m31 four_state0_i = mul(4, full_round_out_0[i]);
                m31 two_cube_fro2_i = mul(2, cube_fro2_width27[i]);
                m31 two_cube_lc_i = mul(2, cube_lc_partial_width27[i]);
                m31 sum_i = add(add(add(add(sub(four_state0_i, two_cube_lc_i), two_cube_fro2_i),
                                       LC_BIAS_N4_COEFS_4_2_M2_1[i]), neg(lc_partial_2_width27[i])),
                               carries_4444[i-1]);
                // Special case for limb 7: subtract p_coef * 136
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_191, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_4444[i] = mul(sum_i, 16);
            }

            // Output range_check_4_4_4_4_0: [p_coef + 3, carry_0 + 3, carry_1 + 3, carry_2 + 3]
            lookup_range_check_4_4_4_4_0[0][row] = add(p_coef_191, 3);
            for (int i = 0; i < 3; i++) {
                lookup_range_check_4_4_4_4_0[1 + i][row] = add(carries_4444[i], 3);
            }

            // Output range_check_4_4_4_4_1: [carry_3 + 3, carry_4 + 3, carry_5 + 3, carry_6 + 3]
            for (int i = 0; i < 4; i++) {
                lookup_range_check_4_4_4_4_1[i][row] = add(carries_4444[3 + i], 3);
            }

            // Output range_check_4_4_0: [carry_7 + 3, carry_8 + 3]
            lookup_range_check_4_4_0[0][row] = add(carries_4444[7], 3);
            lookup_range_check_4_4_0[1][row] = add(carries_4444[8], 3);
        }

        // Step 5: Run partial rounds (27 groups × 3 rounds = 81 rounds)
        // Partial state is in raw form: [cube_fro2, lc_partial, cube_lc_partial, lc_partial_2]
        Felt252Field partial_state[4] = {cube_fro2, lc_partial, cube_lc_partial, lc_partial_2};

        // Array of pointers to the 27 poseidon_3_partial_rounds_chain lookup arrays
        m31 **lookup_partial_chain[27] = {
            lookup_poseidon_3_partial_rounds_chain_0,
            lookup_poseidon_3_partial_rounds_chain_1,
            lookup_poseidon_3_partial_rounds_chain_2,
            lookup_poseidon_3_partial_rounds_chain_3,
            lookup_poseidon_3_partial_rounds_chain_4,
            lookup_poseidon_3_partial_rounds_chain_5,
            lookup_poseidon_3_partial_rounds_chain_6,
            lookup_poseidon_3_partial_rounds_chain_7,
            lookup_poseidon_3_partial_rounds_chain_8,
            lookup_poseidon_3_partial_rounds_chain_9,
            lookup_poseidon_3_partial_rounds_chain_10,
            lookup_poseidon_3_partial_rounds_chain_11,
            lookup_poseidon_3_partial_rounds_chain_12,
            lookup_poseidon_3_partial_rounds_chain_13,
            lookup_poseidon_3_partial_rounds_chain_14,
            lookup_poseidon_3_partial_rounds_chain_15,
            lookup_poseidon_3_partial_rounds_chain_16,
            lookup_poseidon_3_partial_rounds_chain_17,
            lookup_poseidon_3_partial_rounds_chain_18,
            lookup_poseidon_3_partial_rounds_chain_19,
            lookup_poseidon_3_partial_rounds_chain_20,
            lookup_poseidon_3_partial_rounds_chain_21,
            lookup_poseidon_3_partial_rounds_chain_22,
            lookup_poseidon_3_partial_rounds_chain_23,
            lookup_poseidon_3_partial_rounds_chain_24,
            lookup_poseidon_3_partial_rounds_chain_25,
            lookup_poseidon_3_partial_rounds_chain_26
        };

        // Run 27 groups of 3 partial rounds (rounds 4-30)
        // IMPORTANT: Store the INPUT state BEFORE processing each round,
        // matching the SIMD path which stores sub_component_inputs before deduce_output.
        for (int group = 0; group < 27; group++) {
            unsigned round = 4 + group;

            // Populate poseidon_3_partial_rounds_chain lookup for this group
            // Each lookup: [chain_id, round, state[0..3] as Width27 (40 elements)]
            // Total: 42 elements
            // Must store INPUT state BEFORE poseidon_3_partial_rounds mutates it.
            if (lookup_partial_chain[group] != nullptr) {
                // chain_id = row (sequence number)
                lookup_partial_chain[group][0][row] = {row};

                // round = 4 + group (rounds 4 through 30)
                lookup_partial_chain[group][1][row] = {round};

                // Convert partial_state[0..3] to Width27 M31 limbs (10 limbs each = 40 total)
                m31 state_limbs[4][10];
                felt252field_to_m31_width27(partial_state[0], state_limbs[0]);
                felt252field_to_m31_width27(partial_state[1], state_limbs[1]);
                felt252field_to_m31_width27(partial_state[2], state_limbs[2]);
                felt252field_to_m31_width27(partial_state[3], state_limbs[3]);

                // Store state[0] limbs (elements 2-11)
                for (int i = 0; i < 10; i++) {
                    lookup_partial_chain[group][2 + i][row] = state_limbs[0][i];
                }

                // Store state[1] limbs (elements 12-21)
                for (int i = 0; i < 10; i++) {
                    lookup_partial_chain[group][12 + i][row] = state_limbs[1][i];
                }

                // Store state[2] limbs (elements 22-31)
                for (int i = 0; i < 10; i++) {
                    lookup_partial_chain[group][22 + i][row] = state_limbs[2][i];
                }

                // Store state[3] limbs (elements 32-41)
                for (int i = 0; i < 10; i++) {
                    lookup_partial_chain[group][32 + i][row] = state_limbs[3][i];
                }
            }

            // Process 3 partial rounds AFTER storing the input state
            poseidon_3_partial_rounds(partial_state, round);
        }

        // Columns 192-231: Store partial_state after all 27 groups (round 30 output)
        // partial_state[0..3] each gets 10 Width27 limbs
        m31 partial_out_0[10], partial_out_1[10], partial_out_2[10], partial_out_3[10];
        felt252field_to_m31_width27(partial_state[0], partial_out_0);
        felt252field_to_m31_width27(partial_state[1], partial_out_1);
        felt252field_to_m31_width27(partial_state[2], partial_out_2);
        felt252field_to_m31_width27(partial_state[3], partial_out_3);

        for (int i = 0; i < 10; i++) {
            traces[192 + i][row] = partial_out_0[i];  // Columns 192-201
            traces[202 + i][row] = partial_out_1[i];  // Columns 202-211
            traces[212 + i][row] = partial_out_2[i];  // Columns 212-221
            traces[222 + i][row] = partial_out_3[i];  // Columns 222-231
        }

        // Columns 232-253: Two linear combinations for final full rounds setup
        // These use specific bias constants from the constraint system
        // The CPU computes combinations using the Width27 values from trace columns,
        // so we must also use the Width27 values (not the Montgomery representation)

        // Reconstruct Felt252 values from Width27 limbs (same as CPU does)
        Felt252Width27 w27_state0, w27_state1, w27_state2, w27_state3;
        for (int i = 0; i < 10; i++) {
            w27_state0.limbs[i] = partial_out_0[i];
            w27_state1.limbs[i] = partial_out_1[i];
            w27_state2.limbs[i] = partial_out_2[i];
            w27_state3.limbs[i] = partial_out_3[i];
        }
        Felt252Field felt_state0 = width27_m31_to_felt252field(w27_state0);
        Felt252Field felt_state1 = width27_m31_to_felt252field(w27_state1);
        Felt252Field felt_state2 = width27_m31_to_felt252field(w27_state2);
        Felt252Field felt_state3 = width27_m31_to_felt252field(w27_state3);

        // Felt252_3969818800901670911_10562874008078701503_14906396266795319764_223312371439046257
        Felt252Field comb1_bias = {
            (uint32_t)(3969818800901670911ULL & 0xFFFFFFFF),
            (uint32_t)(3969818800901670911ULL >> 32),
            (uint32_t)(10562874008078701503ULL & 0xFFFFFFFF),
            (uint32_t)(10562874008078701503ULL >> 32),
            (uint32_t)(14906396266795319764ULL & 0xFFFFFFFF),
            (uint32_t)(14906396266795319764ULL >> 32),
            (uint32_t)(223312371439046257ULL & 0xFFFFFFFF),
            (uint32_t)(223312371439046257ULL >> 32)
        };
        // Felt252_10310704347937391837_5874215448258336115_2880320859071049537_45350836576946303
        Felt252Field comb2_bias = {
            (uint32_t)(10310704347937391837ULL & 0xFFFFFFFF),
            (uint32_t)(10310704347937391837ULL >> 32),
            (uint32_t)(5874215448258336115ULL & 0xFFFFFFFF),
            (uint32_t)(5874215448258336115ULL >> 32),
            (uint32_t)(2880320859071049537ULL & 0xFFFFFFFF),
            (uint32_t)(2880320859071049537ULL >> 32),
            (uint32_t)(45350836576946303ULL & 0xFFFFFFFF),
            (uint32_t)(45350836576946303ULL >> 32)
        };

        // Convert values to Montgomery form for arithmetic
        felt_state0 = felt_to_mont(felt_state0);
        felt_state1 = felt_to_mont(felt_state1);
        felt_state2 = felt_to_mont(felt_state2);
        felt_state3 = felt_to_mont(felt_state3);
        comb1_bias = felt_to_mont(comb1_bias);
        comb2_bias = felt_to_mont(comb2_bias);

        Felt252Field four = {4, 0, 0, 0, 0, 0, 0, 0};
        Felt252Field two = {2, 0, 0, 0, 0, 0, 0, 0};
        four = felt_to_mont(four);
        two = felt_to_mont(two);

        // combination1 = 4*felt_state0 + 2*felt_state1 + felt_state2 + comb1_bias
        Felt252Field comb1 = felt_add(
            felt_add(
                felt_add(
                    felt_mul(four, felt_state0),
                    felt_mul(two, felt_state1)
                ),
                felt_state2
            ),
            comb1_bias
        );

        // Convert comb1 out of Montgomery form before extracting Width27 limbs
        Felt252Field comb1_normal = felt_from_mont(comb1);
        m31 comb1_width27[10];
        felt252field_to_m31_width27(comb1_normal, comb1_width27);
        for (int i = 0; i < 10; i++) {
            traces[232 + i][row] = comb1_width27[i];  // Columns 232-241
        }

        // p_coef_col242: compute from partial round outputs and combination1
        // Formula: p_coef = ((4*col192 + 2*col202 + col212 + 40454143 - col232 + 134217729) & 0xFFFF) - 1
        {
            uint32_t col192 = partial_out_0[0];
            uint32_t col202 = partial_out_1[0];
            uint32_t col212 = partial_out_2[0];
            uint32_t col232 = comb1_width27[0];
            int64_t biased = 4LL * col192 + 2LL * col202 + col212 + 40454143LL - col232 + 134217729LL;
            uint32_t low16 = (uint32_t)(biased & 0xFFFF);
            traces[242][row] = (m31){low16 - 1};
        }

        // combination2 = 4*felt_state2 + 2*felt_state3 + comb1 + comb2_bias
        Felt252Field comb2 = felt_add(
            felt_add(
                felt_add(
                    felt_mul(four, felt_state2),
                    felt_mul(two, felt_state3)
                ),
                comb1
            ),
            comb2_bias
        );

        // Convert comb2 out of Montgomery form before extracting Width27 limbs
        Felt252Field comb2_normal = felt_from_mont(comb2);
        m31 comb2_width27[10];
        felt252field_to_m31_width27(comb2_normal, comb2_width27);
        for (int i = 0; i < 10; i++) {
            traces[243 + i][row] = comb2_width27[i];  // Columns 243-252
        }

        // Populate poseidon_full_round_chain_1 lookup data
        // This represents the INPUT to the second set of 4 full rounds (round=31).
        // Used by populate_cuda_chain_generators() to compute intermediate pfrc states.
        // SIMD: pfrc_2 = [chain_id=seq*2+1, round=31, cols 243-252, cols 232-241, cols 222-231]
        // State layout: state0=comb2, state1=comb1, state2=partial_out_3
        {
            m31 chain_id_1 = {row * 2 + 1};  // Second chain (seq*2 + 1)
            lookup_poseidon_full_round_chain_1[0][row] = chain_id_1;
            lookup_poseidon_full_round_chain_1[1][row] = M31_31_val;  // round = 31
            for (int i = 0; i < 10; i++) {
                lookup_poseidon_full_round_chain_1[2 + i][row] = comb2_width27[i];     // cols 243-252
                lookup_poseidon_full_round_chain_1[12 + i][row] = comb1_width27[i];    // cols 232-241
                lookup_poseidon_full_round_chain_1[22 + i][row] = partial_out_3[i];    // cols 222-231
            }
        }

        // p_coef_col253: compute from partial round outputs and combinations
        // Formula: p_coef = ((4*col212 + 2*col222 + col232 + 48383197 - col243 + 134217729) & 0xFFFF) - 1
        {
            uint32_t col212 = partial_out_2[0];
            uint32_t col222 = partial_out_3[0];
            uint32_t col232 = comb1_width27[0];
            uint32_t col243 = comb2_width27[0];
            int64_t biased = 4LL * col212 + 2LL * col222 + col232 + 48383197LL - col243 + 134217729LL;
            uint32_t low16 = (uint32_t)(biased & 0xFFFF);
            traces[253][row] = (m31){low16 - 1};
        }

        // After partial rounds, convert back to 3-element state for final 4 full rounds
        // CRITICAL: We must use the Width27 values from trace columns (like CPU does),
        // not the Montgomery form values computed above. The Width27 representation
        // may have slight differences due to the conversion process.

        // Reconstruct state from trace columns for final full rounds:
        // - state[0] = comb2 from cols 243-252
        // - state[1] = comb1 from cols 232-241
        // - state[2] = partial_state[3] from cols 222-231
        Felt252Width27 w27_comb2, w27_comb1, w27_partial3;
        for (int i = 0; i < 10; i++) {
            w27_comb2.limbs[i] = comb2_width27[i];
            w27_comb1.limbs[i] = comb1_width27[i];
            w27_partial3.limbs[i] = partial_out_3[i];
        }

        // Convert Width27 M31 limbs to Felt252Field
        // IMPORTANT: The Width27 representation stores values in Montgomery form (as shown in
        // Rust's Felt252Width27 -> FieldElement conversion which uses from_mont). So the
        // result of width27_m31_to_felt252field IS already in Montgomery form - do NOT call
        // felt_to_mont again!
        Felt252Field final_state0 = width27_m31_to_felt252field(w27_comb2);
        Felt252Field final_state1 = width27_m31_to_felt252field(w27_comb1);
        Felt252Field final_state2 = width27_m31_to_felt252field(w27_partial3);

        state[0] = final_state0;
        state[1] = final_state1;
        state[2] = final_state2;

        // Run final 4 full rounds (rounds 31-34)
        // Unrolled to capture intermediate states for pfrc_5, pfrc_6, pfrc_7
        // Round 31 (input already captured as pfrc_1 above)
        poseidon_full_round(state, 31);
        {
            m31 chain_id_1 = {row * 2 + 1};
            // Capture input to round 32 → pfrc_5
            capture_pfrc_state(lookup_poseidon_full_round_chain_5, state, chain_id_1, 32, row);
        }

        poseidon_full_round(state, 32);
        {
            m31 chain_id_1 = {row * 2 + 1};
            capture_pfrc_state(lookup_poseidon_full_round_chain_6, state, chain_id_1, 33, row);
        }

        poseidon_full_round(state, 33);
        {
            m31 chain_id_1 = {row * 2 + 1};
            capture_pfrc_state(lookup_poseidon_full_round_chain_7, state, chain_id_1, 34, row);
        }

        poseidon_full_round(state, 34);

        // Columns 254-283: Final full round chain output (30 Width27 limbs)
        // IMPORTANT: The Width27 representation stores values in Montgomery form.
        // When converting FieldElement -> Felt252Width27 in Rust, it uses into_mont().
        // So we should extract Width27 limbs directly from the Montgomery form state,
        // NOT after converting out of Montgomery form.
        m31 final_out_0[10], final_out_1[10], final_out_2[10];
        felt252field_to_m31_width27(state[0], final_out_0);
        felt252field_to_m31_width27(state[1], final_out_1);
        felt252field_to_m31_width27(state[2], final_out_2);

        for (int i = 0; i < 10; i++) {
            traces[254 + i][row] = final_out_0[i];   // Columns 254-263
            traces[264 + i][row] = final_out_1[i];   // Columns 264-273
            traces[274 + i][row] = final_out_2[i];   // Columns 274-283
        }

        // Columns 284-301: Unpacked limbs from state[0] (18 selected indices from 28-limb Felt252)
        // Use the Width27 M31 output (final_out_0) for unpacking, which corresponds to
        // poseidon_full_round_chain_output cols 254-263
        m31 felt252_limbs_0[28];
        Felt252Width27 width27_output_0;
        for (int i = 0; i < 10; i++) {
            width27_output_0.limbs[i] = final_out_0[i];  // m31 is uint32_t, no .val field
        }
        unpack_felt252_from_width27(width27_output_0, felt252_limbs_0);

        // Extract 18 specific limbs (skip every 3rd starting from index 2)
        int unpack_indices[18] = {0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25};
        for (int i = 0; i < 18; i++) {
            traces[284 + i][row] = felt252_limbs_0[unpack_indices[i]];  // Columns 284-301
        }

        // ============ Read output state 0 (column 302+) ============
        m31 output_addr_0 = add(instance_addr, M31_3_val);
        m31 output_state_0_id = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            output_addr_0,
            &output_state_0_id
        );
        traces[302][row] = output_state_0_id;
        lookup_memory_address_to_id_3[0][row] = output_addr_0;
        lookup_memory_address_to_id_3[1][row] = output_state_0_id;

        // Read 28 limbs for output state 0
        m31 output_state_0_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            output_state_0_id,
            output_state_0_limbs
        );

        lookup_memory_id_to_big_3[0][row] = output_state_0_id;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_3[1 + i][row] = output_state_0_limbs[i];
        }

        // ============ Read output state 1 (column 321+) ============
        m31 output_addr_1 = add(instance_addr, M31_4_val);
        m31 output_state_1_id = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            output_addr_1,
            &output_state_1_id
        );
        traces[321][row] = output_state_1_id;
        lookup_memory_address_to_id_4[0][row] = output_addr_1;
        lookup_memory_address_to_id_4[1][row] = output_state_1_id;

        // Read 28 limbs for output state 1
        m31 output_state_1_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            output_state_1_id,
            output_state_1_limbs
        );

        lookup_memory_id_to_big_4[0][row] = output_state_1_id;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_4[1 + i][row] = output_state_1_limbs[i];
        }

        // ============ Read output state 2 (column 340) ============
        m31 output_addr_2 = add(instance_addr, M31_5_val);
        m31 output_state_2_id = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            output_addr_2,
            &output_state_2_id
        );
        traces[340][row] = output_state_2_id;
        lookup_memory_address_to_id_5[0][row] = output_addr_2;
        lookup_memory_address_to_id_5[1][row] = output_state_2_id;

        // Read 28 limbs for output state 2
        m31 output_state_2_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            output_state_2_id,
            output_state_2_limbs
        );

        lookup_memory_id_to_big_5[0][row] = output_state_2_id;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_5[1 + i][row] = output_state_2_limbs[i];
        }

        // Columns 303-320: Unpacked limbs from state[1] (18 selected indices)
        m31 felt252_limbs_1[28];
        Felt252Width27 width27_1;
        felt252field_to_width27_m31(state[1], width27_1);
        unpack_felt252_from_width27(width27_1, felt252_limbs_1);

        for (int i = 0; i < 18; i++) {
            traces[303 + i][row] = felt252_limbs_1[unpack_indices[i]];  // Columns 303-320
        }

        // Columns 322-339: Unpacked limbs from state[2] (18 selected indices)
        m31 felt252_limbs_2[28];
        Felt252Width27 width27_2;
        felt252field_to_width27_m31(state[2], width27_2);
        unpack_felt252_from_width27(width27_2, felt252_limbs_2);

        for (int i = 0; i < 18; i++) {
            traces[322 + i][row] = felt252_limbs_2[unpack_indices[i]];  // Columns 322-339
        }

        // Compute carries for combination1 (cols 232-242) -> range_check_4_4_4_4_2, _3, range_check_4_4_1
        // Formula: 4*partial_out_0 + 2*partial_out_1 + partial_out_2 + COMB1_BIAS - comb1_output
        // Bias constants for combination1 (Width27 limbs)
        const m31 COMB1_BIAS[10] = {
            40454143, 49554771, 55508188, 116986206, 88680813,
            45553283, 62360091, 77099918, 22899501, 99
        };
        {
            m31 p_coef_242 = traces[242][row];
            m31 carries_comb1[9];  // M31 values!

            // Carry computation for combination1 using M31 arithmetic
            // carry_0 = (4*state0 + 2*state1 + state2 + BIAS - comb1 - p_coef) * 16
            m31 four_s0_0 = mul(4, partial_out_0[0]);
            m31 two_s1_0 = mul(2, partial_out_1[0]);
            m31 sum_0 = add(add(add(add(four_s0_0, two_s1_0), partial_out_2[0]),
                               COMB1_BIAS[0]), neg(comb1_width27[0]));
            carries_comb1[0] = mul(sub(sum_0, p_coef_242), 16);

            // Compute carries 1-8 using M31 arithmetic
            for (int i = 1; i < 9; i++) {
                m31 four_s0_i = mul(4, partial_out_0[i]);
                m31 two_s1_i = mul(2, partial_out_1[i]);
                m31 sum_i = add(add(add(add(add(four_s0_i, two_s1_i), partial_out_2[i]),
                                       COMB1_BIAS[i]), neg(comb1_width27[i])),
                               carries_comb1[i-1]);
                // Special case for limb 7: subtract p_coef * 136
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_242, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_comb1[i] = mul(sum_i, 16);
            }

            // Output range_check_4_4_4_4_2: [p_coef + 1, carry_0 + 1, carry_1 + 1, carry_2 + 1]
            lookup_range_check_4_4_4_4_2[0][row] = add(p_coef_242, 1);
            for (int i = 0; i < 3; i++) {
                lookup_range_check_4_4_4_4_2[1 + i][row] = add(carries_comb1[i], 1);
            }

            // Output range_check_4_4_4_4_3: [carry_3 + 1, carry_4 + 1, carry_5 + 1, carry_6 + 1]
            for (int i = 0; i < 4; i++) {
                lookup_range_check_4_4_4_4_3[i][row] = add(carries_comb1[3 + i], 1);
            }

            // Output range_check_4_4_1: [carry_7 + 1, carry_8 + 1]
            lookup_range_check_4_4_1[0][row] = add(carries_comb1[7], 1);
            lookup_range_check_4_4_1[1][row] = add(carries_comb1[8], 1);
        }

        // Compute carries for combination2 (cols 243-253) -> range_check_4_4_4_4_4, _5, range_check_4_4_2
        // Formula: 4*partial_out_2 + 2*partial_out_3 + comb1_output + COMB2_BIAS - comb2_output
        // Bias constants for combination2 (Width27 limbs)
        const m31 COMB2_BIAS[10] = {
            48383197, 48193339, 55955004, 65659846, 68491350,
            119023582, 33439011, 58475513, 18765944, 20
        };
        {
            m31 p_coef_253 = traces[253][row];
            m31 carries_comb2[9];  // M31 values!

            // Carry computation for combination2 using M31 arithmetic
            // carry_0 = (4*state2 + 2*state3 + comb1 + BIAS - comb2 - p_coef) * 16
            m31 four_s2_0 = mul(4, partial_out_2[0]);
            m31 two_s3_0 = mul(2, partial_out_3[0]);
            m31 sum_0 = add(add(add(add(four_s2_0, two_s3_0), comb1_width27[0]),
                               COMB2_BIAS[0]), neg(comb2_width27[0]));
            carries_comb2[0] = mul(sub(sum_0, p_coef_253), 16);

            // Compute carries 1-8 using M31 arithmetic
            for (int i = 1; i < 9; i++) {
                m31 four_s2_i = mul(4, partial_out_2[i]);
                m31 two_s3_i = mul(2, partial_out_3[i]);
                m31 sum_i = add(add(add(add(add(four_s2_i, two_s3_i), comb1_width27[i]),
                                       COMB2_BIAS[i]), neg(comb2_width27[i])),
                               carries_comb2[i-1]);
                // Special case for limb 7: subtract p_coef * 136
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_253, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_comb2[i] = mul(sum_i, 16);
            }

            // Output range_check_4_4_4_4_4: [p_coef + 1, carry_0 + 1, carry_1 + 1, carry_2 + 1]
            lookup_range_check_4_4_4_4_4[0][row] = add(p_coef_253, 1);
            for (int i = 0; i < 3; i++) {
                lookup_range_check_4_4_4_4_4[1 + i][row] = add(carries_comb2[i], 1);
            }

            // Output range_check_4_4_4_4_5: [carry_3 + 1, carry_4 + 1, carry_5 + 1, carry_6 + 1]
            for (int i = 0; i < 4; i++) {
                lookup_range_check_4_4_4_4_5[i][row] = add(carries_comb2[3 + i], 1);
            }

            // Output range_check_4_4_2: [carry_7 + 1, carry_8 + 1]
            lookup_range_check_4_4_2[0][row] = add(carries_comb2[7], 1);
            lookup_range_check_4_4_2[1][row] = add(carries_comb2[8], 1);
        }
    }

    // Copy columns 120-283 to lookup_base_trace_cols for interaction kernels
    // This data is needed for interaction kernels that access base_trace columns
    if (row < n_rows) {
        for (int i = 0; i < 164; i++) {
            lookup_base_trace_cols[i][row] = traces[120 + i][row];
        }
    }
}

// ============================================================================
// Interaction trace generation kernels
// ============================================================================

// Global debug flag for column number
__device__ int g_debug_col = -1;

// Kernel template for combining two relation lookups
template<int N_COLS_0, int N_COLS_1>
__global__ void gen_poseidon_interaction_col_gen_add_kernel(
    LookupElementsBasic<N_COLS_0> *relation_0,
    LookupElementsBasic<N_COLS_1> *relation_1,
    m31 **lookup_0,
    m31 **lookup_1,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    int col_debug = -1
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Load lookup values
    m31 values_0[N_COLS_0];
    m31 values_1[N_COLS_1];

    for (int i = 0; i < N_COLS_0; i++) {
        values_0[i] = lookup_0[i][row];
    }
    for (int i = 0; i < N_COLS_1; i++) {
        values_1[i] = lookup_1[i][row];
    }

    // Compute denominators
    qm31 denom0 = relation_0->combine(values_0, N_COLS_0);
    qm31 denom1 = relation_1->combine(values_1, N_COLS_1);

    // Compute combined fraction: (denom0 + denom1) / (denom0 * denom1)
    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    // Store numerator components
    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;

    // Store denominator for batch inverse
    denom[row] = prod;
}

// Kernel template for subtracting two relation lookups (for chain relations)
template<int N_COLS_0, int N_COLS_1>
__global__ void gen_poseidon_interaction_col_gen_sub_kernel(
    LookupElementsBasic<N_COLS_0> *relation_0,
    LookupElementsBasic<N_COLS_1> *relation_1,
    m31 **lookup_0,
    m31 **lookup_1,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Load lookup values
    m31 values_0[N_COLS_0];
    m31 values_1[N_COLS_1];

    for (int i = 0; i < N_COLS_0; i++) {
        values_0[i] = lookup_0[i][row];
    }
    for (int i = 0; i < N_COLS_1; i++) {
        values_1[i] = lookup_1[i][row];
    }

    // Compute denominators
    qm31 denom0 = relation_0->combine(values_0, N_COLS_0);
    qm31 denom1 = relation_1->combine(values_1, N_COLS_1);

    // Compute combined fraction: (denom0 - denom1) / (denom0 * denom1)
    qm31 diff = sub(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    // Store numerator components
    numerator0[row] = diff.a.a;
    numerator1[row] = diff.a.b;
    numerator2[row] = diff.b.a;
    numerator3[row] = diff.b.b;

    // Store denominator for batch inverse
    denom[row] = prod;
}

// Kernel for single lookup (last column with numerator = 1)
template<int N_COLS>
__global__ void gen_poseidon_interaction_col_single_kernel(
    LookupElementsBasic<N_COLS> *relation,
    m31 **lookup,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Load lookup values
    m31 values[N_COLS];
    for (int i = 0; i < N_COLS; i++) {
        values[i] = lookup[i][row];
    }

    // Compute denominator
    qm31 d = relation->combine(values, N_COLS);

    // numerator = 1
    numerator0[row] = M31_1_val;
    numerator1[row] = M31_0_val;
    numerator2[row] = M31_0_val;
    numerator3[row] = M31_0_val;

    // Store denominator for batch inverse
    denom[row] = d;
}

// Comprehensive interaction kernel that reads from base_trace and generates all logup fractions
// Column 3: poseidon_full_round_chain_0 - poseidon_full_round_chain_1 (SUBTRACT)
__global__ void gen_poseidon_interaction_col3_kernel(
    PoseidonFullRoundChain *relation,
    m31 **lookup_pfrc_0,
    m31 **lookup_base_trace_cols,  // base trace cols 120-283, indexed as [col - 120]
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // poseidon_full_round_chain_0: [chain_id, round=0, cols 87-96, cols 98-107, cols 109-118]
    m31 values_0[32];
    for (int i = 0; i < 32; i++) {
        values_0[i] = lookup_pfrc_0[i][row];
    }

    // poseidon_full_round_chain_1: [chain_id=row*2, round=4, cols 120-149]
    // Reconstructed from base trace columns instead of lookup array
    m31 values_1[32];
    values_1[0] = lookup_pfrc_0[0][row];  // chain_id = same as pfrc_0 (row*2)
    values_1[1] = M31_4_val;              // round = 4
    for (int i = 0; i < 30; i++) {
        values_1[2 + i] = lookup_base_trace_cols[i][row];  // cols 120-149
    }

    // Compute denominators
    qm31 denom0 = relation->combine(values_0, 32);
    qm31 denom1 = relation->combine(values_1, 32);

    // numerator = denom0 - denom1 (SUBTRACT for chain relations)
    qm31 diff = sub(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = diff.a.a;
    numerator1[row] = diff.a.b;
    numerator2[row] = diff.b.a;
    numerator3[row] = diff.b.b;
    denom[row] = prod;
}

// Column 4: range_check_felt_252_width_27_0 + range_check_felt_252_width_27_1 (ADD)
// Now uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col4_kernel(
    RangeCheckFelt252Width27 *relation,
    m31 **lookup_base_trace_cols,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_felt_252_width_27_0: [cols 120-129] -> indices 0-9
    m31 values_0[10];
    for (int i = 0; i < 10; i++) values_0[i] = lookup_base_trace_cols[0 + i][row];

    // range_check_felt_252_width_27_1: [cols 130-139] -> indices 10-19
    m31 values_1[10];
    for (int i = 0; i < 10; i++) values_1[i] = lookup_base_trace_cols[10 + i][row];

    qm31 denom0 = relation->combine(values_0, 10);
    qm31 denom1 = relation->combine(values_1, 10);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 5: cube_252_0 + range_check_3_3_3_3_3_0 (ADD)
// Uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col5_kernel(
    Cube252 *cube_rel,
    RangeCheck_3_3_3_3_3 *rc_rel,
    m31 **lookup_base_trace_cols,
    m31 **lookup_rc_33333_0,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // cube_252_0: [cols 140-149 (input), cols 150-159 (output)]
    // cols 140-149 -> indices 20-29, cols 150-159 -> indices 30-39
    m31 cube_values[20];
    for (int i = 0; i < 10; i++) cube_values[i] = lookup_base_trace_cols[20 + i][row];
    for (int i = 0; i < 10; i++) cube_values[10 + i] = lookup_base_trace_cols[30 + i][row];

    // range_check_3_3_3_3_3_0: [p_coef+3, carries+3] from lookup data
    m31 rc_values[5];
    for (int i = 0; i < 5; i++) rc_values[i] = lookup_rc_33333_0[i][row];

    qm31 denom0 = cube_rel->combine(cube_values, 20);
    qm31 denom1 = rc_rel->combine(rc_values, 5);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 6: range_check_3_3_3_3_3_1 + cube_252_1 (ADD)
// Uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col6_kernel(
    RangeCheck_3_3_3_3_3 *rc_rel,
    Cube252 *cube_rel,
    m31 **lookup_base_trace_cols,
    m31 **lookup_rc_33333_1,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_3_3_3_3_3_1: [carries+3] from lookup data
    m31 rc_values[5];
    for (int i = 0; i < 5; i++) rc_values[i] = lookup_rc_33333_1[i][row];

    // cube_252_1: [cols 160-169 (input lc_partial), cols 171-180 (output cube_lc_partial)]
    // cols 160-169 -> indices 40-49, cols 171-180 -> indices 51-60
    m31 cube_values[20];
    for (int i = 0; i < 10; i++) cube_values[i] = lookup_base_trace_cols[40 + i][row];
    for (int i = 0; i < 10; i++) cube_values[10 + i] = lookup_base_trace_cols[51 + i][row];

    qm31 denom0 = rc_rel->combine(rc_values, 5);
    qm31 denom1 = cube_rel->combine(cube_values, 20);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 7: range_check_4_4_4_4_0 + range_check_4_4_4_4_1 (ADD)
__global__ void gen_poseidon_interaction_col7_kernel(
    RangeCheck_4_4_4_4 *relation,
    m31 **lookup_rc_4444_0,
    m31 **lookup_rc_4444_1,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_4_4_4_4_0: [p_coef+3, carries+3] from lookup data
    m31 values_0[4];
    for (int i = 0; i < 4; i++) values_0[i] = lookup_rc_4444_0[i][row];

    // range_check_4_4_4_4_1: [carries+3] from lookup data
    m31 values_1[4];
    for (int i = 0; i < 4; i++) values_1[i] = lookup_rc_4444_1[i][row];

    qm31 denom0 = relation->combine(values_0, 4);
    qm31 denom1 = relation->combine(values_1, 4);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 8: range_check_4_4_0 + poseidon_3_partial_rounds_chain_0 -> (denom1 - denom0)
// Uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col8_kernel(
    RangeCheck_4_4 *rc_rel,
    Poseidon3PartialRoundsChain *chain_rel,
    m31 **lookup_base_trace_cols,
    m31 **lookup_rc_44_0,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_4_4_0: [carries+3] from lookup data
    m31 rc_values[2];
    rc_values[0] = lookup_rc_44_0[0][row];
    rc_values[1] = lookup_rc_44_0[1][row];

    // poseidon_3_partial_rounds_chain_0: [chain_id, round, state[0..3] as Width27]
    // This is 42 elements
    m31 chain_values[42];
    chain_values[0] = row;  // chain_id = seq = row index
    chain_values[1] = M31_4_val;  // round = 4
    // state values from cols 150-159 (cube_state_2), 160-169 (lc_partial), 171-180 (cube_lc_partial), 181-190 (lc_partial_2)
    // cols 150-159 -> indices 30-39, cols 160-169 -> indices 40-49
    // cols 171-180 -> indices 51-60, cols 181-190 -> indices 61-70
    for (int i = 0; i < 10; i++) chain_values[2 + i] = lookup_base_trace_cols[30 + i][row];
    for (int i = 0; i < 10; i++) chain_values[12 + i] = lookup_base_trace_cols[40 + i][row];
    for (int i = 0; i < 10; i++) chain_values[22 + i] = lookup_base_trace_cols[51 + i][row];
    for (int i = 0; i < 10; i++) chain_values[32 + i] = lookup_base_trace_cols[61 + i][row];

    qm31 denom0 = rc_rel->combine(rc_values, 2);
    qm31 denom1 = chain_rel->combine(chain_values, 42);

    // numerator = denom1 - denom0 (SUBTRACT with reversed order)
    qm31 diff = sub(denom1, denom0);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = diff.a.a;
    numerator1[row] = diff.a.b;
    numerator2[row] = diff.b.a;
    numerator3[row] = diff.b.b;
    denom[row] = prod;
}

// Column 9: poseidon_3_partial_rounds_chain_1 + range_check_4_4_4_4_2 (ADD)
// Uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col9_kernel(
    Poseidon3PartialRoundsChain *chain_rel,
    RangeCheck_4_4_4_4 *rc_rel,
    m31 **lookup_base_trace_cols,
    m31 **lookup_rc_4444_2,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // poseidon_3_partial_rounds_chain_1: [chain_id, round=31, state from cols 192-231]
    // cols 192-231 -> indices 72-111
    m31 chain_values[42];
    chain_values[0] = row;  // chain_id = seq = row index
    chain_values[1] = (m31){31};
    for (int i = 0; i < 40; i++) chain_values[2 + i] = lookup_base_trace_cols[72 + i][row];

    // range_check_4_4_4_4_2: from lookup data
    m31 rc_values[4];
    for (int i = 0; i < 4; i++) rc_values[i] = lookup_rc_4444_2[i][row];

    qm31 denom0 = chain_rel->combine(chain_values, 42);
    qm31 denom1 = rc_rel->combine(rc_values, 4);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 10: range_check_4_4_4_4_3 + range_check_4_4_1 (ADD)
__global__ void gen_poseidon_interaction_col10_kernel(
    RangeCheck_4_4_4_4 *rc4_rel,
    RangeCheck_4_4 *rc2_rel,
    m31 **lookup_rc_4444_3,
    m31 **lookup_rc_44_1,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_4_4_4_4_3: from lookup data
    m31 values_0[4];
    for (int i = 0; i < 4; i++) values_0[i] = lookup_rc_4444_3[i][row];

    // range_check_4_4_1: from lookup data
    m31 values_1[2];
    for (int i = 0; i < 2; i++) values_1[i] = lookup_rc_44_1[i][row];

    qm31 denom0 = rc4_rel->combine(values_0, 4);
    qm31 denom1 = rc2_rel->combine(values_1, 2);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 11: range_check_4_4_4_4_4 + range_check_4_4_4_4_5 (ADD)
__global__ void gen_poseidon_interaction_col11_kernel(
    RangeCheck_4_4_4_4 *relation,
    m31 **lookup_rc_4444_4,
    m31 **lookup_rc_4444_5,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_4_4_4_4_4: from lookup data
    m31 values_0[4];
    for (int i = 0; i < 4; i++) values_0[i] = lookup_rc_4444_4[i][row];

    // range_check_4_4_4_4_5: from lookup data
    m31 values_1[4];
    for (int i = 0; i < 4; i++) values_1[i] = lookup_rc_4444_5[i][row];

    qm31 denom0 = relation->combine(values_0, 4);
    qm31 denom1 = relation->combine(values_1, 4);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 12: range_check_4_4_2 + poseidon_full_round_chain_2 -> (denom1 - denom0)
// Uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col12_kernel(
    RangeCheck_4_4 *rc_rel,
    PoseidonFullRoundChain *chain_rel,
    m31 **lookup_base_trace_cols,
    m31 **lookup_rc_44_2,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // range_check_4_4_2: from lookup data
    m31 rc_values[2];
    rc_values[0] = lookup_rc_44_2[0][row];
    rc_values[1] = lookup_rc_44_2[1][row];

    // poseidon_full_round_chain_2: [chain_id, round=31, state from linear combinations]
    // cols 243-252 -> indices 123-132, cols 232-241 -> indices 112-121, cols 222-231 -> indices 102-111
    // Note: poseidon_full_round_chain uses chain_id = seq * 2 + 1 (for chains 2/3)
    m31 chain_values[32];
    chain_values[0] = row * 2 + 1;  // chain_id = seq * 2 + 1
    chain_values[1] = (m31){31};
    // State from cols 232-241 (comb1), 243-252 (comb2 without p_coef), 222-231 (partial_state[3])
    for (int i = 0; i < 10; i++) chain_values[2 + i] = lookup_base_trace_cols[123 + i][row];
    for (int i = 0; i < 10; i++) chain_values[12 + i] = lookup_base_trace_cols[112 + i][row];
    for (int i = 0; i < 10; i++) chain_values[22 + i] = lookup_base_trace_cols[102 + i][row];

    qm31 denom0 = rc_rel->combine(rc_values, 2);
    qm31 denom1 = chain_rel->combine(chain_values, 32);

    // numerator = denom1 - denom0
    qm31 diff = sub(denom1, denom0);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = diff.a.a;
    numerator1[row] = diff.a.b;
    numerator2[row] = diff.b.a;
    numerator3[row] = diff.b.b;
    denom[row] = prod;
}

// Column 13: poseidon_full_round_chain_3 + memory_address_to_id_3 (ADD)
// Uses lookup_base_trace_cols: col X maps to lookup_base_trace_cols[X - 120]
__global__ void gen_poseidon_interaction_col13_kernel(
    PoseidonFullRoundChain *chain_rel,
    MemoryAddressToId *mem_rel,
    m31 **lookup_base_trace_cols,
    m31 **lookup_addr2id_3,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // poseidon_full_round_chain_3: [chain_id, round=35, final output state cols 254-283]
    // cols 254-283 -> indices 134-163
    // Note: poseidon_full_round_chain uses chain_id = seq * 2 + 1 (for chains 2/3)
    m31 chain_values[32];
    chain_values[0] = row * 2 + 1;  // chain_id = seq * 2 + 1
    chain_values[1] = (m31){35};
    for (int i = 0; i < 30; i++) chain_values[2 + i] = lookup_base_trace_cols[134 + i][row];

    // memory_address_to_id_3 from lookup data
    m31 mem_values[2];
    mem_values[0] = lookup_addr2id_3[0][row];
    mem_values[1] = lookup_addr2id_3[1][row];

    qm31 denom0 = chain_rel->combine(chain_values, 32);
    qm31 denom1 = mem_rel->combine(mem_values, 2);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 14: memory_id_to_big_3 + memory_address_to_id_4 (ADD)
__global__ void gen_poseidon_interaction_col14_kernel(
    MemoryIdToBig *id2big_rel,
    MemoryAddressToId *addr2id_rel,
    m31 **lookup_id2big_3,
    m31 **lookup_addr2id_4,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    m31 id2big_values[29];
    for (int i = 0; i < 29; i++) id2big_values[i] = lookup_id2big_3[i][row];

    m31 addr2id_values[2];
    addr2id_values[0] = lookup_addr2id_4[0][row];
    addr2id_values[1] = lookup_addr2id_4[1][row];

    qm31 denom0 = id2big_rel->combine(id2big_values, 29);
    qm31 denom1 = addr2id_rel->combine(addr2id_values, 2);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Column 15: memory_id_to_big_4 + memory_address_to_id_5 (ADD)
__global__ void gen_poseidon_interaction_col15_kernel(
    MemoryIdToBig *id2big_rel,
    MemoryAddressToId *addr2id_rel,
    m31 **lookup_id2big_4,
    m31 **lookup_addr2id_5,
    unsigned n_rows,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    m31 id2big_values[29];
    for (int i = 0; i < 29; i++) id2big_values[i] = lookup_id2big_4[i][row];

    m31 addr2id_values[2];
    addr2id_values[0] = lookup_addr2id_5[0][row];
    addr2id_values[1] = lookup_addr2id_5[1][row];

    qm31 denom0 = id2big_rel->combine(id2big_values, 29);
    qm31 denom1 = addr2id_rel->combine(addr2id_values, 2);

    qm31 sum = add(denom0, denom1);
    qm31 prod = mul(denom0, denom1);

    numerator0[row] = sum.a.a;
    numerator1[row] = sum.a.b;
    numerator2[row] = sum.b.a;
    numerator3[row] = sum.b.b;
    denom[row] = prod;
}

// Kernel to finalize interaction column by multiplying numerator by inverse denominator
// and adding the previous column's value (cumulative sum across columns)
__global__ void gen_poseidon_interaction_finalize_col_kernel(
    unsigned n_rows,
    unsigned col_offset,
    qm31 *denom_inv,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_trace
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    qm31 inv = denom_inv[row];
    qm31 numer = {
        {numerator0[row], numerator1[row]},
        {numerator2[row], numerator3[row]}
    };

    qm31 frac = mul(numer, inv);

    // Add previous column's value for cumulative sum across columns
    qm31 prev_value = {{0, 0}, {0, 0}};
    if (col_offset > 0) {
        prev_value.a.a = interaction_trace[(col_offset - 1) * 4 + 0][row];
        prev_value.a.b = interaction_trace[(col_offset - 1) * 4 + 1][row];
        prev_value.b.a = interaction_trace[(col_offset - 1) * 4 + 2][row];
        prev_value.b.b = interaction_trace[(col_offset - 1) * 4 + 3][row];
    }

    qm31 result = add(frac, prev_value);

    // Store to interaction trace
    interaction_trace[col_offset * 4 + 0][row] = result.a.a;
    interaction_trace[col_offset * 4 + 1][row] = result.a.b;
    interaction_trace[col_offset * 4 + 2][row] = result.b.a;
    interaction_trace[col_offset * 4 + 3][row] = result.b.b;
}

// ============================================================================
// Cumsum shift kernel - computes claimed_sum by summing all elements in last column
// ============================================================================
__global__ void gen_poseidon_interaction_cumsum_shift_kernel(
    uint32_t last_index,
    uint32_t trace_size,
    m31** interaction_traces,
    m31* coordinate_sums
) {
    // Calculate indices for the 4 M31 components of the last logup column
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    // Thread-local accumulators
    m31 sum0 = {0};
    m31 sum1 = {0};
    m31 sum2 = {0};
    m31 sum3 = {0};

    // Grid-stride loop to sum elements
    for (uint32_t i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[idx0][i]);
        sum1 = add(sum1, interaction_traces[idx1][i]);
        sum2 = add(sum2, interaction_traces[idx2][i]);
        sum3 = add(sum3, interaction_traces[idx3][i]);
    }

    // Block-level reduction using shared memory
    extern __shared__ m31 shared[];
    m31* sdata0 = &shared[0];
    m31* sdata1 = &shared[blockDim.x];
    m31* sdata2 = &shared[2 * blockDim.x];
    m31* sdata3 = &shared[3 * blockDim.x];

    sdata0[threadIdx.x] = sum0;
    sdata1[threadIdx.x] = sum1;
    sdata2[threadIdx.x] = sum2;
    sdata3[threadIdx.x] = sum3;

    __syncthreads();

    // Tree reduction within block
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata0[threadIdx.x] = add(sdata0[threadIdx.x], sdata0[threadIdx.x + s]);
            sdata1[threadIdx.x] = add(sdata1[threadIdx.x], sdata1[threadIdx.x + s]);
            sdata2[threadIdx.x] = add(sdata2[threadIdx.x], sdata2[threadIdx.x + s]);
            sdata3[threadIdx.x] = add(sdata3[threadIdx.x], sdata3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // First thread in each block atomically adds to global sum
    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sdata0[0]);
        atomic_add(&coordinate_sums[1], sdata1[0]);
        atomic_add(&coordinate_sums[2], sdata2[0]);
        atomic_add(&coordinate_sums[3], sdata3[0]);
    }
}

// ============================================================================
// Coordinate prefix sum kernel - subtracts claimed_sum/trace_size from last column
// ============================================================================
__global__ void gen_poseidon_interaction_coord_prefix_sum_kernel(
    m31* coordinate_sums,
    uint32_t last_index,
    uint32_t trace_size,
    m31** interaction_traces
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trace_size) return;

    // Compute cumsum_shift = claimed_sum / trace_size
    qm31 claimed_sum = qm31 {
        cm31{coordinate_sums[0], coordinate_sums[1]},
        cm31{coordinate_sums[2], coordinate_sums[3]}
    };
    qm31 cumsum_shift = div(claimed_sum, m31{trace_size});

    // Subtract cumsum_shift from each element in the last column
    interaction_traces[4 * last_index - 4][idx] = sub(interaction_traces[4 * last_index - 4][idx], cumsum_shift.a.a);
    interaction_traces[4 * last_index - 3][idx] = sub(interaction_traces[4 * last_index - 3][idx], cumsum_shift.a.b);
    interaction_traces[4 * last_index - 2][idx] = sub(interaction_traces[4 * last_index - 2][idx], cumsum_shift.b.a);
    interaction_traces[4 * last_index - 1][idx] = sub(interaction_traces[4 * last_index - 1][idx], cumsum_shift.b.b);
}

// ============================================================================
// External C interface for Rust FFI
// ============================================================================

extern "C" void gen_poseidon_builtin_trace(
    m31** traces,
    unsigned log_size,
    unsigned segment_start,
    unsigned* memory_address_to_id_address_to_raw_id,
    unsigned** memory_id_to_big_transposed_big_values,
    unsigned* memory_id_to_big_small_values,

    // Lookup data arrays - memory_address_to_id (6 lookups)
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,
    m31** lookup_memory_address_to_id_3,
    m31** lookup_memory_address_to_id_4,
    m31** lookup_memory_address_to_id_5,
    // Lookup data arrays - memory_id_to_big (6 lookups)
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,
    m31** lookup_memory_id_to_big_3,
    m31** lookup_memory_id_to_big_4,
    m31** lookup_memory_id_to_big_5,
    // Lookup data arrays - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
    m31** lookup_range_check_3_3_3_3_3_0,
    m31** lookup_range_check_3_3_3_3_3_1,
    // Lookup data arrays - range_check_4_4_4_4 (6 lookups, 4 elements each)
    m31** lookup_range_check_4_4_4_4_0,
    m31** lookup_range_check_4_4_4_4_1,
    m31** lookup_range_check_4_4_4_4_2,
    m31** lookup_range_check_4_4_4_4_3,
    m31** lookup_range_check_4_4_4_4_4,
    m31** lookup_range_check_4_4_4_4_5,
    // Lookup data arrays - range_check_4_4 (3 lookups, 2 elements each)
    m31** lookup_range_check_4_4_0,
    m31** lookup_range_check_4_4_1,
    m31** lookup_range_check_4_4_2,
    // Lookup data arrays - poseidon_full_round_chain (8 lookups, 32 elements each)
    m31** lookup_poseidon_full_round_chain_0,
    m31** lookup_poseidon_full_round_chain_1,
    m31** lookup_poseidon_full_round_chain_2,
    m31** lookup_poseidon_full_round_chain_3,
    m31** lookup_poseidon_full_round_chain_4,
    m31** lookup_poseidon_full_round_chain_5,
    m31** lookup_poseidon_full_round_chain_6,
    m31** lookup_poseidon_full_round_chain_7,
    // Lookup data arrays - poseidon_3_partial_rounds_chain (27 lookups, 42 elements each)
    m31** lookup_poseidon_3_partial_rounds_chain_0,
    m31** lookup_poseidon_3_partial_rounds_chain_1,
    m31** lookup_poseidon_3_partial_rounds_chain_2,
    m31** lookup_poseidon_3_partial_rounds_chain_3,
    m31** lookup_poseidon_3_partial_rounds_chain_4,
    m31** lookup_poseidon_3_partial_rounds_chain_5,
    m31** lookup_poseidon_3_partial_rounds_chain_6,
    m31** lookup_poseidon_3_partial_rounds_chain_7,
    m31** lookup_poseidon_3_partial_rounds_chain_8,
    m31** lookup_poseidon_3_partial_rounds_chain_9,
    m31** lookup_poseidon_3_partial_rounds_chain_10,
    m31** lookup_poseidon_3_partial_rounds_chain_11,
    m31** lookup_poseidon_3_partial_rounds_chain_12,
    m31** lookup_poseidon_3_partial_rounds_chain_13,
    m31** lookup_poseidon_3_partial_rounds_chain_14,
    m31** lookup_poseidon_3_partial_rounds_chain_15,
    m31** lookup_poseidon_3_partial_rounds_chain_16,
    m31** lookup_poseidon_3_partial_rounds_chain_17,
    m31** lookup_poseidon_3_partial_rounds_chain_18,
    m31** lookup_poseidon_3_partial_rounds_chain_19,
    m31** lookup_poseidon_3_partial_rounds_chain_20,
    m31** lookup_poseidon_3_partial_rounds_chain_21,
    m31** lookup_poseidon_3_partial_rounds_chain_22,
    m31** lookup_poseidon_3_partial_rounds_chain_23,
    m31** lookup_poseidon_3_partial_rounds_chain_24,
    m31** lookup_poseidon_3_partial_rounds_chain_25,
    m31** lookup_poseidon_3_partial_rounds_chain_26,
    // Base trace cols 120-283 (164 columns) for interaction kernels
    m31** lookup_base_trace_cols
) {
    timer global_timer;
    global_timer.start("generate poseidon_builtin base trace");

    unsigned trace_size = 1 << log_size;
    unsigned num_blocks = (trace_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Clone arrays to device
    m31** device_traces = clone_to_device<m31*>(traces, N_TRACE_COLUMNS);

    // Zero-initialize all trace columns to prevent garbage values
    // This fixes the bug where some rows (e.g., 111, 226) have uninitialized data
    for (int i = 0; i < N_TRACE_COLUMNS; i++) {
        cudaMemsetAsync(traces[i], 0, trace_size * sizeof(m31), 0);
    }

    m31** device_lookup_addr2id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31** device_lookup_addr2id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31** device_lookup_addr2id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31** device_lookup_addr2id_3 = clone_to_device<m31*>(lookup_memory_address_to_id_3, 2);
    m31** device_lookup_addr2id_4 = clone_to_device<m31*>(lookup_memory_address_to_id_4, 2);
    m31** device_lookup_addr2id_5 = clone_to_device<m31*>(lookup_memory_address_to_id_5, 2);

    // Zero-initialize lookup_memory_address_to_id arrays
    for (int j = 0; j < 2; j++) {
        cudaMemsetAsync(lookup_memory_address_to_id_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_address_to_id_1[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_address_to_id_2[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_address_to_id_3[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_address_to_id_4[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_address_to_id_5[j], 0, trace_size * sizeof(m31), 0);
    }

    m31** device_lookup_id2big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31** device_lookup_id2big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31** device_lookup_id2big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31** device_lookup_id2big_3 = clone_to_device<m31*>(lookup_memory_id_to_big_3, 29);
    m31** device_lookup_id2big_4 = clone_to_device<m31*>(lookup_memory_id_to_big_4, 29);
    m31** device_lookup_id2big_5 = clone_to_device<m31*>(lookup_memory_id_to_big_5, 29);

    // Zero-initialize lookup_memory_id_to_big arrays
    for (int j = 0; j < 29; j++) {
        cudaMemsetAsync(lookup_memory_id_to_big_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_id_to_big_1[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_id_to_big_2[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_id_to_big_3[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_id_to_big_4[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_memory_id_to_big_5[j], 0, trace_size * sizeof(m31), 0);
    }

    // Clone range_check_3_3_3_3_3 lookup data to device (2 lookups, 5 elements each)
    m31** device_lookup_rc_33333_0 = clone_to_device<m31*>(lookup_range_check_3_3_3_3_3_0, 5);
    m31** device_lookup_rc_33333_1 = clone_to_device<m31*>(lookup_range_check_3_3_3_3_3_1, 5);

    // Zero-initialize range_check_3_3_3_3_3 arrays
    for (int j = 0; j < 5; j++) {
        cudaMemsetAsync(lookup_range_check_3_3_3_3_3_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_3_3_3_3_3_1[j], 0, trace_size * sizeof(m31), 0);
    }

    // Clone range_check_4_4_4_4 lookup data to device (6 lookups, 4 elements each)
    m31** device_lookup_rc_4444_0 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_0, 4);
    m31** device_lookup_rc_4444_1 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_1, 4);
    m31** device_lookup_rc_4444_2 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_2, 4);
    m31** device_lookup_rc_4444_3 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_3, 4);
    m31** device_lookup_rc_4444_4 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_4, 4);
    m31** device_lookup_rc_4444_5 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_5, 4);

    // Zero-initialize range_check_4_4_4_4 arrays
    for (int j = 0; j < 4; j++) {
        cudaMemsetAsync(lookup_range_check_4_4_4_4_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_4_4_1[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_4_4_2[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_4_4_3[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_4_4_4[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_4_4_5[j], 0, trace_size * sizeof(m31), 0);
    }

    // Clone range_check_4_4 lookup data to device (3 lookups, 2 elements each)
    m31** device_lookup_rc_44_0 = clone_to_device<m31*>(lookup_range_check_4_4_0, 2);
    m31** device_lookup_rc_44_1 = clone_to_device<m31*>(lookup_range_check_4_4_1, 2);
    m31** device_lookup_rc_44_2 = clone_to_device<m31*>(lookup_range_check_4_4_2, 2);

    // Zero-initialize range_check_4_4 arrays
    for (int j = 0; j < 2; j++) {
        cudaMemsetAsync(lookup_range_check_4_4_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_1[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_range_check_4_4_2[j], 0, trace_size * sizeof(m31), 0);
    }

    // Clone poseidon_full_round_chain lookup data to device (8 lookups, 32 elements each)
    m31** device_lookup_pfrc_0 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_0, 32);
    m31** device_lookup_pfrc_1 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_1, 32);
    m31** device_lookup_pfrc_2 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_2, 32);
    m31** device_lookup_pfrc_3 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_3, 32);
    m31** device_lookup_pfrc_4 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_4, 32);
    m31** device_lookup_pfrc_5 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_5, 32);
    m31** device_lookup_pfrc_6 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_6, 32);
    m31** device_lookup_pfrc_7 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_7, 32);

    // Zero-initialize poseidon_full_round_chain arrays
    for (int j = 0; j < 32; j++) {
        cudaMemsetAsync(lookup_poseidon_full_round_chain_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_1[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_2[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_3[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_4[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_5[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_6[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_full_round_chain_7[j], 0, trace_size * sizeof(m31), 0);
    }

    // Clone poseidon_3_partial_rounds_chain lookup data to device (27 lookups, 42 elements each)
    m31** device_lookup_p3prc_0 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_0, 42);
    m31** device_lookup_p3prc_1 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_1, 42);
    m31** device_lookup_p3prc_2 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_2, 42);
    m31** device_lookup_p3prc_3 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_3, 42);
    m31** device_lookup_p3prc_4 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_4, 42);
    m31** device_lookup_p3prc_5 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_5, 42);
    m31** device_lookup_p3prc_6 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_6, 42);
    m31** device_lookup_p3prc_7 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_7, 42);
    m31** device_lookup_p3prc_8 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_8, 42);
    m31** device_lookup_p3prc_9 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_9, 42);
    m31** device_lookup_p3prc_10 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_10, 42);
    m31** device_lookup_p3prc_11 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_11, 42);
    m31** device_lookup_p3prc_12 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_12, 42);
    m31** device_lookup_p3prc_13 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_13, 42);
    m31** device_lookup_p3prc_14 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_14, 42);
    m31** device_lookup_p3prc_15 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_15, 42);
    m31** device_lookup_p3prc_16 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_16, 42);
    m31** device_lookup_p3prc_17 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_17, 42);
    m31** device_lookup_p3prc_18 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_18, 42);
    m31** device_lookup_p3prc_19 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_19, 42);
    m31** device_lookup_p3prc_20 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_20, 42);
    m31** device_lookup_p3prc_21 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_21, 42);
    m31** device_lookup_p3prc_22 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_22, 42);
    m31** device_lookup_p3prc_23 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_23, 42);
    m31** device_lookup_p3prc_24 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_24, 42);
    m31** device_lookup_p3prc_25 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_25, 42);
    m31** device_lookup_p3prc_26 = clone_to_device<m31*>(lookup_poseidon_3_partial_rounds_chain_26, 42);

    // Zero-initialize poseidon_3_partial_rounds_chain arrays
    for (int j = 0; j < 42; j++) {
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_0[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_1[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_2[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_3[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_4[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_5[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_6[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_7[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_8[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_9[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_10[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_11[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_12[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_13[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_14[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_15[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_16[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_17[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_18[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_19[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_20[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_21[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_22[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_23[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_24[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_25[j], 0, trace_size * sizeof(m31), 0);
        cudaMemsetAsync(lookup_poseidon_3_partial_rounds_chain_26[j], 0, trace_size * sizeof(m31), 0);
    }

    // Clone base_trace_cols lookup data to device (164 columns: 120-283)
    m31** device_lookup_base_trace_cols = clone_to_device<m31*>(lookup_base_trace_cols, 164);

    // Zero-initialize base_trace_cols arrays
    for (int j = 0; j < 164; j++) {
        cudaMemsetAsync(lookup_base_trace_cols[j], 0, trace_size * sizeof(m31), 0);
    }


    // Clone memory_id_to_big_transposed_big_values to device (array of 8 pointers)
    unsigned** device_transposed_big_values = clone_to_device<unsigned*>(memory_id_to_big_transposed_big_values, 8);

    // Launch base trace kernel
    generate_poseidon_builtin_base_trace_kernel<<<num_blocks, BLOCK_SIZE>>>(
        device_traces,
        device_lookup_addr2id_0,
        device_lookup_addr2id_1,
        device_lookup_addr2id_2,
        device_lookup_addr2id_3,
        device_lookup_addr2id_4,
        device_lookup_addr2id_5,
        device_lookup_id2big_0,
        device_lookup_id2big_1,
        device_lookup_id2big_2,
        device_lookup_id2big_3,
        device_lookup_id2big_4,
        device_lookup_id2big_5,
        device_lookup_rc_33333_0,
        device_lookup_rc_33333_1,
        device_lookup_rc_4444_0,
        device_lookup_rc_4444_1,
        device_lookup_rc_4444_2,
        device_lookup_rc_4444_3,
        device_lookup_rc_4444_4,
        device_lookup_rc_4444_5,
        device_lookup_rc_44_0,
        device_lookup_rc_44_1,
        device_lookup_rc_44_2,
        device_lookup_pfrc_0,
        device_lookup_pfrc_1,
        device_lookup_pfrc_2,
        device_lookup_pfrc_3,
        device_lookup_pfrc_4,
        device_lookup_pfrc_5,
        device_lookup_pfrc_6,
        device_lookup_pfrc_7,
        device_lookup_p3prc_0,
        device_lookup_p3prc_1,
        device_lookup_p3prc_2,
        device_lookup_p3prc_3,
        device_lookup_p3prc_4,
        device_lookup_p3prc_5,
        device_lookup_p3prc_6,
        device_lookup_p3prc_7,
        device_lookup_p3prc_8,
        device_lookup_p3prc_9,
        device_lookup_p3prc_10,
        device_lookup_p3prc_11,
        device_lookup_p3prc_12,
        device_lookup_p3prc_13,
        device_lookup_p3prc_14,
        device_lookup_p3prc_15,
        device_lookup_p3prc_16,
        device_lookup_p3prc_17,
        device_lookup_p3prc_18,
        device_lookup_p3prc_19,
        device_lookup_p3prc_20,
        device_lookup_p3prc_21,
        device_lookup_p3prc_22,
        device_lookup_p3prc_23,
        device_lookup_p3prc_24,
        device_lookup_p3prc_25,
        device_lookup_p3prc_26,
        device_lookup_base_trace_cols,
        segment_start,
        memory_address_to_id_address_to_raw_id,
        device_transposed_big_values,
        memory_id_to_big_small_values,
        trace_size,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Cleanup device memory
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_addr2id_1);
    cuda_free_memory(device_lookup_addr2id_2);
    cuda_free_memory(device_lookup_addr2id_3);
    cuda_free_memory(device_lookup_addr2id_4);
    cuda_free_memory(device_lookup_addr2id_5);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_lookup_id2big_1);
    cuda_free_memory(device_lookup_id2big_2);
    cuda_free_memory(device_lookup_id2big_3);
    cuda_free_memory(device_lookup_id2big_4);
    cuda_free_memory(device_lookup_id2big_5);
    cuda_free_memory(device_lookup_rc_33333_0);
    cuda_free_memory(device_lookup_rc_33333_1);
    cuda_free_memory(device_lookup_rc_4444_0);
    cuda_free_memory(device_lookup_rc_4444_1);
    cuda_free_memory(device_lookup_rc_4444_2);
    cuda_free_memory(device_lookup_rc_4444_3);
    cuda_free_memory(device_lookup_rc_4444_4);
    cuda_free_memory(device_lookup_rc_4444_5);
    cuda_free_memory(device_lookup_rc_44_0);
    cuda_free_memory(device_lookup_rc_44_1);
    cuda_free_memory(device_lookup_rc_44_2);
    cuda_free_memory(device_lookup_pfrc_0);
    cuda_free_memory(device_lookup_pfrc_1);
    cuda_free_memory(device_lookup_pfrc_2);
    cuda_free_memory(device_lookup_pfrc_3);
    cuda_free_memory(device_lookup_pfrc_4);
    cuda_free_memory(device_lookup_pfrc_5);
    cuda_free_memory(device_lookup_pfrc_6);
    cuda_free_memory(device_lookup_pfrc_7);
    cuda_free_memory(device_lookup_p3prc_0);
    cuda_free_memory(device_lookup_p3prc_1);
    cuda_free_memory(device_lookup_p3prc_2);
    cuda_free_memory(device_lookup_p3prc_3);
    cuda_free_memory(device_lookup_p3prc_4);
    cuda_free_memory(device_lookup_p3prc_5);
    cuda_free_memory(device_lookup_p3prc_6);
    cuda_free_memory(device_lookup_p3prc_7);
    cuda_free_memory(device_lookup_p3prc_8);
    cuda_free_memory(device_lookup_p3prc_9);
    cuda_free_memory(device_lookup_p3prc_10);
    cuda_free_memory(device_lookup_p3prc_11);
    cuda_free_memory(device_lookup_p3prc_12);
    cuda_free_memory(device_lookup_p3prc_13);
    cuda_free_memory(device_lookup_p3prc_14);
    cuda_free_memory(device_lookup_p3prc_15);
    cuda_free_memory(device_lookup_p3prc_16);
    cuda_free_memory(device_lookup_p3prc_17);
    cuda_free_memory(device_lookup_p3prc_18);
    cuda_free_memory(device_lookup_p3prc_19);
    cuda_free_memory(device_lookup_p3prc_20);
    cuda_free_memory(device_lookup_p3prc_21);
    cuda_free_memory(device_lookup_p3prc_22);
    cuda_free_memory(device_lookup_p3prc_23);
    cuda_free_memory(device_lookup_p3prc_24);
    cuda_free_memory(device_lookup_p3prc_25);
    cuda_free_memory(device_lookup_p3prc_26);
    cuda_free_memory(device_lookup_base_trace_cols);
    cuda_free_memory(device_transposed_big_values);

    global_timer.end("generate poseidon_builtin base trace");
}

extern "C" void gen_poseidon_builtin_interaction_trace(
    m31** interaction_trace,
    unsigned log_size,

    // Relation elements
    MemoryAddressToId* memory_address_to_id_relation,
    MemoryIdToBig* memory_id_to_big_relation,
    PoseidonFullRoundChain* poseidon_full_round_chain_relation,
    RangeCheckFelt252Width27* range_check_felt_252_width_27_relation,
    Cube252* cube_252_relation,
    RangeCheck_3_3_3_3_3* range_check_3_3_3_3_3_relation,
    RangeCheck_4_4_4_4* range_check_4_4_4_4_relation,
    RangeCheck_4_4* range_check_4_4_relation,
    Poseidon3PartialRoundsChain* poseidon_3_partial_rounds_chain_relation,

    // Base trace columns for reconstructing lookup data
    m31** base_trace,

    // Lookup data arrays (6x memory_address_to_id)
    m31** lookup_memory_address_to_id_0,
    m31** lookup_memory_address_to_id_1,
    m31** lookup_memory_address_to_id_2,
    m31** lookup_memory_address_to_id_3,
    m31** lookup_memory_address_to_id_4,
    m31** lookup_memory_address_to_id_5,

    // Lookup data arrays (6x memory_id_to_big)
    m31** lookup_memory_id_to_big_0,
    m31** lookup_memory_id_to_big_1,
    m31** lookup_memory_id_to_big_2,
    m31** lookup_memory_id_to_big_3,
    m31** lookup_memory_id_to_big_4,
    m31** lookup_memory_id_to_big_5,

    // Lookup data arrays - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
    m31** lookup_range_check_3_3_3_3_3_0,
    m31** lookup_range_check_3_3_3_3_3_1,

    // Lookup data arrays - range_check_4_4_4_4 (6 lookups, 4 elements each)
    m31** lookup_range_check_4_4_4_4_0,
    m31** lookup_range_check_4_4_4_4_1,
    m31** lookup_range_check_4_4_4_4_2,
    m31** lookup_range_check_4_4_4_4_3,
    m31** lookup_range_check_4_4_4_4_4,
    m31** lookup_range_check_4_4_4_4_5,

    // Lookup data arrays - range_check_4_4 (3 lookups, 2 elements each)
    m31** lookup_range_check_4_4_0,
    m31** lookup_range_check_4_4_1,
    m31** lookup_range_check_4_4_2,

    // Lookup data arrays - poseidon_full_round_chain (2 lookups, 32 elements each)
    m31** lookup_poseidon_full_round_chain_0,
    m31** lookup_poseidon_full_round_chain_1,

    // Base trace cols 120-283 (164 columns) for interaction kernels
    m31** lookup_base_trace_cols,

    unsigned* claimed_sum
) {
    timer global_timer;
    global_timer.start("generate poseidon_builtin interaction trace");

    unsigned n_rows = 1 << log_size;
    unsigned num_blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary buffers for logup computation
    qm31* d_denom = cuda_malloc<qm31>(n_rows);
    qm31* d_denom_inv = cuda_malloc<qm31>(n_rows);
    m31* d_numerator0 = cuda_malloc<m31>(n_rows);
    m31* d_numerator1 = cuda_malloc<m31>(n_rows);
    m31* d_numerator2 = cuda_malloc<m31>(n_rows);
    m31* d_numerator3 = cuda_malloc<m31>(n_rows);

    // Clone relation structs to device
    MemoryAddressToId* d_mem_addr2id_rel = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig* d_mem_id2big_rel = cuda_malloc<MemoryIdToBig>(1);

    cudaMemcpyAsync(d_mem_addr2id_rel, memory_address_to_id_relation, sizeof(MemoryAddressToId), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_mem_id2big_rel, memory_id_to_big_relation, sizeof(MemoryIdToBig), cudaMemcpyHostToDevice, 0);

    // Clone lookup data to device
    m31** d_interaction_trace = clone_to_device<m31*>(interaction_trace, N_LOGUP_COLS * 4);

    m31** d_lookup_addr2id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31** d_lookup_addr2id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31** d_lookup_addr2id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31** d_lookup_addr2id_3 = clone_to_device<m31*>(lookup_memory_address_to_id_3, 2);
    m31** d_lookup_addr2id_4 = clone_to_device<m31*>(lookup_memory_address_to_id_4, 2);
    m31** d_lookup_addr2id_5 = clone_to_device<m31*>(lookup_memory_address_to_id_5, 2);

    m31** d_lookup_id2big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31** d_lookup_id2big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31** d_lookup_id2big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31** d_lookup_id2big_3 = clone_to_device<m31*>(lookup_memory_id_to_big_3, 29);
    m31** d_lookup_id2big_4 = clone_to_device<m31*>(lookup_memory_id_to_big_4, 29);
    m31** d_lookup_id2big_5 = clone_to_device<m31*>(lookup_memory_id_to_big_5, 29);

    // Clone range_check_3_3_3_3_3 lookup data to device (2 lookups, 5 elements each)
    m31** d_lookup_rc_33333_0 = clone_to_device<m31*>(lookup_range_check_3_3_3_3_3_0, 5);
    m31** d_lookup_rc_33333_1 = clone_to_device<m31*>(lookup_range_check_3_3_3_3_3_1, 5);

    // Clone range_check_4_4_4_4 lookup data to device (6 lookups, 4 elements each)
    m31** d_lookup_rc_4444_0 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_0, 4);
    m31** d_lookup_rc_4444_1 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_1, 4);
    m31** d_lookup_rc_4444_2 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_2, 4);
    m31** d_lookup_rc_4444_3 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_3, 4);
    m31** d_lookup_rc_4444_4 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_4, 4);
    m31** d_lookup_rc_4444_5 = clone_to_device<m31*>(lookup_range_check_4_4_4_4_5, 4);

    // Clone range_check_4_4 lookup data to device (3 lookups, 2 elements each)
    m31** d_lookup_rc_44_0 = clone_to_device<m31*>(lookup_range_check_4_4_0, 2);
    m31** d_lookup_rc_44_1 = clone_to_device<m31*>(lookup_range_check_4_4_1, 2);
    m31** d_lookup_rc_44_2 = clone_to_device<m31*>(lookup_range_check_4_4_2, 2);

    // Clone poseidon_full_round_chain lookup data to device (2 lookups, 32 elements each)
    m31** d_lookup_pfrc_0 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_0, 32);
    m31** d_lookup_pfrc_1 = clone_to_device<m31*>(lookup_poseidon_full_round_chain_1, 32);

    // Clone base_trace_cols lookup data to device (164 columns: 120-283)
    m31** d_lookup_base_trace_cols = clone_to_device<m31*>(lookup_base_trace_cols, 164);

    // Process logup columns - combine memory_address_to_id with memory_id_to_big
    // Column 0: memory_address_to_id_0 + memory_id_to_big_0
    gen_poseidon_interaction_col_gen_add_kernel<2, 29><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_addr2id_rel, d_mem_id2big_rel,
        d_lookup_addr2id_0, d_lookup_id2big_0,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        0  // Debug column 0
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 0, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 1: memory_address_to_id_1 + memory_id_to_big_1
    gen_poseidon_interaction_col_gen_add_kernel<2, 29><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_addr2id_rel, d_mem_id2big_rel,
        d_lookup_addr2id_1, d_lookup_id2big_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        1  // Debug column 1
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 1, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 2: memory_address_to_id_2 + memory_id_to_big_2
    gen_poseidon_interaction_col_gen_add_kernel<2, 29><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_addr2id_rel, d_mem_id2big_rel,
        d_lookup_addr2id_2, d_lookup_id2big_2,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 2, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Clone additional relation structs to device
    PoseidonFullRoundChain* d_poseidon_chain_rel = cuda_malloc<PoseidonFullRoundChain>(1);
    RangeCheckFelt252Width27* d_rc_felt252_rel = cuda_malloc<RangeCheckFelt252Width27>(1);
    Cube252* d_cube_rel = cuda_malloc<Cube252>(1);
    RangeCheck_3_3_3_3_3* d_rc_33333_rel = cuda_malloc<RangeCheck_3_3_3_3_3>(1);
    RangeCheck_4_4_4_4* d_rc_4444_rel = cuda_malloc<RangeCheck_4_4_4_4>(1);
    RangeCheck_4_4* d_rc_44_rel = cuda_malloc<RangeCheck_4_4>(1);
    Poseidon3PartialRoundsChain* d_partial_chain_rel = cuda_malloc<Poseidon3PartialRoundsChain>(1);

    cudaMemcpyAsync(d_poseidon_chain_rel, poseidon_full_round_chain_relation, sizeof(PoseidonFullRoundChain), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_felt252_rel, range_check_felt_252_width_27_relation, sizeof(RangeCheckFelt252Width27), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_cube_rel, cube_252_relation, sizeof(Cube252), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_33333_rel, range_check_3_3_3_3_3_relation, sizeof(RangeCheck_3_3_3_3_3), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_4444_rel, range_check_4_4_4_4_relation, sizeof(RangeCheck_4_4_4_4), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_44_rel, range_check_4_4_relation, sizeof(RangeCheck_4_4), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_partial_chain_rel, poseidon_3_partial_rounds_chain_relation, sizeof(Poseidon3PartialRoundsChain), cudaMemcpyHostToDevice, 0);

    // Column 3: poseidon_full_round_chain_0 - poseidon_full_round_chain_1 (SUBTRACT)
    // pfrc_1 is reconstructed from base trace cols 120-149 (indices 0-29 in lookup_base_trace_cols)
    gen_poseidon_interaction_col3_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_poseidon_chain_rel, d_lookup_pfrc_0, d_lookup_base_trace_cols,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 3, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 4: range_check_felt_252_width_27_0 + range_check_felt_252_width_27_1 (ADD)
    // Now using d_lookup_base_trace_cols: col 120-139 maps to indices 0-19
    gen_poseidon_interaction_col4_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_felt252_rel, d_lookup_base_trace_cols,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 4, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 5: cube_252_0 + range_check_3_3_3_3_3_0 (ADD)
    // col 140-159 maps to indices 20-39
    gen_poseidon_interaction_col5_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_cube_rel, d_rc_33333_rel, d_lookup_base_trace_cols, d_lookup_rc_33333_0,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 5, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 6: range_check_3_3_3_3_3_1 + cube_252_1 (ADD)
    // col 160-180 maps to indices 40-60
    gen_poseidon_interaction_col6_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_33333_rel, d_cube_rel, d_lookup_base_trace_cols, d_lookup_rc_33333_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 6, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 7: range_check_4_4_4_4_0 + range_check_4_4_4_4_1 (ADD)
    gen_poseidon_interaction_col7_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_4444_rel, d_lookup_rc_4444_0, d_lookup_rc_4444_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 7, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 8: range_check_4_4_0 + poseidon_3_partial_rounds_chain_0 -> (denom1 - denom0)
    // col 150-190 maps to indices 30-70
    gen_poseidon_interaction_col8_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_44_rel, d_partial_chain_rel, d_lookup_base_trace_cols, d_lookup_rc_44_0,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 8, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 9: poseidon_3_partial_rounds_chain_1 + range_check_4_4_4_4_2 (ADD)
    // col 192-231 maps to indices 72-111
    gen_poseidon_interaction_col9_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_partial_chain_rel, d_rc_4444_rel, d_lookup_base_trace_cols, d_lookup_rc_4444_2,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 9, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 10: range_check_4_4_4_4_3 + range_check_4_4_1 (ADD)
    gen_poseidon_interaction_col10_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_4444_rel, d_rc_44_rel, d_lookup_rc_4444_3, d_lookup_rc_44_1,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 10, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 11: range_check_4_4_4_4_4 + range_check_4_4_4_4_5 (ADD)
    gen_poseidon_interaction_col11_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_4444_rel, d_lookup_rc_4444_4, d_lookup_rc_4444_5,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 11, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 12: range_check_4_4_2 + poseidon_full_round_chain_2 -> (denom1 - denom0)
    // col 232-252 maps to indices 112-132
    gen_poseidon_interaction_col12_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rc_44_rel, d_poseidon_chain_rel, d_lookup_base_trace_cols, d_lookup_rc_44_2,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 12, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 13: poseidon_full_round_chain_3 + memory_address_to_id_3 (ADD)
    // col 254-283 maps to indices 134-163
    gen_poseidon_interaction_col13_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_poseidon_chain_rel, d_mem_addr2id_rel, d_lookup_base_trace_cols, d_lookup_addr2id_3,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 13, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 14: memory_id_to_big_3 + memory_address_to_id_4 (ADD)
    gen_poseidon_interaction_col14_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_mem_id2big_rel, d_mem_addr2id_rel, d_lookup_id2big_3, d_lookup_addr2id_4,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 14, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 15: memory_id_to_big_4 + memory_address_to_id_5 (ADD)
    gen_poseidon_interaction_col15_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_mem_id2big_rel, d_mem_addr2id_rel, d_lookup_id2big_4, d_lookup_addr2id_5,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 15, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Column 16: memory_id_to_big_5 (single lookup, numerator = 1)
    gen_poseidon_interaction_col_single_kernel<29><<<num_blocks, BLOCK_SIZE>>>(
        d_mem_id2big_rel, d_lookup_id2big_5,
        n_rows, d_denom, d_numerator0, d_numerator1, d_numerator2, d_numerator3
    );
    batch_inverse_secure_field(d_denom, d_denom_inv, n_rows);
    gen_poseidon_interaction_finalize_col_kernel<<<num_blocks, BLOCK_SIZE>>>(
        n_rows, 16, d_denom_inv, d_numerator0, d_numerator1, d_numerator2, d_numerator3,
        d_interaction_trace
    );

    // Step 1: Compute claimed_sum by summing all elements in the last column
    // Allocate and zero out coordinate_sums buffer
    m31* d_coordinate_sums = cuda_malloc<m31>(4);
    cudaMemsetAsync(d_coordinate_sums, 0, 4 * sizeof(m31), 0);

    // Launch cumsum_shift kernel to sum all values
    // Use smaller number of blocks for reduction, with shared memory
    unsigned reduction_blocks = min(num_blocks, 256u);
    size_t shared_mem_size = 4 * BLOCK_SIZE * sizeof(m31);
    gen_poseidon_interaction_cumsum_shift_kernel<<<reduction_blocks, BLOCK_SIZE, shared_mem_size>>>(
        N_LOGUP_COLS, n_rows, d_interaction_trace, d_coordinate_sums
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 2: Subtract cumsum_shift from all elements in the last column
    gen_poseidon_interaction_coord_prefix_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_coordinate_sums, N_LOGUP_COLS, n_rows, d_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 3: Apply inclusive prefix sum on the shifted last logup column
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 4], n_rows);
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 3], n_rows);
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 2], n_rows);
    inclusive_prefix_sum(interaction_trace[4 * N_LOGUP_COLS - 1], n_rows);

    // Step 4: Copy coordinate_sums (claimed_sum) back to host
    cudaMemcpyAsync(claimed_sum, d_coordinate_sums, 4 * sizeof(m31), cudaMemcpyDeviceToHost, 0);

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Cleanup
    cuda_free_memory(d_denom);
    cuda_free_memory(d_denom_inv);
    cuda_free_memory(d_numerator0);
    cuda_free_memory(d_numerator1);
    cuda_free_memory(d_numerator2);
    cuda_free_memory(d_numerator3);
    cuda_free_memory(d_mem_addr2id_rel);
    cuda_free_memory(d_mem_id2big_rel);
    cuda_free_memory(d_interaction_trace);
    cuda_free_memory(d_lookup_addr2id_0);
    cuda_free_memory(d_lookup_addr2id_1);
    cuda_free_memory(d_lookup_addr2id_2);
    cuda_free_memory(d_lookup_addr2id_3);
    cuda_free_memory(d_lookup_addr2id_4);
    cuda_free_memory(d_lookup_addr2id_5);
    cuda_free_memory(d_lookup_id2big_0);
    cuda_free_memory(d_lookup_id2big_1);
    cuda_free_memory(d_lookup_id2big_2);
    cuda_free_memory(d_lookup_id2big_3);
    cuda_free_memory(d_lookup_id2big_4);
    cuda_free_memory(d_lookup_id2big_5);
    cuda_free_memory(d_coordinate_sums);
    cuda_free_memory(d_poseidon_chain_rel);
    cuda_free_memory(d_rc_felt252_rel);
    cuda_free_memory(d_cube_rel);
    cuda_free_memory(d_rc_33333_rel);
    cuda_free_memory(d_rc_4444_rel);
    cuda_free_memory(d_rc_44_rel);
    cuda_free_memory(d_partial_chain_rel);
    cuda_free_memory(d_lookup_base_trace_cols);
    cuda_free_memory(d_lookup_rc_33333_0);
    cuda_free_memory(d_lookup_rc_33333_1);
    cuda_free_memory(d_lookup_rc_4444_0);
    cuda_free_memory(d_lookup_rc_4444_1);
    cuda_free_memory(d_lookup_rc_4444_2);
    cuda_free_memory(d_lookup_rc_4444_3);
    cuda_free_memory(d_lookup_rc_4444_4);
    cuda_free_memory(d_lookup_rc_4444_5);
    cuda_free_memory(d_lookup_rc_44_0);
    cuda_free_memory(d_lookup_rc_44_1);
    cuda_free_memory(d_lookup_rc_44_2);
    cuda_free_memory(d_lookup_pfrc_0);
    cuda_free_memory(d_lookup_pfrc_1);

    global_timer.end("generate poseidon_builtin interaction trace");
}
