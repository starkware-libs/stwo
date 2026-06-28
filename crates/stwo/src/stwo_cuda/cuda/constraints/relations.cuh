#ifndef RELATIONS_H
#define RELATIONS_H

#include "logup.cuh"
#include "utils.cuh"

// =============================================================================
// Legacy per-relation lookup element types (kept for backward compatibility
// with existing kernels like fibonacci_example, poseidon_example).
// =============================================================================
typedef LookupElementsBasic<20> BlakeG;
typedef LookupElementsBasic<35> BlakeRound;
typedef LookupElementsBasic<17> BlakeRoundSigma;
typedef LookupElementsBasic<20> Cube252;
typedef LookupElementsBasic<2> MemoryAddressToId;
typedef LookupElementsBasic<29> MemoryIdToBig;
typedef LookupElementsBasic<3> Opcodes;
typedef LookupElementsBasic<73> PartialEcMul;
typedef LookupElementsBasic<57> PedersenPointsTable;
typedef LookupElementsBasic<42> Poseidon3PartialRoundsChain;
typedef LookupElementsBasic<32> PoseidonFullRoundChain;
typedef LookupElementsBasic<31> PoseidonRoundKeys;
typedef LookupElementsBasic<1> RangeCheck_6;
typedef LookupElementsBasic<1> RangeCheck_8;
typedef LookupElementsBasic<1> RangeCheck_11;
typedef LookupElementsBasic<1> RangeCheck_12;
typedef LookupElementsBasic<1> RangeCheck_18;
typedef LookupElementsBasic<1> RangeCheck_18_B;
typedef LookupElementsBasic<1> RangeCheck_19;
typedef LookupElementsBasic<1> RangeCheck_19_B;
typedef LookupElementsBasic<1> RangeCheck_19_C;
typedef LookupElementsBasic<1> RangeCheck_19_D;
typedef LookupElementsBasic<1> RangeCheck_19_E;
typedef LookupElementsBasic<1> RangeCheck_19_F;
typedef LookupElementsBasic<1> RangeCheck_19_G;
typedef LookupElementsBasic<1> RangeCheck_19_H;
typedef LookupElementsBasic<2> RangeCheck_3_6;
typedef LookupElementsBasic<2> RangeCheck_4_3;
typedef LookupElementsBasic<2> RangeCheck_4_4;
typedef LookupElementsBasic<2> RangeCheck_5_4;
typedef LookupElementsBasic<2> RangeCheck_9_9;
typedef LookupElementsBasic<2> RangeCheck_9_9_B;
typedef LookupElementsBasic<2> RangeCheck_9_9_C;
typedef LookupElementsBasic<2> RangeCheck_9_9_D;
typedef LookupElementsBasic<2> RangeCheck_9_9_E;
typedef LookupElementsBasic<2> RangeCheck_9_9_F;
typedef LookupElementsBasic<2> RangeCheck_9_9_G;
typedef LookupElementsBasic<2> RangeCheck_9_9_H;
typedef LookupElementsBasic<3> RangeCheck_7_2_5;
typedef LookupElementsBasic<4> RangeCheck_3_6_6_3;
typedef LookupElementsBasic<4> RangeCheck_4_4_4_4;
typedef LookupElementsBasic<5> RangeCheck_3_3_3_3_3;
typedef LookupElementsBasic<10> RangeCheckFelt252Width27;
typedef LookupElementsBasic<7> VerifyInstruction;
typedef LookupElementsBasic<3> VerifyBitwiseXor_4;
typedef LookupElementsBasic<3> VerifyBitwiseXor_7;
typedef LookupElementsBasic<3> VerifyBitwiseXor_8;
typedef LookupElementsBasic<3> VerifyBitwiseXor_8_B;
typedef LookupElementsBasic<3> VerifyBitwiseXor_9;
typedef LookupElementsBasic<3> VerifyBitwiseXor_12;
typedef LookupElementsBasic<8> TripleXor32;

// =============================================================================
// CommonLookupElements: Single shared lookup elements for all relations.
// In the "now" architecture, all relations use the same LookupElements<128>,
// with a unique RELATION_ID (M31 constant) prepended as the first value in
// each relation tuple.
// =============================================================================
typedef LookupElementsBasic<128> CommonLookupElements;

// RELATION_ID constants — match the Rust values in cairo-air/src/relations.rs.
// These are prepended as the first M31 value in every relation tuple when using
// CommonLookupElements.
static constexpr m31 MEMORY_ADDRESS_TO_ID_RELATION_ID = m31(1444891767);
static constexpr m31 MEMORY_ID_TO_BIG_RELATION_ID = m31(1662111297);
static constexpr m31 OPCODES_RELATION_ID = m31(428564188);
static constexpr m31 RANGE_CHECK_9_9_RELATION_ID = m31(517791011);
static constexpr m31 RANGE_CHECK_9_9_B_RELATION_ID = m31(1897792095);
static constexpr m31 RANGE_CHECK_9_9_C_RELATION_ID = m31(1881014476);
static constexpr m31 RANGE_CHECK_9_9_D_RELATION_ID = m31(1864236857);
static constexpr m31 RANGE_CHECK_9_9_E_RELATION_ID = m31(1847459238);
static constexpr m31 RANGE_CHECK_9_9_F_RELATION_ID = m31(1830681619);
static constexpr m31 RANGE_CHECK_9_9_G_RELATION_ID = m31(1813904000);
static constexpr m31 RANGE_CHECK_9_9_H_RELATION_ID = m31(2065568285);
static constexpr m31 VERIFY_BITWISE_XOR_12_RELATION_ID = m31(648362599);

// Additional RELATION_IDs used by subroutines and components.
// These are the inline M31 constants used as first values in relation entries.
static constexpr m31 VERIFY_INSTRUCTION_RELATION_ID = m31(1719106205);
static constexpr m31 RANGE_CHECK_3_6_6_3_RELATION_ID = m31(1005786011);
static constexpr m31 RANGE_CHECK_4_3_RELATION_ID = m31(1567323731);
static constexpr m31 RANGE_CHECK_4_4_RELATION_ID = m31(1651211826);
static constexpr m31 RANGE_CHECK_5_4_RELATION_ID = m31(1735099921);
static constexpr m31 RANGE_CHECK_3_3_3_3_3_RELATION_ID = m31(502259093);
static constexpr m31 RANGE_CHECK_7_2_5_RELATION_ID = m31(371240602);
static constexpr m31 RANGE_CHECK_6_RELATION_ID = m31(1185356339);
static constexpr m31 RANGE_CHECK_8_RELATION_ID = m31(1420243005);
static constexpr m31 RANGE_CHECK_11_RELATION_ID = m31(991608089);
static constexpr m31 RANGE_CHECK_12_RELATION_ID = m31(941275232);
static constexpr m31 RANGE_CHECK_18_RELATION_ID = m31(1109051422);
static constexpr m31 RANGE_CHECK_18_B_RELATION_ID = m31(1424798916);
static constexpr m31 RANGE_CHECK_20_RELATION_ID = m31(1410849886);
static constexpr m31 RANGE_CHECK_252_WIDTH_27_RELATION_ID = m31(1090315331);
static constexpr m31 VERIFY_BITWISE_XOR_4_RELATION_ID = m31(45448144);
static constexpr m31 VERIFY_BITWISE_XOR_7_RELATION_ID = m31(62225763);
static constexpr m31 VERIFY_BITWISE_XOR_8_RELATION_ID = m31(112558620);
static constexpr m31 VERIFY_BITWISE_XOR_8_B_RELATION_ID = m31(521092554);
static constexpr m31 VERIFY_BITWISE_XOR_9_RELATION_ID = m31(95781001);
static constexpr m31 BLAKE_G_RELATION_ID = m31(1139985212);
static constexpr m31 BLAKE_ROUND_SIGMA_RELATION_ID = m31(1805967942);
static constexpr m31 BLAKE_ROUND_RELATION_ID = m31(40528774);
// blake_compress_opcode uses the BlakeRound relation for its lookup entries
static constexpr m31 BLAKE_COMPRESS_OPCODE_RELATION_ID = BLAKE_ROUND_RELATION_ID;
static constexpr m31 TRIPLE_XOR_32_RELATION_ID = m31(990559919);
static constexpr m31 CUBE_252_RELATION_ID = m31(1987997202);
static constexpr m31 RANGE_CHECK_4_4_4_4_RELATION_ID = m31(1027333874);
static constexpr m31 RANGE_CHECK_20_B_RELATION_ID = m31(514232941);
static constexpr m31 RANGE_CHECK_20_C_RELATION_ID = m31(531010560);
static constexpr m31 RANGE_CHECK_20_D_RELATION_ID = m31(480677703);
static constexpr m31 RANGE_CHECK_20_E_RELATION_ID = m31(497455322);
static constexpr m31 RANGE_CHECK_20_F_RELATION_ID = m31(447122465);
static constexpr m31 RANGE_CHECK_20_G_RELATION_ID = m31(463900084);
static constexpr m31 RANGE_CHECK_20_H_RELATION_ID = m31(682009131);

// Poseidon context component relation IDs
static constexpr m31 POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID = m31(1343313504);
static constexpr m31 POSEIDON_FULL_ROUND_CHAIN_RELATION_ID = m31(1480369132);
static constexpr m31 POSEIDON_ROUND_KEYS_RELATION_ID = m31(1024310512);
static constexpr m31 POSEIDON_AGGREGATOR_RELATION_ID = m31(1551892206);

// Pedersen context component relation IDs
static constexpr m31 PARTIAL_EC_MUL_RELATION_ID = m31(1621226978);
static constexpr m31 PARTIAL_EC_MUL_WINDOW_BITS_9_RELATION_ID = m31(2038149019);
static constexpr m31 PARTIAL_EC_MUL_WINDOW_BITS_18_RELATION_ID = m31(1621226978);
static constexpr m31 PEDERSEN_POINTS_TABLE_RELATION_ID = m31(1444721856);
static constexpr m31 PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_RELATION_ID = m31(1791500038);
static constexpr m31 PEDERSEN_POINTS_TABLE_WINDOW_BITS_18_RELATION_ID = m31(1444721856);
static constexpr m31 PEDERSEN_AGGREGATOR_WINDOW_BITS_18_RELATION_ID = m31(520578465);
static constexpr m31 PEDERSEN_AGGREGATOR_WINDOW_BITS_9_RELATION_ID = m31(194336987);

#endif // RELATIONS_H
