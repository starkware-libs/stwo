// CUDA trace generation for poseidon_aggregator component
// 342 trace columns, multiple lookups for Poseidon hash computation
//
// Key differences from poseidon_builtin:
// - No memory_address_to_id lookups (inputs are IDs directly, cols 0-5)
// - 342 columns (vs 341 for builtin)
// - No segment_start parameter
// - Output states unpacked from computed Poseidon output (not from memory)
// - Column offsets shifted by 3 relative to builtin (LC starts at 90 vs 87)
//
// Lookups:
// - 6 memory_id_to_big (29 elements each)
// - 8 poseidon_full_round_chain (32 elements each)
// - 2 range_check_3_3_3_3_3 (5 elements each)
// - 6 range_check_4_4_4_4 (4 elements each)
// - 3 range_check_4_4 (2 elements each)
// - 27 poseidon_3_partial_rounds_chain (42 elements each)

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "gen_memory_id_to_big_trace.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "../constraints/relations.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include <cstdint>
#include <cstdio>

// ============================================================================
// Constants
// ============================================================================

#define AGG_N_TRACE_COLUMNS 342
#define AGG_N_LOGUP_COLS 14
#define AGG_BLOCK_SIZE 256

// ============================================================================
// Type definitions (identical to poseidon_builtin — safe to redefine)
// ============================================================================

struct PosAggFelt252Width27 {
    m31 limbs[10];
};

typedef ff_storage<8> PosAggFelt252Field;

// ============================================================================
// Inline field operations (static to avoid link conflicts with poseidon_builtin)
// ============================================================================

static __device__ __forceinline__ PosAggFelt252Field posagg_felt_add(
    const PosAggFelt252Field& a, const PosAggFelt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

static __device__ __forceinline__ PosAggFelt252Field posagg_felt_sub(
    const PosAggFelt252Field& a, const PosAggFelt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

static __device__ __forceinline__ PosAggFelt252Field posagg_felt_mul(
    const PosAggFelt252Field& a, const PosAggFelt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

static __device__ __forceinline__ PosAggFelt252Field posagg_felt_to_mont(
    const PosAggFelt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

static __device__ __forceinline__ PosAggFelt252Field posagg_felt_from_mont(
    const PosAggFelt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

// ============================================================================
// Constants (own copies to avoid link conflicts)
// ============================================================================

__device__ __constant__ PosAggFelt252Field POSAGG_MONT_CUBE_FACTOR = {{
    0x406DF18E, 0xCC7177D1, 0x77FFCC06, 0x75457066,
    0x36300018, 0xF47D84F8, 0x873C0A6D, 0x038E5F79
}};

__device__ __constant__ uint32_t POSAGG_LINEAR_COMB_BIAS_0[10] = {
    74972783, 117420501, 112795138, 91013252, 60709090,
    44848225, 108487870, 44781849, 102193642, 208
};
__device__ __constant__ uint32_t POSAGG_LINEAR_COMB_BIAS_1[10] = {
    41224388, 90391646, 36279186, 129717753, 94624323,
    75104388, 133303902, 48945103, 41320857, 112
};
__device__ __constant__ uint32_t POSAGG_LINEAR_COMB_BIAS_2[10] = {
    4883209, 28820206, 79012328, 49157069, 78826183,
    72285071, 33413160, 90842759, 60124463, 116
};

__device__ __constant__ uint32_t POSAGG_LC_BIAS_N4_COEFS_1_1_M2_1[10] = {
    103094260, 121146754, 95050340, 16173996, 50758155,
    54415179, 19292069, 45351266, 122233508, 248
};
__device__ __constant__ uint32_t POSAGG_LC_BIAS_N4_COEFS_4_2_M2_1[10] = {
    121657377, 112479959, 130418270, 4974792, 59852719,
    120369218, 62439890, 50468641, 86573645, 154
};

// Round keys: 35 rounds, 3 constants per round (4 x u64 each)
__device__ __constant__ uint64_t POSAGG_ROUND_KEYS[35][3][4] = {
    // Round 0 (full round)
    {{9808894619969057997ULL, 2962375666393338310ULL, 17382841788414994265ULL, 443257643709112289ULL},
     {12537484503666775718ULL, 3256805997184644908ULL, 6617722259049010207ULL, 543112534054733059ULL},
     {1454046077829682943ULL, 14331133962181073949ULL, 2327346812919484995ULL, 379005027604567203ULL}},
    // Round 1
    {{316912300309518807ULL, 9546737057323600779ULL, 4990939959663297477ULL, 409555158710929193ULL},
     {14375050875883784322ULL, 3258765518491372314ULL, 6123276414968091301ULL, 564945574506516589ULL},
     {16399159056375946558ULL, 12401617009203210820ULL, 11251954111719545950ULL, 433429710746163456ULL}},
    // Round 2
    {{7006359817426385501ULL, 17203056170488800107ULL, 10266463410669146573ULL, 302632258003414824ULL},
     {14549101708442237159ULL, 5447808788302094550ULL, 5985460154360671870ULL, 377474158904626280ULL},
     {2016536305309843132ULL, 7819086366821881070ULL, 6549900492473011498ULL, 375041666190951811ULL}},
    // Round 3
    {{10958522786453251491ULL, 9697564334799485296ULL, 12061515884908749995ULL, 382992757552468126ULL},
     {3123128533353263104ULL, 1101320594306927398ULL, 12277506650622088974ULL, 151394834303922635ULL},
     {10558209943415701402ULL, 3761550961988184469ULL, 3770582263098070207ULL, 337917216919135628ULL}},
    // Round 4
    {{11059652905750625380ULL, 13475195141561210865ULL, 13294395003503408798ULL, 543485850395037306ULL},
     {2130930448912281502ULL, 11333634387982439184ULL, 8850610548639699306ULL, 457475955817445493ULL},
     {13832015370839427617ULL, 3536623570151876469ULL, 6528270901734940966ULL, 1727329127918258ULL}},
    // Round 5
    {{8115407971155582787ULL, 9978560128345434391ULL, 5056408649803520810ULL, 262112615523232333ULL},
     {15132830981034848104ULL, 221062278661831986ULL, 642558393488344280ULL, 294435853161867218ULL},
     {17720328485765873457ULL, 906259302840864515ULL, 9886887042042513701ULL, 91167054851387838ULL}},
    // Round 6
    {{17686069520976319126ULL, 357140690021361429ULL, 4698816318705416205ULL, 393981709058899502ULL},
     {11422699654326280778ULL, 16059236267229938280ULL, 14304086719964370791ULL, 408074902120160445ULL},
     {10098039853740591407ULL, 18346213706869023683ULL, 9856189649941491293ULL, 184406899276982606ULL}},
    // Round 7
    {{4111625807920905640ULL, 13925198121954558587ULL, 12310073145155562618ULL, 235927056615592132ULL},
     {6384587362686501122ULL, 8879249956632890389ULL, 16548116661510213280ULL, 336779148206005613ULL},
     {16031192111936359076ULL, 5992351855224207619ULL, 12627781605286612627ULL, 62096344547068532ULL}},
    // Round 8
    {{7132509749599922320ULL, 6162523224461213171ULL, 8812310904075333603ULL, 485641259025949528ULL},
     {9608543551775247566ULL, 5196481376567266799ULL, 4241060526105290574ULL, 127878632248644222ULL},
     {5678830138638110916ULL, 17150803417208083936ULL, 3818159621611526901ULL, 334934582699708306ULL}},
    // Round 9
    {{2411984091808734148ULL, 4676885686514770974ULL, 16024545775274701230ULL, 35144855618233700ULL},
     {3764367625527066390ULL, 16203185340231970163ULL, 3091249709657140605ULL, 246768161927392025ULL},
     {14145295330358598665ULL, 8189373517343951529ULL, 9185965492588609992ULL, 84513112494072013ULL}},
    // Round 10
    {{11207056279882910057ULL, 8343632314543039012ULL, 11825707279115803854ULL, 290028553136950168ULL},
     {8324430733814960957ULL, 1295757100659713237ULL, 13793768882092694703ULL, 170786252004278897ULL},
     {3470195965229650799ULL, 7014342387715054610ULL, 8068066979310954905ULL, 29052085287755578ULL}},
    // Round 11
    {{16182857524744847037ULL, 13508798581767021717ULL, 11609421482229546503ULL, 516546051298708222ULL},
     {15033308840249833218ULL, 2943419432621774691ULL, 12721848833217608341ULL, 495114643959326355ULL},
     {15900588831390415422ULL, 13235287741266617180ULL, 199968676028566291ULL, 31464852864192972ULL}},
    // Round 12
    {{1620930607287324279ULL, 5691881440575675201ULL, 14029655266804088527ULL, 85281494397439074ULL},
     {14358430905948328990ULL, 8174075507093900204ULL, 4259719420728355987ULL, 198633356278451635ULL},
     {9809328616759644792ULL, 9452995911559605327ULL, 14571337138054143278ULL, 30595350758349726ULL}},
    // Round 13
    {{5052814846190951527ULL, 2564319269931470445ULL, 11667947324380057882ULL, 381968969372291382ULL},
     {10270535945835093416ULL, 7013539859298536233ULL, 12880625276280589921ULL, 423512085516371736ULL},
     {9247593602616013019ULL, 9800080834126373385ULL, 15154714092547675637ULL, 85949681409890944ULL}},
    // Round 14
    {{16691816165224508898ULL, 7632563883998222450ULL, 7283702476287088318ULL, 122927053010244988ULL},
     {5281932702559195717ULL, 3912411525754476767ULL, 2751980448518808692ULL, 449246095011271218ULL},
     {2154533012881281233ULL, 12108066475824676498ULL, 3101185982842383519ULL, 23082295823839220ULL}},
    // Round 15
    {{7605923575436520679ULL, 17775553940505278137ULL, 12354955681295648587ULL, 509503557137574051ULL},
     {6859516056437622059ULL, 15185460371714151768ULL, 11379739398280558941ULL, 467020453759984910ULL},
     {16906035870412618946ULL, 2048289172670790831ULL, 3913835398798558993ULL, 247123888097172708ULL}},
    // Round 16
    {{14861675641163720274ULL, 13490102368184285242ULL, 7347027430097237399ULL, 12293023119986689ULL},
     {17015804524472763158ULL, 2030415039026408622ULL, 17809691364575575612ULL, 373903064080638948ULL},
     {8022416475434186219ULL, 17815483025186149958ULL, 17841645611508634712ULL, 214671237987350594ULL}},
    // Round 17
    {{12836252995927977384ULL, 12965348847163767059ULL, 16404178258733598267ULL, 90570121994582570ULL},
     {3307613700375182919ULL, 6181136657427428089ULL, 13131983874186228376ULL, 111501226499533359ULL},
     {17976156798761747365ULL, 5762323702974965097ULL, 2597451573586851781ULL, 505236697901381580ULL}},
    // Round 18
    {{10381335841935625777ULL, 14975760611930379479ULL, 14435427058050060920ULL, 398310795157470149ULL},
     {7159397558615159963ULL, 7188734421404393410ULL, 1719787959693584840ULL, 314760383409924924ULL},
     {2143638283124439842ULL, 13645456387251540292ULL, 13644498560249152495ULL, 40707004462247922ULL}},
    // Round 19
    {{4076286098075094248ULL, 6047377110922245529ULL, 10310252161423143831ULL, 4203916682452144ULL},
     {16974446701450911254ULL, 13817121135003214195ULL, 2110477587661043379ULL, 408404362623939579ULL},
     {2551943238291520253ULL, 7863747909853014700ULL, 10038172819555036178ULL, 498557445795487158ULL}},
    // Round 20
    {{2539457924220004741ULL, 6798810574668326903ULL, 734801439896130781ULL, 197318323104987578ULL},
     {5630865015115736926ULL, 6924395279250128121ULL, 6087898613499446423ULL, 97920604124542022ULL},
     {8238671520399847831ULL, 17200586436505834896ULL, 18050188643002787777ULL, 522418299476559161ULL}},
    // Round 21
    {{16671769425050870867ULL, 9818859487908268561ULL, 14628982416326270968ULL, 105495391240150891ULL},
     {15264270982597349937ULL, 3214172504508351054ULL, 96620451664846254ULL, 107082329324411929ULL},
     {14217429574808978483ULL, 128541115086728122ULL, 9630653827036234478ULL, 337586787343095831ULL}},
    // Round 22
    {{8212059728564478324ULL, 2347043101088709486ULL, 6567058597747348925ULL, 136303161555124818ULL},
     {9215571201366006957ULL, 18390624930749960250ULL, 12318590206736769157ULL, 94289926799047171ULL},
     {2681199449952507734ULL, 2490827916210922767ULL, 17337862272405306868ULL, 167761531076143152ULL}},
    // Round 23
    {{12798085655163047736ULL, 13696387070792973059ULL, 5787352356986496426ULL, 499982807322000917ULL},
     {11741293120501172298ULL, 2334843635281516844ULL, 168280946537445205ULL, 83199885793358504ULL},
     {12631247450401696760ULL, 1333500347809883553ULL, 7960218164236031817ULL, 545118190714259783ULL}},
    // Round 24
    {{10647924774016649377ULL, 7324763009135962770ULL, 16081867801897155361ULL, 428325268027940951ULL},
     {9916218426490841056ULL, 1030092679665695813ULL, 1263314736050787503ULL, 189382575143994932ULL},
     {13675857909985031082ULL, 12446306110735075056ULL, 5033197056310105549ULL, 437182141684164304ULL}},
    // Round 25
    {{16017091423340680337ULL, 16608552316205419099ULL, 8224269358142049653ULL, 421369926480120239ULL},
     {9595998208446022457ULL, 16134937223929573489ULL, 14473485045201252070ULL, 492654444776713988ULL},
     {5219139941060837063ULL, 15853778231351784921ULL, 17809532253019979818ULL, 417855011179639500ULL}},
    // Round 26
    {{10403233932030073012ULL, 16730279770404598818ULL, 5644362027553147371ULL, 71735369062479113ULL},
     {9631935352839043273ULL, 17054291492818491992ULL, 17392960852478338690ULL, 460451922811752815ULL},
     {8477235527901859940ULL, 9814894372396933310ULL, 4799990685272759189ULL, 400282443455003622ULL}},
    // Round 27
    {{1576843956219139158ULL, 702430080765383770ULL, 17965431827539789120ULL, 396822748194700979ULL},
     {6534711024274141327ULL, 7130143603302325050ULL, 16132773823416316657ULL, 291127484699867608ULL},
     {4689783592352337689ULL, 13240213583589885272ULL, 9538724043191226625ULL, 53501278744087361ULL}},
    // Round 28
    {{5303664022074446245ULL, 11845428113003974855ULL, 12846417839211241712ULL, 261206892901600457ULL},
     {7213274845438153874ULL, 3182604331988734929ULL, 3825798403644353861ULL, 498424027422114180ULL},
     {16374080072108357107ULL, 12602249521880983826ULL, 16789038177262469212ULL, 434844701481670312ULL}},
    // Round 29
    {{5424818379886973048ULL, 12246700522389374047ULL, 16105275706111921973ULL, 171092744151981801ULL},
     {6597251438734065845ULL, 2269299153703490182ULL, 1681550773894047556ULL, 497581200702741603ULL},
     {448123443688089620ULL, 2093069703231428951ULL, 10690354368775868807ULL, 47970718237487928ULL}},
    // Round 30
    {{5663590943682895948ULL, 1657980836047728369ULL, 14473564132866859515ULL, 63033233261729097ULL},
     {10253436943656002480ULL, 496762026162837186ULL, 4416882861358244138ULL, 410250456421992694ULL},
     {13092039789419326359ULL, 12701778245175598192ULL, 10053832990213334033ULL, 99263799527290004ULL}},
    // Round 31 (final full round)
    {{17612850421305358241ULL, 8270527177451683327ULL, 3824004781155827930ULL, 23420660055616514ULL},
     {14792590694225191985ULL, 10340526750113228103ULL, 13907692663639317222ULL, 419027522602786902ULL},
     {6537420399978421861ULL, 13000095247594509242ULL, 4689550804574686720ULL, 216508944422249833ULL}},
    // Round 32
    {{5284804903132760421ULL, 6021193890942533180ULL, 12919709475177128988ULL, 388552658857864838ULL},
     {14487504387541221402ULL, 8671715521049733970ULL, 11505630672145478718ULL, 340569955046273223ULL},
     {15605227016987613715ULL, 4467780628446399859ULL, 1916547247173479880ULL, 360408062797054196ULL}},
    // Round 33
    {{2318724554222443904ULL, 12462857660673509117ULL, 1043912215626002694ULL, 444903370426104614ULL},
     {13771333397279872933ULL, 8504629457688506196ULL, 17402104297977249580ULL, 365482958936297625ULL},
     {9663191740751744091ULL, 17588211649412389869ULL, 8849849772044756264ULL, 441288247586026977ULL}},
    // Round 34 (unused - zeros)
    {{0ULL, 0ULL, 0ULL, 0ULL},
     {0ULL, 0ULL, 0ULL, 0ULL},
     {0ULL, 0ULL, 0ULL, 0ULL}}
};

// ============================================================================
// Device helper functions
// ============================================================================

__device__ void posagg_pack_felt252_to_width27(const m31* src_limbs, PosAggFelt252Width27& dst) {
    for (int i = 0; i < 9; i++) {
        m31 l0 = src_limbs[i * 3];
        m31 l1 = src_limbs[i * 3 + 1];
        m31 l2 = src_limbs[i * 3 + 2];
        dst.limbs[i] = add(add(l0, mul(l1, (m31){512})), mul(l2, (m31){262144}));
    }
    dst.limbs[9] = src_limbs[27];
}

__device__ void posagg_unpack_felt252_from_width27(const PosAggFelt252Width27& src, m31* dst_limbs) {
    for (int i = 0; i < 9; i++) {
        m31 val = src.limbs[i];
        dst_limbs[i * 3] = val & 0x1FF;
        dst_limbs[i * 3 + 1] = (val >> 9) & 0x1FF;
        dst_limbs[i * 3 + 2] = (val >> 18) & 0x1FF;
    }
    dst_limbs[27] = src.limbs[9];
}

__device__ PosAggFelt252Field posagg_width27_to_felt252field(const uint64_t* width27) {
    PosAggFelt252Field result;
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

__device__ void posagg_felt252field_to_width27_m31(const PosAggFelt252Field& felt, PosAggFelt252Width27& dst) {
    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    dst.limbs[0] = (m31){(uint32_t)(val0 & 0x7FFFFFF)};
    dst.limbs[1] = (m31){(uint32_t)((val0 >> 27) & 0x7FFFFFF)};

    uint64_t cross01 = (val0 >> 54) | (val1 << 10);
    dst.limbs[2] = (m31){(uint32_t)(cross01 & 0x7FFFFFF)};
    dst.limbs[3] = (m31){(uint32_t)((val1 >> 17) & 0x7FFFFFF)};

    uint64_t cross12 = (val1 >> 44) | (val2 << 20);
    dst.limbs[4] = (m31){(uint32_t)(cross12 & 0x7FFFFFF)};
    dst.limbs[5] = (m31){(uint32_t)((val2 >> 7) & 0x7FFFFFF)};
    dst.limbs[6] = (m31){(uint32_t)((val2 >> 34) & 0x7FFFFFF)};

    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    dst.limbs[7] = (m31){(uint32_t)(cross23 & 0x7FFFFFF)};
    dst.limbs[8] = (m31){(uint32_t)((val3 >> 24) & 0x7FFFFFF)};
    dst.limbs[9] = (m31){(uint32_t)((val3 >> 51) & 0x1FF)};
}

__device__ PosAggFelt252Field posagg_width27_m31_to_felt252field(const PosAggFelt252Width27& src) {
    uint64_t val0 = 0, val1 = 0, val2 = 0, val3 = 0;

    val0 = (uint64_t)src.limbs[0];
    val0 |= ((uint64_t)src.limbs[1]) << 27;
    val0 |= ((uint64_t)src.limbs[2]) << 54;

    val1 = ((uint64_t)src.limbs[2]) >> 10;
    val1 |= ((uint64_t)src.limbs[3]) << 17;
    val1 |= ((uint64_t)src.limbs[4]) << 44;

    val2 = ((uint64_t)src.limbs[4]) >> 20;
    val2 |= ((uint64_t)src.limbs[5]) << 7;
    val2 |= ((uint64_t)src.limbs[6]) << 34;
    val2 |= ((uint64_t)src.limbs[7]) << 61;

    val3 = ((uint64_t)src.limbs[7]) >> 3;
    val3 |= ((uint64_t)src.limbs[8]) << 24;
    val3 |= ((uint64_t)src.limbs[9]) << 51;

    PosAggFelt252Field result;
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

__device__ PosAggFelt252Field posagg_bias_to_felt252field(const uint32_t* bias) {
    PosAggFelt252Width27 w27;
    for (int i = 0; i < 10; i++) {
        w27.limbs[i] = (m31){bias[i]};
    }
    return posagg_width27_m31_to_felt252field(w27);
}

__device__ void posagg_compute_linear_combination_with_p_coef(
    const PosAggFelt252Width27& input_width27,
    const uint32_t* bias,
    PosAggFelt252Width27& combination_width27,
    m31& p_coef
) {
    PosAggFelt252Field input_felt = posagg_width27_m31_to_felt252field(input_width27);
    PosAggFelt252Field bias_felt = posagg_bias_to_felt252field(bias);
    PosAggFelt252Field combination_felt = posagg_felt_add(input_felt, bias_felt);
    posagg_felt252field_to_width27_m31(combination_felt, combination_width27);

    uint32_t input_0 = input_width27.limbs[0];
    uint32_t bias_0 = bias[0];
    uint32_t combo_0 = combination_width27.limbs[0];
    uint32_t biased = input_0 + bias_0 - combo_0 + 134217729;
    uint32_t low_16 = biased & 0xFFFF;
    p_coef = (m31){low_16 - 1};
}

__device__ void posagg_felt252field_to_m31_width27(const PosAggFelt252Field& felt, m31* limbs) {
    PosAggFelt252Width27 w27;
    posagg_felt252field_to_width27_m31(felt, w27);
    for (int i = 0; i < 10; i++) {
        limbs[i] = w27.limbs[i];
    }
}

__device__ PosAggFelt252Field posagg_felt_cube(const PosAggFelt252Field& x) {
    PosAggFelt252Field x2 = posagg_felt_mul(x, x);
    PosAggFelt252Field x3 = posagg_felt_mul(x2, x);
    PosAggFelt252Field local_factor = POSAGG_MONT_CUBE_FACTOR;
    return posagg_felt_mul(x3, local_factor);
}

__device__ PosAggFelt252Field posagg_get_round_key(unsigned round, unsigned state_idx) {
    return posagg_width27_to_felt252field(
        (const uint64_t*)POSAGG_ROUND_KEYS[round][state_idx]);
}

__device__ void posagg_mds_mix(PosAggFelt252Field state[3]) {
    PosAggFelt252Field x = state[0];
    PosAggFelt252Field y = state[1];
    PosAggFelt252Field z = state[2];

    PosAggFelt252Field y1_zm1 = posagg_felt_sub(y, z);
    PosAggFelt252Field x1_ym1_z1 = posagg_felt_sub(x, y1_zm1);
    PosAggFelt252Field x1_y1_zm1 = posagg_felt_add(x, y1_zm1);
    PosAggFelt252Field x1_y1 = posagg_felt_add(x, y);
    PosAggFelt252Field x2_y2 = posagg_felt_add(x1_y1, x1_y1);

    state[0] = posagg_felt_add(x2_y2, x1_ym1_z1);
    state[1] = x1_ym1_z1;
    state[2] = posagg_felt_sub(x1_y1_zm1, z);
}

__device__ void posagg_full_round(PosAggFelt252Field state[3], unsigned round) {
    state[0] = posagg_felt_cube(state[0]);
    state[1] = posagg_felt_cube(state[1]);
    state[2] = posagg_felt_cube(state[2]);
    posagg_mds_mix(state);
    state[0] = posagg_felt_add(state[0], posagg_get_round_key(round, 0));
    state[1] = posagg_felt_add(state[1], posagg_get_round_key(round, 1));
    state[2] = posagg_felt_add(state[2], posagg_get_round_key(round, 2));
}

__device__ void posagg_partial_round(PosAggFelt252Field state[4], PosAggFelt252Field half_key) {
    PosAggFelt252Field z23 = posagg_felt_cube(state[3]);
    PosAggFelt252Field z03 = state[0];
    PosAggFelt252Field z1 = state[1];
    PosAggFelt252Field z13 = state[2];
    PosAggFelt252Field z2 = state[3];

    PosAggFelt252Field z03_z13 = posagg_felt_add(z03, z13);
    PosAggFelt252Field z03_z13_z1 = posagg_felt_add(z03_z13, z1);
    PosAggFelt252Field longsum = posagg_felt_add(
        posagg_felt_sub(posagg_felt_add(z03_z13_z1, z2), z23), half_key);
    PosAggFelt252Field half_z3 = posagg_felt_add(
        posagg_felt_add(posagg_felt_add(longsum, z03_z13_z1), z03_z13), z03);
    PosAggFelt252Field z3 = posagg_felt_add(half_z3, half_z3);

    state[0] = z13;
    state[1] = z2;
    state[2] = z23;
    state[3] = z3;
}

__device__ void posagg_3_partial_rounds(PosAggFelt252Field state[4], unsigned round) {
    for (int i = 0; i < 3; i++) {
        PosAggFelt252Field half_key = posagg_get_round_key(round, i);
        posagg_partial_round(state, half_key);
    }
}

__device__ void posagg_capture_pfrc_state(m31** lookup, const PosAggFelt252Field state[3],
                                           m31 chain_id, int round, int row) {
    lookup[0][row] = chain_id;
    lookup[1][row] = (m31){(uint32_t)round};
    m31 w27[10];
    posagg_felt252field_to_m31_width27(state[0], w27);
    for (int i = 0; i < 10; i++) lookup[2 + i][row] = w27[i];
    posagg_felt252field_to_m31_width27(state[1], w27);
    for (int i = 0; i < 10; i++) lookup[12 + i][row] = w27[i];
    posagg_felt252field_to_m31_width27(state[2], w27);
    for (int i = 0; i < 10; i++) lookup[22 + i][row] = w27[i];
}

// ============================================================================
// Base trace generation kernel for poseidon_aggregator
// ============================================================================
__launch_bounds__(AGG_BLOCK_SIZE, 2)
__global__ void generate_poseidon_aggregator_base_trace_kernel(
    m31 **traces,

    // Inputs: 6 ID arrays + mults
    unsigned *input_ids_0,
    unsigned *input_ids_1,
    unsigned *input_ids_2,
    unsigned *input_ids_3,
    unsigned *input_ids_4,
    unsigned *input_ids_5,
    unsigned *mults_in,

    // Memory tables
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    // Lookup data outputs
    // 6 memory_id_to_big (29 elements each)
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,
    // 2 range_check_3_3_3_3_3 (5 elements each)
    m31 **lookup_range_check_3_3_3_3_3_0,
    m31 **lookup_range_check_3_3_3_3_3_1,
    // 6 range_check_4_4_4_4 (4 elements each)
    m31 **lookup_range_check_4_4_4_4_0,
    m31 **lookup_range_check_4_4_4_4_1,
    m31 **lookup_range_check_4_4_4_4_2,
    m31 **lookup_range_check_4_4_4_4_3,
    m31 **lookup_range_check_4_4_4_4_4,
    m31 **lookup_range_check_4_4_4_4_5,
    // 3 range_check_4_4 (2 elements each)
    m31 **lookup_range_check_4_4_0,
    m31 **lookup_range_check_4_4_1,
    m31 **lookup_range_check_4_4_2,
    // 10 poseidon_full_round_chain (32 elements each)
    // pfrc 0-7: sub-component feeds (rounds 0,1,2,3,31,32,33,34)
    // pfrc 8-9: IT boundary entries (round 4 = chain 0 exit, round 35 = chain 1 exit)
    m31 **lookup_poseidon_full_round_chain_0,
    m31 **lookup_poseidon_full_round_chain_1,
    m31 **lookup_poseidon_full_round_chain_2,
    m31 **lookup_poseidon_full_round_chain_3,
    m31 **lookup_poseidon_full_round_chain_4,
    m31 **lookup_poseidon_full_round_chain_5,
    m31 **lookup_poseidon_full_round_chain_6,
    m31 **lookup_poseidon_full_round_chain_7,
    m31 **lookup_poseidon_full_round_chain_8,
    m31 **lookup_poseidon_full_round_chain_9,
    // 28 poseidon_3_partial_rounds_chain (42 elements each)
    // p3prc 0-26: sub-component feeds (input to each of 27 groups)
    // p3prc 27: IT boundary (round=31, chain exit = output after all groups)
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
    m31 **lookup_poseidon_3_partial_rounds_chain_27,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0_val = {0};
    const m31 M31_1_val = {1};
    const m31 M31_3_val = {3};
    const m31 M31_4_val = {4};
    const m31 M31_31_val = {31};

    if (row < trace_size) {
        // ============ Read 6 input IDs (cols 0-5) ============
        m31 id0 = {input_ids_0[row]};
        m31 id1 = {input_ids_1[row]};
        m31 id2 = {input_ids_2[row]};
        m31 id3 = {input_ids_3[row]};
        m31 id4 = {input_ids_4[row]};
        m31 id5 = {input_ids_5[row]};

        traces[0][row] = id0;
        traces[1][row] = id1;
        traces[2][row] = id2;
        traces[3][row] = id3;
        traces[4][row] = id4;
        traces[5][row] = id5;

        // ============ Read 3 input state limbs via memory_id_to_big (cols 6-89) ============
        // State 0: cols 6-33 (28 limbs)
        m31 input_state_0_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            id0, input_state_0_limbs);
        for (int i = 0; i < 28; i++) traces[6 + i][row] = input_state_0_limbs[i];

        lookup_memory_id_to_big_0[0][row] = id0;
        for (int i = 0; i < 28; i++) lookup_memory_id_to_big_0[1 + i][row] = input_state_0_limbs[i];

        // State 1: cols 34-61 (28 limbs)
        m31 input_state_1_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            id1, input_state_1_limbs);
        for (int i = 0; i < 28; i++) traces[34 + i][row] = input_state_1_limbs[i];

        lookup_memory_id_to_big_1[0][row] = id1;
        for (int i = 0; i < 28; i++) lookup_memory_id_to_big_1[1 + i][row] = input_state_1_limbs[i];

        // State 2: cols 62-89 (28 limbs)
        m31 input_state_2_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            id2, input_state_2_limbs);
        for (int i = 0; i < 28; i++) traces[62 + i][row] = input_state_2_limbs[i];

        lookup_memory_id_to_big_2[0][row] = id2;
        for (int i = 0; i < 28; i++) lookup_memory_id_to_big_2[1 + i][row] = input_state_2_limbs[i];

        // ============ Pack input states to Width27 ============
        PosAggFelt252Width27 packed_state_0, packed_state_1, packed_state_2;
        posagg_pack_felt252_to_width27(input_state_0_limbs, packed_state_0);
        posagg_pack_felt252_to_width27(input_state_1_limbs, packed_state_1);
        posagg_pack_felt252_to_width27(input_state_2_limbs, packed_state_2);

        // ============ Linear combinations (cols 90-122) ============
        PosAggFelt252Width27 lc_result_0, lc_result_1, lc_result_2;
        m31 p_coef_0, p_coef_1, p_coef_2;

        posagg_compute_linear_combination_with_p_coef(
            packed_state_0, POSAGG_LINEAR_COMB_BIAS_0, lc_result_0, p_coef_0);
        posagg_compute_linear_combination_with_p_coef(
            packed_state_1, POSAGG_LINEAR_COMB_BIAS_1, lc_result_1, p_coef_1);
        posagg_compute_linear_combination_with_p_coef(
            packed_state_2, POSAGG_LINEAR_COMB_BIAS_2, lc_result_2, p_coef_2);

        // Cols 90-99: LC result 0, col 100: p_coef 0
        for (int i = 0; i < 10; i++) traces[90 + i][row] = lc_result_0.limbs[i];
        traces[100][row] = p_coef_0;
        // Cols 101-110: LC result 1, col 111: p_coef 1
        for (int i = 0; i < 10; i++) traces[101 + i][row] = lc_result_1.limbs[i];
        traces[111][row] = p_coef_1;
        // Cols 112-121: LC result 2, col 122: p_coef 2
        for (int i = 0; i < 10; i++) traces[112 + i][row] = lc_result_2.limbs[i];
        traces[122][row] = p_coef_2;

        // ============ 4 Full Rounds (cols 123-152) ============
        PosAggFelt252Field lc_state_0 = posagg_width27_m31_to_felt252field(lc_result_0);
        PosAggFelt252Field lc_state_1 = posagg_width27_m31_to_felt252field(lc_result_1);
        PosAggFelt252Field lc_state_2 = posagg_width27_m31_to_felt252field(lc_result_2);

        PosAggFelt252Field state[3] = {lc_state_0, lc_state_1, lc_state_2};
        m31 chain_id = {row * 2};

        // pfrc_0: input to round 0
        lookup_poseidon_full_round_chain_0[0][row] = chain_id;
        lookup_poseidon_full_round_chain_0[1][row] = M31_0_val;
        for (int i = 0; i < 10; i++) {
            lookup_poseidon_full_round_chain_0[2 + i][row] = lc_result_0.limbs[i];
            lookup_poseidon_full_round_chain_0[12 + i][row] = lc_result_1.limbs[i];
            lookup_poseidon_full_round_chain_0[22 + i][row] = lc_result_2.limbs[i];
        }

        // Round 0
        posagg_full_round(state, 0);
        // pfrc_1: sub-component feed (round=1, state after round 0)
        posagg_capture_pfrc_state(lookup_poseidon_full_round_chain_1, state, chain_id, 1, row);

        // Round 1
        posagg_full_round(state, 1);
        // pfrc_2: sub-component feed (round=2, state after round 1)
        posagg_capture_pfrc_state(lookup_poseidon_full_round_chain_2, state, chain_id, 2, row);

        // Round 2
        posagg_full_round(state, 2);
        // pfrc_3: sub-component feed (round=3, state after round 2)
        posagg_capture_pfrc_state(lookup_poseidon_full_round_chain_3, state, chain_id, 3, row);

        // Round 3
        posagg_full_round(state, 3);

        // Store full round chain output (30 Width27 limbs = 3 states × 10)
        m31 full_round_out_0[10], full_round_out_1[10], full_round_out_2[10];
        posagg_felt252field_to_m31_width27(state[0], full_round_out_0);
        posagg_felt252field_to_m31_width27(state[1], full_round_out_1);
        posagg_felt252field_to_m31_width27(state[2], full_round_out_2);

        // Cols 123-132: State 0, 133-142: State 1, 143-152: State 2
        for (int i = 0; i < 10; i++) {
            traces[123 + i][row] = full_round_out_0[i];
            traces[133 + i][row] = full_round_out_1[i];
            traces[143 + i][row] = full_round_out_2[i];
        }

        // pfrc_8: IT boundary (round=4, chain 0 exit = output of 4 full rounds)
        lookup_poseidon_full_round_chain_8[0][row] = chain_id;
        lookup_poseidon_full_round_chain_8[1][row] = M31_4_val;
        for (int i = 0; i < 10; i++) {
            lookup_poseidon_full_round_chain_8[2 + i][row] = full_round_out_0[i];
            lookup_poseidon_full_round_chain_8[12 + i][row] = full_round_out_1[i];
            lookup_poseidon_full_round_chain_8[22 + i][row] = full_round_out_2[i];
        }

        // ============ Cube of full_round_out_2 (cols 153-162) ============
        PosAggFelt252Width27 fro2_w27;
        for (int i = 0; i < 10; i++) fro2_w27.limbs[i] = full_round_out_2[i];
        PosAggFelt252Field fro2_felt = posagg_width27_m31_to_felt252field(fro2_w27);
        PosAggFelt252Field cube_fro2 = posagg_felt_cube(fro2_felt);
        m31 cube_fro2_width27[10];
        posagg_felt252field_to_m31_width27(cube_fro2, cube_fro2_width27);
        for (int i = 0; i < 10; i++) traces[153 + i][row] = cube_fro2_width27[i];

        // ============ Linear combination: state0 + state1 - 2*cube + BIAS (cols 163-173) ============
        PosAggFelt252Width27 fro0_w27, fro1_w27;
        for (int i = 0; i < 10; i++) {
            fro0_w27.limbs[i] = full_round_out_0[i];
            fro1_w27.limbs[i] = full_round_out_1[i];
        }
        PosAggFelt252Field fro0_felt = posagg_width27_m31_to_felt252field(fro0_w27);
        PosAggFelt252Field fro1_felt = posagg_width27_m31_to_felt252field(fro1_w27);

        PosAggFelt252Field bias_felt = posagg_bias_to_felt252field(POSAGG_LC_BIAS_N4_COEFS_1_1_M2_1);
        PosAggFelt252Field two_cube = posagg_felt_add(cube_fro2, cube_fro2);
        PosAggFelt252Field lc_partial = posagg_felt_add(
            posagg_felt_sub(posagg_felt_add(fro0_felt, fro1_felt), two_cube), bias_felt);
        m31 lc_partial_width27[10];
        posagg_felt252field_to_m31_width27(lc_partial, lc_partial_width27);
        for (int i = 0; i < 10; i++) traces[163 + i][row] = lc_partial_width27[i];

        // Col 173: p_coef for LC N4 Coefs 1 1 M2 1
        m31 p_coef_173;
        m31 carries_33333_0[9];
        {
            int64_t biased_val = (int64_t)full_round_out_0[0] + (int64_t)full_round_out_1[0]
                        - 2 * (int64_t)cube_fro2_width27[0]
                        + (int64_t)POSAGG_LC_BIAS_N4_COEFS_1_1_M2_1[0]
                        - (int64_t)lc_partial_width27[0]
                        + 402653187LL;
            uint32_t low16 = (uint32_t)(biased_val & 0xFFFF);
            p_coef_173 = sub((m31)low16, M31_3_val);
            traces[173][row] = p_coef_173;

            m31 two_cube_0 = mul(2, cube_fro2_width27[0]);
            m31 sum_0 = add(add(add(sub(full_round_out_0[0], two_cube_0), full_round_out_1[0]),
                              POSAGG_LC_BIAS_N4_COEFS_1_1_M2_1[0]), neg(lc_partial_width27[0]));
            carries_33333_0[0] = mul(sub(sum_0, p_coef_173), 16);

            for (int i = 1; i < 9; i++) {
                m31 two_cube_i = mul(2, cube_fro2_width27[i]);
                m31 sum_i = add(add(add(add(sub(full_round_out_0[i], two_cube_i), full_round_out_1[i]),
                                       POSAGG_LC_BIAS_N4_COEFS_1_1_M2_1[i]), neg(lc_partial_width27[i])),
                               carries_33333_0[i-1]);
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_173, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_33333_0[i] = mul(sum_i, 16);
            }

            lookup_range_check_3_3_3_3_3_0[0][row] = add(p_coef_173, 3);
            for (int i = 0; i < 4; i++)
                lookup_range_check_3_3_3_3_3_0[1 + i][row] = add(carries_33333_0[i], 3);

            for (int i = 0; i < 5; i++)
                lookup_range_check_3_3_3_3_3_1[i][row] = add(carries_33333_0[4 + i], 3);
        }

        // ============ Cube of linear combination (cols 174-183) ============
        PosAggFelt252Field cube_lc_partial = posagg_felt_cube(lc_partial);
        m31 cube_lc_partial_width27[10];
        posagg_felt252field_to_m31_width27(cube_lc_partial, cube_lc_partial_width27);
        for (int i = 0; i < 10; i++) traces[174 + i][row] = cube_lc_partial_width27[i];

        // ============ LC: 4*state0 + 2*cube_fro2 - 2*cube_lc + BIAS (cols 184-194) ============
        PosAggFelt252Field bias2_felt = posagg_bias_to_felt252field(POSAGG_LC_BIAS_N4_COEFS_4_2_M2_1);
        PosAggFelt252Field four_fro0 = posagg_felt_add(
            posagg_felt_add(fro0_felt, fro0_felt), posagg_felt_add(fro0_felt, fro0_felt));
        PosAggFelt252Field two_cube_fro2 = posagg_felt_add(cube_fro2, cube_fro2);
        PosAggFelt252Field two_cube_lc = posagg_felt_add(cube_lc_partial, cube_lc_partial);
        PosAggFelt252Field lc_partial_2 = posagg_felt_add(
            posagg_felt_sub(posagg_felt_add(four_fro0, two_cube_fro2), two_cube_lc), bias2_felt);
        m31 lc_partial_2_width27[10];
        posagg_felt252field_to_m31_width27(lc_partial_2, lc_partial_2_width27);
        for (int i = 0; i < 10; i++) traces[184 + i][row] = lc_partial_2_width27[i];

        // Col 194: p_coef for LC N4 Coefs 4 2 M2 1
        m31 p_coef_194;
        m31 carries_4444[9];
        {
            int64_t biased_val = 4 * (int64_t)full_round_out_0[0]
                        + 2 * (int64_t)cube_fro2_width27[0]
                        - 2 * (int64_t)cube_lc_partial_width27[0]
                        + (int64_t)POSAGG_LC_BIAS_N4_COEFS_4_2_M2_1[0]
                        - (int64_t)lc_partial_2_width27[0]
                        + 402653187LL;
            uint32_t low16 = (uint32_t)(biased_val & 0xFFFF);
            p_coef_194 = sub((m31)low16, M31_3_val);
            traces[194][row] = p_coef_194;

            m31 four_state0_0 = mul(4, full_round_out_0[0]);
            m31 two_cube_fro2_0 = mul(2, cube_fro2_width27[0]);
            m31 two_cube_lc_0 = mul(2, cube_lc_partial_width27[0]);
            m31 sum_0 = add(add(add(sub(four_state0_0, two_cube_lc_0), two_cube_fro2_0),
                              POSAGG_LC_BIAS_N4_COEFS_4_2_M2_1[0]), neg(lc_partial_2_width27[0]));
            carries_4444[0] = mul(sub(sum_0, p_coef_194), 16);

            for (int i = 1; i < 9; i++) {
                m31 four_state0_i = mul(4, full_round_out_0[i]);
                m31 two_cube_fro2_i = mul(2, cube_fro2_width27[i]);
                m31 two_cube_lc_i = mul(2, cube_lc_partial_width27[i]);
                m31 sum_i = add(add(add(add(sub(four_state0_i, two_cube_lc_i), two_cube_fro2_i),
                                       POSAGG_LC_BIAS_N4_COEFS_4_2_M2_1[i]), neg(lc_partial_2_width27[i])),
                               carries_4444[i-1]);
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_194, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_4444[i] = mul(sum_i, 16);
            }

            lookup_range_check_4_4_4_4_0[0][row] = add(p_coef_194, 3);
            for (int i = 0; i < 3; i++)
                lookup_range_check_4_4_4_4_0[1 + i][row] = add(carries_4444[i], 3);

            for (int i = 0; i < 4; i++)
                lookup_range_check_4_4_4_4_1[i][row] = add(carries_4444[3 + i], 3);

            lookup_range_check_4_4_0[0][row] = add(carries_4444[7], 3);
            lookup_range_check_4_4_0[1][row] = add(carries_4444[8], 3);
        }

        // ============ 27 groups of 3 partial rounds (cols 195-234) ============
        PosAggFelt252Field partial_state[4] = {cube_fro2, lc_partial, cube_lc_partial, lc_partial_2};

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

        for (int group = 0; group < 27; group++) {
            unsigned round = 4 + group;

            if (lookup_partial_chain[group] != nullptr) {
                lookup_partial_chain[group][0][row] = {row};
                lookup_partial_chain[group][1][row] = {round};

                m31 state_limbs[4][10];
                posagg_felt252field_to_m31_width27(partial_state[0], state_limbs[0]);
                posagg_felt252field_to_m31_width27(partial_state[1], state_limbs[1]);
                posagg_felt252field_to_m31_width27(partial_state[2], state_limbs[2]);
                posagg_felt252field_to_m31_width27(partial_state[3], state_limbs[3]);

                for (int s = 0; s < 4; s++)
                    for (int i = 0; i < 10; i++)
                        lookup_partial_chain[group][2 + s * 10 + i][row] = state_limbs[s][i];
            }

            posagg_3_partial_rounds(partial_state, round);
        }

        // Cols 195-234: Store partial_state after all 27 groups
        m31 partial_out_0[10], partial_out_1[10], partial_out_2[10], partial_out_3[10];
        posagg_felt252field_to_m31_width27(partial_state[0], partial_out_0);
        posagg_felt252field_to_m31_width27(partial_state[1], partial_out_1);
        posagg_felt252field_to_m31_width27(partial_state[2], partial_out_2);
        posagg_felt252field_to_m31_width27(partial_state[3], partial_out_3);

        for (int i = 0; i < 10; i++) {
            traces[195 + i][row] = partial_out_0[i];
            traces[205 + i][row] = partial_out_1[i];
            traces[215 + i][row] = partial_out_2[i];
            traces[225 + i][row] = partial_out_3[i];
        }

        // p3prc[27]: IT boundary (round=31, chain exit = output after all 27 groups)
        lookup_poseidon_3_partial_rounds_chain_27[0][row] = {row};
        lookup_poseidon_3_partial_rounds_chain_27[1][row] = {31};
        for (int s = 0; s < 4; s++)
            for (int i = 0; i < 10; i++)
                lookup_poseidon_3_partial_rounds_chain_27[2 + s * 10 + i][row] =
                    (s == 0) ? partial_out_0[i] : (s == 1) ? partial_out_1[i] :
                    (s == 2) ? partial_out_2[i] : partial_out_3[i];

        // ============ Two linear combinations for final full rounds (cols 235-256) ============
        PosAggFelt252Width27 w27_state0, w27_state1, w27_state2, w27_state3;
        for (int i = 0; i < 10; i++) {
            w27_state0.limbs[i] = partial_out_0[i];
            w27_state1.limbs[i] = partial_out_1[i];
            w27_state2.limbs[i] = partial_out_2[i];
            w27_state3.limbs[i] = partial_out_3[i];
        }
        PosAggFelt252Field felt_state0 = posagg_width27_m31_to_felt252field(w27_state0);
        PosAggFelt252Field felt_state1 = posagg_width27_m31_to_felt252field(w27_state1);
        PosAggFelt252Field felt_state2 = posagg_width27_m31_to_felt252field(w27_state2);
        PosAggFelt252Field felt_state3 = posagg_width27_m31_to_felt252field(w27_state3);

        // Bias constants (Felt252 u64 representation)
        PosAggFelt252Field comb1_bias = {
            (uint32_t)(3969818800901670911ULL & 0xFFFFFFFF),
            (uint32_t)(3969818800901670911ULL >> 32),
            (uint32_t)(10562874008078701503ULL & 0xFFFFFFFF),
            (uint32_t)(10562874008078701503ULL >> 32),
            (uint32_t)(14906396266795319764ULL & 0xFFFFFFFF),
            (uint32_t)(14906396266795319764ULL >> 32),
            (uint32_t)(223312371439046257ULL & 0xFFFFFFFF),
            (uint32_t)(223312371439046257ULL >> 32)
        };
        PosAggFelt252Field comb2_bias = {
            (uint32_t)(10310704347937391837ULL & 0xFFFFFFFF),
            (uint32_t)(10310704347937391837ULL >> 32),
            (uint32_t)(5874215448258336115ULL & 0xFFFFFFFF),
            (uint32_t)(5874215448258336115ULL >> 32),
            (uint32_t)(2880320859071049537ULL & 0xFFFFFFFF),
            (uint32_t)(2880320859071049537ULL >> 32),
            (uint32_t)(45350836576946303ULL & 0xFFFFFFFF),
            (uint32_t)(45350836576946303ULL >> 32)
        };

        felt_state0 = posagg_felt_to_mont(felt_state0);
        felt_state1 = posagg_felt_to_mont(felt_state1);
        felt_state2 = posagg_felt_to_mont(felt_state2);
        felt_state3 = posagg_felt_to_mont(felt_state3);
        comb1_bias = posagg_felt_to_mont(comb1_bias);
        comb2_bias = posagg_felt_to_mont(comb2_bias);

        PosAggFelt252Field four = {4, 0, 0, 0, 0, 0, 0, 0};
        PosAggFelt252Field two_val = {2, 0, 0, 0, 0, 0, 0, 0};
        four = posagg_felt_to_mont(four);
        two_val = posagg_felt_to_mont(two_val);

        // comb1 = 4*state0 + 2*state1 + state2 + comb1_bias
        PosAggFelt252Field comb1 = posagg_felt_add(
            posagg_felt_add(
                posagg_felt_add(posagg_felt_mul(four, felt_state0), posagg_felt_mul(two_val, felt_state1)),
                felt_state2),
            comb1_bias);

        PosAggFelt252Field comb1_normal = posagg_felt_from_mont(comb1);
        m31 comb1_width27[10];
        posagg_felt252field_to_m31_width27(comb1_normal, comb1_width27);
        for (int i = 0; i < 10; i++) traces[235 + i][row] = comb1_width27[i];

        // Width27 bias constants for carry computation
        const m31 COMB1_BIAS[10] = {
            40454143, 49554771, 55508188, 116986206, 88680813,
            45553283, 62360091, 77099918, 22899501, 99
        };

        // Col 245: p_coef for combination1
        {
            int64_t biased = 4LL * (int64_t)partial_out_0[0]
                        + 2LL * (int64_t)partial_out_1[0]
                        + (int64_t)partial_out_2[0]
                        + (int64_t)COMB1_BIAS[0]
                        - (int64_t)comb1_width27[0]
                        + 134217729LL;
            uint32_t low16 = (uint32_t)(biased & 0xFFFF);
            traces[245][row] = (m31){low16 - 1};
        }

        // comb2 = 4*state2 + 2*state3 + comb1 + comb2_bias
        PosAggFelt252Field comb2 = posagg_felt_add(
            posagg_felt_add(
                posagg_felt_add(posagg_felt_mul(four, felt_state2), posagg_felt_mul(two_val, felt_state3)),
                comb1),
            comb2_bias);

        PosAggFelt252Field comb2_normal = posagg_felt_from_mont(comb2);
        m31 comb2_width27[10];
        posagg_felt252field_to_m31_width27(comb2_normal, comb2_width27);
        for (int i = 0; i < 10; i++) traces[246 + i][row] = comb2_width27[i];

        // pfrc_4: sub-component feed + IT boundary (round=31, chain 1 entry)
        // state0=comb2, state1=comb1, state2=partial_out_3
        {
            m31 chain_id_1 = {row * 2 + 1};
            lookup_poseidon_full_round_chain_4[0][row] = chain_id_1;
            lookup_poseidon_full_round_chain_4[1][row] = M31_31_val;
            for (int i = 0; i < 10; i++) {
                lookup_poseidon_full_round_chain_4[2 + i][row] = comb2_width27[i];
                lookup_poseidon_full_round_chain_4[12 + i][row] = comb1_width27[i];
                lookup_poseidon_full_round_chain_4[22 + i][row] = partial_out_3[i];
            }
        }

        // Col 256: p_coef for combination2
        const m31 COMB2_BIAS[10] = {
            48383197, 48193339, 55955004, 65659846, 68491350,
            119023582, 33439011, 58475513, 18765944, 20
        };
        {
            int64_t biased = 4LL * (int64_t)partial_out_2[0]
                        + 2LL * (int64_t)partial_out_3[0]
                        + (int64_t)comb1_width27[0]
                        + (int64_t)COMB2_BIAS[0]
                        - (int64_t)comb2_width27[0]
                        + 134217729LL;
            uint32_t low16 = (uint32_t)(biased & 0xFFFF);
            traces[256][row] = (m31){low16 - 1};
        }

        // ============ Final 4 full rounds (cols 257-286) ============
        PosAggFelt252Width27 w27_comb2, w27_comb1, w27_partial3;
        for (int i = 0; i < 10; i++) {
            w27_comb2.limbs[i] = comb2_width27[i];
            w27_comb1.limbs[i] = comb1_width27[i];
            w27_partial3.limbs[i] = partial_out_3[i];
        }

        PosAggFelt252Field final_state0 = posagg_width27_m31_to_felt252field(w27_comb2);
        PosAggFelt252Field final_state1 = posagg_width27_m31_to_felt252field(w27_comb1);
        PosAggFelt252Field final_state2 = posagg_width27_m31_to_felt252field(w27_partial3);

        state[0] = final_state0;
        state[1] = final_state1;
        state[2] = final_state2;

        // Round 31
        posagg_full_round(state, 31);
        {
            m31 chain_id_1 = {row * 2 + 1};
            // pfrc_5: sub-component feed (round=32, state after round 31)
            posagg_capture_pfrc_state(lookup_poseidon_full_round_chain_5, state, chain_id_1, 32, row);
        }

        // Round 32
        posagg_full_round(state, 32);
        {
            m31 chain_id_1 = {row * 2 + 1};
            // pfrc_6: sub-component feed (round=33, state after round 32)
            posagg_capture_pfrc_state(lookup_poseidon_full_round_chain_6, state, chain_id_1, 33, row);
        }

        // Round 33
        posagg_full_round(state, 33);
        {
            m31 chain_id_1 = {row * 2 + 1};
            // pfrc_7: sub-component feed (round=34, state after round 33)
            posagg_capture_pfrc_state(lookup_poseidon_full_round_chain_7, state, chain_id_1, 34, row);
        }

        // Round 34 — note: round key is zeros
        posagg_full_round(state, 34);

        // Store final full round output (cols 257-286)
        m31 final_out_0[10], final_out_1[10], final_out_2[10];
        posagg_felt252field_to_m31_width27(state[0], final_out_0);
        posagg_felt252field_to_m31_width27(state[1], final_out_1);
        posagg_felt252field_to_m31_width27(state[2], final_out_2);

        for (int i = 0; i < 10; i++) {
            traces[257 + i][row] = final_out_0[i];
            traces[267 + i][row] = final_out_1[i];
            traces[277 + i][row] = final_out_2[i];
        }

        // pfrc_9: IT boundary (round=35, chain 1 exit = output of final 4 full rounds)
        {
            m31 chain_id_1 = {row * 2 + 1};
            lookup_poseidon_full_round_chain_9[0][row] = chain_id_1;
            lookup_poseidon_full_round_chain_9[1][row] = {35};
            for (int i = 0; i < 10; i++) {
                lookup_poseidon_full_round_chain_9[2 + i][row] = final_out_0[i];
                lookup_poseidon_full_round_chain_9[12 + i][row] = final_out_1[i];
                lookup_poseidon_full_round_chain_9[22 + i][row] = final_out_2[i];
            }
        }

        // ============ Unpack output states (cols 287-340) ============
        // 3 states × 18 selected indices from 28-limb Felt252
        int unpack_indices[18] = {0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25};

        // Output state 0: cols 287-304
        m31 felt252_limbs_0[28];
        PosAggFelt252Width27 w27_out_0;
        for (int i = 0; i < 10; i++) w27_out_0.limbs[i] = final_out_0[i];
        posagg_unpack_felt252_from_width27(w27_out_0, felt252_limbs_0);
        for (int i = 0; i < 18; i++) traces[287 + i][row] = felt252_limbs_0[unpack_indices[i]];

        // Output state 1: cols 305-322
        m31 felt252_limbs_1[28];
        PosAggFelt252Width27 w27_out_1;
        posagg_felt252field_to_width27_m31(state[1], w27_out_1);
        posagg_unpack_felt252_from_width27(w27_out_1, felt252_limbs_1);
        for (int i = 0; i < 18; i++) traces[305 + i][row] = felt252_limbs_1[unpack_indices[i]];

        // Output state 2: cols 323-340
        m31 felt252_limbs_2[28];
        PosAggFelt252Width27 w27_out_2;
        posagg_felt252field_to_width27_m31(state[2], w27_out_2);
        posagg_unpack_felt252_from_width27(w27_out_2, felt252_limbs_2);
        for (int i = 0; i < 18; i++) traces[323 + i][row] = felt252_limbs_2[unpack_indices[i]];

        // ============ Multiplicity (col 341) ============
        traces[341][row] = {mults_in[row]};

        // ============ Memory_id_to_big for output states (ids 3-5) ============
        // Read output state 0 limbs via id3
        m31 output_state_0_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            id3, output_state_0_limbs);


        lookup_memory_id_to_big_3[0][row] = id3;
        for (int i = 0; i < 28; i++) lookup_memory_id_to_big_3[1 + i][row] = output_state_0_limbs[i];

        // Read output state 1 limbs via id4
        m31 output_state_1_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            id4, output_state_1_limbs);

        lookup_memory_id_to_big_4[0][row] = id4;
        for (int i = 0; i < 28; i++) lookup_memory_id_to_big_4[1 + i][row] = output_state_1_limbs[i];

        // Read output state 2 limbs via id5
        m31 output_state_2_limbs[28];
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            id5, output_state_2_limbs);


        lookup_memory_id_to_big_5[0][row] = id5;
        for (int i = 0; i < 28; i++) lookup_memory_id_to_big_5[1 + i][row] = output_state_2_limbs[i];

        // ============ Carries for combination1 and combination2 ============
        // Comb1 carries: rc_4_4_4_4_2, rc_4_4_4_4_3, rc_4_4_1
        {
            m31 p_coef_245 = traces[245][row];
            m31 carries_comb1[9];

            m31 four_s0_0 = mul(4, partial_out_0[0]);
            m31 two_s1_0 = mul(2, partial_out_1[0]);
            m31 sum_0 = add(add(add(add(four_s0_0, two_s1_0), partial_out_2[0]),
                               COMB1_BIAS[0]), neg(comb1_width27[0]));
            carries_comb1[0] = mul(sub(sum_0, p_coef_245), 16);

            for (int i = 1; i < 9; i++) {
                m31 four_s0_i = mul(4, partial_out_0[i]);
                m31 two_s1_i = mul(2, partial_out_1[i]);
                m31 sum_i = add(add(add(add(add(four_s0_i, two_s1_i), partial_out_2[i]),
                                       COMB1_BIAS[i]), neg(comb1_width27[i])),
                               carries_comb1[i-1]);
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_245, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_comb1[i] = mul(sum_i, 16);
            }

            lookup_range_check_4_4_4_4_2[0][row] = add(p_coef_245, 1);
            for (int i = 0; i < 3; i++)
                lookup_range_check_4_4_4_4_2[1 + i][row] = add(carries_comb1[i], 1);

            for (int i = 0; i < 4; i++)
                lookup_range_check_4_4_4_4_3[i][row] = add(carries_comb1[3 + i], 1);

            lookup_range_check_4_4_1[0][row] = add(carries_comb1[7], 1);
            lookup_range_check_4_4_1[1][row] = add(carries_comb1[8], 1);
        }

        // Comb2 carries: rc_4_4_4_4_4, rc_4_4_4_4_5, rc_4_4_2
        {
            m31 p_coef_256 = traces[256][row];
            m31 carries_comb2[9];

            m31 four_s2_0 = mul(4, partial_out_2[0]);
            m31 two_s3_0 = mul(2, partial_out_3[0]);
            m31 sum_0 = add(add(add(add(four_s2_0, two_s3_0), comb1_width27[0]),
                               COMB2_BIAS[0]), neg(comb2_width27[0]));
            carries_comb2[0] = mul(sub(sum_0, p_coef_256), 16);

            for (int i = 1; i < 9; i++) {
                m31 four_s2_i = mul(4, partial_out_2[i]);
                m31 two_s3_i = mul(2, partial_out_3[i]);
                m31 sum_i = add(add(add(add(add(four_s2_i, two_s3_i), comb1_width27[i]),
                                       COMB2_BIAS[i]), neg(comb2_width27[i])),
                               carries_comb2[i-1]);
                if (i == 7) {
                    m31 p_coef_term = mul(p_coef_256, 136);
                    sum_i = sub(sum_i, p_coef_term);
                }
                carries_comb2[i] = mul(sum_i, 16);
            }

            lookup_range_check_4_4_4_4_4[0][row] = add(p_coef_256, 1);
            for (int i = 0; i < 3; i++)
                lookup_range_check_4_4_4_4_4[1 + i][row] = add(carries_comb2[i], 1);

            for (int i = 0; i < 4; i++)
                lookup_range_check_4_4_4_4_5[i][row] = add(carries_comb2[3 + i], 1);

            lookup_range_check_4_4_2[0][row] = add(carries_comb2[7], 1);
            lookup_range_check_4_4_2[1][row] = add(carries_comb2[8], 1);
        }
    }
}

// ============================================================================
// Interaction trace generation kernels (using CommonLookupElements)
// ============================================================================
//
// All kernels use LookupElementsBasic<128> (= CommonLookupElements).
// Lookup data from the base trace does NOT include relation_id, so we
// prepend relid as a scalar parameter before calling combine().
//
// N0, N1 = element count WITHOUT relation_id.
// ============================================================================

// ADD pair kernel: frac = (d0 + d1) / (d0 * d1)
template <int N0, int N1>
__launch_bounds__(AGG_BLOCK_SIZE, 2)
__global__ void posagg_it_add_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31 relid0, m31** data_0,
    m31 relid1, m31** data_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)trace_size) return;

    m31 vals0[N0 + 1], vals1[N1 + 1];
    vals0[0] = relid0;
    for (int i = 0; i < N0; i++) vals0[1 + i] = data_0[i][idx];
    vals1[0] = relid1;
    for (int i = 0; i < N1; i++) vals1[1 + i] = data_1[i][idx];

    qm31 d0 = lookup_elements->combine(vals0, N0 + 1);
    qm31 d1 = lookup_elements->combine(vals1, N1 + 1);

    logup_col_write_frac(idx, add(d0, d1), mul(d0, d1),
                        denom_ptr, numer0, numer1, numer2, numer3);
}

// SUB pair kernel (d1 - d0): frac = (d1 - d0) / (d0 * d1)
template <int N0, int N1>
__launch_bounds__(AGG_BLOCK_SIZE, 2)
__global__ void posagg_it_sub_d1d0_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31 relid0, m31** data_0,
    m31 relid1, m31** data_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)trace_size) return;

    m31 vals0[N0 + 1], vals1[N1 + 1];
    vals0[0] = relid0;
    for (int i = 0; i < N0; i++) vals0[1 + i] = data_0[i][idx];
    vals1[0] = relid1;
    for (int i = 0; i < N1; i++) vals1[1 + i] = data_1[i][idx];

    qm31 d0 = lookup_elements->combine(vals0, N0 + 1);
    qm31 d1 = lookup_elements->combine(vals1, N1 + 1);

    logup_col_write_frac(idx, sub(d1, d0), mul(d0, d1),
                        denom_ptr, numer0, numer1, numer2, numer3);
}

// SUB pair kernel (d0 - d1): frac = (d0 - d1) / (d0 * d1)
template <int N0, int N1>
__launch_bounds__(AGG_BLOCK_SIZE, 2)
__global__ void posagg_it_sub_d0d1_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31 relid0, m31** data_0,
    m31 relid1, m31** data_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)trace_size) return;

    m31 vals0[N0 + 1], vals1[N1 + 1];
    vals0[0] = relid0;
    for (int i = 0; i < N0; i++) vals0[1 + i] = data_0[i][idx];
    vals1[0] = relid1;
    for (int i = 0; i < N1; i++) vals1[1 + i] = data_1[i][idx];

    qm31 d0 = lookup_elements->combine(vals0, N0 + 1);
    qm31 d1 = lookup_elements->combine(vals1, N1 + 1);

    logup_col_write_frac(idx, sub(d0, d1), mul(d0, d1),
                        denom_ptr, numer0, numer1, numer2, numer3);
}

// MULT kernel: frac = (d1 - d0 * mult) / (d0 * d1)
template <int N0, int N1>
__launch_bounds__(AGG_BLOCK_SIZE, 2)
__global__ void posagg_it_mult_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31 relid0, m31** data_0,
    m31 relid1, m31** data_1,
    m31* mults,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)trace_size) return;

    m31 vals0[N0 + 1], vals1[N1 + 1];
    vals0[0] = relid0;
    for (int i = 0; i < N0; i++) vals0[1 + i] = data_0[i][idx];
    vals1[0] = relid1;
    for (int i = 0; i < N1; i++) vals1[1 + i] = data_1[i][idx];

    qm31 d0 = lookup_elements->combine(vals0, N0 + 1);
    qm31 d1 = lookup_elements->combine(vals1, N1 + 1);

    m31 m = mults[idx];
    qm31 scaled_d0 = mul(m, d0);

    logup_col_write_frac(idx, sub(d1, scaled_d0), mul(d0, d1),
                        denom_ptr, numer0, numer1, numer2, numer3);
}

// Finalize kernel: multiply numerator by inverse denominator and accumulate
__global__ void posagg_it_finalize_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = rep_index - 1;

    if (idx < (int)trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[idx], numerator1[idx]},
                cm31{numerator2[idx], numerator3[idx]}
            },
            denom_inv_ptr[idx]
        );

        if (pre_index == -1) {
            qm31 tmp = value;
            numerator0[idx] = tmp.a.a;
            numerator1[idx] = tmp.a.b;
            numerator2[idx] = tmp.b.a;
            numerator3[idx] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][idx], interaction_traces[pre_index * 4 + 1][idx]},
                cm31{interaction_traces[pre_index * 4 + 2][idx], interaction_traces[pre_index * 4 + 3][idx]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[idx] = tmp.a.a;
            numerator1[idx] = tmp.a.b;
            numerator2[idx] = tmp.b.a;
            numerator3[idx] = tmp.b.b;
        }

        interaction_traces[rep_index * 4 + 0][idx] = numerator0[idx];
        interaction_traces[rep_index * 4 + 1][idx] = numerator1[idx];
        interaction_traces[rep_index * 4 + 2][idx] = numerator2[idx];
        interaction_traces[rep_index * 4 + 3][idx] = numerator3[idx];
    }
}

// Cumsum shift kernel — computes claimed_sum from last column
__global__ void posagg_it_cumsum_shift(
    unsigned n_cols,
    unsigned trace_size,
    m31** interactive_traces,
    m31* coordinate_sums
) {
    int idx0 = 4 * n_cols - 4;
    int idx1 = 4 * n_cols - 3;
    int idx2 = 4 * n_cols - 2;
    int idx3 = 4 * n_cols - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = tid; i < (int)trace_size; i += gridSize) {
        s0 = add(s0, interactive_traces[idx0][i]);
        s1 = add(s1, interactive_traces[idx1][i]);
        s2 = add(s2, interactive_traces[idx2][i]);
        s3 = add(s3, interactive_traces[idx3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sd0 = &shared[0];
    m31* sd1 = &shared[blockDim.x];
    m31* sd2 = &shared[2 * blockDim.x];
    m31* sd3 = &shared[3 * blockDim.x];

    sd0[threadIdx.x] = s0;
    sd1[threadIdx.x] = s1;
    sd2[threadIdx.x] = s2;
    sd3[threadIdx.x] = s3;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sd0[threadIdx.x] = add(sd0[threadIdx.x], sd0[threadIdx.x + s]);
            sd1[threadIdx.x] = add(sd1[threadIdx.x], sd1[threadIdx.x + s]);
            sd2[threadIdx.x] = add(sd2[threadIdx.x], sd2[threadIdx.x + s]);
            sd3[threadIdx.x] = add(sd3[threadIdx.x], sd3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sd0[0]);
        atomic_add(&coordinate_sums[1], sd1[0]);
        atomic_add(&coordinate_sums[2], sd2[0]);
        atomic_add(&coordinate_sums[3], sd3[0]);
    }
}

// Coordinate prefix sum kernel — subtracts shift from last column
__global__ void posagg_it_coord_prefix_sum(
    m31* coordinate_sums,
    unsigned n_cols,
    unsigned trace_size,
    m31** interactive_traces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (int)trace_size) {
        qm31 cs = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 shift = div(cs, m31(trace_size));

        interactive_traces[4 * n_cols - 4][idx] = sub(interactive_traces[4 * n_cols - 4][idx], shift.a.a);
        interactive_traces[4 * n_cols - 3][idx] = sub(interactive_traces[4 * n_cols - 3][idx], shift.a.b);
        interactive_traces[4 * n_cols - 2][idx] = sub(interactive_traces[4 * n_cols - 2][idx], shift.b.a);
        interactive_traces[4 * n_cols - 1][idx] = sub(interactive_traces[4 * n_cols - 1][idx], shift.b.b);
    }
}

// Macros for processing logup columns
#define POSAGG_IT_PROCESS_ADD(col, N0, N1, r0, d0, r1, d1) \
    posagg_it_add_kernel<N0, N1><<<num_blocks, block_dim>>>( \
        d_lookup, r0, d0, r1, d1, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    posagg_it_finalize_kernel<<<num_blocks, block_dim>>>(col, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

#define POSAGG_IT_PROCESS_SUB_D1D0(col, N0, N1, r0, d0, r1, d1) \
    posagg_it_sub_d1d0_kernel<N0, N1><<<num_blocks, block_dim>>>( \
        d_lookup, r0, d0, r1, d1, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    posagg_it_finalize_kernel<<<num_blocks, block_dim>>>(col, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

#define POSAGG_IT_PROCESS_SUB_D0D1(col, N0, N1, r0, d0, r1, d1) \
    posagg_it_sub_d0d1_kernel<N0, N1><<<num_blocks, block_dim>>>( \
        d_lookup, r0, d0, r1, d1, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    posagg_it_finalize_kernel<<<num_blocks, block_dim>>>(col, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

// ============================================================================
// C wrapper: gen_poseidon_aggregator_trace (base trace)
// ============================================================================

extern "C" void gen_poseidon_aggregator_trace(
    unsigned *traces,
    unsigned log_size,
    unsigned *input_ids_0,
    unsigned *input_ids_1,
    unsigned *input_ids_2,
    unsigned *input_ids_3,
    unsigned *input_ids_4,
    unsigned *input_ids_5,
    unsigned *mults_in,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,
    unsigned *lookup_memory_id_to_big_0,
    unsigned *lookup_memory_id_to_big_1,
    unsigned *lookup_memory_id_to_big_2,
    unsigned *lookup_memory_id_to_big_3,
    unsigned *lookup_memory_id_to_big_4,
    unsigned *lookup_memory_id_to_big_5,
    unsigned *lookup_range_check_3_3_3_3_3_0,
    unsigned *lookup_range_check_3_3_3_3_3_1,
    unsigned *lookup_range_check_4_4_4_4_0,
    unsigned *lookup_range_check_4_4_4_4_1,
    unsigned *lookup_range_check_4_4_4_4_2,
    unsigned *lookup_range_check_4_4_4_4_3,
    unsigned *lookup_range_check_4_4_4_4_4,
    unsigned *lookup_range_check_4_4_4_4_5,
    unsigned *lookup_range_check_4_4_0,
    unsigned *lookup_range_check_4_4_1,
    unsigned *lookup_range_check_4_4_2,
    unsigned *lookup_poseidon_full_round_chain_0,
    unsigned *lookup_poseidon_full_round_chain_1,
    unsigned *lookup_poseidon_full_round_chain_2,
    unsigned *lookup_poseidon_full_round_chain_3,
    unsigned *lookup_poseidon_full_round_chain_4,
    unsigned *lookup_poseidon_full_round_chain_5,
    unsigned *lookup_poseidon_full_round_chain_6,
    unsigned *lookup_poseidon_full_round_chain_7,
    unsigned *lookup_poseidon_full_round_chain_8,
    unsigned *lookup_poseidon_full_round_chain_9,
    unsigned *lookup_poseidon_3_partial_rounds_chain_0,
    unsigned *lookup_poseidon_3_partial_rounds_chain_1,
    unsigned *lookup_poseidon_3_partial_rounds_chain_2,
    unsigned *lookup_poseidon_3_partial_rounds_chain_3,
    unsigned *lookup_poseidon_3_partial_rounds_chain_4,
    unsigned *lookup_poseidon_3_partial_rounds_chain_5,
    unsigned *lookup_poseidon_3_partial_rounds_chain_6,
    unsigned *lookup_poseidon_3_partial_rounds_chain_7,
    unsigned *lookup_poseidon_3_partial_rounds_chain_8,
    unsigned *lookup_poseidon_3_partial_rounds_chain_9,
    unsigned *lookup_poseidon_3_partial_rounds_chain_10,
    unsigned *lookup_poseidon_3_partial_rounds_chain_11,
    unsigned *lookup_poseidon_3_partial_rounds_chain_12,
    unsigned *lookup_poseidon_3_partial_rounds_chain_13,
    unsigned *lookup_poseidon_3_partial_rounds_chain_14,
    unsigned *lookup_poseidon_3_partial_rounds_chain_15,
    unsigned *lookup_poseidon_3_partial_rounds_chain_16,
    unsigned *lookup_poseidon_3_partial_rounds_chain_17,
    unsigned *lookup_poseidon_3_partial_rounds_chain_18,
    unsigned *lookup_poseidon_3_partial_rounds_chain_19,
    unsigned *lookup_poseidon_3_partial_rounds_chain_20,
    unsigned *lookup_poseidon_3_partial_rounds_chain_21,
    unsigned *lookup_poseidon_3_partial_rounds_chain_22,
    unsigned *lookup_poseidon_3_partial_rounds_chain_23,
    unsigned *lookup_poseidon_3_partial_rounds_chain_24,
    unsigned *lookup_poseidon_3_partial_rounds_chain_25,
    unsigned *lookup_poseidon_3_partial_rounds_chain_26,
    unsigned *lookup_poseidon_3_partial_rounds_chain_27
) {
    unsigned trace_size = 1 << log_size;
    unsigned n_rows = trace_size;
    int blocks = (trace_size + AGG_BLOCK_SIZE - 1) / AGG_BLOCK_SIZE;

    // Clone all pointer arrays from HOST to DEVICE
    // (HOST arrays contain DEVICE pointers; CUDA kernels need DEVICE arrays of DEVICE pointers)
    m31** d_traces = clone_to_device<m31*>((m31**)traces, AGG_N_TRACE_COLUMNS);
    unsigned** d_mem_big = clone_to_device<unsigned*>(memory_id_to_big_transposed_big_values, 8);

    m31** d_lk_mem[6] = {
        clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29),
        clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29),
        clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29),
        clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29),
        clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29),
        clone_to_device<m31*>((m31**)lookup_memory_id_to_big_5, 29),
    };
    m31** d_lk_rc33333[2] = {
        clone_to_device<m31*>((m31**)lookup_range_check_3_3_3_3_3_0, 5),
        clone_to_device<m31*>((m31**)lookup_range_check_3_3_3_3_3_1, 5),
    };
    m31** d_lk_rc4444[6] = {
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_0, 4),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_1, 4),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_2, 4),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_3, 4),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_4, 4),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_5, 4),
    };
    m31** d_lk_rc44[3] = {
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_0, 2),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_1, 2),
        clone_to_device<m31*>((m31**)lookup_range_check_4_4_2, 2),
    };

    unsigned* pfrc_host_ptrs[10] = {
        lookup_poseidon_full_round_chain_0, lookup_poseidon_full_round_chain_1,
        lookup_poseidon_full_round_chain_2, lookup_poseidon_full_round_chain_3,
        lookup_poseidon_full_round_chain_4, lookup_poseidon_full_round_chain_5,
        lookup_poseidon_full_round_chain_6, lookup_poseidon_full_round_chain_7,
        lookup_poseidon_full_round_chain_8, lookup_poseidon_full_round_chain_9,
    };
    m31** d_lk_pfrc[10];
    for (int i = 0; i < 10; i++)
        d_lk_pfrc[i] = clone_to_device<m31*>((m31**)pfrc_host_ptrs[i], 32);

    unsigned* p3prc_host_ptrs[28] = {
        lookup_poseidon_3_partial_rounds_chain_0,  lookup_poseidon_3_partial_rounds_chain_1,
        lookup_poseidon_3_partial_rounds_chain_2,  lookup_poseidon_3_partial_rounds_chain_3,
        lookup_poseidon_3_partial_rounds_chain_4,  lookup_poseidon_3_partial_rounds_chain_5,
        lookup_poseidon_3_partial_rounds_chain_6,  lookup_poseidon_3_partial_rounds_chain_7,
        lookup_poseidon_3_partial_rounds_chain_8,  lookup_poseidon_3_partial_rounds_chain_9,
        lookup_poseidon_3_partial_rounds_chain_10, lookup_poseidon_3_partial_rounds_chain_11,
        lookup_poseidon_3_partial_rounds_chain_12, lookup_poseidon_3_partial_rounds_chain_13,
        lookup_poseidon_3_partial_rounds_chain_14, lookup_poseidon_3_partial_rounds_chain_15,
        lookup_poseidon_3_partial_rounds_chain_16, lookup_poseidon_3_partial_rounds_chain_17,
        lookup_poseidon_3_partial_rounds_chain_18, lookup_poseidon_3_partial_rounds_chain_19,
        lookup_poseidon_3_partial_rounds_chain_20, lookup_poseidon_3_partial_rounds_chain_21,
        lookup_poseidon_3_partial_rounds_chain_22, lookup_poseidon_3_partial_rounds_chain_23,
        lookup_poseidon_3_partial_rounds_chain_24, lookup_poseidon_3_partial_rounds_chain_25,
        lookup_poseidon_3_partial_rounds_chain_26, lookup_poseidon_3_partial_rounds_chain_27,
    };
    m31** d_lk_p3prc[28];
    for (int i = 0; i < 28; i++)
        d_lk_p3prc[i] = clone_to_device<m31*>((m31**)p3prc_host_ptrs[i], 42);

    // Launch base trace kernel
    generate_poseidon_aggregator_base_trace_kernel<<<blocks, AGG_BLOCK_SIZE>>>(
        d_traces,
        input_ids_0, input_ids_1, input_ids_2,
        input_ids_3, input_ids_4, input_ids_5,
        mults_in,
        d_mem_big,
        memory_id_to_big_small_values,
        d_lk_mem[0], d_lk_mem[1], d_lk_mem[2],
        d_lk_mem[3], d_lk_mem[4], d_lk_mem[5],
        d_lk_rc33333[0], d_lk_rc33333[1],
        d_lk_rc4444[0], d_lk_rc4444[1], d_lk_rc4444[2],
        d_lk_rc4444[3], d_lk_rc4444[4], d_lk_rc4444[5],
        d_lk_rc44[0], d_lk_rc44[1], d_lk_rc44[2],
        d_lk_pfrc[0], d_lk_pfrc[1], d_lk_pfrc[2], d_lk_pfrc[3],
        d_lk_pfrc[4], d_lk_pfrc[5], d_lk_pfrc[6], d_lk_pfrc[7],
        d_lk_pfrc[8], d_lk_pfrc[9],
        d_lk_p3prc[0],  d_lk_p3prc[1],  d_lk_p3prc[2],  d_lk_p3prc[3],
        d_lk_p3prc[4],  d_lk_p3prc[5],  d_lk_p3prc[6],  d_lk_p3prc[7],
        d_lk_p3prc[8],  d_lk_p3prc[9],  d_lk_p3prc[10], d_lk_p3prc[11],
        d_lk_p3prc[12], d_lk_p3prc[13], d_lk_p3prc[14], d_lk_p3prc[15],
        d_lk_p3prc[16], d_lk_p3prc[17], d_lk_p3prc[18], d_lk_p3prc[19],
        d_lk_p3prc[20], d_lk_p3prc[21], d_lk_p3prc[22], d_lk_p3prc[23],
        d_lk_p3prc[24], d_lk_p3prc[25], d_lk_p3prc[26], d_lk_p3prc[27],
        n_rows,
        trace_size
    );

    // Cleanup device pointer arrays
    cuda_free_memory(d_traces);
    cuda_free_memory(d_mem_big);
    for (int i = 0; i < 6; i++) cuda_free_memory(d_lk_mem[i]);
    for (int i = 0; i < 2; i++) cuda_free_memory(d_lk_rc33333[i]);
    for (int i = 0; i < 6; i++) cuda_free_memory(d_lk_rc4444[i]);
    for (int i = 0; i < 3; i++) cuda_free_memory(d_lk_rc44[i]);
    for (int i = 0; i < 10; i++) cuda_free_memory(d_lk_pfrc[i]);
    for (int i = 0; i < 28; i++) cuda_free_memory(d_lk_p3prc[i]);
}

// ============================================================================
// C wrapper: gen_poseidon_aggregator_interaction_trace
// ============================================================================

extern "C" void gen_poseidon_aggregator_interaction_trace(
    void* lookup_elements,
    // 6 memory_id_to_big (29 elements each)
    unsigned *lookup_memory_id_to_big_0,
    unsigned *lookup_memory_id_to_big_1,
    unsigned *lookup_memory_id_to_big_2,
    unsigned *lookup_memory_id_to_big_3,
    unsigned *lookup_memory_id_to_big_4,
    unsigned *lookup_memory_id_to_big_5,
    // 2 rc_3_3_3_3_3 (5 elements each)
    unsigned *lookup_range_check_3_3_3_3_3_0,
    unsigned *lookup_range_check_3_3_3_3_3_1,
    // 6 rc_4_4_4_4 (4 elements each)
    unsigned *lookup_range_check_4_4_4_4_0,
    unsigned *lookup_range_check_4_4_4_4_1,
    unsigned *lookup_range_check_4_4_4_4_2,
    unsigned *lookup_range_check_4_4_4_4_3,
    unsigned *lookup_range_check_4_4_4_4_4,
    unsigned *lookup_range_check_4_4_4_4_5,
    // 3 rc_4_4 (2 elements each)
    unsigned *lookup_range_check_4_4_0,
    unsigned *lookup_range_check_4_4_1,
    unsigned *lookup_range_check_4_4_2,
    // 4 pfrc (32 elements each)
    unsigned *lookup_poseidon_full_round_chain_0,
    unsigned *lookup_poseidon_full_round_chain_1,
    unsigned *lookup_poseidon_full_round_chain_2,
    unsigned *lookup_poseidon_full_round_chain_3,
    // 2 p3prc (42 elements each)
    unsigned *lookup_poseidon_3_partial_rounds_chain_0,
    unsigned *lookup_poseidon_3_partial_rounds_chain_1,
    // Base trace (342 columns)
    unsigned *base_trace,
    unsigned log_size,
    // Output
    unsigned *interaction_trace_columns,
    unsigned *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    // 1. Copy CommonLookupElements to device
    LookupElementsBasic<128>* d_lookup = cuda_malloc<LookupElementsBasic<128>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<128>>(
        (LookupElementsBasic<128>*)lookup_elements, d_lookup, 1);

    // 2. Clone lookup data pointer arrays to device
    m31** d_mem_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);
    m31** d_mem_1 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29);
    m31** d_mem_2 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29);
    m31** d_mem_3 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29);
    m31** d_mem_4 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29);
    m31** d_mem_5 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_5, 29);

    m31** d_rc33333_0 = clone_to_device<m31*>((m31**)lookup_range_check_3_3_3_3_3_0, 5);
    m31** d_rc33333_1 = clone_to_device<m31*>((m31**)lookup_range_check_3_3_3_3_3_1, 5);

    m31** d_rc4444_0 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_0, 4);
    m31** d_rc4444_1 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_1, 4);
    m31** d_rc4444_2 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_2, 4);
    m31** d_rc4444_3 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_3, 4);
    m31** d_rc4444_4 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_4, 4);
    m31** d_rc4444_5 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_4_4_5, 4);

    m31** d_rc44_0 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_0, 2);
    m31** d_rc44_1 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_1, 2);
    m31** d_rc44_2 = clone_to_device<m31*>((m31**)lookup_range_check_4_4_2, 2);

    m31** d_pfrc_0 = clone_to_device<m31*>((m31**)lookup_poseidon_full_round_chain_0, 32);
    m31** d_pfrc_1 = clone_to_device<m31*>((m31**)lookup_poseidon_full_round_chain_1, 32);
    m31** d_pfrc_2 = clone_to_device<m31*>((m31**)lookup_poseidon_full_round_chain_2, 32);
    m31** d_pfrc_3 = clone_to_device<m31*>((m31**)lookup_poseidon_full_round_chain_3, 32);

    m31** d_p3prc_0 = clone_to_device<m31*>((m31**)lookup_poseidon_3_partial_rounds_chain_0, 42);
    m31** d_p3prc_1 = clone_to_device<m31*>((m31**)lookup_poseidon_3_partial_rounds_chain_1, 42);

    // 3. Build cube_252, rc_252w27, and poseidon_aggregator arrays from base_trace
    m31** bt = (m31**)base_trace;  // HOST array of 342 DEVICE pointers

    // cube_252_0: input=trace[143..152], output=trace[153..162] (20 elements)
    m31* cube0_host[20];
    for (int i = 0; i < 10; i++) cube0_host[i] = bt[143 + i];
    for (int i = 0; i < 10; i++) cube0_host[10 + i] = bt[153 + i];
    m31** d_cube_0 = clone_to_device<m31*>(cube0_host, 20);

    // cube_252_1: input=trace[163..172], output=trace[174..183] (20 elements)
    m31* cube1_host[20];
    for (int i = 0; i < 10; i++) cube1_host[i] = bt[163 + i];
    for (int i = 0; i < 10; i++) cube1_host[10 + i] = bt[174 + i];
    m31** d_cube_1 = clone_to_device<m31*>(cube1_host, 20);

    // rc_252_width_27_0: trace[123..132] (10 elements)
    m31** d_rc252_0 = clone_to_device<m31*>(&bt[123], 10);

    // rc_252_width_27_1: trace[133..142] (10 elements)
    m31** d_rc252_1 = clone_to_device<m31*>(&bt[133], 10);

    // poseidon_aggregator_0: trace[0..5] (6 elements)
    m31** d_agg_0 = clone_to_device<m31*>(&bt[0], 6);

    // mults: trace col 341
    m31* mults = bt[341];

    // 4. Allocate working memory
    qm31* device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31* denom_inv = cuda_malloc<qm31>(trace_size);
    m31* numer0 = cuda_malloc<m31>(trace_size);
    m31* numer1 = cuda_malloc<m31>(trace_size);
    m31* numer2 = cuda_malloc<m31>(trace_size);
    m31* numer3 = cuda_malloc<m31>(trace_size);

    m31** device_it = clone_to_device<m31*>(
        (m31**)interaction_trace_columns, 4 * AGG_N_LOGUP_COLS);

    int block_dim = trace_size < AGG_BLOCK_SIZE ? trace_size : AGG_BLOCK_SIZE;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Relation ID constants (from constraints/relations.cuh)
    m31 relid_mem     = MEMORY_ID_TO_BIG_RELATION_ID;
    m31 relid_pfrc    = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
    m31 relid_rc252   = RANGE_CHECK_252_WIDTH_27_RELATION_ID;
    m31 relid_cube    = CUBE_252_RELATION_ID;
    m31 relid_rc33333 = RANGE_CHECK_3_3_3_3_3_RELATION_ID;
    m31 relid_rc4444  = RANGE_CHECK_4_4_4_4_RELATION_ID;
    m31 relid_rc44    = RANGE_CHECK_4_4_RELATION_ID;
    m31 relid_p3prc   = POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID;
    m31 relid_agg     = POSEIDON_AGGREGATOR_RELATION_ID;

    // 5. Process 14 logup columns
    // Col 0: mem_id_to_big_0 + mem_id_to_big_1 (ADD)
    POSAGG_IT_PROCESS_ADD(0, 29, 29, relid_mem, d_mem_0, relid_mem, d_mem_1);

    // Col 1: mem_id_to_big_2 + pfrc_0 (SUB d1-d0: numerator = pfrc - mem)
    POSAGG_IT_PROCESS_SUB_D1D0(1, 29, 32, relid_mem, d_mem_2, relid_pfrc, d_pfrc_0);

    // Col 2: pfrc_1 + rc_252w27_0 (ADD)
    POSAGG_IT_PROCESS_ADD(2, 32, 10, relid_pfrc, d_pfrc_1, relid_rc252, d_rc252_0);

    // Col 3: rc_252w27_1 + cube_252_0 (ADD)
    POSAGG_IT_PROCESS_ADD(3, 10, 20, relid_rc252, d_rc252_1, relid_cube, d_cube_0);

    // Col 4: rc_3_3_3_3_3_0 + rc_3_3_3_3_3_1 (ADD)
    POSAGG_IT_PROCESS_ADD(4, 5, 5, relid_rc33333, d_rc33333_0, relid_rc33333, d_rc33333_1);

    // Col 5: cube_252_1 + rc_4_4_4_4_0 (ADD)
    POSAGG_IT_PROCESS_ADD(5, 20, 4, relid_cube, d_cube_1, relid_rc4444, d_rc4444_0);

    // Col 6: rc_4_4_4_4_1 + rc_4_4_0 (ADD)
    POSAGG_IT_PROCESS_ADD(6, 4, 2, relid_rc4444, d_rc4444_1, relid_rc44, d_rc44_0);

    // Col 7: p3prc_0 + p3prc_1 (SUB d0-d1)
    POSAGG_IT_PROCESS_SUB_D0D1(7, 42, 42, relid_p3prc, d_p3prc_0, relid_p3prc, d_p3prc_1);

    // Col 8: rc_4_4_4_4_2 + rc_4_4_4_4_3 (ADD)
    POSAGG_IT_PROCESS_ADD(8, 4, 4, relid_rc4444, d_rc4444_2, relid_rc4444, d_rc4444_3);

    // Col 9: rc_4_4_1 + rc_4_4_4_4_4 (ADD)
    POSAGG_IT_PROCESS_ADD(9, 2, 4, relid_rc44, d_rc44_1, relid_rc4444, d_rc4444_4);

    // Col 10: rc_4_4_4_4_5 + rc_4_4_2 (ADD)
    POSAGG_IT_PROCESS_ADD(10, 4, 2, relid_rc4444, d_rc4444_5, relid_rc44, d_rc44_2);

    // Col 11: pfrc_2 + pfrc_3 (SUB d0-d1)
    POSAGG_IT_PROCESS_SUB_D0D1(11, 32, 32, relid_pfrc, d_pfrc_2, relid_pfrc, d_pfrc_3);

    // Col 12: mem_id_to_big_3 + mem_id_to_big_4 (ADD)
    POSAGG_IT_PROCESS_ADD(12, 29, 29, relid_mem, d_mem_3, relid_mem, d_mem_4);

    // Col 13: mem_id_to_big_5 + poseidon_aggregator_0 (MULT: d1 - d0 * mult)
    posagg_it_mult_kernel<29, 6><<<num_blocks, block_dim>>>(
        d_lookup, relid_mem, d_mem_5, relid_agg, d_agg_0, mults, trace_size,
        device_logup_denom, numer0, numer1, numer2, numer3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    posagg_it_finalize_kernel<<<num_blocks, block_dim>>>(13, trace_size, denom_inv,
        numer0, numer1, numer2, numer3, device_it);

    // 6. Finalization
    cudaMemsetAsync(claimed_sum, 0, 4 * sizeof(m31), 0);

    size_t shared_size = 4 * block_dim * sizeof(m31);
    posagg_it_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        AGG_N_LOGUP_COLS, trace_size, device_it, (m31*)claimed_sum);

    posagg_it_coord_prefix_sum<<<num_blocks, block_dim>>>(
        (m31*)claimed_sum, AGG_N_LOGUP_COLS, trace_size, device_it);

    // Inclusive prefix sum on last 4 interaction columns
    m31** it_cols = (m31**)interaction_trace_columns;
    inclusive_prefix_sum(it_cols[4 * AGG_N_LOGUP_COLS - 4], trace_size);
    inclusive_prefix_sum(it_cols[4 * AGG_N_LOGUP_COLS - 3], trace_size);
    inclusive_prefix_sum(it_cols[4 * AGG_N_LOGUP_COLS - 2], trace_size);
    inclusive_prefix_sum(it_cols[4 * AGG_N_LOGUP_COLS - 1], trace_size);

    // 7. Cleanup
    cuda_free_memory(d_lookup);
    cuda_free_memory(d_mem_0); cuda_free_memory(d_mem_1);
    cuda_free_memory(d_mem_2); cuda_free_memory(d_mem_3);
    cuda_free_memory(d_mem_4); cuda_free_memory(d_mem_5);
    cuda_free_memory(d_rc33333_0); cuda_free_memory(d_rc33333_1);
    cuda_free_memory(d_rc4444_0); cuda_free_memory(d_rc4444_1);
    cuda_free_memory(d_rc4444_2); cuda_free_memory(d_rc4444_3);
    cuda_free_memory(d_rc4444_4); cuda_free_memory(d_rc4444_5);
    cuda_free_memory(d_rc44_0); cuda_free_memory(d_rc44_1); cuda_free_memory(d_rc44_2);
    cuda_free_memory(d_pfrc_0); cuda_free_memory(d_pfrc_1);
    cuda_free_memory(d_pfrc_2); cuda_free_memory(d_pfrc_3);
    cuda_free_memory(d_p3prc_0); cuda_free_memory(d_p3prc_1);
    cuda_free_memory(d_cube_0); cuda_free_memory(d_cube_1);
    cuda_free_memory(d_rc252_0); cuda_free_memory(d_rc252_1);
    cuda_free_memory(d_agg_0);
    cuda_free_memory(device_logup_denom); cuda_free_memory(denom_inv);
    cuda_free_memory(numer0); cuda_free_memory(numer1);
    cuda_free_memory(numer2); cuda_free_memory(numer3);
    cuda_free_memory(device_it);
}
