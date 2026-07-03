use std::ffi::c_void;

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::vcs::blake2_hash::Blake2sHash;

#[link(name = "stwo_cuda")]
extern "C" {

    pub fn verify_bitwise_xor_4_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn verify_bitwise_xor_7_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn verify_bitwise_xor_8_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn verify_bitwise_xor_8_b_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn verify_bitwise_xor_9_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn verify_bitwise_xor_12_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const *const u32,
        mults_col_size: u32,
        mults_row_log_size: u32,
    );

    // Interaction trace generation for verify_bitwise_xor components
    pub fn verify_bitwise_xor_4_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn verify_bitwise_xor_7_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn verify_bitwise_xor_8_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn verify_bitwise_xor_8_paired_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities_0: *const u32,
        multiplicities_1: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    pub fn verify_bitwise_xor_8_b_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn verify_bitwise_xor_9_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn verify_bitwise_xor_12_interaction_trace(
        lookup_elements: *mut c_void,
        multiplicities: *const *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    pub fn generate_blake_g_traces(
        traces: *const *const u32,
        lookup_blake_g_0: *const *const u32,
        lookup_verify_bitwise_xor_12_0: *const *const u32,
        lookup_verify_bitwise_xor_12_1: *const *const u32,
        lookup_verify_bitwise_xor_4_0: *const *const u32,
        lookup_verify_bitwise_xor_4_1: *const *const u32,
        lookup_verify_bitwise_xor_7_0: *const *const u32,
        lookup_verify_bitwise_xor_7_1: *const *const u32,
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,
        lookup_verify_bitwise_xor_8_4: *const *const u32,
        lookup_verify_bitwise_xor_8_5: *const *const u32,
        lookup_verify_bitwise_xor_8_6: *const *const u32,
        lookup_verify_bitwise_xor_8_7: *const *const u32,
        lookup_verify_bitwise_xor_9_0: *const *const u32,
        lookup_verify_bitwise_xor_9_1: *const *const u32,

        sub_componet_input_verify_bitwise_xor_8: *const *const u32,
        sub_componet_input_verify_bitwise_xor_12: *const *const u32,
        sub_componet_input_verify_bitwise_xor_4: *const *const u32,
        sub_componet_input_verify_bitwise_xor_7: *const *const u32,
        sub_componet_input_verify_bitwise_xor_9: *const *const u32,

        blake_g_input: *const *const u32,

        trace_log_len: u32,
    );

    pub fn generate_blake_g_interaction_traces(
        blake_g: *mut c_void,
        verify_bitwise_xor_12: *mut c_void,
        verify_bitwise_xor_4: *mut c_void,
        verify_bitwise_xor_7: *mut c_void,
        verify_bitwise_xor_8: *mut c_void,
        verify_bitwise_xor_8_b: *mut c_void,
        verify_bitwise_xor_9: *mut c_void,

        lookup_blake_g_0: *const *const u32,
        lookup_verify_bitwise_xor_12_0: *const *const u32,
        lookup_verify_bitwise_xor_12_1: *const *const u32,
        lookup_verify_bitwise_xor_4_0: *const *const u32,
        lookup_verify_bitwise_xor_4_1: *const *const u32,
        lookup_verify_bitwise_xor_7_0: *const *const u32,
        lookup_verify_bitwise_xor_7_1: *const *const u32,
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,
        lookup_verify_bitwise_xor_8_4: *const *const u32,
        lookup_verify_bitwise_xor_8_5: *const *const u32,
        lookup_verify_bitwise_xor_8_6: *const *const u32,
        lookup_verify_bitwise_xor_8_7: *const *const u32,
        lookup_verify_bitwise_xor_9_0: *const *const u32,
        lookup_verify_bitwise_xor_9_1: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn generate_triple_xor_32_traces(
        traces: *const *const u32,
        lookup_triple_xor_32: *const *const u32,
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,
        lookup_verify_bitwise_xor_8_b_0: *const *const u32,
        lookup_verify_bitwise_xor_8_b_1: *const *const u32,
        lookup_verify_bitwise_xor_8_b_2: *const *const u32,
        lookup_verify_bitwise_xor_8_b_3: *const *const u32,

        sub_componet_input_verify_bitwise_xor_8: *const *const u32,
        sub_componet_input_verify_bitwise_xor_8_b: *const *const u32,

        triple_xor_32_input: *const *const u32,

        trace_log_len: u32,
    );

    pub fn generate_triple_xor_32_interaction_traces(
        triple_xor_32: *mut c_void,
        verify_bitwise_xor_8: *mut c_void,
        verify_bitwise_xor_8_b: *mut c_void,

        lookup_triple_xor_32: *const *const u32,
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,
        lookup_verify_bitwise_xor_8_b_0: *const *const u32,
        lookup_verify_bitwise_xor_8_b_1: *const *const u32,
        lookup_verify_bitwise_xor_8_b_2: *const *const u32,
        lookup_verify_bitwise_xor_8_b_3: *const *const u32,

        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    // Blake round trace generation
    pub fn generate_blake_round_traces(
        traces: *const *const u32,

        lookup_blake_g_0: *const *const u32,
        lookup_blake_g_1: *const *const u32,
        lookup_blake_g_2: *const *const u32,
        lookup_blake_g_3: *const *const u32,
        lookup_blake_g_4: *const *const u32,
        lookup_blake_g_5: *const *const u32,
        lookup_blake_g_6: *const *const u32,
        lookup_blake_g_7: *const *const u32,
        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,
        lookup_blake_round_sigma_0: *const *const u32,
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,

        sub_component_inputs_blake_round_sigma: *const *const u32,
        sub_component_inputs_range_check_7_2_5: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_blake_g: *const *const u32,

        blake_round_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transpose_big_value_ptr: *const *const u32,
        memory_id_to_big_small_value_ptr: *const u32,

        n_rows: u32,
        trace_log_len: u32,
    );

    pub fn generate_blake_round_interaction_traces(
        blake_g: *mut c_void,
        blake_round: *mut c_void,
        blake_round_sigma: *mut c_void,
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        range_check_7_2_5: *mut c_void,

        lookup_blake_g_0: *const *const u32,
        lookup_blake_g_1: *const *const u32,
        lookup_blake_g_2: *const *const u32,
        lookup_blake_g_3: *const *const u32,
        lookup_blake_g_4: *const *const u32,
        lookup_blake_g_5: *const *const u32,
        lookup_blake_g_6: *const *const u32,
        lookup_blake_g_7: *const *const u32,

        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,

        lookup_blake_round_sigma_0: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,

        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,

        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    pub fn blake_round_sigma_mults_init(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const *const u32,
        mults_cols_sizes: u32,
        mults_row_log_size: u32,
    );

    pub fn generate_blake_round_sigma_interaction_traces(
        blake_round_sigma: *mut c_void,

        lookup_blake_round_sigma: *const *const u32,

        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    // Memory address to ID functions
    pub fn memory_address_to_id_add_inputs(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn generate_memory_address_to_id_traces(
        traces: *const *const u32,
        interaction_traces: *const *const u32,
        address_to_raw_id: *const u32,
        multiplicities: *const u32,
        total_size_log: u32,
        log_size: u32,
        lookup_elements: *mut c_void,
    );

    // Memory address to ID interaction trace generation (full CUDA)
    pub fn memory_address_to_id_generate_interaction_trace(
        lookup_elements: *mut c_void,
        padded_ids: *const u32,
        padded_mults: *const u32,
        total_size_log: u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    // Memory ID to big functions
    pub fn memory_id_to_big_deduce_finese_cuda(
        transpose_big_value_ptr: *const *const u32,
        small_value_ptr: *const u32,
        id: u32,
        felt252_out: *const u32,
    );

    pub fn memory_id_to_big_add_inputs(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        big_mults: *const u32,
        big_mults_row_log_size: u32,
        small_mults: *const u32,
        small_mults_row_log_size: u32,
    );

    // Memory ID to big trace generation
    pub fn memory_id_to_big_generate_big_trace(
        transposed_big_values: *const *const u32,
        multiplicities: *const u32,
        trace_size: u32,
        n_values: u32,
        trace_columns: *const *const u32,
    );

    pub fn memory_id_to_big_generate_small_trace(
        small_values: *const u128,
        multiplicities: *const u32,
        trace_size: u32,
        n_values: u32,
        trace_columns: *const *const u32,
    );

    // Memory ID to big interaction trace generation
    pub fn memory_id_to_big_generate_big_interaction_trace(
        lookup_element_ptr: *mut std::os::raw::c_void,
        rc_9_9_ptr: *mut std::os::raw::c_void,
        rc_9_9_b_ptr: *mut std::os::raw::c_void,
        rc_9_9_c_ptr: *mut std::os::raw::c_void,
        rc_9_9_d_ptr: *mut std::os::raw::c_void,
        rc_9_9_e_ptr: *mut std::os::raw::c_void,
        rc_9_9_f_ptr: *mut std::os::raw::c_void,
        rc_9_9_g_ptr: *mut std::os::raw::c_void,
        rc_9_9_h_ptr: *mut std::os::raw::c_void,
        value_columns: *const *const u32,
        multiplicities: *const u32,
        trace_size: u32,
        id_offset: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    pub fn memory_id_to_big_generate_small_interaction_trace(
        lookup_element_ptr: *mut std::os::raw::c_void,
        rc_9_9_ptr: *mut std::os::raw::c_void,
        rc_9_9_b_ptr: *mut std::os::raw::c_void,
        rc_9_9_c_ptr: *mut std::os::raw::c_void,
        rc_9_9_d_ptr: *mut std::os::raw::c_void,
        value_columns: *const *const u32,
        multiplicities: *const u32,
        trace_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    // Add opcode trace generation
    pub fn generate_add_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // add_opcode_small functions
    pub fn generate_add_opcode_small_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_opcode_small_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // assert_eq_opcode functions
    pub fn generate_assert_eq_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_assert_eq_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // assert_eq_opcode_double_deref functions
    pub fn generate_assert_eq_opcode_double_deref_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,

        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_assert_eq_opcode_double_deref_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,

        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // assert_eq_opcode_imm functions
    pub fn generate_assert_eq_opcode_imm_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_assert_eq_opcode_imm_interaction_traces(
        memory_address_to_id: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // add_ap_opcode functions
    pub fn generate_add_ap_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_18_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        opcodes_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_ap_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        range_check_11: *mut c_void,
        range_check_18: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_18_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // Range check vector functions
    pub fn range_check_vector_add_inputs(
        inputs: *const *const u32,
        input_col_sizes: u32,
        input_row_sizes: u32,
        n_range: u32,
        ranges: *const u32,
        mults: *const u32,
        mults_row_log_size: u32,
    );

    pub fn partition_into_bit_segments_cuda(
        total_size: u32,
        n_range: u32,
        n_bits_per_segments: *const u32,
        output_value: *const *const u32,
    );

    pub fn range_check_vector_generate_interaction_trace(
        lookup_element_ptr: *mut std::os::raw::c_void,
        multiplicities: *const u32,
        n_range: u32,
        ranges: *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *mut u32,
    );

    /// Multi-relation range check interaction trace generation entirely on GPU.
    /// Processes n_pairs relation pairs, reusing the cumsum_shift + prefix_sum
    /// finalization from the single-relation kernel.
    pub fn range_check_multi_relation_interaction_trace(
        n_pairs: u32,
        n_range: u32,
        ranges: *const u32,
        lookup_elements: *const *mut std::os::raw::c_void, // 2*n_pairs
        multiplicities: *const *const u32,                 // 2*n_pairs device ptrs
        log_size: u32,
        interaction_trace_columns: *const *const u32, // 4*n_pairs output cols
        claimed_sum: *mut u32,                        // 4 m31s for qm31
    );

    // blake_compress_opcode functions
    pub fn generate_blake_compress_opcode_traces(
        traces: *const *const u32,

        // Lookup data - blake_round (2 lookups)
        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,

        // Lookup data - memory_address_to_id (20 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,

        // Lookup data - memory_id_to_big (20 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_7_2_5 (17 lookups)
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,
        lookup_range_check_7_2_5_16: *const *const u32,

        // Lookup data - triple_xor_32 (8 lookups)
        lookup_triple_xor_32_0: *const *const u32,
        lookup_triple_xor_32_1: *const *const u32,
        lookup_triple_xor_32_2: *const *const u32,
        lookup_triple_xor_32_3: *const *const u32,
        lookup_triple_xor_32_4: *const *const u32,
        lookup_triple_xor_32_5: *const *const u32,
        lookup_triple_xor_32_6: *const *const u32,
        lookup_triple_xor_32_7: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (4 lookups)
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_7_2_5: *const *const u32,
        sub_component_inputs_verify_bitwise_xor_8: *const *const u32,
        sub_component_inputs_blake_round: *const *const u32,
        sub_component_inputs_triple_xor_32: *const *const u32,

        // Opcode inputs
        blake_compress_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_blake_compress_opcode_interaction_traces(
        // Relations
        blake_round: *mut c_void,
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        range_check_7_2_5: *mut c_void,
        triple_xor_32: *mut c_void,
        verify_bitwise_xor_8: *mut c_void,
        verify_instruction: *mut c_void,

        // Lookup data - blake_round (2 lookups)
        lookup_blake_round_0: *const *const u32,
        lookup_blake_round_1: *const *const u32,

        // Lookup data - memory_address_to_id (20 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,

        // Lookup data - memory_id_to_big (20 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_7_2_5 (17 lookups)
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_7_2_5_1: *const *const u32,
        lookup_range_check_7_2_5_2: *const *const u32,
        lookup_range_check_7_2_5_3: *const *const u32,
        lookup_range_check_7_2_5_4: *const *const u32,
        lookup_range_check_7_2_5_5: *const *const u32,
        lookup_range_check_7_2_5_6: *const *const u32,
        lookup_range_check_7_2_5_7: *const *const u32,
        lookup_range_check_7_2_5_8: *const *const u32,
        lookup_range_check_7_2_5_9: *const *const u32,
        lookup_range_check_7_2_5_10: *const *const u32,
        lookup_range_check_7_2_5_11: *const *const u32,
        lookup_range_check_7_2_5_12: *const *const u32,
        lookup_range_check_7_2_5_13: *const *const u32,
        lookup_range_check_7_2_5_14: *const *const u32,
        lookup_range_check_7_2_5_15: *const *const u32,
        lookup_range_check_7_2_5_16: *const *const u32,

        // Lookup data - triple_xor_32 (8 lookups)
        lookup_triple_xor_32_0: *const *const u32,
        lookup_triple_xor_32_1: *const *const u32,
        lookup_triple_xor_32_2: *const *const u32,
        lookup_triple_xor_32_3: *const *const u32,
        lookup_triple_xor_32_4: *const *const u32,
        lookup_triple_xor_32_5: *const *const u32,
        lookup_triple_xor_32_6: *const *const u32,
        lookup_triple_xor_32_7: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (4 lookups)
        lookup_verify_bitwise_xor_8_0: *const *const u32,
        lookup_verify_bitwise_xor_8_1: *const *const u32,
        lookup_verify_bitwise_xor_8_2: *const *const u32,
        lookup_verify_bitwise_xor_8_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // call_opcode functions
    pub fn generate_call_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        call_opcode_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_call_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // call_opcode_rel_imm functions
    pub fn generate_call_opcode_rel_imm_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        call_opcode_rel_imm_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_call_opcode_rel_imm_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // generic_opcode functions (244 trace columns, 34 interaction columns, 67 lookups)
    pub fn generate_generic_opcode_traces(
        traces: *const *const u32,

        // memory_address_to_id (3 lookups × 2 fields)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // memory_id_to_big (3 lookups × 29 fields)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // opcodes (2 lookups × 3 fields)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // range_check_9_9 (4 lookups × 2 fields)
        lookup_range_check_9_9_0: *const *const u32,
        lookup_range_check_9_9_1: *const *const u32,
        lookup_range_check_9_9_2: *const *const u32,
        lookup_range_check_9_9_3: *const *const u32,

        // range_check_9_9_b (4 lookups × 2 fields)
        lookup_range_check_9_9_b_0: *const *const u32,
        lookup_range_check_9_9_b_1: *const *const u32,
        lookup_range_check_9_9_b_2: *const *const u32,
        lookup_range_check_9_9_b_3: *const *const u32,

        // range_check_9_9_c (4 lookups × 2 fields)
        lookup_range_check_9_9_c_0: *const *const u32,
        lookup_range_check_9_9_c_1: *const *const u32,
        lookup_range_check_9_9_c_2: *const *const u32,
        lookup_range_check_9_9_c_3: *const *const u32,

        // range_check_9_9_d (4 lookups × 2 fields)
        lookup_range_check_9_9_d_0: *const *const u32,
        lookup_range_check_9_9_d_1: *const *const u32,
        lookup_range_check_9_9_d_2: *const *const u32,
        lookup_range_check_9_9_d_3: *const *const u32,

        // range_check_9_9_e (4 lookups × 2 fields)
        lookup_range_check_9_9_e_0: *const *const u32,
        lookup_range_check_9_9_e_1: *const *const u32,
        lookup_range_check_9_9_e_2: *const *const u32,
        lookup_range_check_9_9_e_3: *const *const u32,

        // range_check_9_9_f (4 lookups × 2 fields)
        lookup_range_check_9_9_f_0: *const *const u32,
        lookup_range_check_9_9_f_1: *const *const u32,
        lookup_range_check_9_9_f_2: *const *const u32,
        lookup_range_check_9_9_f_3: *const *const u32,

        // range_check_9_9_g (2 lookups × 2 fields)
        lookup_range_check_9_9_g_0: *const *const u32,
        lookup_range_check_9_9_g_1: *const *const u32,

        // range_check_9_9_h (2 lookups × 2 fields)
        lookup_range_check_9_9_h_0: *const *const u32,
        lookup_range_check_9_9_h_1: *const *const u32,

        // range_check_19 (4 lookups × 1 field)
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,

        // range_check_19_b (4 lookups × 1 field)
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,

        // range_check_19_c (4 lookups × 1 field)
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,

        // range_check_19_d (3 lookups × 1 field)
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,

        // range_check_19_e (3 lookups × 1 field)
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,

        // range_check_19_f (3 lookups × 1 field)
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,

        // range_check_19_g (3 lookups × 1 field)
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,

        // range_check_19_h (4 lookups × 1 field)
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,

        // range_check_18 (1 lookup × 1 field)
        lookup_range_check_18_0: *const *const u32,

        // range_check_11 (1 lookup × 1 field)
        lookup_range_check_11_0: *const *const u32,

        // verify_instruction (1 lookup × 7 fields)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Opcode inputs
        generic_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_generic_opcode_interaction_traces(
        // Relation pointers (22 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        range_check_9_9: *mut c_void,
        range_check_9_9_b: *mut c_void,
        range_check_9_9_c: *mut c_void,
        range_check_9_9_d: *mut c_void,
        range_check_9_9_e: *mut c_void,
        range_check_9_9_f: *mut c_void,
        range_check_9_9_g: *mut c_void,
        range_check_9_9_h: *mut c_void,
        range_check_19: *mut c_void,
        range_check_19_b: *mut c_void,
        range_check_19_c: *mut c_void,
        range_check_19_d: *mut c_void,
        range_check_19_e: *mut c_void,
        range_check_19_f: *mut c_void,
        range_check_19_g: *mut c_void,
        range_check_19_h: *mut c_void,
        range_check_18: *mut c_void,
        range_check_11: *mut c_void,
        verify_instruction: *mut c_void,

        // All lookup data (67 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_range_check_9_9_0: *const *const u32,
        lookup_range_check_9_9_1: *const *const u32,
        lookup_range_check_9_9_2: *const *const u32,
        lookup_range_check_9_9_3: *const *const u32,
        lookup_range_check_9_9_b_0: *const *const u32,
        lookup_range_check_9_9_b_1: *const *const u32,
        lookup_range_check_9_9_b_2: *const *const u32,
        lookup_range_check_9_9_b_3: *const *const u32,
        lookup_range_check_9_9_c_0: *const *const u32,
        lookup_range_check_9_9_c_1: *const *const u32,
        lookup_range_check_9_9_c_2: *const *const u32,
        lookup_range_check_9_9_c_3: *const *const u32,
        lookup_range_check_9_9_d_0: *const *const u32,
        lookup_range_check_9_9_d_1: *const *const u32,
        lookup_range_check_9_9_d_2: *const *const u32,
        lookup_range_check_9_9_d_3: *const *const u32,
        lookup_range_check_9_9_e_0: *const *const u32,
        lookup_range_check_9_9_e_1: *const *const u32,
        lookup_range_check_9_9_e_2: *const *const u32,
        lookup_range_check_9_9_e_3: *const *const u32,
        lookup_range_check_9_9_f_0: *const *const u32,
        lookup_range_check_9_9_f_1: *const *const u32,
        lookup_range_check_9_9_f_2: *const *const u32,
        lookup_range_check_9_9_f_3: *const *const u32,
        lookup_range_check_9_9_g_0: *const *const u32,
        lookup_range_check_9_9_g_1: *const *const u32,
        lookup_range_check_9_9_h_0: *const *const u32,
        lookup_range_check_9_9_h_1: *const *const u32,
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,
        lookup_range_check_18_0: *const *const u32,
        lookup_range_check_11_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jnz_opcode functions
    pub fn generate_jnz_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jnz_opcode_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jnz_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jnz_opcode_taken functions
    pub fn generate_jnz_opcode_taken_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jnz_opcode_taken_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jnz_opcode_taken_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode functions
    pub fn generate_jump_opcode_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode_double_deref functions
    pub fn generate_jump_opcode_double_deref_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_double_deref_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_double_deref_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode_rel functions
    pub fn generate_jump_opcode_rel_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_rel_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_rel_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // jump_opcode_rel_imm functions
    pub fn generate_jump_opcode_rel_imm_traces(
        traces: *const *const u32,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        jump_opcode_rel_imm_input: *const *const u32,

        memory_address_to_id_address_to_raw_id: *const u32,

        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_jump_opcode_rel_imm_interaction_traces(
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // mul_opcode functions (130 trace columns, 19 interaction columns)
    pub fn generate_mul_opcode_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_19 (4 lookups)
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,

        // Lookup data - range_check_19_b (4 lookups)
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,

        // Lookup data - range_check_19_c (4 lookups)
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,

        // Lookup data - range_check_19_d (3 lookups)
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,

        // Lookup data - range_check_19_e (3 lookups)
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,

        // Lookup data - range_check_19_f (3 lookups)
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,

        // Lookup data - range_check_19_g (3 lookups)
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,

        // Lookup data - range_check_19_h (4 lookups)
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_19: *const *const u32,
        sub_component_inputs_range_check_19_b: *const *const u32,
        sub_component_inputs_range_check_19_c: *const *const u32,
        sub_component_inputs_range_check_19_d: *const *const u32,
        sub_component_inputs_range_check_19_e: *const *const u32,
        sub_component_inputs_range_check_19_f: *const *const u32,
        sub_component_inputs_range_check_19_g: *const *const u32,
        sub_component_inputs_range_check_19_h: *const *const u32,

        // Opcode inputs
        mul_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_mul_opcode_interaction_traces(
        // Relation pointers (12 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,
        range_check_19: *mut c_void,
        range_check_19_b: *mut c_void,
        range_check_19_c: *mut c_void,
        range_check_19_d: *mut c_void,
        range_check_19_e: *mut c_void,
        range_check_19_f: *mut c_void,
        range_check_19_g: *mut c_void,
        range_check_19_h: *mut c_void,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_19 (4 lookups)
        lookup_range_check_19_0: *const *const u32,
        lookup_range_check_19_1: *const *const u32,
        lookup_range_check_19_2: *const *const u32,
        lookup_range_check_19_3: *const *const u32,

        // Lookup data - range_check_19_b (4 lookups)
        lookup_range_check_19_b_0: *const *const u32,
        lookup_range_check_19_b_1: *const *const u32,
        lookup_range_check_19_b_2: *const *const u32,
        lookup_range_check_19_b_3: *const *const u32,

        // Lookup data - range_check_19_c (4 lookups)
        lookup_range_check_19_c_0: *const *const u32,
        lookup_range_check_19_c_1: *const *const u32,
        lookup_range_check_19_c_2: *const *const u32,
        lookup_range_check_19_c_3: *const *const u32,

        // Lookup data - range_check_19_d (3 lookups)
        lookup_range_check_19_d_0: *const *const u32,
        lookup_range_check_19_d_1: *const *const u32,
        lookup_range_check_19_d_2: *const *const u32,

        // Lookup data - range_check_19_e (3 lookups)
        lookup_range_check_19_e_0: *const *const u32,
        lookup_range_check_19_e_1: *const *const u32,
        lookup_range_check_19_e_2: *const *const u32,

        // Lookup data - range_check_19_f (3 lookups)
        lookup_range_check_19_f_0: *const *const u32,
        lookup_range_check_19_f_1: *const *const u32,
        lookup_range_check_19_f_2: *const *const u32,

        // Lookup data - range_check_19_g (3 lookups)
        lookup_range_check_19_g_0: *const *const u32,
        lookup_range_check_19_g_1: *const *const u32,
        lookup_range_check_19_g_2: *const *const u32,

        // Lookup data - range_check_19_h (4 lookups)
        lookup_range_check_19_h_0: *const *const u32,
        lookup_range_check_19_h_1: *const *const u32,
        lookup_range_check_19_h_2: *const *const u32,
        lookup_range_check_19_h_3: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // mul_opcode_small functions (37 trace columns, 6 interaction columns)
    pub fn generate_mul_opcode_small_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_11 (3 lookups)
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_11_1: *const *const u32,
        lookup_range_check_11_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_11: *const *const u32,

        // Opcode inputs
        mul_opcode_small_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_mul_opcode_small_interaction_traces(
        // Relation pointers (5 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,
        range_check_11: *mut c_void,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_11 (3 lookups)
        lookup_range_check_11_0: *const *const u32,
        lookup_range_check_11_1: *const *const u32,
        lookup_range_check_11_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === QM_31_ADD_MUL_OPCODE ===
    pub fn generate_qm_31_add_mul_opcode_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (3 lookups)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_4_4_4_4: *const *const u32,

        // Opcode inputs
        qm_31_add_mul_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_qm_31_add_mul_opcode_interaction_traces(
        // Relation pointers (5 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,
        range_check_4_4_4_4: *mut c_void,

        // Lookup data - memory_address_to_id (3 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,

        // Lookup data - memory_id_to_big (3 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (3 lookups)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === RET_OPCODE ===
    pub fn generate_ret_opcode_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (2 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,

        // Lookup data - memory_id_to_big (2 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_verify_instruction: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Opcode inputs
        ret_opcode_input: *const *const u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_ret_opcode_interaction_traces(
        // Relation pointers (4 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        opcodes: *mut c_void,
        verify_instruction: *mut c_void,

        // Lookup data - memory_address_to_id (2 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,

        // Lookup data - memory_id_to_big (2 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,

        // Lookup data - opcodes (2 lookups)
        lookup_opcodes_0: *const *const u32,
        lookup_opcodes_1: *const *const u32,

        // Lookup data - verify_instruction (1 lookup)
        lookup_verify_instruction_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === ADD_MOD_BUILTIN ===
    pub fn generate_add_mod_builtin_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_add_mod_builtin_interaction_traces(
        // Relation pointers (2 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === BITWISE_BUILTIN ===
    pub fn generate_bitwise_builtin_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (5 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,

        // Lookup data - memory_id_to_big (5 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,

        // Lookup data - verify_bitwise_xor_9 (27 lookups)
        lookup_verify_bitwise_xor_9_0: *const *const u32,
        lookup_verify_bitwise_xor_9_1: *const *const u32,
        lookup_verify_bitwise_xor_9_2: *const *const u32,
        lookup_verify_bitwise_xor_9_3: *const *const u32,
        lookup_verify_bitwise_xor_9_4: *const *const u32,
        lookup_verify_bitwise_xor_9_5: *const *const u32,
        lookup_verify_bitwise_xor_9_6: *const *const u32,
        lookup_verify_bitwise_xor_9_7: *const *const u32,
        lookup_verify_bitwise_xor_9_8: *const *const u32,
        lookup_verify_bitwise_xor_9_9: *const *const u32,
        lookup_verify_bitwise_xor_9_10: *const *const u32,
        lookup_verify_bitwise_xor_9_11: *const *const u32,
        lookup_verify_bitwise_xor_9_12: *const *const u32,
        lookup_verify_bitwise_xor_9_13: *const *const u32,
        lookup_verify_bitwise_xor_9_14: *const *const u32,
        lookup_verify_bitwise_xor_9_15: *const *const u32,
        lookup_verify_bitwise_xor_9_16: *const *const u32,
        lookup_verify_bitwise_xor_9_17: *const *const u32,
        lookup_verify_bitwise_xor_9_18: *const *const u32,
        lookup_verify_bitwise_xor_9_19: *const *const u32,
        lookup_verify_bitwise_xor_9_20: *const *const u32,
        lookup_verify_bitwise_xor_9_21: *const *const u32,
        lookup_verify_bitwise_xor_9_22: *const *const u32,
        lookup_verify_bitwise_xor_9_23: *const *const u32,
        lookup_verify_bitwise_xor_9_24: *const *const u32,
        lookup_verify_bitwise_xor_9_25: *const *const u32,
        lookup_verify_bitwise_xor_9_26: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (1 lookup)
        lookup_verify_bitwise_xor_8_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_verify_bitwise_xor_9: *const *const u32,
        sub_component_inputs_verify_bitwise_xor_8: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_bitwise_builtin_interaction_traces(
        // Relation pointers (4 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        verify_bitwise_xor_9: *mut c_void,
        verify_bitwise_xor_8: *mut c_void,

        // Lookup data - memory_address_to_id (5 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,

        // Lookup data - memory_id_to_big (5 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,

        // Lookup data - verify_bitwise_xor_9 (27 lookups)
        lookup_verify_bitwise_xor_9_0: *const *const u32,
        lookup_verify_bitwise_xor_9_1: *const *const u32,
        lookup_verify_bitwise_xor_9_2: *const *const u32,
        lookup_verify_bitwise_xor_9_3: *const *const u32,
        lookup_verify_bitwise_xor_9_4: *const *const u32,
        lookup_verify_bitwise_xor_9_5: *const *const u32,
        lookup_verify_bitwise_xor_9_6: *const *const u32,
        lookup_verify_bitwise_xor_9_7: *const *const u32,
        lookup_verify_bitwise_xor_9_8: *const *const u32,
        lookup_verify_bitwise_xor_9_9: *const *const u32,
        lookup_verify_bitwise_xor_9_10: *const *const u32,
        lookup_verify_bitwise_xor_9_11: *const *const u32,
        lookup_verify_bitwise_xor_9_12: *const *const u32,
        lookup_verify_bitwise_xor_9_13: *const *const u32,
        lookup_verify_bitwise_xor_9_14: *const *const u32,
        lookup_verify_bitwise_xor_9_15: *const *const u32,
        lookup_verify_bitwise_xor_9_16: *const *const u32,
        lookup_verify_bitwise_xor_9_17: *const *const u32,
        lookup_verify_bitwise_xor_9_18: *const *const u32,
        lookup_verify_bitwise_xor_9_19: *const *const u32,
        lookup_verify_bitwise_xor_9_20: *const *const u32,
        lookup_verify_bitwise_xor_9_21: *const *const u32,
        lookup_verify_bitwise_xor_9_22: *const *const u32,
        lookup_verify_bitwise_xor_9_23: *const *const u32,
        lookup_verify_bitwise_xor_9_24: *const *const u32,
        lookup_verify_bitwise_xor_9_25: *const *const u32,
        lookup_verify_bitwise_xor_9_26: *const *const u32,

        // Lookup data - verify_bitwise_xor_8 (1 lookup)
        lookup_verify_bitwise_xor_8_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === MUL_MOD_BUILTIN ===
    pub fn generate_mul_mod_builtin_traces(
        traces: *const *const u32,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        // Lookup data - range_check_12 (32 lookups)
        lookup_range_check_12_0: *const *const u32,
        lookup_range_check_12_1: *const *const u32,
        lookup_range_check_12_2: *const *const u32,
        lookup_range_check_12_3: *const *const u32,
        lookup_range_check_12_4: *const *const u32,
        lookup_range_check_12_5: *const *const u32,
        lookup_range_check_12_6: *const *const u32,
        lookup_range_check_12_7: *const *const u32,
        lookup_range_check_12_8: *const *const u32,
        lookup_range_check_12_9: *const *const u32,
        lookup_range_check_12_10: *const *const u32,
        lookup_range_check_12_11: *const *const u32,
        lookup_range_check_12_12: *const *const u32,
        lookup_range_check_12_13: *const *const u32,
        lookup_range_check_12_14: *const *const u32,
        lookup_range_check_12_15: *const *const u32,
        lookup_range_check_12_16: *const *const u32,
        lookup_range_check_12_17: *const *const u32,
        lookup_range_check_12_18: *const *const u32,
        lookup_range_check_12_19: *const *const u32,
        lookup_range_check_12_20: *const *const u32,
        lookup_range_check_12_21: *const *const u32,
        lookup_range_check_12_22: *const *const u32,
        lookup_range_check_12_23: *const *const u32,
        lookup_range_check_12_24: *const *const u32,
        lookup_range_check_12_25: *const *const u32,
        lookup_range_check_12_26: *const *const u32,
        lookup_range_check_12_27: *const *const u32,
        lookup_range_check_12_28: *const *const u32,
        lookup_range_check_12_29: *const *const u32,
        lookup_range_check_12_30: *const *const u32,
        lookup_range_check_12_31: *const *const u32,

        // Lookup data - range_check_18 (62 lookups)
        lookup_range_check_18_0: *const *const u32,
        lookup_range_check_18_1: *const *const u32,
        lookup_range_check_18_2: *const *const u32,
        lookup_range_check_18_3: *const *const u32,
        lookup_range_check_18_4: *const *const u32,
        lookup_range_check_18_5: *const *const u32,
        lookup_range_check_18_6: *const *const u32,
        lookup_range_check_18_7: *const *const u32,
        lookup_range_check_18_8: *const *const u32,
        lookup_range_check_18_9: *const *const u32,
        lookup_range_check_18_10: *const *const u32,
        lookup_range_check_18_11: *const *const u32,
        lookup_range_check_18_12: *const *const u32,
        lookup_range_check_18_13: *const *const u32,
        lookup_range_check_18_14: *const *const u32,
        lookup_range_check_18_15: *const *const u32,
        lookup_range_check_18_16: *const *const u32,
        lookup_range_check_18_17: *const *const u32,
        lookup_range_check_18_18: *const *const u32,
        lookup_range_check_18_19: *const *const u32,
        lookup_range_check_18_20: *const *const u32,
        lookup_range_check_18_21: *const *const u32,
        lookup_range_check_18_22: *const *const u32,
        lookup_range_check_18_23: *const *const u32,
        lookup_range_check_18_24: *const *const u32,
        lookup_range_check_18_25: *const *const u32,
        lookup_range_check_18_26: *const *const u32,
        lookup_range_check_18_27: *const *const u32,
        lookup_range_check_18_28: *const *const u32,
        lookup_range_check_18_29: *const *const u32,
        lookup_range_check_18_30: *const *const u32,
        lookup_range_check_18_31: *const *const u32,
        lookup_range_check_18_32: *const *const u32,
        lookup_range_check_18_33: *const *const u32,
        lookup_range_check_18_34: *const *const u32,
        lookup_range_check_18_35: *const *const u32,
        lookup_range_check_18_36: *const *const u32,
        lookup_range_check_18_37: *const *const u32,
        lookup_range_check_18_38: *const *const u32,
        lookup_range_check_18_39: *const *const u32,
        lookup_range_check_18_40: *const *const u32,
        lookup_range_check_18_41: *const *const u32,
        lookup_range_check_18_42: *const *const u32,
        lookup_range_check_18_43: *const *const u32,
        lookup_range_check_18_44: *const *const u32,
        lookup_range_check_18_45: *const *const u32,
        lookup_range_check_18_46: *const *const u32,
        lookup_range_check_18_47: *const *const u32,
        lookup_range_check_18_48: *const *const u32,
        lookup_range_check_18_49: *const *const u32,
        lookup_range_check_18_50: *const *const u32,
        lookup_range_check_18_51: *const *const u32,
        lookup_range_check_18_52: *const *const u32,
        lookup_range_check_18_53: *const *const u32,
        lookup_range_check_18_54: *const *const u32,
        lookup_range_check_18_55: *const *const u32,
        lookup_range_check_18_56: *const *const u32,
        lookup_range_check_18_57: *const *const u32,
        lookup_range_check_18_58: *const *const u32,
        lookup_range_check_18_59: *const *const u32,
        lookup_range_check_18_60: *const *const u32,
        lookup_range_check_18_61: *const *const u32,

        // Lookup data - range_check_3_6_6_3 (40 lookups)
        lookup_range_check_3_6_6_3_0: *const *const u32,
        lookup_range_check_3_6_6_3_1: *const *const u32,
        lookup_range_check_3_6_6_3_2: *const *const u32,
        lookup_range_check_3_6_6_3_3: *const *const u32,
        lookup_range_check_3_6_6_3_4: *const *const u32,
        lookup_range_check_3_6_6_3_5: *const *const u32,
        lookup_range_check_3_6_6_3_6: *const *const u32,
        lookup_range_check_3_6_6_3_7: *const *const u32,
        lookup_range_check_3_6_6_3_8: *const *const u32,
        lookup_range_check_3_6_6_3_9: *const *const u32,
        lookup_range_check_3_6_6_3_10: *const *const u32,
        lookup_range_check_3_6_6_3_11: *const *const u32,
        lookup_range_check_3_6_6_3_12: *const *const u32,
        lookup_range_check_3_6_6_3_13: *const *const u32,
        lookup_range_check_3_6_6_3_14: *const *const u32,
        lookup_range_check_3_6_6_3_15: *const *const u32,
        lookup_range_check_3_6_6_3_16: *const *const u32,
        lookup_range_check_3_6_6_3_17: *const *const u32,
        lookup_range_check_3_6_6_3_18: *const *const u32,
        lookup_range_check_3_6_6_3_19: *const *const u32,
        lookup_range_check_3_6_6_3_20: *const *const u32,
        lookup_range_check_3_6_6_3_21: *const *const u32,
        lookup_range_check_3_6_6_3_22: *const *const u32,
        lookup_range_check_3_6_6_3_23: *const *const u32,
        lookup_range_check_3_6_6_3_24: *const *const u32,
        lookup_range_check_3_6_6_3_25: *const *const u32,
        lookup_range_check_3_6_6_3_26: *const *const u32,
        lookup_range_check_3_6_6_3_27: *const *const u32,
        lookup_range_check_3_6_6_3_28: *const *const u32,
        lookup_range_check_3_6_6_3_29: *const *const u32,
        lookup_range_check_3_6_6_3_30: *const *const u32,
        lookup_range_check_3_6_6_3_31: *const *const u32,
        lookup_range_check_3_6_6_3_32: *const *const u32,
        lookup_range_check_3_6_6_3_33: *const *const u32,
        lookup_range_check_3_6_6_3_34: *const *const u32,
        lookup_range_check_3_6_6_3_35: *const *const u32,
        lookup_range_check_3_6_6_3_36: *const *const u32,
        lookup_range_check_3_6_6_3_37: *const *const u32,
        lookup_range_check_3_6_6_3_38: *const *const u32,
        lookup_range_check_3_6_6_3_39: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_12: *const *const u32,
        sub_component_inputs_range_check_18: *const *const u32,
        sub_component_inputs_range_check_3_6_6_3: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_mul_mod_builtin_interaction_traces(
        // Relation pointers (5 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        range_check_12: *mut c_void,
        range_check_18: *mut c_void,
        range_check_3_6_6_3: *mut c_void,

        // Lookup data - memory_address_to_id (29 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,
        lookup_memory_address_to_id_6: *const *const u32,
        lookup_memory_address_to_id_7: *const *const u32,
        lookup_memory_address_to_id_8: *const *const u32,
        lookup_memory_address_to_id_9: *const *const u32,
        lookup_memory_address_to_id_10: *const *const u32,
        lookup_memory_address_to_id_11: *const *const u32,
        lookup_memory_address_to_id_12: *const *const u32,
        lookup_memory_address_to_id_13: *const *const u32,
        lookup_memory_address_to_id_14: *const *const u32,
        lookup_memory_address_to_id_15: *const *const u32,
        lookup_memory_address_to_id_16: *const *const u32,
        lookup_memory_address_to_id_17: *const *const u32,
        lookup_memory_address_to_id_18: *const *const u32,
        lookup_memory_address_to_id_19: *const *const u32,
        lookup_memory_address_to_id_20: *const *const u32,
        lookup_memory_address_to_id_21: *const *const u32,
        lookup_memory_address_to_id_22: *const *const u32,
        lookup_memory_address_to_id_23: *const *const u32,
        lookup_memory_address_to_id_24: *const *const u32,
        lookup_memory_address_to_id_25: *const *const u32,
        lookup_memory_address_to_id_26: *const *const u32,
        lookup_memory_address_to_id_27: *const *const u32,
        lookup_memory_address_to_id_28: *const *const u32,

        // Lookup data - memory_id_to_big (24 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,
        lookup_memory_id_to_big_6: *const *const u32,
        lookup_memory_id_to_big_7: *const *const u32,
        lookup_memory_id_to_big_8: *const *const u32,
        lookup_memory_id_to_big_9: *const *const u32,
        lookup_memory_id_to_big_10: *const *const u32,
        lookup_memory_id_to_big_11: *const *const u32,
        lookup_memory_id_to_big_12: *const *const u32,
        lookup_memory_id_to_big_13: *const *const u32,
        lookup_memory_id_to_big_14: *const *const u32,
        lookup_memory_id_to_big_15: *const *const u32,
        lookup_memory_id_to_big_16: *const *const u32,
        lookup_memory_id_to_big_17: *const *const u32,
        lookup_memory_id_to_big_18: *const *const u32,
        lookup_memory_id_to_big_19: *const *const u32,
        lookup_memory_id_to_big_20: *const *const u32,
        lookup_memory_id_to_big_21: *const *const u32,
        lookup_memory_id_to_big_22: *const *const u32,
        lookup_memory_id_to_big_23: *const *const u32,

        // Lookup data - range_check_12 (32 lookups)
        lookup_range_check_12_0: *const *const u32,
        lookup_range_check_12_1: *const *const u32,
        lookup_range_check_12_2: *const *const u32,
        lookup_range_check_12_3: *const *const u32,
        lookup_range_check_12_4: *const *const u32,
        lookup_range_check_12_5: *const *const u32,
        lookup_range_check_12_6: *const *const u32,
        lookup_range_check_12_7: *const *const u32,
        lookup_range_check_12_8: *const *const u32,
        lookup_range_check_12_9: *const *const u32,
        lookup_range_check_12_10: *const *const u32,
        lookup_range_check_12_11: *const *const u32,
        lookup_range_check_12_12: *const *const u32,
        lookup_range_check_12_13: *const *const u32,
        lookup_range_check_12_14: *const *const u32,
        lookup_range_check_12_15: *const *const u32,
        lookup_range_check_12_16: *const *const u32,
        lookup_range_check_12_17: *const *const u32,
        lookup_range_check_12_18: *const *const u32,
        lookup_range_check_12_19: *const *const u32,
        lookup_range_check_12_20: *const *const u32,
        lookup_range_check_12_21: *const *const u32,
        lookup_range_check_12_22: *const *const u32,
        lookup_range_check_12_23: *const *const u32,
        lookup_range_check_12_24: *const *const u32,
        lookup_range_check_12_25: *const *const u32,
        lookup_range_check_12_26: *const *const u32,
        lookup_range_check_12_27: *const *const u32,
        lookup_range_check_12_28: *const *const u32,
        lookup_range_check_12_29: *const *const u32,
        lookup_range_check_12_30: *const *const u32,
        lookup_range_check_12_31: *const *const u32,

        // Lookup data - range_check_18 (62 lookups)
        lookup_range_check_18_0: *const *const u32,
        lookup_range_check_18_1: *const *const u32,
        lookup_range_check_18_2: *const *const u32,
        lookup_range_check_18_3: *const *const u32,
        lookup_range_check_18_4: *const *const u32,
        lookup_range_check_18_5: *const *const u32,
        lookup_range_check_18_6: *const *const u32,
        lookup_range_check_18_7: *const *const u32,
        lookup_range_check_18_8: *const *const u32,
        lookup_range_check_18_9: *const *const u32,
        lookup_range_check_18_10: *const *const u32,
        lookup_range_check_18_11: *const *const u32,
        lookup_range_check_18_12: *const *const u32,
        lookup_range_check_18_13: *const *const u32,
        lookup_range_check_18_14: *const *const u32,
        lookup_range_check_18_15: *const *const u32,
        lookup_range_check_18_16: *const *const u32,
        lookup_range_check_18_17: *const *const u32,
        lookup_range_check_18_18: *const *const u32,
        lookup_range_check_18_19: *const *const u32,
        lookup_range_check_18_20: *const *const u32,
        lookup_range_check_18_21: *const *const u32,
        lookup_range_check_18_22: *const *const u32,
        lookup_range_check_18_23: *const *const u32,
        lookup_range_check_18_24: *const *const u32,
        lookup_range_check_18_25: *const *const u32,
        lookup_range_check_18_26: *const *const u32,
        lookup_range_check_18_27: *const *const u32,
        lookup_range_check_18_28: *const *const u32,
        lookup_range_check_18_29: *const *const u32,
        lookup_range_check_18_30: *const *const u32,
        lookup_range_check_18_31: *const *const u32,
        lookup_range_check_18_32: *const *const u32,
        lookup_range_check_18_33: *const *const u32,
        lookup_range_check_18_34: *const *const u32,
        lookup_range_check_18_35: *const *const u32,
        lookup_range_check_18_36: *const *const u32,
        lookup_range_check_18_37: *const *const u32,
        lookup_range_check_18_38: *const *const u32,
        lookup_range_check_18_39: *const *const u32,
        lookup_range_check_18_40: *const *const u32,
        lookup_range_check_18_41: *const *const u32,
        lookup_range_check_18_42: *const *const u32,
        lookup_range_check_18_43: *const *const u32,
        lookup_range_check_18_44: *const *const u32,
        lookup_range_check_18_45: *const *const u32,
        lookup_range_check_18_46: *const *const u32,
        lookup_range_check_18_47: *const *const u32,
        lookup_range_check_18_48: *const *const u32,
        lookup_range_check_18_49: *const *const u32,
        lookup_range_check_18_50: *const *const u32,
        lookup_range_check_18_51: *const *const u32,
        lookup_range_check_18_52: *const *const u32,
        lookup_range_check_18_53: *const *const u32,
        lookup_range_check_18_54: *const *const u32,
        lookup_range_check_18_55: *const *const u32,
        lookup_range_check_18_56: *const *const u32,
        lookup_range_check_18_57: *const *const u32,
        lookup_range_check_18_58: *const *const u32,
        lookup_range_check_18_59: *const *const u32,
        lookup_range_check_18_60: *const *const u32,
        lookup_range_check_18_61: *const *const u32,

        // Lookup data - range_check_3_6_6_3 (40 lookups)
        lookup_range_check_3_6_6_3_0: *const *const u32,
        lookup_range_check_3_6_6_3_1: *const *const u32,
        lookup_range_check_3_6_6_3_2: *const *const u32,
        lookup_range_check_3_6_6_3_3: *const *const u32,
        lookup_range_check_3_6_6_3_4: *const *const u32,
        lookup_range_check_3_6_6_3_5: *const *const u32,
        lookup_range_check_3_6_6_3_6: *const *const u32,
        lookup_range_check_3_6_6_3_7: *const *const u32,
        lookup_range_check_3_6_6_3_8: *const *const u32,
        lookup_range_check_3_6_6_3_9: *const *const u32,
        lookup_range_check_3_6_6_3_10: *const *const u32,
        lookup_range_check_3_6_6_3_11: *const *const u32,
        lookup_range_check_3_6_6_3_12: *const *const u32,
        lookup_range_check_3_6_6_3_13: *const *const u32,
        lookup_range_check_3_6_6_3_14: *const *const u32,
        lookup_range_check_3_6_6_3_15: *const *const u32,
        lookup_range_check_3_6_6_3_16: *const *const u32,
        lookup_range_check_3_6_6_3_17: *const *const u32,
        lookup_range_check_3_6_6_3_18: *const *const u32,
        lookup_range_check_3_6_6_3_19: *const *const u32,
        lookup_range_check_3_6_6_3_20: *const *const u32,
        lookup_range_check_3_6_6_3_21: *const *const u32,
        lookup_range_check_3_6_6_3_22: *const *const u32,
        lookup_range_check_3_6_6_3_23: *const *const u32,
        lookup_range_check_3_6_6_3_24: *const *const u32,
        lookup_range_check_3_6_6_3_25: *const *const u32,
        lookup_range_check_3_6_6_3_26: *const *const u32,
        lookup_range_check_3_6_6_3_27: *const *const u32,
        lookup_range_check_3_6_6_3_28: *const *const u32,
        lookup_range_check_3_6_6_3_29: *const *const u32,
        lookup_range_check_3_6_6_3_30: *const *const u32,
        lookup_range_check_3_6_6_3_31: *const *const u32,
        lookup_range_check_3_6_6_3_32: *const *const u32,
        lookup_range_check_3_6_6_3_33: *const *const u32,
        lookup_range_check_3_6_6_3_34: *const *const u32,
        lookup_range_check_3_6_6_3_35: *const *const u32,
        lookup_range_check_3_6_6_3_36: *const *const u32,
        lookup_range_check_3_6_6_3_37: *const *const u32,
        lookup_range_check_3_6_6_3_38: *const *const u32,
        lookup_range_check_3_6_6_3_39: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === PEDERSEN_BUILTIN (simplified: 3 columns, no EC math) ===
    pub fn gen_pedersen_builtin_trace(
        traces: *const *const u32,     // 3 output columns
        lk_mem_0: *const *const u32,   // 3 arrays
        lk_mem_1: *const *const u32,   // 3 arrays
        lk_mem_2: *const *const u32,   // 3 arrays
        lk_agg_0: *const *const u32,   // 4 arrays
        sub_mem: *const *const u32,    // 3 arrays
        sub_agg: *const *const u32,    // 3 arrays
        address_to_raw_id: *const u32, // GPU memory table
        segment_start: u32,
        n_rows: u32,
        log_size: u32,
    );

    pub fn gen_pedersen_builtin_interaction_trace(
        lookup_elements: *mut std::os::raw::c_void,
        lk_mem_0: *const *const u32, // 3 arrays
        lk_mem_1: *const *const u32, // 3 arrays
        lk_mem_2: *const *const u32, // 3 arrays
        lk_agg_0: *const *const u32, // 4 arrays
        log_size: u32,
        interaction_trace_columns: *const *const u32, // 8 columns (4*2)
        claimed_sum: *mut u32,                        // 4 m31s for qm31
    );

    // === PEDERSEN_BUILTIN_NARROW (simplified: 3 columns, window_bits_9) ===
    pub fn gen_pedersen_builtin_narrow_trace(
        traces: *const *const u32,     // 3 output columns
        lk_mem_0: *const *const u32,   // 3 arrays
        lk_mem_1: *const *const u32,   // 3 arrays
        lk_mem_2: *const *const u32,   // 3 arrays
        lk_agg_0: *const *const u32,   // 4 arrays
        sub_mem: *const *const u32,    // 3 arrays
        sub_agg: *const *const u32,    // 3 arrays
        address_to_raw_id: *const u32, // GPU memory table
        segment_start: u32,
        n_rows: u32,
        log_size: u32,
    );

    pub fn gen_pedersen_builtin_narrow_interaction_trace(
        lookup_elements: *mut std::os::raw::c_void,
        lk_mem_0: *const *const u32, // 3 arrays
        lk_mem_1: *const *const u32, // 3 arrays
        lk_mem_2: *const *const u32, // 3 arrays
        lk_agg_0: *const *const u32, // 4 arrays
        log_size: u32,
        interaction_trace_columns: *const *const u32, // 8 columns (4*2)
        claimed_sum: *mut u32,                        // 4 m31s for qm31
    );

    // === POSEIDON_BUILTIN_SPLIT (6 columns, split AIR) ===
    pub fn gen_poseidon_builtin_split_trace(
        traces: *const *const u32,     // 6 output columns
        lk_mem_0: *const *const u32,   // 3 arrays
        lk_mem_1: *const *const u32,   // 3 arrays
        lk_mem_2: *const *const u32,   // 3 arrays
        lk_mem_3: *const *const u32,   // 3 arrays
        lk_mem_4: *const *const u32,   // 3 arrays
        lk_mem_5: *const *const u32,   // 3 arrays
        lk_agg_0: *const *const u32,   // 7 arrays
        sub_mem: *const *const u32,    // 6 arrays
        sub_agg: *const *const u32,    // 6 arrays
        address_to_raw_id: *const u32, // GPU memory table
        segment_start: u32,
        n_rows: u32,
        log_size: u32,
    );

    pub fn gen_poseidon_builtin_split_interaction_trace(
        lookup_elements: *mut std::os::raw::c_void,
        lk_mem_0: *const *const u32, // 3 arrays
        lk_mem_1: *const *const u32, // 3 arrays
        lk_mem_2: *const *const u32, // 3 arrays
        lk_mem_3: *const *const u32, // 3 arrays
        lk_mem_4: *const *const u32, // 3 arrays
        lk_mem_5: *const *const u32, // 3 arrays
        lk_agg_0: *const *const u32, // 7 arrays
        log_size: u32,
        interaction_trace_columns: *const *const u32, // 16 columns (4*4)
        claimed_sum: *mut u32,                        // 4 m31s for qm31
    );

    // GPU-native pedersen table generation (generates table directly on GPU)
    pub fn initialize_pedersen_table();
    pub fn is_pedersen_table_initialized() -> bool;
    pub fn free_pedersen_table();

    // Get device pointers for the pedersen table columns (must be initialized first).
    pub fn get_pedersen_table_column_ptrs(
        output_ptrs: *mut *const u32, // Array of 56 device pointers
        out_n_rows: *mut u32,         // Padded row count
    );

    // Generate preprocessed columns directly on GPU
    // Seq column: output[i] = i for i in 0..(1 << log_size)
    pub fn gen_seq_column_on_gpu(output: *const u32, log_size: u32);

    // RangeCheck column generation on GPU
    // Generates partitioned enumeration for range check columns
    pub fn gen_range_check_columns_on_gpu(
        output_columns: *const *const u32, // Array of column pointers
        n_columns: u32,                    // Number of columns
        bits_per_segment: *const u32,      // Array of bit widths per segment
        n_segments: u32,                   // Number of segments
    );

    // BitwiseXor column generation on GPU
    // Generates XOR lookup table columns
    pub fn gen_bitwise_xor_columns_on_gpu(
        output_columns: *const *const u32, // Array of 3 column pointers (a, b, a^b)
        n_bits: u32,                       // Number of bits (4, 7, 8, 9, or 10)
    );

    // === POSEIDON_BUILTIN ===
    pub fn gen_poseidon_builtin_trace(
        traces: *const *const u32,
        log_size: u32,
        segment_start: u32,
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        // Lookup data - memory_address_to_id (6 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,

        // Lookup data - memory_id_to_big (6 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,

        // Lookup data - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
        lookup_range_check_3_3_3_3_3_0: *const *const u32,
        lookup_range_check_3_3_3_3_3_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (6 lookups, 4 elements each)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,
        lookup_range_check_4_4_4_4_3: *const *const u32,
        lookup_range_check_4_4_4_4_4: *const *const u32,
        lookup_range_check_4_4_4_4_5: *const *const u32,

        // Lookup data - range_check_4_4 (3 lookups, 2 elements each)
        lookup_range_check_4_4_0: *const *const u32,
        lookup_range_check_4_4_1: *const *const u32,
        lookup_range_check_4_4_2: *const *const u32,

        // Lookup data - poseidon_full_round_chain (8 lookups, 32 elements each)
        lookup_poseidon_full_round_chain_0: *const *const u32,
        lookup_poseidon_full_round_chain_1: *const *const u32,
        lookup_poseidon_full_round_chain_2: *const *const u32,
        lookup_poseidon_full_round_chain_3: *const *const u32,
        lookup_poseidon_full_round_chain_4: *const *const u32,
        lookup_poseidon_full_round_chain_5: *const *const u32,
        lookup_poseidon_full_round_chain_6: *const *const u32,
        lookup_poseidon_full_round_chain_7: *const *const u32,

        // Lookup data - poseidon_3_partial_rounds_chain (27 lookups, 42 elements each)
        lookup_poseidon_3_partial_rounds_chain_0: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_1: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_2: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_3: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_4: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_5: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_6: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_7: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_8: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_9: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_10: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_11: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_12: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_13: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_14: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_15: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_16: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_17: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_18: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_19: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_20: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_21: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_22: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_23: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_24: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_25: *const *const u32,
        lookup_poseidon_3_partial_rounds_chain_26: *const *const u32,

        // Base trace cols 120-283 (164 columns) for interaction kernels
        lookup_base_trace_cols: *const *const u32,
    );

    pub fn gen_poseidon_builtin_interaction_trace(
        interaction_trace: *const *const u32,
        log_size: u32,

        // Lookup elements (relations)
        memory_address_to_id_relation: *const u32,
        memory_id_to_big_relation: *const u32,
        poseidon_full_round_chain_relation: *const u32,
        range_check_felt_252_width_27_relation: *const u32,
        cube_252_relation: *const u32,
        range_check_3_3_3_3_3_relation: *const u32,
        range_check_4_4_4_4_relation: *const u32,
        range_check_4_4_relation: *const u32,
        poseidon_3_partial_rounds_chain_relation: *const u32,

        // Base trace columns for reconstructing all lookup data
        base_trace: *const *const u32,

        // Lookup data - memory_address_to_id (6 lookups)
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_address_to_id_1: *const *const u32,
        lookup_memory_address_to_id_2: *const *const u32,
        lookup_memory_address_to_id_3: *const *const u32,
        lookup_memory_address_to_id_4: *const *const u32,
        lookup_memory_address_to_id_5: *const *const u32,

        // Lookup data - memory_id_to_big (6 lookups)
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_memory_id_to_big_1: *const *const u32,
        lookup_memory_id_to_big_2: *const *const u32,
        lookup_memory_id_to_big_3: *const *const u32,
        lookup_memory_id_to_big_4: *const *const u32,
        lookup_memory_id_to_big_5: *const *const u32,

        // Lookup data - range_check_3_3_3_3_3 (2 lookups, 5 elements each)
        lookup_range_check_3_3_3_3_3_0: *const *const u32,
        lookup_range_check_3_3_3_3_3_1: *const *const u32,

        // Lookup data - range_check_4_4_4_4 (6 lookups, 4 elements each)
        lookup_range_check_4_4_4_4_0: *const *const u32,
        lookup_range_check_4_4_4_4_1: *const *const u32,
        lookup_range_check_4_4_4_4_2: *const *const u32,
        lookup_range_check_4_4_4_4_3: *const *const u32,
        lookup_range_check_4_4_4_4_4: *const *const u32,
        lookup_range_check_4_4_4_4_5: *const *const u32,

        // Lookup data - range_check_4_4 (3 lookups, 2 elements each)
        lookup_range_check_4_4_0: *const *const u32,
        lookup_range_check_4_4_1: *const *const u32,
        lookup_range_check_4_4_2: *const *const u32,

        // Lookup data - poseidon_full_round_chain (2 lookups, 32 elements each)
        lookup_poseidon_full_round_chain_0: *const *const u32,
        lookup_poseidon_full_round_chain_1: *const *const u32,

        // Base trace cols 120-283 (164 columns) for interaction kernels
        lookup_base_trace_cols: *const *const u32,

        claimed_sum: *const u32,
    );

    // === RANGE_CHECK_BUILTIN_BITS_96 ===
    pub fn generate_range_check_builtin_bits_96_traces(
        traces: *const *const u32,

        // Lookup data - 1 MemoryAddressToId lookup (2 elements)
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup (29 elements)
        lookup_memory_id_to_big_0: *const *const u32,

        // Lookup data - 1 RangeCheck_6 lookup (1 element)
        lookup_range_check_6_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_6: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_range_check_builtin_bits_96_interaction_traces(
        // Relation pointers (3 relations)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,
        range_check_6: *mut c_void,

        // Lookup data - 1 MemoryAddressToId lookup
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup
        lookup_memory_id_to_big_0: *const *const u32,

        // Lookup data - 1 RangeCheck_6 lookup
        lookup_range_check_6_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // === RANGE_CHECK_BUILTIN_BITS_128 ===
    pub fn generate_range_check_builtin_bits_128_traces(
        traces: *const *const u32,

        // Lookup data - 1 MemoryAddressToId lookup (2 elements)
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup (29 elements)
        lookup_memory_id_to_big_0: *const *const u32,

        // Sub-component inputs
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,

        // Builtin segment info
        segment_start: u32,

        // Memory lookup tables
        memory_address_to_id_address_to_raw_id: *const u32,
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,

        n_rows: u32,
        log_size: u32,
    );

    pub fn generate_range_check_builtin_bits_128_interaction_traces(
        // Relation pointers (2 relations - no range_check_6)
        memory_address_to_id: *mut c_void,
        memory_id_to_big: *mut c_void,

        // Lookup data - 1 MemoryAddressToId lookup
        lookup_memory_address_to_id_0: *const *const u32,

        // Lookup data - 1 MemoryIdToBig lookup
        lookup_memory_id_to_big_0: *const *const u32,

        n_rows: u32,
        log_size: u32,
        interaction_trace: *const *const u32,
        claimed_sum: *const u32,
    );

    // verify_instruction CUDA trace generation
    pub fn generate_verify_instruction_trace(
        traces: *const *const u32,
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_4_3_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,
        sub_component_inputs_memory_address_to_id: *const *const u32,
        sub_component_inputs_memory_id_to_big: *const *const u32,
        sub_component_inputs_range_check_7_2_5: *const *const u32,
        sub_component_inputs_range_check_4_3: *const *const u32,
        verify_instruction_inputs: *const *const u32,
        multiplicities: *const u32,
        memory_address_to_id_address_to_raw_id: *const u32,
        n_rows: u32,
        log_size: u32,
    );

    /// Full CUDA interaction trace generation for verify_instruction with logup accumulation.
    /// Generates 12 columns (3 logup × 4 BaseField) with proper prefix sum and claimed_sum.
    pub fn generate_verify_instruction_interaction_trace(
        interaction_traces: *const *const u32,
        lookup_range_check_7_2_5_0: *const *const u32,
        lookup_range_check_4_3_0: *const *const u32,
        lookup_memory_address_to_id_0: *const *const u32,
        lookup_memory_id_to_big_0: *const *const u32,
        lookup_verify_instruction_0: *const *const u32,
        multiplicities: *const u32,
        range_check_7_2_5_lookup: *mut c_void,
        range_check_4_3_lookup: *mut c_void,
        memory_address_to_id_lookup: *mut c_void,
        memory_id_to_big_lookup: *mut c_void,
        verify_instruction_lookup: *mut c_void,
        log_size: u32,
        claimed_sum: *mut u32,
    );

    // cube_252 CUDA trace generation (following blake_g pattern)
    pub fn generate_cube_252_trace(
        // Output: trace columns (141 columns)
        trace_columns: *const *const u32,
        // Output: lookup data for cube_252 self-lookup (20 elements)
        lookup_cube_252_0: *const *const u32,
        // Output: lookup data for range_check_9_9 variants (2 elements each)
        lookup_rc_9_9_0: *const *const u32,
        lookup_rc_9_9_1: *const *const u32,
        lookup_rc_9_9_2: *const *const u32,
        lookup_rc_9_9_3: *const *const u32,
        lookup_rc_9_9_4: *const *const u32,
        lookup_rc_9_9_5: *const *const u32,
        lookup_rc_9_9_b_0: *const *const u32,
        lookup_rc_9_9_b_1: *const *const u32,
        lookup_rc_9_9_b_2: *const *const u32,
        lookup_rc_9_9_b_3: *const *const u32,
        lookup_rc_9_9_b_4: *const *const u32,
        lookup_rc_9_9_b_5: *const *const u32,
        lookup_rc_9_9_c_0: *const *const u32,
        lookup_rc_9_9_c_1: *const *const u32,
        lookup_rc_9_9_c_2: *const *const u32,
        lookup_rc_9_9_c_3: *const *const u32,
        lookup_rc_9_9_c_4: *const *const u32,
        lookup_rc_9_9_c_5: *const *const u32,
        lookup_rc_9_9_d_0: *const *const u32,
        lookup_rc_9_9_d_1: *const *const u32,
        lookup_rc_9_9_d_2: *const *const u32,
        lookup_rc_9_9_d_3: *const *const u32,
        lookup_rc_9_9_d_4: *const *const u32,
        lookup_rc_9_9_d_5: *const *const u32,
        lookup_rc_9_9_e_0: *const *const u32,
        lookup_rc_9_9_e_1: *const *const u32,
        lookup_rc_9_9_e_2: *const *const u32,
        lookup_rc_9_9_e_3: *const *const u32,
        lookup_rc_9_9_e_4: *const *const u32,
        lookup_rc_9_9_e_5: *const *const u32,
        lookup_rc_9_9_f_0: *const *const u32,
        lookup_rc_9_9_f_1: *const *const u32,
        lookup_rc_9_9_f_2: *const *const u32,
        lookup_rc_9_9_f_3: *const *const u32,
        lookup_rc_9_9_f_4: *const *const u32,
        lookup_rc_9_9_f_5: *const *const u32,
        lookup_rc_9_9_g_0: *const *const u32,
        lookup_rc_9_9_g_1: *const *const u32,
        lookup_rc_9_9_g_2: *const *const u32,
        lookup_rc_9_9_h_0: *const *const u32,
        lookup_rc_9_9_h_1: *const *const u32,
        lookup_rc_9_9_h_2: *const *const u32,
        // Output: lookup data for range_check_19 variants (1 element each)
        lookup_rc_19_0: *const *const u32,
        lookup_rc_19_1: *const *const u32,
        lookup_rc_19_2: *const *const u32,
        lookup_rc_19_3: *const *const u32,
        lookup_rc_19_4: *const *const u32,
        lookup_rc_19_5: *const *const u32,
        lookup_rc_19_6: *const *const u32,
        lookup_rc_19_7: *const *const u32,
        lookup_rc_19_b_0: *const *const u32,
        lookup_rc_19_b_1: *const *const u32,
        lookup_rc_19_b_2: *const *const u32,
        lookup_rc_19_b_3: *const *const u32,
        lookup_rc_19_b_4: *const *const u32,
        lookup_rc_19_b_5: *const *const u32,
        lookup_rc_19_b_6: *const *const u32,
        lookup_rc_19_b_7: *const *const u32,
        lookup_rc_19_c_0: *const *const u32,
        lookup_rc_19_c_1: *const *const u32,
        lookup_rc_19_c_2: *const *const u32,
        lookup_rc_19_c_3: *const *const u32,
        lookup_rc_19_c_4: *const *const u32,
        lookup_rc_19_c_5: *const *const u32,
        lookup_rc_19_c_6: *const *const u32,
        lookup_rc_19_c_7: *const *const u32,
        lookup_rc_19_d_0: *const *const u32,
        lookup_rc_19_d_1: *const *const u32,
        lookup_rc_19_d_2: *const *const u32,
        lookup_rc_19_d_3: *const *const u32,
        lookup_rc_19_d_4: *const *const u32,
        lookup_rc_19_d_5: *const *const u32,
        lookup_rc_19_e_0: *const *const u32,
        lookup_rc_19_e_1: *const *const u32,
        lookup_rc_19_e_2: *const *const u32,
        lookup_rc_19_e_3: *const *const u32,
        lookup_rc_19_e_4: *const *const u32,
        lookup_rc_19_e_5: *const *const u32,
        lookup_rc_19_f_0: *const *const u32,
        lookup_rc_19_f_1: *const *const u32,
        lookup_rc_19_f_2: *const *const u32,
        lookup_rc_19_f_3: *const *const u32,
        lookup_rc_19_f_4: *const *const u32,
        lookup_rc_19_f_5: *const *const u32,
        lookup_rc_19_g_0: *const *const u32,
        lookup_rc_19_g_1: *const *const u32,
        lookup_rc_19_g_2: *const *const u32,
        lookup_rc_19_g_3: *const *const u32,
        lookup_rc_19_g_4: *const *const u32,
        lookup_rc_19_g_5: *const *const u32,
        lookup_rc_19_h_0: *const *const u32,
        lookup_rc_19_h_1: *const *const u32,
        lookup_rc_19_h_2: *const *const u32,
        lookup_rc_19_h_3: *const *const u32,
        lookup_rc_19_h_4: *const *const u32,
        lookup_rc_19_h_5: *const *const u32,
        lookup_rc_19_h_6: *const *const u32,
        lookup_rc_19_h_7: *const *const u32,
        // Sub-component inputs for range_check_9_9 variants (flattened: [count][2] elements)
        sub_rc_9_9: *const *const u32,   // 6 * 2 = 12 pointers
        sub_rc_9_9_b: *const *const u32, // 6 * 2 = 12 pointers
        sub_rc_9_9_c: *const *const u32, // 6 * 2 = 12 pointers
        sub_rc_9_9_d: *const *const u32, // 6 * 2 = 12 pointers
        sub_rc_9_9_e: *const *const u32, // 6 * 2 = 12 pointers
        sub_rc_9_9_f: *const *const u32, // 6 * 2 = 12 pointers
        sub_rc_9_9_g: *const *const u32, // 3 * 2 = 6 pointers
        sub_rc_9_9_h: *const *const u32, // 3 * 2 = 6 pointers
        // Sub-component inputs for range_check_19 variants (flattened: [count][1] elements)
        sub_rc_19: *const *const u32,   // 8 * 1 = 8 pointers
        sub_rc_19_b: *const *const u32, // 8 * 1 = 8 pointers
        sub_rc_19_c: *const *const u32, // 8 * 1 = 8 pointers
        sub_rc_19_d: *const *const u32, // 6 * 1 = 6 pointers
        sub_rc_19_e: *const *const u32, // 6 * 1 = 6 pointers
        sub_rc_19_f: *const *const u32, // 6 * 1 = 6 pointers
        sub_rc_19_g: *const *const u32, // 6 * 1 = 6 pointers
        sub_rc_19_h: *const *const u32, // 8 * 1 = 8 pointers
        // Input: 10 columns (Width27 format)
        inputs: *const *const u32,
        // Input: log size of trace
        trace_log_size: u32,
    );

    pub fn generate_cube_252_interaction_trace(
        trace_columns: *const *const u32, // Base trace (141 columns)
        trace_size: u32,
        // Lookup elements for each relation (order matches SIMD and C function)
        cube_252_lookup_elements: *mut c_void,
        rc_19_lookup_elements: *mut c_void,
        rc_19_b_lookup_elements: *mut c_void,
        rc_19_c_lookup_elements: *mut c_void,
        rc_19_d_lookup_elements: *mut c_void,
        rc_19_e_lookup_elements: *mut c_void,
        rc_19_f_lookup_elements: *mut c_void,
        rc_19_g_lookup_elements: *mut c_void,
        rc_19_h_lookup_elements: *mut c_void,
        rc_9_9_lookup_elements: *mut c_void,
        rc_9_9_b_lookup_elements: *mut c_void,
        rc_9_9_c_lookup_elements: *mut c_void,
        rc_9_9_d_lookup_elements: *mut c_void,
        rc_9_9_e_lookup_elements: *mut c_void,
        rc_9_9_f_lookup_elements: *mut c_void,
        rc_9_9_g_lookup_elements: *mut c_void,
        rc_9_9_h_lookup_elements: *mut c_void,
        interaction_trace_columns: *const *const u32, // Output interaction trace
        claimed_sum: *mut u32,                        // Output claimed sum (4 u32s for qm31)
    );

    // Poseidon Full Round Chain
    pub fn poseidon_full_round_chain_generate_trace(
        input_limb_0: *const u32,   // Index values
        input_limb_1: *const u32,   // Round number values
        state_0: *const *const u32, // State[0]: 10 input columns (Width27 format)
        state_1: *const *const u32, // State[1]: 10 input columns (Width27 format)
        state_2: *const *const u32, // State[2]: 10 input columns (Width27 format)
        n_rows: u32,
        trace_columns: *const *const u32, // 126 output trace columns
        poseidon_round_keys_table: *const *const u32, // 30 columns of round keys
    );

    pub fn poseidon_full_round_chain_add_to_multiplicities(
        trace_columns: *const *const u32,
        n_rows: u32,
        // Cube252 multiplicities (3 lookups per row)
        cube_252_mults: *const u32,
        cube_252_log_size: u32,
        // PoseidonRoundKeys multiplicities (1 lookup per row)
        poseidon_round_keys_mults: *const u32,
        // RangeCheck_3_3_3_3_3 multiplicities (6 lookups per row)
        rc_3_3_3_3_3_mults: *const u32,
        rc_3_3_3_3_3_log_size: u32,
    );

    /// Compute RC_3_3_3_3_3 inputs on GPU from trace columns.
    /// Reads trace columns 32-124, computes carry chains for 3 linear combinations,
    /// and writes 30 biased carry arrays (6 sets × 5 columns).
    pub fn poseidon_full_round_chain_compute_rc_inputs(
        trace_columns: *const *const u32,
        n_rows: u32,
        output_arrays: *const *const u32, // 30 output device pointers
    );

    pub fn poseidon_full_round_chain_generate_interaction_trace(
        trace_columns: *const *const u32, // Base trace (126 columns)
        trace_size: u32,
        // Lookup elements for each relation
        cube_252_lookup_elements: *mut c_void,
        poseidon_round_keys_lookup_elements: *mut c_void,
        range_check_3_3_3_3_3_lookup_elements: *mut c_void,
        poseidon_full_round_chain_lookup_elements: *mut c_void,
        interaction_trace_columns: *const *const u32, // Output interaction trace (24 columns)
        claimed_sum: *mut u32,                        // Output claimed sum (4 u32s for qm31)
    );

    // Poseidon 3 Partial Rounds Chain
    pub fn poseidon_3_partial_rounds_chain_generate_trace(
        input_limb_0: *const u32,   // Index values
        input_limb_1: *const u32,   // Round number values
        state_0: *const *const u32, // State[0]: 10 input columns (Width27 format)
        state_1: *const *const u32, // State[1]: 10 input columns (Width27 format)
        state_2: *const *const u32, // State[2]: 10 input columns (Width27 format)
        state_3: *const *const u32, // State[3]: 10 input columns (Width27 format)
        n_rows: u32,
        actual_n_rows: u32,               // Number of actual (non-padding) rows
        trace_columns: *const *const u32, // 169 output trace columns
        poseidon_round_keys_table: *const *const u32, // 30 columns of round keys
    );

    pub fn poseidon_3_partial_rounds_chain_add_to_multiplicities(
        trace_columns: *const *const u32,
        n_rows: u32,
        // Cube252 multiplicities (3 lookups per row)
        cube_252_mults: *const u32,
        cube_252_log_size: u32,
        // PoseidonRoundKeys multiplicities
        poseidon_round_keys_mults: *const u32,
        // RangeCheckFelt252Width27 multiplicities (3 lookups per row)
        rc_felt_252_width_27_mults: *const u32,
        rc_felt_252_width_27_log_size: u32,
        // RangeCheck_4_4 multiplicities (3 lookups per row)
        rc_4_4_mults: *const u32,
        rc_4_4_log_size: u32,
        // RangeCheck_4_4_4_4 multiplicities (6 lookups per row)
        rc_4_4_4_4_mults: *const u32,
        rc_4_4_4_4_log_size: u32,
    );

    pub fn poseidon_3_partial_rounds_chain_generate_interaction_trace(
        trace_columns: *const *const u32, // Base trace (169 columns)
        trace_size: u32,
        // Lookup elements for each relation
        cube_252_lookup_elements: *mut c_void,
        poseidon_round_keys_lookup_elements: *mut c_void,
        range_check_felt_252_width_27_lookup_elements: *mut c_void,
        range_check_4_4_lookup_elements: *mut c_void,
        range_check_4_4_4_4_lookup_elements: *mut c_void,
        poseidon_3_partial_rounds_chain_lookup_elements: *mut c_void,
        interaction_trace_columns: *const *const u32, // Output interaction trace (36 columns)
        claimed_sum: *mut u32,                        // Output claimed sum (4 u32s for qm31)
    );

    // RangeCheckFelt252Width27 CUDA trace generation
    pub fn range_check_felt_252_width_27_generate_trace(
        input_limbs: *const *const u32,   // 10 input columns (Width27 format)
        n_rows: u32,                      // Padded size (power of 2)
        actual_n_rows: u32,               // Actual data rows (before padding)
        trace_columns: *const *const u32, // 20 output trace columns
    );

    pub fn range_check_felt_252_width_27_add_to_multiplicities(
        trace_columns: *const *const u32,
        n_rows: u32,
        // RangeCheck_18 multiplicities (7 lookups per row)
        rc_18_mults: *const u32,
        rc_18_log_size: u32,
        // RangeCheck_18_B multiplicities (2 lookups per row)
        rc_18_b_mults: *const u32,
        rc_18_b_log_size: u32,
        // RangeCheck_9_9 multiplicities (1 lookup per row)
        rc_9_9_mults: *const u32,
        rc_9_9_log_size: u32,
        // RangeCheck_9_9_B multiplicities (1 lookup per row)
        rc_9_9_b_mults: *const u32,
        rc_9_9_b_log_size: u32,
        // RangeCheck_9_9_C multiplicities (1 lookup per row)
        rc_9_9_c_mults: *const u32,
        rc_9_9_c_log_size: u32,
        // RangeCheck_9_9_D multiplicities (1 lookup per row)
        rc_9_9_d_mults: *const u32,
        rc_9_9_d_log_size: u32,
        // RangeCheck_9_9_E multiplicities (1 lookup per row)
        rc_9_9_e_mults: *const u32,
        rc_9_9_e_log_size: u32,
    );

    pub fn range_check_felt_252_width_27_generate_interaction_trace(
        trace_columns: *const *const u32, // Base trace (20 columns)
        trace_size: u32,
        // Lookup elements for each relation
        rc_9_9_lookup_elements: *mut c_void,
        rc_18_lookup_elements: *mut c_void,
        rc_9_9_b_lookup_elements: *mut c_void,
        rc_18_b_lookup_elements: *mut c_void,
        rc_9_9_c_lookup_elements: *mut c_void,
        rc_9_9_d_lookup_elements: *mut c_void,
        rc_9_9_e_lookup_elements: *mut c_void,
        range_check_felt_252_width_27_lookup_elements: *mut c_void,
        interaction_trace_columns: *const *const u32, // Output interaction trace (32 columns)
        claimed_sum: *mut u32,                        // Output claimed sum (4 u32s for qm31)
    );

    // partial_ec_mul CUDA trace generation
    pub fn partial_ec_mul_generate_trace(
        input_columns: *const *const u32, // 73 input columns
        n_rows: u32,                      // Number of valid (non-padding) rows
        log_size: u32,                    // Log2 of padded trace size
        trace_columns: *const *const u32, // 472 output trace columns
    );

    pub fn partial_ec_mul_add_to_multiplicities(
        trace_columns: *const *const u32,
        n_rows: u32,
        log_size: u32, // Log2 of trace size for proper padding handling
        // PedersenPointsTable multiplicities (1 lookup per row)
        pedersen_points_table_mults: *const u32,
        pedersen_points_table_log_size: u32,
        // RangeCheck_9_9 variants (8)
        rc_9_9_mults: *const u32,
        rc_9_9_log_size: u32,
        rc_9_9_b_mults: *const u32,
        rc_9_9_b_log_size: u32,
        rc_9_9_c_mults: *const u32,
        rc_9_9_c_log_size: u32,
        rc_9_9_d_mults: *const u32,
        rc_9_9_d_log_size: u32,
        rc_9_9_e_mults: *const u32,
        rc_9_9_e_log_size: u32,
        rc_9_9_f_mults: *const u32,
        rc_9_9_f_log_size: u32,
        rc_9_9_g_mults: *const u32,
        rc_9_9_g_log_size: u32,
        rc_9_9_h_mults: *const u32,
        rc_9_9_h_log_size: u32,
        // RangeCheck_19 variants (8)
        rc_19_mults: *const u32,
        rc_19_log_size: u32,
        rc_19_b_mults: *const u32,
        rc_19_b_log_size: u32,
        rc_19_c_mults: *const u32,
        rc_19_c_log_size: u32,
        rc_19_d_mults: *const u32,
        rc_19_d_log_size: u32,
        rc_19_e_mults: *const u32,
        rc_19_e_log_size: u32,
        rc_19_f_mults: *const u32,
        rc_19_f_log_size: u32,
        rc_19_g_mults: *const u32,
        rc_19_g_log_size: u32,
        rc_19_h_mults: *const u32,
        rc_19_h_log_size: u32,
    );

    pub fn partial_ec_mul_generate_interaction_trace(
        trace_columns: *const *const u32, // Base trace (472 columns)
        n_rows: u32,                      // Number of valid (non-padding) rows
        log_size: u32,                    // Log2 of padded trace size
        // Lookup elements for each relation
        pedersen_points_table_lookup_elements: *mut c_void,
        rc_9_9_lookup_elements: *mut c_void,
        rc_9_9_b_lookup_elements: *mut c_void,
        rc_9_9_c_lookup_elements: *mut c_void,
        rc_9_9_d_lookup_elements: *mut c_void,
        rc_9_9_e_lookup_elements: *mut c_void,
        rc_9_9_f_lookup_elements: *mut c_void,
        rc_9_9_g_lookup_elements: *mut c_void,
        rc_9_9_h_lookup_elements: *mut c_void,
        rc_19_lookup_elements: *mut c_void,
        rc_19_b_lookup_elements: *mut c_void,
        rc_19_c_lookup_elements: *mut c_void,
        rc_19_d_lookup_elements: *mut c_void,
        rc_19_e_lookup_elements: *mut c_void,
        rc_19_f_lookup_elements: *mut c_void,
        rc_19_g_lookup_elements: *mut c_void,
        rc_19_h_lookup_elements: *mut c_void,
        partial_ec_mul_lookup_elements: *mut c_void,
        interaction_trace_columns: *const *const u32, // Output interaction trace
        claimed_sum: *mut u32,                        // Output claimed sum (4 u32s for qm31)
    );

    /// Merged CUDA trace generation for partial_ec_mul.
    /// Generates trace, lookup_data, and sub_component_inputs in a single kernel call.
    /// This follows the blake_g pattern of integrated trace generation.
    pub fn generate_partial_ec_mul_trace(
        traces: *const *const u32, // 472 trace output columns
        // Lookup data pointers
        lookup_partial_ec_mul_0: *const *const u32, // 73 arrays
        lookup_partial_ec_mul_1: *const *const u32, // 73 arrays
        lookup_pedersen_points_table_0: *const *const u32, // 57 arrays
        // Range check 19 lookup pointers (all variants)
        lookup_rc_19: *const *const u32,   // 12 arrays
        lookup_rc_19_b: *const *const u32, // 12 arrays
        lookup_rc_19_c: *const *const u32, // 12 arrays
        lookup_rc_19_d: *const *const u32, // 9 arrays
        lookup_rc_19_e: *const *const u32, // 9 arrays
        lookup_rc_19_f: *const *const u32, // 9 arrays
        lookup_rc_19_g: *const *const u32, // 9 arrays
        lookup_rc_19_h: *const *const u32, // 12 arrays
        // Range check 9_9 lookup pointers (all variants)
        lookup_rc_9_9: *const *const u32,   // 18*2 arrays
        lookup_rc_9_9_b: *const *const u32, // 18*2 arrays
        lookup_rc_9_9_c: *const *const u32, // 18*2 arrays
        lookup_rc_9_9_d: *const *const u32, // 18*2 arrays
        lookup_rc_9_9_e: *const *const u32, // 18*2 arrays
        lookup_rc_9_9_f: *const *const u32, // 18*2 arrays
        lookup_rc_9_9_g: *const *const u32, // 9*2 arrays
        lookup_rc_9_9_h: *const *const u32, // 9*2 arrays
        // Sub component inputs pointers
        sub_inputs_ppt: *const *const u32, // 1*1 arrays (pedersen_points_table)
        sub_inputs_rc_9_9: *const *const u32, // 18*2 arrays
        sub_inputs_rc_9_9_b: *const *const u32, // 18*2 arrays
        sub_inputs_rc_9_9_c: *const *const u32, // 18*2 arrays
        sub_inputs_rc_9_9_d: *const *const u32, // 18*2 arrays
        sub_inputs_rc_9_9_e: *const *const u32, // 18*2 arrays
        sub_inputs_rc_9_9_f: *const *const u32, // 18*2 arrays
        sub_inputs_rc_9_9_g: *const *const u32, // 9*2 arrays
        sub_inputs_rc_9_9_h: *const *const u32, // 9*2 arrays
        sub_inputs_rc_19_h: *const *const u32, // 12*1 arrays
        sub_inputs_rc_19: *const *const u32, // 12*1 arrays
        sub_inputs_rc_19_b: *const *const u32, // 12*1 arrays
        sub_inputs_rc_19_c: *const *const u32, // 12*1 arrays
        sub_inputs_rc_19_d: *const *const u32, // 9*1 arrays
        sub_inputs_rc_19_e: *const *const u32, // 9*1 arrays
        sub_inputs_rc_19_f: *const *const u32, // 9*1 arrays
        sub_inputs_rc_19_g: *const *const u32, // 9*1 arrays
        // Inputs
        inputs: *const *const u32, // 73 input columns
        n_rows: u32,               // Number of valid rows
        log_size: u32,             // Log2 of trace size
    );

    /// Generate interaction trace from lookup_data (instead of trace columns).
    /// Uses all lookup data arrays to compute the 107 LogUp columns.
    pub fn generate_partial_ec_mul_interaction_traces(
        // Lookup elements for each relation (18 relations total)
        pedersen_points_table_lookup_elements: *mut c_void,
        rc_9_9_lookup_elements: *mut c_void,
        rc_9_9_b_lookup_elements: *mut c_void,
        rc_9_9_c_lookup_elements: *mut c_void,
        rc_9_9_d_lookup_elements: *mut c_void,
        rc_9_9_e_lookup_elements: *mut c_void,
        rc_9_9_f_lookup_elements: *mut c_void,
        rc_9_9_g_lookup_elements: *mut c_void,
        rc_9_9_h_lookup_elements: *mut c_void,
        rc_19_lookup_elements: *mut c_void,
        rc_19_b_lookup_elements: *mut c_void,
        rc_19_c_lookup_elements: *mut c_void,
        rc_19_d_lookup_elements: *mut c_void,
        rc_19_e_lookup_elements: *mut c_void,
        rc_19_f_lookup_elements: *mut c_void,
        rc_19_g_lookup_elements: *mut c_void,
        rc_19_h_lookup_elements: *mut c_void,
        partial_ec_mul_lookup_elements: *mut c_void,
        // Lookup data pointers - main relations
        lookup_partial_ec_mul_0: *const *const u32, // 73 arrays
        lookup_partial_ec_mul_1: *const *const u32, // 73 arrays
        lookup_pedersen_points_table_0: *const *const u32, // 57 arrays
        // Lookup data pointers - range_check_19 variants (1 element each)
        lookup_rc_19: *const *const u32,   // 12 arrays
        lookup_rc_19_b: *const *const u32, // 12 arrays
        lookup_rc_19_c: *const *const u32, // 12 arrays
        lookup_rc_19_d: *const *const u32, // 9 arrays
        lookup_rc_19_e: *const *const u32, // 9 arrays
        lookup_rc_19_f: *const *const u32, // 9 arrays
        lookup_rc_19_g: *const *const u32, // 9 arrays
        lookup_rc_19_h: *const *const u32, // 12 arrays
        // Lookup data pointers - range_check_9_9 variants (2 elements each)
        lookup_rc_9_9: *const *const u32,   // 18*2 = 36 arrays
        lookup_rc_9_9_b: *const *const u32, // 36 arrays
        lookup_rc_9_9_c: *const *const u32, // 36 arrays
        lookup_rc_9_9_d: *const *const u32, // 36 arrays
        lookup_rc_9_9_e: *const *const u32, // 36 arrays
        lookup_rc_9_9_f: *const *const u32, // 36 arrays
        lookup_rc_9_9_g: *const *const u32, // 9*2 = 18 arrays
        lookup_rc_9_9_h: *const *const u32, // 18 arrays
        // Sizes
        n_rows: u32,   // Number of valid (non-padding) rows
        log_size: u32, // Log2 of padded trace size
        // Output
        interaction_trace_columns: *const *const u32, /* Output interaction trace (4*107 = 428
                                                       * cols) */
        claimed_sum: *mut u32, // Output claimed sum (4 u32s for qm31)
    );

    /// Add inputs to pedersen_points_table multiplicities.
    /// Takes table indices and atomically adds 1 to the multiplicity at each index.
    pub fn pedersen_points_table_add_inputs(
        indices: *const u32, // Table indices (one per row)
        n_rows: u32,         // Number of rows
        mults: *const u32,   // Output multiplicities (atomically updated)
        mults_log_size: u32, // Log2 of multiplicities table size
    );

    // Interaction trace generation for pedersen_points_table (pure CUDA path).
    // Generates logup interaction trace using GPU-resident pedersen table columns.
    pub fn pedersen_points_table_interaction_trace(
        lookup_elements: *mut c_void, // LookupElementsBasic<58> from Rust
        multiplicities: *const u32,   // GPU multiplicities
        log_size: u32,                // Log2 of table size (23)
        interaction_traces: *const *const u32, // 4 output columns (qm31 components)
        claimed_sum: *const u32,      // Output claimed sum (4 x m31)
    );

    // ========================================================================
    // partial_ec_mul_window_bits_18 (297-col "now" architecture)
    // ========================================================================

    /// Merged CUDA trace generation for partial_ec_mul_wb18.
    /// Generates 297-col trace, lookup_data, and sub_component_inputs in a single kernel.
    pub fn gen_partial_ec_mul_wb18_trace(
        traces: *const *const u32, // 297 trace output columns
        // Lookup data - self-interaction
        lookup_partial_ec_mul_0: *const *const u32, // 73 arrays
        lookup_partial_ec_mul_1: *const *const u32, // 73 arrays
        // Lookup data - pedersen_points_table
        lookup_ppt_0: *const *const u32, // 58 arrays
        // Lookup data - rc_20 variants (2 elements per entry)
        lookup_rc_20: *const *const u32,   // 12*2=24 flat ptrs
        lookup_rc_20_b: *const *const u32, // 12*2=24
        lookup_rc_20_c: *const *const u32, // 12*2=24
        lookup_rc_20_d: *const *const u32, // 12*2=24
        lookup_rc_20_e: *const *const u32, // 9*2=18
        lookup_rc_20_f: *const *const u32, // 9*2=18
        lookup_rc_20_g: *const *const u32, // 9*2=18
        lookup_rc_20_h: *const *const u32, // 9*2=18
        // Lookup data - rc_9_9 variants (3 elements per entry)
        lookup_rc_9_9: *const *const u32,   // 6*3=18 flat ptrs
        lookup_rc_9_9_b: *const *const u32, // 6*3=18
        lookup_rc_9_9_c: *const *const u32, // 6*3=18
        lookup_rc_9_9_d: *const *const u32, // 6*3=18
        lookup_rc_9_9_e: *const *const u32, // 6*3=18
        lookup_rc_9_9_f: *const *const u32, // 6*3=18
        lookup_rc_9_9_g: *const *const u32, // 3*3=9
        lookup_rc_9_9_h: *const *const u32, // 3*3=9
        // Sub-component inputs - pedersen_points_table
        sub_inputs_ppt: *const *const u32, // 1*1=1 flat ptrs
        // Sub-component inputs - rc_9_9 variants (2 cols per feed)
        sub_inputs_rc_9_9: *const *const u32, // 6*2=12 flat ptrs
        sub_inputs_rc_9_9_b: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_c: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_d: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_e: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_f: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_g: *const *const u32, // 3*2=6
        sub_inputs_rc_9_9_h: *const *const u32, // 3*2=6
        // Sub-component inputs - rc_20 variants (1 col per feed)
        sub_inputs_rc_20: *const *const u32, // 12*1=12 flat ptrs
        sub_inputs_rc_20_b: *const *const u32, // 12*1=12
        sub_inputs_rc_20_c: *const *const u32, // 12*1=12
        sub_inputs_rc_20_d: *const *const u32, // 12*1=12
        sub_inputs_rc_20_e: *const *const u32, // 9*1=9
        sub_inputs_rc_20_f: *const *const u32, // 9*1=9
        sub_inputs_rc_20_g: *const *const u32, // 9*1=9
        sub_inputs_rc_20_h: *const *const u32, // 9*1=9
        // Inputs
        inputs: *const *const u32, // 72 input columns
        n_rows: u32,               // Number of valid rows
        log_size: u32,             // Log2 of trace size
    );

    /// Generate interaction trace for partial_ec_mul_wb18 from lookup_data.
    /// Uses all lookup data arrays to compute the 65 LogUp columns.
    pub fn gen_partial_ec_mul_wb18_interaction_trace(
        // Lookup elements for each relation (18 total: pem + ppt + 8 rc_20 + 8 rc_9_9)
        partial_ec_mul_lookup_elements: *mut c_void,
        pedersen_points_table_lookup_elements: *mut c_void,
        rc_20_lookup_elements: *mut c_void,
        rc_20_b_lookup_elements: *mut c_void,
        rc_20_c_lookup_elements: *mut c_void,
        rc_20_d_lookup_elements: *mut c_void,
        rc_20_e_lookup_elements: *mut c_void,
        rc_20_f_lookup_elements: *mut c_void,
        rc_20_g_lookup_elements: *mut c_void,
        rc_20_h_lookup_elements: *mut c_void,
        rc_9_9_lookup_elements: *mut c_void,
        rc_9_9_b_lookup_elements: *mut c_void,
        rc_9_9_c_lookup_elements: *mut c_void,
        rc_9_9_d_lookup_elements: *mut c_void,
        rc_9_9_e_lookup_elements: *mut c_void,
        rc_9_9_f_lookup_elements: *mut c_void,
        rc_9_9_g_lookup_elements: *mut c_void,
        rc_9_9_h_lookup_elements: *mut c_void,
        // Lookup data pointers
        lookup_partial_ec_mul_0: *const *const u32, // 73 arrays
        lookup_partial_ec_mul_1: *const *const u32, // 73 arrays
        lookup_ppt_0: *const *const u32,            // 58 arrays
        lookup_rc_20: *const *const u32,            // 12*2=24 flat ptrs
        lookup_rc_20_b: *const *const u32,          // 12*2=24
        lookup_rc_20_c: *const *const u32,          // 12*2=24
        lookup_rc_20_d: *const *const u32,          // 12*2=24
        lookup_rc_20_e: *const *const u32,          // 9*2=18
        lookup_rc_20_f: *const *const u32,          // 9*2=18
        lookup_rc_20_g: *const *const u32,          // 9*2=18
        lookup_rc_20_h: *const *const u32,          // 9*2=18
        lookup_rc_9_9: *const *const u32,           // 6*3=18 flat ptrs
        lookup_rc_9_9_b: *const *const u32,         // 6*3=18
        lookup_rc_9_9_c: *const *const u32,         // 6*3=18
        lookup_rc_9_9_d: *const *const u32,         // 6*3=18
        lookup_rc_9_9_e: *const *const u32,         // 6*3=18
        lookup_rc_9_9_f: *const *const u32,         // 6*3=18
        lookup_rc_9_9_g: *const *const u32,         // 3*3=9
        lookup_rc_9_9_h: *const *const u32,         // 3*3=9
        // Sizes
        n_rows: u32,
        log_size: u32,
        // Output
        interaction_trace_columns: *const *const u32, // 4*65 = 260 cols
        claimed_sum: *const u32,                      // 4 u32s for qm31
    );

    // ========================================================================
    // pedersen_aggregator_window_bits_18 (206-col native CUDA)
    // ========================================================================

    /// Native CUDA trace generation for pedersen_aggregator_wb18.
    /// Generates 206-col trace, lookup_data, and sub_component_inputs in a single kernel.
    /// Uses the GPU-resident pedersen table directly, eliminating the CPU table.
    pub fn gen_pedersen_aggregator_wb18_trace(
        traces: *const *const u32, // 206 trace output columns
        // Lookup data
        lk_mem_0: *const *const u32, // 30 arrays (memory_id_to_big #0)
        lk_mem_1: *const *const u32, // 30 arrays (memory_id_to_big #1)
        lk_mem_2: *const *const u32, // 30 arrays (memory_id_to_big #2)
        lk_rc8_0: *const *const u32, // 2 arrays (range_check_8 #0)
        lk_rc8_1: *const *const u32, // 2 arrays (range_check_8 #1)
        lk_rc8_2: *const *const u32, // 2 arrays (range_check_8 #2)
        lk_rc8_3: *const *const u32, // 2 arrays (range_check_8 #3)
        lk_pem_0: *const *const u32, // 73 arrays (PEM chain 0 input)
        lk_pem_1: *const *const u32, // 73 arrays (PEM chain 0 output)
        lk_pem_2: *const *const u32, // 73 arrays (PEM chain 1 input)
        lk_pem_3: *const *const u32, // 73 arrays (PEM chain 1 output)
        lk_agg_0: *const *const u32, // 4 arrays (self-lookup)
        mults: *const u32,           // multiplicity data
        // Sub-component inputs
        sub_mem: *const *const u32, // 3 arrays
        sub_rc8: *const *const u32, // 4 arrays
        sub_pem: *const *const u32, // 72 arrays (each 28*trace_size)
        // Inputs
        inputs: *const *const u32, // 3 input columns
        // Memory state
        transpose_big_value_ptr: *const *const u32, // memory_id_to_big transpose ptrs
        small_value_ptr: *const u32,                // memory_id_to_big small values
        // Sizes
        n_rows: u32,
        log_size: u32,
    );

    /// Native CUDA interaction trace generation for pedersen_aggregator_wb18.
    /// Uses lookup data (already on GPU) to compute 6 logup columns entirely on GPU.
    pub fn gen_pedersen_aggregator_wb18_interaction_trace(
        // CommonLookupElements (= LookupElements<128>)
        lookup_elements: *mut std::os::raw::c_void,
        // Lookup data (all device pointers)
        lk_mem_0: *const *const u32, // 30 arrays
        lk_mem_1: *const *const u32, // 30 arrays
        lk_mem_2: *const *const u32, // 30 arrays
        lk_rc8_0: *const *const u32, // 2 arrays
        lk_rc8_1: *const *const u32, // 2 arrays
        lk_rc8_2: *const *const u32, // 2 arrays
        lk_rc8_3: *const *const u32, // 2 arrays
        lk_pem_0: *const *const u32, // 73 arrays
        lk_pem_1: *const *const u32, // 73 arrays
        lk_pem_2: *const *const u32, // 73 arrays
        lk_pem_3: *const *const u32, // 73 arrays
        lk_agg_0: *const *const u32, // 4 arrays
        mults: *const u32,           // multiplicities
        // Sizes
        log_size: u32,
        // Output
        interaction_trace_columns: *const *const u32, // 4*6 = 24 columns
        claimed_sum: *mut u32,                        // 4 m31s for qm31
    );

    // ========================================================================
    // Pedersen window_bits_9 (small table) components
    // ========================================================================

    // GPU-native small pedersen table generation (window_bits_9, ~7MB)
    pub fn initialize_pedersen_table_small();
    pub fn is_pedersen_table_small_initialized() -> bool;
    pub fn free_pedersen_table_small();

    /// Add inputs to small pedersen_points_table multiplicity tracking on GPU.
    pub fn pedersen_points_table_small_add_inputs(
        input_table_indices: *const u32, // table index column
        n_rows: u32,                     // number of rows
        multiplicities: *const u32,      // multiplicity array (atomicAdd target)
        log_size: u32,                   // log2 of table size
    );

    /// Generate interaction trace for pedersen_points_table_wb9 on GPU.
    pub fn pedersen_points_table_wb9_interaction_trace(
        lookup_elements: *mut c_void,                 // CommonLookupElements
        multiplicities: *const u32,                   // multiplicity data
        log_size: u32,                                // log2 of table size
        interaction_trace_columns: *const *const u32, // 4 columns (4*1 logup)
        claimed_sum: *const u32,                      // 4 m31s for qm31
    );

    // ========================================================================
    // partial_ec_mul_window_bits_9 (311-col "now" architecture)
    // ========================================================================

    /// Merged CUDA trace generation for partial_ec_mul_wb9.
    /// Generates 311-col trace, lookup_data, and sub_component_inputs in a single kernel.
    pub fn gen_partial_ec_mul_wb9_trace(
        traces: *const *const u32, // 311 trace output columns
        // Lookup data - self-interaction
        lookup_partial_ec_mul_0: *const *const u32, // 87 arrays
        lookup_partial_ec_mul_1: *const *const u32, // 87 arrays
        // Lookup data - pedersen_points_table
        lookup_ppt_0: *const *const u32, // 58 arrays
        // Lookup data - rc_20 variants (2 elements per entry)
        lookup_rc_20: *const *const u32,   // 12*2=24 flat ptrs
        lookup_rc_20_b: *const *const u32, // 12*2=24
        lookup_rc_20_c: *const *const u32, // 12*2=24
        lookup_rc_20_d: *const *const u32, // 12*2=24
        lookup_rc_20_e: *const *const u32, // 9*2=18
        lookup_rc_20_f: *const *const u32, // 9*2=18
        lookup_rc_20_g: *const *const u32, // 9*2=18
        lookup_rc_20_h: *const *const u32, // 9*2=18
        // Lookup data - rc_9_9 variants (3 elements per entry)
        lookup_rc_9_9: *const *const u32,   // 6*3=18 flat ptrs
        lookup_rc_9_9_b: *const *const u32, // 6*3=18
        lookup_rc_9_9_c: *const *const u32, // 6*3=18
        lookup_rc_9_9_d: *const *const u32, // 6*3=18
        lookup_rc_9_9_e: *const *const u32, // 6*3=18
        lookup_rc_9_9_f: *const *const u32, // 6*3=18
        lookup_rc_9_9_g: *const *const u32, // 3*3=9
        lookup_rc_9_9_h: *const *const u32, // 3*3=9
        // Sub-component inputs - pedersen_points_table
        sub_inputs_ppt: *const *const u32, // 1*1=1 flat ptrs
        // Sub-component inputs - rc_9_9 variants (2 cols per feed)
        sub_inputs_rc_9_9: *const *const u32, // 6*2=12 flat ptrs
        sub_inputs_rc_9_9_b: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_c: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_d: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_e: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_f: *const *const u32, // 6*2=12
        sub_inputs_rc_9_9_g: *const *const u32, // 3*2=6
        sub_inputs_rc_9_9_h: *const *const u32, // 3*2=6
        // Sub-component inputs - rc_20 variants (1 col per feed)
        sub_inputs_rc_20: *const *const u32, // 12*1=12 flat ptrs
        sub_inputs_rc_20_b: *const *const u32, // 12*1=12
        sub_inputs_rc_20_c: *const *const u32, // 12*1=12
        sub_inputs_rc_20_d: *const *const u32, // 12*1=12
        sub_inputs_rc_20_e: *const *const u32, // 9*1=9
        sub_inputs_rc_20_f: *const *const u32, // 9*1=9
        sub_inputs_rc_20_g: *const *const u32, // 9*1=9
        sub_inputs_rc_20_h: *const *const u32, // 9*1=9
        // Inputs
        inputs: *const *const u32, // 86 input columns
        n_rows: u32,               // Number of valid rows
        log_size: u32,             // Log2 of trace size
    );

    /// Generate interaction trace for partial_ec_mul_wb9 from lookup_data.
    pub fn gen_partial_ec_mul_wb9_interaction_trace(
        // Lookup elements for each relation (18 total: pem + ppt + 8 rc_20 + 8 rc_9_9)
        partial_ec_mul_lookup_elements: *mut c_void,
        pedersen_points_table_lookup_elements: *mut c_void,
        rc_20_lookup_elements: *mut c_void,
        rc_20_b_lookup_elements: *mut c_void,
        rc_20_c_lookup_elements: *mut c_void,
        rc_20_d_lookup_elements: *mut c_void,
        rc_20_e_lookup_elements: *mut c_void,
        rc_20_f_lookup_elements: *mut c_void,
        rc_20_g_lookup_elements: *mut c_void,
        rc_20_h_lookup_elements: *mut c_void,
        rc_9_9_lookup_elements: *mut c_void,
        rc_9_9_b_lookup_elements: *mut c_void,
        rc_9_9_c_lookup_elements: *mut c_void,
        rc_9_9_d_lookup_elements: *mut c_void,
        rc_9_9_e_lookup_elements: *mut c_void,
        rc_9_9_f_lookup_elements: *mut c_void,
        rc_9_9_g_lookup_elements: *mut c_void,
        rc_9_9_h_lookup_elements: *mut c_void,
        // Lookup data pointers
        lookup_partial_ec_mul_0: *const *const u32, // 87 arrays
        lookup_partial_ec_mul_1: *const *const u32, // 87 arrays
        lookup_ppt_0: *const *const u32,            // 58 arrays
        lookup_rc_20: *const *const u32,            // 12*2=24 flat ptrs
        lookup_rc_20_b: *const *const u32,          // 12*2=24
        lookup_rc_20_c: *const *const u32,          // 12*2=24
        lookup_rc_20_d: *const *const u32,          // 12*2=24
        lookup_rc_20_e: *const *const u32,          // 9*2=18
        lookup_rc_20_f: *const *const u32,          // 9*2=18
        lookup_rc_20_g: *const *const u32,          // 9*2=18
        lookup_rc_20_h: *const *const u32,          // 9*2=18
        lookup_rc_9_9: *const *const u32,           // 6*3=18 flat ptrs
        lookup_rc_9_9_b: *const *const u32,         // 6*3=18
        lookup_rc_9_9_c: *const *const u32,         // 6*3=18
        lookup_rc_9_9_d: *const *const u32,         // 6*3=18
        lookup_rc_9_9_e: *const *const u32,         // 6*3=18
        lookup_rc_9_9_f: *const *const u32,         // 6*3=18
        lookup_rc_9_9_g: *const *const u32,         // 3*3=9
        lookup_rc_9_9_h: *const *const u32,         // 3*3=9
        // Sizes
        n_rows: u32,
        log_size: u32,
        // Output
        interaction_trace_columns: *const *const u32, // 4*65 = 260 cols
        claimed_sum: *const u32,                      // 4 u32s for qm31
    );

    // ========================================================================
    // pedersen_aggregator_window_bits_9 (234-col native CUDA)
    // ========================================================================

    /// Native CUDA trace generation for pedersen_aggregator_wb9.
    /// Generates 234-col trace, lookup_data, and sub_component_inputs in a single kernel.
    /// Uses the GPU-resident small pedersen table directly.
    pub fn gen_pedersen_aggregator_wb9_trace(
        traces: *const *const u32, // 234 trace output columns
        // Lookup data
        lk_mem_0: *const *const u32, // 30 arrays (memory_id_to_big #0)
        lk_mem_1: *const *const u32, // 30 arrays (memory_id_to_big #1)
        lk_mem_2: *const *const u32, // 30 arrays (memory_id_to_big #2)
        lk_rc8_0: *const *const u32, // 2 arrays (range_check_8 #0)
        lk_rc8_1: *const *const u32, // 2 arrays (range_check_8 #1)
        lk_rc8_2: *const *const u32, // 2 arrays (range_check_8 #2)
        lk_rc8_3: *const *const u32, // 2 arrays (range_check_8 #3)
        lk_pem_0: *const *const u32, // 87 arrays (PEM chain 0 input)
        lk_pem_1: *const *const u32, // 87 arrays (PEM chain 0 output)
        lk_pem_2: *const *const u32, // 87 arrays (PEM chain 1 input)
        lk_pem_3: *const *const u32, // 87 arrays (PEM chain 1 output)
        lk_agg_0: *const *const u32, // 4 arrays (self-lookup)
        mults: *const u32,           // multiplicity data
        // Sub-component inputs
        sub_mem: *const *const u32, // 3 arrays
        sub_rc8: *const *const u32, // 4 arrays
        sub_pem: *const *const u32, // 86 arrays (each 56*trace_size)
        // Inputs
        inputs: *const *const u32, // 3 input columns
        // Memory state
        transpose_big_value_ptr: *const *const u32, // memory_id_to_big transpose ptrs
        small_value_ptr: *const u32,                // memory_id_to_big small values
        // Sizes
        n_rows: u32,
        log_size: u32,
    );

    /// Native CUDA interaction trace generation for pedersen_aggregator_wb9.
    /// Uses lookup data (already on GPU) to compute 6 logup columns entirely on GPU.
    pub fn gen_pedersen_aggregator_wb9_interaction_trace(
        // CommonLookupElements (= LookupElements<128>)
        lookup_elements: *mut std::os::raw::c_void,
        // Lookup data (all device pointers)
        lk_mem_0: *const *const u32, // 30 arrays
        lk_mem_1: *const *const u32, // 30 arrays
        lk_mem_2: *const *const u32, // 30 arrays
        lk_rc8_0: *const *const u32, // 2 arrays
        lk_rc8_1: *const *const u32, // 2 arrays
        lk_rc8_2: *const *const u32, // 2 arrays
        lk_rc8_3: *const *const u32, // 2 arrays
        lk_pem_0: *const *const u32, // 87 arrays
        lk_pem_1: *const *const u32, // 87 arrays
        lk_pem_2: *const *const u32, // 87 arrays
        lk_pem_3: *const *const u32, // 87 arrays
        lk_agg_0: *const *const u32, // 4 arrays
        mults: *const u32,           // multiplicities
        // Sizes
        log_size: u32,
        // Output
        interaction_trace_columns: *const *const u32, // 4*6 = 24 columns
        claimed_sum: *mut u32,                        // 4 m31s for qm31
    );

    // === POSEIDON_AGGREGATOR (342 columns, native CUDA) ===
    pub fn gen_poseidon_aggregator_trace(
        traces: *const u32, // 342 output columns (m31**)
        log_size: u32,
        // Inputs (6 ID arrays + mults)
        input_ids_0: *const u32,
        input_ids_1: *const u32,
        input_ids_2: *const u32,
        input_ids_3: *const u32,
        input_ids_4: *const u32,
        input_ids_5: *const u32,
        mults_in: *const u32,
        // Memory tables
        memory_id_to_big_transposed_big_values: *const *const u32,
        memory_id_to_big_small_values: *const u32,
        // Lookup data outputs (6 memory_id_to_big, 29 elements each)
        lookup_memory_id_to_big_0: *const u32,
        lookup_memory_id_to_big_1: *const u32,
        lookup_memory_id_to_big_2: *const u32,
        lookup_memory_id_to_big_3: *const u32,
        lookup_memory_id_to_big_4: *const u32,
        lookup_memory_id_to_big_5: *const u32,
        // 2 range_check_3_3_3_3_3 (5 elements each)
        lookup_range_check_3_3_3_3_3_0: *const u32,
        lookup_range_check_3_3_3_3_3_1: *const u32,
        // 6 range_check_4_4_4_4 (4 elements each)
        lookup_range_check_4_4_4_4_0: *const u32,
        lookup_range_check_4_4_4_4_1: *const u32,
        lookup_range_check_4_4_4_4_2: *const u32,
        lookup_range_check_4_4_4_4_3: *const u32,
        lookup_range_check_4_4_4_4_4: *const u32,
        lookup_range_check_4_4_4_4_5: *const u32,
        // 3 range_check_4_4 (2 elements each)
        lookup_range_check_4_4_0: *const u32,
        lookup_range_check_4_4_1: *const u32,
        lookup_range_check_4_4_2: *const u32,
        // 10 poseidon_full_round_chain (32 elements each)
        // pfrc 0-7: sub-component feeds, pfrc 8-9: IT boundary (round 4, round 35)
        lookup_poseidon_full_round_chain_0: *const u32,
        lookup_poseidon_full_round_chain_1: *const u32,
        lookup_poseidon_full_round_chain_2: *const u32,
        lookup_poseidon_full_round_chain_3: *const u32,
        lookup_poseidon_full_round_chain_4: *const u32,
        lookup_poseidon_full_round_chain_5: *const u32,
        lookup_poseidon_full_round_chain_6: *const u32,
        lookup_poseidon_full_round_chain_7: *const u32,
        lookup_poseidon_full_round_chain_8: *const u32,
        lookup_poseidon_full_round_chain_9: *const u32,
        // 28 poseidon_3_partial_rounds_chain (42 elements each)
        // p3prc 0-26: sub-component feeds, p3prc 27: IT boundary (round 31)
        lookup_poseidon_3_partial_rounds_chain_0: *const u32,
        lookup_poseidon_3_partial_rounds_chain_1: *const u32,
        lookup_poseidon_3_partial_rounds_chain_2: *const u32,
        lookup_poseidon_3_partial_rounds_chain_3: *const u32,
        lookup_poseidon_3_partial_rounds_chain_4: *const u32,
        lookup_poseidon_3_partial_rounds_chain_5: *const u32,
        lookup_poseidon_3_partial_rounds_chain_6: *const u32,
        lookup_poseidon_3_partial_rounds_chain_7: *const u32,
        lookup_poseidon_3_partial_rounds_chain_8: *const u32,
        lookup_poseidon_3_partial_rounds_chain_9: *const u32,
        lookup_poseidon_3_partial_rounds_chain_10: *const u32,
        lookup_poseidon_3_partial_rounds_chain_11: *const u32,
        lookup_poseidon_3_partial_rounds_chain_12: *const u32,
        lookup_poseidon_3_partial_rounds_chain_13: *const u32,
        lookup_poseidon_3_partial_rounds_chain_14: *const u32,
        lookup_poseidon_3_partial_rounds_chain_15: *const u32,
        lookup_poseidon_3_partial_rounds_chain_16: *const u32,
        lookup_poseidon_3_partial_rounds_chain_17: *const u32,
        lookup_poseidon_3_partial_rounds_chain_18: *const u32,
        lookup_poseidon_3_partial_rounds_chain_19: *const u32,
        lookup_poseidon_3_partial_rounds_chain_20: *const u32,
        lookup_poseidon_3_partial_rounds_chain_21: *const u32,
        lookup_poseidon_3_partial_rounds_chain_22: *const u32,
        lookup_poseidon_3_partial_rounds_chain_23: *const u32,
        lookup_poseidon_3_partial_rounds_chain_24: *const u32,
        lookup_poseidon_3_partial_rounds_chain_25: *const u32,
        lookup_poseidon_3_partial_rounds_chain_26: *const u32,
        lookup_poseidon_3_partial_rounds_chain_27: *const u32,
    );

    pub fn gen_poseidon_aggregator_interaction_trace(
        lookup_elements: *mut std::os::raw::c_void, // opaque CommonLookupElements*
        // 6 memory_id_to_big lookup data
        lookup_memory_id_to_big_0: *const u32,
        lookup_memory_id_to_big_1: *const u32,
        lookup_memory_id_to_big_2: *const u32,
        lookup_memory_id_to_big_3: *const u32,
        lookup_memory_id_to_big_4: *const u32,
        lookup_memory_id_to_big_5: *const u32,
        // 2 rc_3_3_3_3_3
        lookup_range_check_3_3_3_3_3_0: *const u32,
        lookup_range_check_3_3_3_3_3_1: *const u32,
        // 6 rc_4_4_4_4
        lookup_range_check_4_4_4_4_0: *const u32,
        lookup_range_check_4_4_4_4_1: *const u32,
        lookup_range_check_4_4_4_4_2: *const u32,
        lookup_range_check_4_4_4_4_3: *const u32,
        lookup_range_check_4_4_4_4_4: *const u32,
        lookup_range_check_4_4_4_4_5: *const u32,
        // 3 rc_4_4
        lookup_range_check_4_4_0: *const u32,
        lookup_range_check_4_4_1: *const u32,
        lookup_range_check_4_4_2: *const u32,
        // 4 pfrc (for interaction trace)
        lookup_poseidon_full_round_chain_0: *const u32,
        lookup_poseidon_full_round_chain_1: *const u32,
        lookup_poseidon_full_round_chain_2: *const u32,
        lookup_poseidon_full_round_chain_3: *const u32,
        // 2 p3prc (for interaction trace)
        lookup_poseidon_3_partial_rounds_chain_0: *const u32,
        lookup_poseidon_3_partial_rounds_chain_1: *const u32,
        // Base trace column pointers
        base_trace: *const u32,
        log_size: u32,
        // Output
        interaction_trace_columns: *const u32, // 4 * 14 = 56 columns
        claimed_sum: *mut u32,                 // 4 m31s for qm31
    );

}
