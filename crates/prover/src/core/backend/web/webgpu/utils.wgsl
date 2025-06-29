// This shader contains utility functions for bit manipulation and index transformations.
// It is stateless and can be used as a library in other shaders.

/// Returns the bit reversed index of `i` which is represented by `log_size` bits.
fn bit_reverse_index(i: u32, log_size: u32) -> u32 {
    if (log_size == 0u) {
        return i;
    }
    let bits = reverse_bits_u32(i);
    return bits >> (32u - log_size);
}

fn reverse_bits_u32(x: u32) -> u32 {
    var x_mut = x;
    var result = 0u;
    
    for (var i = 0u; i < 32u; i = i + 1u) {
        result = (result << 1u) | (x_mut & 1u);
        x_mut = x_mut >> 1u;
    }
    
    return result;
}

/// Returns the index of the offset element in a bit reversed circle evaluation
/// of log size `eval_log_size` relative to a smaller domain of size `domain_log_size`.
fn offset_bit_reversed_circle_domain_index(
    i: u32,
    domain_log_size: u32,
    eval_log_size: u32,
    offset: i32,
) -> u32 {
    var prev_index = bit_reverse_index(i, eval_log_size);
    let half_size = 1u << (eval_log_size - 1u);
    let step_size = i32(1u << (eval_log_size - domain_log_size - 1u)) * offset;
    
    if (prev_index < half_size) {
        let temp = i32(prev_index) + step_size;
        // Implement rem_euclid for positive modulo
        let m = i32(half_size);
        let rem = temp % m;
        prev_index = u32(select(rem + m, rem, rem >= 0));
    } else {
        let temp = i32(prev_index - half_size) - step_size;
        // Implement rem_euclid for positive modulo
        let m = i32(half_size);
        let rem = temp % m;
        prev_index = u32(select(rem + m, rem, rem >= 0)) + half_size;
    }
    
    return bit_reverse_index(prev_index, eval_log_size);
}

fn circle_domain_index_to_coset_index(i: u32, n: u32) -> u32 {
    if (i < (n / 2u)) {
        return 2u * i;
    } else {
        return 2u * (n - 1u - i) + 1u;
    }
}

fn coset_index_to_circle_domain_index(coset_index: u32, log_domain_size: u32) -> u32 {
    if (coset_index % 2u == 0u) {
        return coset_index / 2u;
    } else {
        return ((2u << log_domain_size) - coset_index) / 2u;
    }
}