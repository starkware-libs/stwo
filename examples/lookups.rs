#![feature(array_windows, exact_size_is_empty, array_chunks)]
#![allow(dead_code, unused_variables, unused_imports)]

use std::array;
use std::iter::{repeat, successors, zip, Sum};
use std::ops::{Add, Mul, Neg, Sub};
use std::process::Output;
use std::time::Instant;

use num_traits::{One, Zero};
use prover_research::core::channel;
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::ExtensionField;
use prover_research::core::fields::Field;
use rand::{thread_rng, Rng};

fn main() {
    const LOG_N: u32 = 17;

    // Create a new 16-bit range check pool with 2^LOG_N values.
    let rc_pool = {
        let mut pool = RangeCheckPool::new();
        let mut rng = thread_rng();

        for _ in 0..1 << LOG_N {
            pool.push(rng.gen());
        }

        pool
    };

    let (ordered_values, _) = rc_pool.get_ordered_values_with_padding();
    let unordered_values = rc_pool.get_unordered_values_with_padding();

    let ordered_values = ordered_values.into_iter().map(|v| [v]).collect();
    let unordered_values = unordered_values.into_iter().map(|v| [v]).collect();

    let grand_product = GrandProduct::new(ordered_values, unordered_values);

    let challenge = ExtensionField::from_m31(
        BaseField::from_u32_unchecked(183912),
        BaseField::from_u32_unchecked(183912),
        BaseField::from_u32_unchecked(9283535),
        BaseField::from_u32_unchecked(23489483),
    );

    let now = Instant::now();
    let cumulative_product = grand_product.gen_commutative_product_column([challenge]);
    println!("Grand product: {:?}", now.elapsed());
    assert_eq!(cumulative_product.last(), Some(&ExtensionField::one()));

    let multiplicities = rc_pool.get_multiplicities().to_vec();
    let ordered_values = (0..=u16::MAX).map(|v| [BaseField::from(v)]).collect();
    let unordered_values = rc_pool
        .get_unordered_values()
        .into_iter()
        .map(|v| [v])
        .collect();

    let log_up = LogUp::new(ordered_values, unordered_values, multiplicities);

    let now = Instant::now();
    let cumulative_sum = log_up.gen_commutative_sum_column::<32>([challenge]);
    println!("LogUp: {:?}", now.elapsed());
    assert_eq!(cumulative_sum.last(), Some(&ExtensionField::zero()));

    let mut summed_reciprocals = log_up.get_summed_reciprocals([challenge]);
    summed_reciprocals.resize(summed_reciprocals.len().next_power_of_two(), Zero::zero());

    let cumulative = summed_reciprocals
        .iter()
        .copied()
        .sum::<ProjectiveFraction<ExtensionField>>();
    println!("{:?}", cumulative);
    println!("{:?}", cumulative.is_zero());

    let gkr_prover = GkrProver::new(summed_reciprocals);

    println!("Res: {}", GkrVerifier::verify(&gkr_prover));
}

struct GkrProver {
    layers: Vec<Vec<ProjectiveFraction<ExtensionField>>>,
}

impl GkrProver {
    fn new(bottom_layer: Vec<ProjectiveFraction<ExtensionField>>) -> Self {
        assert!(bottom_layer.len().is_power_of_two());

        let layers = successors(Some(bottom_layer), |prev_layer| {
            if prev_layer.len() > 1 {
                // Generate the next layer (of half the size) by summing neighbors.
                Some(prev_layer.array_chunks().map(|&[a, b]| a + b).collect())
            } else {
                None
            }
        })
        .collect();

        Self { layers }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn get_layer(&self, i: usize) -> &Vec<ProjectiveFraction<ExtensionField>> {
        &self.layers[self.num_layers() - i - 1]
    }

    /// Returns the evaluation of `p_k(x) + lambda * q_k(x)` where `k` is the `layer_index`.
    ///
    /// Output of the form `(p_k(challenges), q_k(challenges)`
    ///
    /// <https://eprint.iacr.org/2023/1284.pdf> (Page 4)
    fn evaluate_layer(
        &self,
        layer_index: usize,
        challenges: &[ExtensionField],
        // lambda: ExtensionField,
    ) -> (ExtensionField, ExtensionField) {
        assert_eq!(layer_index, challenges.len());

        let layer = self.get_layer(layer_index);
        let next_layer = self.get_layer(layer_index + 1);

        let mut p_eval = ExtensionField::zero();
        let mut q_eval = ExtensionField::zero();

        let hypercube = BooleanHypercube::new(layer_index as u32);

        for (i, y) in hypercube.iter().enumerate() {
            let neighbour_sum = next_layer[i * 2] + next_layer[i * 2 + 1];
            let lagrange_kernel_eval = eval_lagrange_kernel(challenges, &y);

            p_eval += lagrange_kernel_eval * neighbour_sum.numerator;
            q_eval += lagrange_kernel_eval * neighbour_sum.denominator;
        }

        // p_eval + lambda * q_eval
        (p_eval, q_eval)
    }

    fn circuit_output(&self) -> ProjectiveFraction<ExtensionField> {
        self.layers.last().unwrap()[0]
    }
}

struct GkrVerifier;

impl GkrVerifier {
    fn verify(prover: &GkrProver) -> bool {
        let num_layers = prover.num_layers();

        let mut rng = thread_rng();
        let challenges = (0..num_layers)
            .map(|_| rng.gen())
            .collect::<Vec<ExtensionField>>();

        // Prover sends output of the circle p_0 and q_0.
        let circuit_output = prover.circuit_output();
        println!("Claimed circuit output: {:?}", circuit_output);

        // Verifier checks the point fraction is zero.
        if !circuit_output.is_zero() {
            return false;
        }

        // First round:
        // Verifier wants to be sent p_1(1) and p_1(-1) also q_1(1) and q_1(-1)
        // Since p (and q) is linear then alpha * p(x) = p(alpha * x)
        //
        // (1 + x0 * y0) / 2
        // (1 + x0 * 1) / 2
        let (p_one, q_one) = prover.evaluate_layer(1, &[ExtensionField::one()]);
        let (p_neg_one, q_neg_one) = prover.evaluate_layer(1, &[-ExtensionField::one()]);
        println!("p_val: {}", p_one + (challenges[0] * p_neg_one));

        let (p_expected, _) = prover.evaluate_layer(1, &[ExtensionField::one() - challenges[0]]);
        println!("p_val: {}", p_expected);

        // let one = ExtensionField::one();
        // println!(
        //     "Hmmm: {}",
        //     eval_lagrange_kernel(&[one], &[one])
        //         + challenges[0] * eval_lagrange_kernel(&[-one], &[-one])
        // );
        // println!(
        //     "Hmmm: {}",
        //     eval_lagrange_kernel(&[one - challenges[0]], &[one])
        // );

        false
    }
}

/// Boolean hypercube with values in `{+-1}^dimension`.
#[derive(Debug, Clone, Copy)]
struct BooleanHypercube {
    dimension: u32,
}

impl BooleanHypercube {
    fn new(dimension: u32) -> Self {
        Self { dimension }
    }

    fn iter(self) -> impl Iterator<Item = Vec<ExtensionField>> {
        (0..1 << self.dimension).map(move |i| {
            (0..self.dimension)
                .map(|bit| {
                    if i & (1 << bit) != 0 {
                        ExtensionField::one()
                    } else {
                        -ExtensionField::one()
                    }
                })
                .collect()
        })
    }
}

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// When y is an elements of `{+-1}^n` then this function evaluates the Lagrange polynomial which is
/// the unique multilinear polynomial equal to 1 if `x = y` and equal to 0 whenever x is an element
/// of `{+-1}^n`.
///
/// From: <https://eprint.iacr.org/2023/1284.pdf>.
fn eval_lagrange_kernel(
    x_assignments: &[ExtensionField],
    y_assignments: &[ExtensionField],
) -> ExtensionField {
    assert_eq!(x_assignments.len(), y_assignments.len());

    let n = x_assignments.len();
    let norm = BaseField::from_u32_unchecked(2u32.pow(n as u32));
    println!("n: {}", n);

    zip(x_assignments, y_assignments)
        .map(|(&x, &y)| ExtensionField::one() + x * y)
        .product::<ExtensionField>()
        / norm
}

struct RangeCheckPool {
    values: Vec<u16>,
}

impl RangeCheckPool {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn push(&mut self, v: u16) {
        self.values.push(v);
    }

    /// Calculates the multiplicities of each element in the pool.
    ///
    /// Each index of the output array corresponds to a value, and the value at that index
    /// represents the number of occurrences of that value in the pool.
    fn get_multiplicities(&self) -> Box<[BaseField; 1 << u16::BITS]> {
        let mut res = [0; 1 << u16::BITS];
        self.values.iter().for_each(|&v| res[v as usize] += 1);
        Box::new(res.map(BaseField::from_u32_unchecked))
    }

    /// Returns an ordered list of the values in the pool with padding so elements are contiguous.
    ///
    /// Output is of the form `(ordered_vals_with_padding, padding_vals)`.
    fn get_ordered_values_with_padding(&self) -> (Vec<BaseField>, Vec<BaseField>) {
        let mut ordered_vals = self.values.clone();
        ordered_vals.sort_unstable();

        // range check values need to be continuos therefore any gaps
        // e.g. [..., 3, 4, 7, 8, ...] need to be filled with [5, 6] as padding.
        let mut padding_vals = Vec::new();
        for &[a, b] in ordered_vals.array_windows() {
            for v in u16::saturating_add(a, 1)..b {
                padding_vals.push(v);
            }
        }

        // Add padding to the ordered vals (res)
        for v in &padding_vals {
            ordered_vals.push(*v);
        }

        // re-sort the values.
        ordered_vals.sort_unstable();

        (
            ordered_vals.into_iter().map(BaseField::from).collect(),
            padding_vals.into_iter().map(BaseField::from).collect(),
        )
    }

    /// Returns unordered pool values with gap padding values appended to the end.
    fn get_unordered_values_with_padding(&self) -> Vec<BaseField> {
        let (_, padding) = self.get_ordered_values_with_padding();
        self.values
            .iter()
            .copied()
            .map(BaseField::from)
            .chain(padding)
            .collect()
    }

    /// Returns an ordered list of pool values without padding.
    fn get_ordered_values(&self) -> Vec<BaseField> {
        let mut ordered_vals = self.values.clone();
        ordered_vals.sort_unstable();
        ordered_vals.into_iter().map(BaseField::from).collect()
    }

    /// Returns an ordered list of pool values without padding.
    fn get_unordered_values(&self) -> Vec<BaseField> {
        self.values.iter().copied().map(BaseField::from).collect()
    }
}

struct GrandProduct<const N_COLS: usize> {
    ordered_rows: Vec<[BaseField; N_COLS]>,
    unordered_rows: Vec<[BaseField; N_COLS]>,
}

impl<const N_COLS: usize> GrandProduct<N_COLS> {
    fn new(
        ordered_rows: Vec<[BaseField; N_COLS]>,
        unordered_rows: Vec<[BaseField; N_COLS]>,
    ) -> Self {
        assert_eq!(ordered_rows.len(), unordered_rows.len());
        Self {
            ordered_rows,
            unordered_rows,
        }
    }

    fn gen_commutative_product_column(
        &self,
        challenges: [ExtensionField; N_COLS],
    ) -> Vec<ExtensionField> {
        let mut numerators = Vec::new();
        let mut denominators = Vec::new();

        let mut numerator_acc = ExtensionField::one();
        let mut denominators_acc = ExtensionField::one();

        for (unordered_row, ordered_row) in zip(&self.unordered_rows, &self.ordered_rows) {
            numerator_acc *= eval_lookup_instance(unordered_row, &challenges);
            denominators_acc *= eval_lookup_instance(ordered_row, &challenges);
            numerators.push(numerator_acc);
            denominators.push(denominators_acc);
        }

        let mut cumulative_product_column = Vec::new();

        for (numerator, denominator_inv) in zip(numerators, batch_inverse(denominators)) {
            cumulative_product_column.push(numerator * denominator_inv);
        }

        cumulative_product_column
    }
}

struct LogUp<const N_COLS: usize> {
    /// Multiplicities of the ordered rows that appear in the unordered rows.
    multiplicities: Vec<BaseField>,
    ordered_rows: Vec<[BaseField; N_COLS]>,
    unordered_rows: Vec<[BaseField; N_COLS]>,
}

impl<const N_COLS: usize> LogUp<N_COLS> {
    fn new(
        ordered_rows: Vec<[BaseField; N_COLS]>,
        unordered_rows: Vec<[BaseField; N_COLS]>,
        multiplicities: Vec<BaseField>,
    ) -> Self {
        assert_eq!(multiplicities.len(), ordered_rows.len());
        Self {
            multiplicities,
            ordered_rows,
            unordered_rows,
        }
    }

    /// Output is of the form `(unordered_reciprocals, ordered_reciprocals)`
    ///
    /// These are the helper columns `h(x) = 1/(z - t(x))` and `h_1(x) = 1/(z - w_1(x))`:
    /// <https://eprint.iacr.org/2023/1284.pdf> (page 2).
    fn gen_reciprocal_columns(
        &self,
        challenges: [ExtensionField; N_COLS],
    ) -> (Vec<ExtensionField>, Vec<ExtensionField>) {
        let unordered_instances = self
            .unordered_rows
            .iter()
            .map(|row| eval_lookup_instance(row, &challenges))
            .collect();

        // Compute all `1 / unordered_instance_i`.
        let unordered_reciprocals = batch_inverse(unordered_instances);

        let ordered_instances = self
            .ordered_rows
            .iter()
            .map(|row| eval_lookup_instance(row, &challenges))
            .collect();

        // Compute all `1 / ordered_instance_i`.
        let ordered_reciprocals = batch_inverse(ordered_instances);

        (unordered_reciprocals, ordered_reciprocals)
    }

    fn get_summed_reciprocals(
        &self,
        challenges: [ExtensionField; N_COLS],
    ) -> Vec<ProjectiveFraction<ExtensionField>> {
        let (unordered_reciprocals, ordered_reciprocals) = self.gen_reciprocal_columns(challenges);

        let mut unordered_reciprocals =
            unordered_reciprocals
                .into_iter()
                .map(|denominator| ProjectiveFraction {
                    numerator: ExtensionField::one(),
                    denominator,
                });
        let mut ordered_reciprocals =
            zip(&self.multiplicities, ordered_reciprocals).map(|(&numerator, denominator)| {
                ProjectiveFraction {
                    numerator: numerator.into(),
                    denominator,
                }
            });

        let mut summed_reciprocals = Vec::new();

        while !unordered_reciprocals.is_empty() || !ordered_reciprocals.is_empty() {
            let unordered_reciprocal = unordered_reciprocals.next().unwrap_or(Zero::zero());
            let ordered_reciprocal = ordered_reciprocals.next().unwrap_or(Zero::zero());

            summed_reciprocals.push(unordered_reciprocal - ordered_reciprocal)
        }

        summed_reciprocals
    }

    /// Cumulative sum is batched. `CHUNK_SIZE` many instances are aggregated together at each cell.
    /// Possible since unlike permutation product adding doesn't increase the constraint degree.
    fn gen_commutative_sum_column<const CHUNK_SIZE: usize>(
        &self,
        challenges: [ExtensionField; N_COLS],
    ) -> Vec<ExtensionField> {
        let (unordered_reciprocals, ordered_reciprocals) = self.gen_reciprocal_columns(challenges);

        let mut unordered_reciprocals = unordered_reciprocals.into_iter();
        let mut ordered_reciprocals = zip(ordered_reciprocals, &self.multiplicities)
            .map(|(reciprocal, &multiplicity)| reciprocal * multiplicity);

        let mut cumulative_sum_column = Vec::new();
        let mut acc = ExtensionField::zero();

        while !unordered_reciprocals.is_empty() || !ordered_reciprocals.is_empty() {
            let unordered_sum = (&mut unordered_reciprocals)
                .take(CHUNK_SIZE)
                .sum::<ExtensionField>();

            let ordered_sum = (&mut ordered_reciprocals)
                .take(CHUNK_SIZE)
                .sum::<ExtensionField>();

            acc += unordered_sum - ordered_sum;

            cumulative_sum_column.push(acc);
        }

        cumulative_sum_column
    }
}

#[derive(Debug, Clone, Copy)]
struct ProjectiveFraction<T> {
    numerator: T,
    denominator: T,
}

impl<T: Copy + Add<Output = T> + Mul<Output = T>> Add for ProjectiveFraction<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            numerator: self.numerator * rhs.denominator + rhs.numerator * self.denominator,
            denominator: self.denominator * rhs.denominator,
        }
    }
}

impl<T: Copy + Sub<Output = T> + Mul<Output = T>> Sub for ProjectiveFraction<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            numerator: self.numerator * rhs.denominator - rhs.numerator * self.denominator,
            denominator: self.denominator * rhs.denominator,
        }
    }
}

impl<T: Neg<Output = T>> Neg for ProjectiveFraction<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            numerator: self.numerator,
            denominator: -self.denominator,
        }
    }
}

impl<T: Mul<Output = T>> Mul<T> for ProjectiveFraction<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            numerator: self.numerator * rhs,
            denominator: self.denominator,
        }
    }
}

impl<T: Mul<Output = T>> Mul for ProjectiveFraction<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            numerator: self.numerator * rhs.numerator,
            denominator: self.denominator * rhs.denominator,
        }
    }
}

impl<T: Zero + One> Zero for ProjectiveFraction<T>
where
    ProjectiveFraction<T>: Add<Output = ProjectiveFraction<T>>,
{
    fn zero() -> Self {
        Self {
            numerator: T::zero(),
            denominator: T::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero() && !self.denominator.is_zero()
    }
}

impl<T> Sum for ProjectiveFraction<T>
where
    ProjectiveFraction<T>: Zero,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

/// Evaluates an instance of a lookup argument.
///
/// Let `[v_1, ..., v_n]` represent row and `[z, alpha_2, ..., alpha_n]` represent challenges sent
/// by a verifier. Returns `z - (v_1 + alpha_2 * v_2 + ... + alpha_n * v_n)`.
fn eval_lookup_instance<const N: usize>(
    row: &[BaseField; N],
    challenges: &[ExtensionField; N],
) -> ExtensionField {
    challenges[0]
        - row[0]
        - zip(&challenges[1..], &row[1..])
            .map(|(&alpha, &v)| alpha * v)
            .sum::<ExtensionField>()
}

/// Computes the inverse of all items in `v`.
///
/// Inversions are expensive but their cost can be amortized by batching inversions together.
// TODO: move to utils
fn batch_inverse<F: Field, U: AsMut<[F]>>(mut v: U) -> U {
    // 1. `[1, a, ab, abc]`
    let mut acc = F::one();
    let n = v.as_mut().len();
    let mut prods = Vec::with_capacity(n);
    for (v, prod) in zip(v.as_mut(), prods.spare_capacity_mut()) {
        prod.write(acc);
        acc *= *v;
    }

    // SAFETY: all values have been initialized
    unsafe { prods.set_len(n) }

    // 2. `1/abcd`
    let mut acc_inv = acc.inverse();

    // 3. `[1/a, a/ab, ab/abc, abc/abcd] = [1/a, 1/b, 1/c, 1/d]`
    for (v, prod) in zip(v.as_mut().iter_mut().rev(), prods.into_iter().rev()) {
        let acc_inv_next = *v * acc_inv;
        *v = acc_inv * prod;
        acc_inv = acc_inv_next;
    }

    v
}
