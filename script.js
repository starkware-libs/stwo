function fibonacci(n) {
  if (n === 0) return 0;
  if (n === 1) return 1;

  let a = 0n;
  let b = 1n;

  for (let i = 2; i <= n; i++) {
    let c = a + b;
    a = b;
    b = c;
  }

  return b;
}
const huge = fibonacci(65537);
const M31 = 2147483647n;
console.log("f(65537) mod M31 =", (huge % M31).toString());
