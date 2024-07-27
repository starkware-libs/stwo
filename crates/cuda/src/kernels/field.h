const unsigned int P = 0x7fffffff;

__device__ unsigned int add31(unsigned int a, unsigned int b)
{
    unsigned int s = a + b;
    return (s > P) ? (s - P) : s;
}

// TODO: Do this better.
__device__ unsigned int mul31(unsigned int a, unsigned int b)
{
    unsigned long long mul = ((unsigned long long)a) * b;
    unsigned int h = mul >> 31;
    unsigned int l = mul & P;
    return add31(l, h);
}
