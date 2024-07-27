const unsigned int P = 0x7fffffff;

struct M31
{
    unsigned int val;

    __device__ M31 add(M31 other)
    {
        unsigned int s = this->val + other.val;
        return M31{(s > P) ? (s - P) : s};
    }

    __device__ M31 mul(M31 other)
    {
        unsigned long long mul = ((unsigned long long)this->val) * other.val;
        unsigned int h = mul >> 31;
        unsigned int l = mul & P;
        return M31{M31{l}.add(M31{h})};
    }
};
