const unsigned int P = 0x7fffffff;

struct M31
{
    unsigned int val;

    __device__ M31 add(M31 other)
    {
        unsigned int s = this->val + other.val;
        return M31{(s > P) ? (s - P) : s};
    }

    __device__ M31 sub(M31 other)
    {
        unsigned int s = this->val - other.val;
        return M31{(s > P) ? (s + P) : s};
    }

    __device__ M31 mul(M31 other)
    {
        unsigned long long mul = ((unsigned long long)this->val) * other.val;
        unsigned int h = mul >> 31;
        unsigned int l = mul & P;
        return M31{M31{l}.add(M31{h})};
    }
};

struct CM31
{
    M31 a;
    M31 b;

    __device__ CM31 add(CM31 other)
    {
        return CM31{this->a.add(other.a), this->b.add(other.b)};
    }

    __device__ CM31 sub(CM31 other)
    {
        return CM31{this->a.sub(other.a), this->b.sub(other.b)};
    }

    __device__ CM31 mul(CM31 other)
    {
        M31 ac = this->a.mul(other.a);
        M31 bd = this->b.mul(other.b);
        // Computes (a + b) * (c + d).
        M31 ab_t_cd = this->a.add(this->b).mul(other.a.add(other.b));
        // (ac - bd) + (ad + bc)i.
        return CM31{ac.sub(bd), ab_t_cd.sub(ac.add(bd))};
    }
};

struct QM31
{
    CM31 a;
    CM31 b;

    __device__ QM31 add(QM31 other)
    {
        return QM31{this->a.add(other.a), this->b.add(other.b)};
    }

    __device__ QM31 sub(QM31 other)
    {
        return QM31{this->a.sub(other.a), this->b.sub(other.b)};
    }

    __device__ QM31 mul(QM31 other)
    {
        // Compute using Karatsuba.
        //   (a + ub) * (c + ud) =
        //   (ac + (2+i)bd) + (ad + bc)u =
        //   ac + 2bd + ibd + (ad + bc)u.
        auto ac = this->a.mul(other.a);
        auto bd = this->b.mul(other.b);
        auto bd_times_1_plus_i = CM31{bd.a.sub(bd.b), bd.a.add(bd.b)};
        // Computes ac + bd.
        auto ac_p_bd = ac.add(bd);
        // Computes ad + bc.
        auto ad_p_bc = this->a.add(this->b).mul(other.a.add(other.b)).sub(ac_p_bd);
        // ac + 2bd + ibd =
        // ac + bd + bd + ibd
        auto l = CM31{
            ac_p_bd.a.add(bd_times_1_plus_i.a),
            ac_p_bd.b.add(bd_times_1_plus_i.b),
        };
        return QM31{l, ad_p_bc};
    }
};