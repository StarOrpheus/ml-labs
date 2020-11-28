#pragma once

#include <cstddef>
#include <iomanip>
#include <vector>

template<typename T>
struct vec
{
    template<typename ForwardIt>
    vec(ForwardIt start, ForwardIt last)
        : impl(start, last)
    {}

    vec(size_t n)
        : impl(n)
    {}

    vec(vec&& other) noexcept
        : impl(std::move(other.impl))
    {}

    size_t size() const
    noexcept(noexcept(std::declval<std::vector<T>>().size()))
    {
        return impl.size();
    }

    template<typename T2>
    void fill(T2&& val)
    {
        std::fill(impl.begin(), impl.end(), std::forward<T2>(val));
    }

    vec copy() const
    {
        return vec(impl.begin(), impl.end());
    }

    vec& operator=(vec&& other) noexcept
    {
        impl = std::move(other.impl);
        return *this;
    }

    vec& operator+=(vec const& other)
        noexcept(noexcept(std::declval<std::vector<T>>()[0]))
    {
        assert(impl.size() == other.size());

        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] += other.impl[i];
    }

    vec& operator-=(vec const& other)
        noexcept(noexcept(std::declval<std::vector<T>>()[0]))
    {
        assert(impl.size() == other.size());
        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] -= other.impl[i];
        return *this;
    }

    template<typename Mult>
    vec& operator*=(Mult d)
    noexcept(noexcept(std::declval<std::vector<T>>()[0]))
    {
        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] *= d;
        return *this;
    }

    template<typename Mult>
    vec& operator/=(Mult d)
        noexcept(noexcept(std::declval<std::vector<T>>()[0]))
    {
        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] /= d;
        return *this;
    }

    /// Scalar multiplication of vectors
    T operator*(vec const& other) const
    {
        assert(other.size() == size());
        return std::transform_reduce(impl.begin(), impl.end(), other.impl.begin(), T{0});
    }

    T& operator[](size_t index)
        noexcept(noexcept(std::declval<std::vector<T>>()[index]))
    {
        return impl[index];
    }

    T operator[](size_t index) const
        noexcept(noexcept(std::declval<std::vector<T>>()[index]))
    {
        return impl[index];
    }

    T back() const
        noexcept(noexcept(std::declval<std::vector<T>>().back()))
    {
        return impl.back();
    }


    friend std::ostream& operator<<(std::ostream& out, vec const& v)
    {
        auto first = v.cbegin();
        auto last = v.cend();
        for (auto it = first; it != last; ++it)
        {
            out << std::fixed << std::setprecision(9) << v[it - first];
            if ((it + 1) != last)
                out << " ";
        }
        return out;
    }

private:
    std::vector<T> impl;

public:
    using iterator = typename decltype(impl)::iterator;
    using const_iterator = typename decltype(impl)::const_iterator;

    iterator begin()
    {
        return impl.begin();
    }

    iterator end()
    {
        return impl.end();
    }

    const_iterator cbegin() const
    {
        return impl.cbegin();
    }

    const_iterator cend() const
    {
        return impl.cend();
    }
};
