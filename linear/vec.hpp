#pragma once

#include <cstddef>
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
        : impl(std::move(other))
    {}

    vec& operator=(vec&& other) noexcept
    {
        impl = std::move(other.impl);
    }

    vec& operator+=(vec const& other)
    {
        assert(impl.size() == other.size());

        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] += other.impl[i];
    }

    vec& operator-=(vec const& other)
    {
        assert(impl.size() == other.size());
        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] -= other.impl[i];
    }

    template<typename Mult>
    vec& operator*=(Mult d)
    {
        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] *= d;
    }

    template<typename Mult>
    vec& operator/=(Mult d)
    {
        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] /= d;
    }

    template<typename Mult>
    vec& operator*=(vec const& other)
    {
        assert(impl.size() == other.size());

        #pragma omp simd
        for (size_t i = 0; i < impl.size(); ++i)
            impl[i] *= other.impl[i];
    }

    T& operator[](size_t index)
    {
        return impl[index];
    }

    T operator[](size_t index) const
    {
        return impl[index];
    }

private:
    std::vector<T> impl;
};
