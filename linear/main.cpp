#include <istream>
#include <ostream>

#include "vec.hpp"

using precise_t = double;

struct training_object
{
    training_object(size_t n, size_t m)
        : features(n),
          target(m)
    {}

    vec<precise_t> features;
    vec<precise_t> target;
};

void solution(std::istream& in,
              std::ostream& out)
{
    std::vector<training_object> objects;
    size_t n, m;

    in >> n >> m;

    objects.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        objects.emplace_back(m, 1);
        for (size_t j = 0; j < m; ++j)
            in >> objects.back().features[j];
        in >> objects.back().target[0];
    }
}

int main()
{

}