#include <istream>
#include <ostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <string>
#include <sstream>

using precise_t = double;

using feature_vec = std::vector<precise_t>;

precise_t scalar(feature_vec const& lhs,
                 feature_vec const& rhs)
{
    precise_t sum = 0; // std::transform_reduce is not available on CF
    assert(lhs.size() == rhs.size());
//    #pragma omp simd
    for (size_t i = 0; i < lhs.size(); ++i)
        sum += lhs[i] * rhs[i];
    return sum;
}

void muls(feature_vec& lhs,
          precise_t rhs)
{
//    #pragma omp simd
    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] *= rhs;
}

std::ostream& operator<<(std::ostream& out, feature_vec const& v)
{
    auto first = v.cbegin();
    auto last = v.cend();
    for (auto it = first; it != last; ++it)
    {
        out << std::fixed << std::setprecision(9) << *it;
        if ((it + 1) != last)
            out << " ";
    }
    return out;
}

struct training_object
{
    feature_vec features;
    precise_t target;
};

struct single_run_result
{
    precise_t   s; // snape error result
    feature_vec w;
};

precise_t nz(precise_t x)
{
    if (x == 0)
        return 1.0;
    else
        return x;
}

/// The SMAPE error function
precise_t smape(std::vector<training_object> const& objects,
                feature_vec const& W)
{
    precise_t result = 0;
    for (auto&& obj : objects)
    {
        auto cur_target = scalar(obj.features, W);
        result += std::abs(cur_target - obj.target)
                    / nz(std::abs(cur_target) + std::abs(obj.target));
    }
    result /= objects.size();
    return result;
}

precise_t copy_sign(precise_t x_, precise_t y)
{
    auto x = std::abs(x_);
    if (y < 0)
        return -x;
    else
        return x;
}

precise_t dSMAPE2(feature_vec const& x,
                  precise_t E,
                  feature_vec const& W,
                  size_t i)
{
    assert(i < x.size());
    auto F = scalar(W, x);
    auto num = x[i] * (copy_sign(1, F - E) * (std::abs(F) + std::abs(E)) - copy_sign(F - E, F));
    auto a = std::abs(F) + std::abs(E);
    auto den = nz(a * a);
    return num / den;
}

feature_vec solution(std::vector<training_object> objects,
                     size_t m)
{
    std::mt19937_64 rnd_dev{1337228};
    std::uniform_real_distribution<> dis((precise_t) -1.0, (precise_t) 1.0);

    single_run_result result{1000, feature_vec{0}};

    size_t run_cnt = 5;
    auto start = std::chrono::system_clock::now();
    for (size_t run_id = 0; run_id < run_cnt; ++run_id)
    {
        std::shuffle(objects.begin(), objects.end(), rnd_dev);

        feature_vec W(m);
        std::generate(W.begin(), W.end(), std::bind(dis, rnd_dev));
        muls(W, 1 / (2 * (precise_t) m));

        for (size_t iter = 0; iter < 100; ++iter)
        {
            training_object& obj = objects[iter % objects.size()];
            for (size_t i = 0; i < m; ++i)
            {
                auto mu = 1 / (precise_t) (iter + 1);
//                precise_t d1 = dSMAPE1(obj.features, obj.target, W, i);
                precise_t d1 = dSMAPE2(obj.features, obj.target, W, i);
                W[i] -= mu * d1;
            }
        }

        assert(W.size() == m);
        auto current_S = smape(objects, W);

        if (current_S < result.s)
        {
            result.s = current_S;
            result.w = std::move(W);
        }

        if (std::chrono::system_clock::now() - start >= std::chrono::milliseconds(300))
            break;
    }

    return result.w;
}

void read_objects(std::istream& in,
                  std::vector<training_object>& objects,
                  size_t m, size_t n)
{
    objects.resize(n);
    for (auto&& obj : objects)
    {
        obj.features.reserve(m + 1);
        obj.features.resize(m);
        for (auto&& x : obj.features)
            in >> x;
        obj.features.push_back(1);
        in >> obj.target;
    }
}

int main(int argc, char** argv)
{
    std::ios_base::sync_with_stdio(false);

    if (argc == 2 && std::string(argv[1]) == "test")
    {
//        #pragma omp parallel for
//        for (size_t i = 1; i <= 8; ++i)
//        {
//            auto test_start = std::chrono::system_clock::now();
//
//            std::ifstream in("assets/test " + std::to_string(i) + ".txt");
//
//            size_t m, n1, n2;
//            std::vector<training_object> train_objects;
//            std::vector<training_object> test_objects;
//
//            in >> m >> n1;
//            read_objects(in, train_objects, m, n1);
//            in >> n2;
//            read_objects(in, test_objects, m, n2);
//
//            ++m;
//
//            auto W = solution(train_objects, m);
//
//            std::chrono::duration<double> dur = std::chrono::system_clock::now() - test_start;
//
//            auto smape_train = smape(train_objects, W);
//            auto smape_test = smape(test_objects, W);
//
//            std::stringstream ss;
//            ss << "#############\n"
//               << "test #" << i << " train_objects_num=" << n1 << " test_objects_num=" << n2 << "\n"
//               << "smape on train set = " << smape_train << "\n"
//               << "smape on test set = " << smape_test << "\n"
//               << "solution run took " << dur.count() << "s";
//            std::cout << std::move(ss).str() << std::endl;
//        }
    }
    else
    {
        size_t n, m;
        std::vector<training_object> objects;
        std::cin >> n >> m;
        read_objects(std::cin, objects, m, n);
        ++m;
        auto W = solution(objects, m);
        std::cout << W << std::endl;
    }
}