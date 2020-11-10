#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include <functional>
#include <cmath>

#include "csv.hpp"

using confusion_m_t = std::vector<std::vector<int>>;
using precise_t = long double;

constexpr double Pi = 3.14159265358979323846;

constexpr static precise_t PRECISION = 1e-9;

auto get_class_total(confusion_m_t const& cm, size_t index)
{
    return std::accumulate(cm[index].begin(), cm[index].end(), 0);
}

std::pair<precise_t, precise_t>
get_precision_and_recall(confusion_m_t const& cm, size_t index)
{
    int tp_fp = 0;
    for (size_t j = 0; j < cm.size(); ++j)
        tp_fp += cm[j][index];
    auto tp_fn = get_class_total(cm, index);
    return {
            (tp_fp == 0) ? 0 : cm[index][index] / (precise_t) tp_fp,
            (tp_fn == 0) ? 0 : cm[index][index] / (precise_t) tp_fn
    };
}

precise_t f_score(precise_t precision, precise_t recall)
{
    auto sum = precision + recall;
    return (std::abs(sum) < PRECISION)
           ? 0
           : 2 * precision * recall / sum;
}

precise_t f_score(confusion_m_t const& cm, size_t index)
{
    auto[precision, recall] = get_precision_and_recall(cm, index);
    return f_score(precision, recall);
}

std::pair<precise_t, precise_t>
micro_macro_f_scores(confusion_m_t const& cm)
{
    int total = 0;
    precise_t macro_f_score = 0, micro_f_score = 0;
    precise_t micro_precision = 0, micro_recall = 0;

    for (size_t i = 0; i < cm.size(); ++i)
    {
        auto class_total = get_class_total(cm, i);
        auto[precision, recall] = get_precision_and_recall(cm, i);
        macro_f_score += f_score(cm, i) * class_total;
        micro_precision += precision * class_total;
        micro_recall += recall * class_total;
        total += class_total;
    }

    macro_f_score /= total;
    micro_precision /= total;
    micro_recall /= total;

    micro_f_score = f_score(micro_precision, micro_recall);

    return {micro_f_score, macro_f_score};
}

using object_feature_t = precise_t;
using object_features_t = std::vector<object_feature_t>;

struct training_obj
{
    object_features_t Xs;
    object_features_t Ys;
};

enum distance_label_t
{
    MANHATTAN,
    EUCLIDEAN,
    CHEBYSHEV
};

distance_label_t str_to_dist_label(std::string const& str)
{
    static const std::unordered_map<std::string, distance_label_t> mp{
            {"manhattan", MANHATTAN},
            {"euclidean", EUCLIDEAN},
            {"chebyshev", CHEBYSHEV}
    };

    auto it = mp.find(str);
    assert(it != mp.end());
    return it->second;
}

enum kernel_label_t
{
    UNIFORM,
    TRIANGULAR,
    EPANECHNIKOV,
    QUARTIC,
    TRIWEIGHT,
    TRICUBE,
    GAUSSIAN,
    COSINE,
    LOGISTIC,
    SIGMOID
};

kernel_label_t str_to_kernel_label(std::string const& str)
{
    static const std::unordered_map<std::string, kernel_label_t> mp{
            {"uniform",      UNIFORM},
            {"triangular",   TRIANGULAR},
            {"epanechnikov", EPANECHNIKOV},
            {"quartic",      QUARTIC},
            {"triweight",    TRIWEIGHT},
            {"tricube",      TRICUBE},
            {"gaussian",     GAUSSIAN},
            {"cosine",       COSINE},
            {"logistic",     LOGISTIC},
            {"sigmoid",      SIGMOID}
    };

    auto it = mp.find(str);
    assert(it != mp.end());
    return it->second;
}

enum window_label_t
{
    FIXED_WINDOW,
    FLOATING_WINDOW
};

window_label_t str_to_window_label(std::string const& str)
{
    static const std::unordered_map<std::string, window_label_t> mp{
            {"fixed",    FIXED_WINDOW},
            {"variable", FLOATING_WINDOW},
    };

    auto it = mp.find(str);
    assert(it != mp.end());
    return it->second;
}

using distance_function = std::function<
        precise_t(object_features_t const&, object_features_t const&)
>;

using kernel_function = std::function<
        precise_t(precise_t, precise_t)
>;

distance_function get_distance_fn(distance_label_t label)
{
    switch (label)
    {
        case MANHATTAN:
            return [](auto&& lhs, auto&& rhs) -> precise_t
            {
                precise_t result = 0;
                assert(lhs.size() == rhs.size());
                for (size_t i = 0; i < lhs.size(); ++i)
                    result += std::abs(lhs[i] - rhs[i]);
                return result;
            };
        case EUCLIDEAN:
            return [](auto&& lhs, auto&& rhs) -> precise_t
            {
                precise_t result = 0;
                assert(lhs.size() == rhs.size());
                for (size_t i = 0; i < lhs.size(); ++i)
                {
                    precise_t temp = lhs[i] - rhs[i];
                    result += temp * temp;
                }
                return std::sqrt(result);
            };
        case CHEBYSHEV:
            return [](auto&& lhs, auto&& rhs) -> precise_t
            {
                precise_t result = -0.1;
                assert(lhs.size() == rhs.size());
                for (size_t i = 0; i < lhs.size(); ++i)
                    result = std::max(
                            result, (precise_t) std::abs(lhs[i] - rhs[i]));
                return result;
            };
        default:
            assert(false);
            exit(-1);
    }
}

kernel_function get_kernel_function(kernel_label_t label)
{
    switch (label)
    {
        case UNIFORM:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? 1 / precise_t(2) : 0;
            };
        case TRIANGULAR:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? 1 - precise_t(temp) : 0;
            };
        case EPANECHNIKOV:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? ((precise_t) 3 / 4) * (1 - temp * temp) : 0;
            };
        case QUARTIC:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                precise_t t2 = 1 - temp * temp;
                return (temp < 1) ? ((precise_t) 15 / 16) * t2 * t2 : 0;
            };
        case TRIWEIGHT:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                precise_t t2 = 1 - temp * temp;
                return (temp < 1) ? ((precise_t) 35 / 32) * t2 * t2 * t2 : 0;
            };
        case TRICUBE:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                precise_t t2 = 1 - temp * temp * temp;
                return (temp < 1) ? ((precise_t) 70 / 81) * t2 * t2 * t2 : 0;
            };
        case GAUSSIAN:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (1 / (std::sqrt(2 * Pi)))
                       * std::exp(((precise_t) -1 / 2) * temp * temp);
            };
        case COSINE:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? Pi / 4 * std::cos(Pi * temp / 2) : 0;
            };
        case LOGISTIC:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return 1 / (std::exp(temp) + 2 + std::exp(-temp));
            };
        case SIGMOID:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return 2 / (Pi * (std::exp(temp) + std::exp(-temp)));
            };
        default:
            assert(false);
            exit(-1);
    }
}

bool equals(object_features_t const& lhs, object_features_t const& rhs)
{
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i)
        if (std::abs(lhs[i] - rhs[i]) < PRECISION)
            return false;
    return true;
}

void add(object_features_t& lhs, object_features_t const& rhs)
{
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] += rhs[i];
}

void sub(object_features_t& lhs, object_features_t const& rhs)
{
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] -= rhs[i];
}

void div(object_features_t& lhs, precise_t d)
{
    for (auto& lh : lhs)
        lh /= d;
}

void mul(object_features_t& lhs, precise_t d)
{
    for (auto& lh : lhs)
        lh *= d;
}

object_features_t knn_predict(std::vector<training_obj> objects,
                              distance_label_t distance_label,
                              kernel_label_t kernel_label,
                              window_label_t window_label,
                              size_t window_k,
                              object_features_t const& q)
{
    for (auto&& obj : objects)
        assert(obj.Ys.size() == objects.back().Ys.size());
    size_t target_sz = objects.back().Ys.size();

    auto distance_fn = get_distance_fn(distance_label);
    auto kernel_fn = get_kernel_function(kernel_label);

    std::sort(
            objects.begin(), objects.end(),
            [&q, &distance_fn]
             (auto&& lhs, auto&& rhs)
            {
                return distance_fn(lhs.Xs, q) < distance_fn(rhs.Xs, q);
            }
    );

    precise_t window_p;
    switch (window_label)
    {
        case FIXED_WINDOW:
            window_p = window_k;
            break;
        case FLOATING_WINDOW:
            window_p = distance_fn(objects.at(window_k).Xs, q);
            break;
        default:
            assert(false);
            throw std::runtime_error("Bad window label");
    }

    if (std::abs(window_p) < PRECISION)
    {
        object_features_t result(target_sz, 0);
        if (equals(objects[0].Xs, q))
        {
            int cnt = 0;
            for (auto&& obj : objects)
            {
                if (!equals(obj.Xs, q))
                    break;
                add(result, obj.Ys);
                cnt++;
            }
            div(result, cnt);
        }
        else
        {
            for (auto&& obj : objects)
                add(result, obj.Ys);
            div(result, objects.size());
        }
        return result;
    }
    else
    {
        object_features_t X(target_sz, 0);
        object_features_t y_sum(target_sz, 0);
        precise_t Y = 0;

        for (auto&& obj : objects)
        {
            auto kernel_call = kernel_fn(distance_fn(obj.Xs, q), window_p);
            auto ys = obj.Ys;
            mul(ys, kernel_call);
            add(X, ys);
            Y += kernel_call;
            add(y_sum, obj.Ys);
        }

        if (std::abs(Y) < PRECISION)
        {
            div(y_sum, objects.size());
            return y_sum;
        }
        else
        {
            div(X, Y);
            return X;
        }
    }
}

void min(object_features_t& lhs,
         object_features_t const& rhs)
{
    assert(lhs.size() == rhs.size());
    for(size_t i = 0; i < lhs.size(); ++i)
        lhs[i] = lhs[i] < rhs[i] ? lhs[i] : rhs[i];
}

void max(object_features_t& lhs,
         object_features_t const& rhs)
{
    assert(lhs.size() == rhs.size());
    for(size_t i = 0; i < lhs.size(); ++i)
        lhs[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
}

void divps(object_features_t& lhs,
           object_features_t const& rhs)
{
    assert(lhs.size() == rhs.size());
    for(size_t i = 0; i < lhs.size(); ++i)
        lhs[i] /= rhs[i];
}

int main()
{
    csv::CSVReader rdr("assets/wine_dataset.csv");
    std::vector<training_obj> objects;
    objects.reserve(rdr.n_rows());

    for (auto&& line : rdr)
    {
        object_features_t Xs, Ys;

        auto it = line.begin();
        auto target = it->get<int>();
        switch (target)
        {
            case 1:
                Ys = {1.0, 0.0, 0.0};
                break;
            case 2:
                Ys = {0.0, 1.0, 0.0};
                break;
            case 3:
                Ys = {1.0, 0.0, 1.0};
                break;
            default:
                assert(false);
                throw std::runtime_error("Bad target class: " + std::to_string(target));
        }
        it++;

        for (; it != line.end(); ++it)
            Xs.emplace_back(it->get<precise_t>());

        objects.push_back({Xs, Ys});
    }

    auto mins = objects.back().Xs;
    auto maxs = objects.back().Xs;
    for (auto&& obj : objects)
    {
        min(mins, obj.Xs);
        max(maxs, obj.Xs);
    }
    sub(maxs, mins);

    for (auto& obj : objects)
    {
        sub(obj.Xs, mins);
        divps(obj.Xs, maxs);
    }

    for (auto&& obj : objects)
    {
        std::cout << "Xs: ";
        for (auto&& x : obj.Xs)
            std::cout << x << " ";
        std::cout << "\nYs: ";
        for (auto&& y : obj.Ys)
            std::cout << y << " ";
        std::cout << std::endl << std::endl;
    }

    std::array kernel_labels{UNIFORM, TRIANGULAR, QUARTIC};
    std::array dist_labels{EUCLIDEAN, MANHATTAN, CHEBYSHEV};

    std::tuple<precise_t, kernel_label_t, distance_label_t, window_label_t, size_t>
        best_option{0, UNIFORM, EUCLIDEAN, FIXED_WINDOW, 0};

    for (auto kernel_l : kernel_labels)
        for (auto dist_l : dist_labels)
            for (size_t k_neighbors = 1; k_neighbors < std::sqrt(objects.size()); ++k_neighbors)
            {

            }


}