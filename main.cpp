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

constexpr enum class f_score {
    MICRO_F,
    MACRO_F
} f_score_type = f_score::MACRO_F;

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

std::ostream& operator<<(std::ostream& os,
                         training_obj const& obj)
{
    os << "{!";
    for (auto&& x : obj.Xs)
        os << x << " ";
    os << "->";
    for (auto&& y : obj.Ys)
        os << " " << y;
    os << "!}";
    return os;
}

enum class distance_l
{
    MANHATTAN,
    EUCLIDEAN,
    CHEBYSHEV
};

distance_l str_to_dist_label(std::string const& str)
{
    static const std::unordered_map<std::string, distance_l> mp{
            {"manhattan", distance_l::MANHATTAN},
            {"euclidean", distance_l::EUCLIDEAN},
            {"chebyshev", distance_l::CHEBYSHEV}
    };

    auto it = mp.find(str);
    assert(it != mp.end());
    return it->second;
}

enum class kernel_l
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

kernel_l str_to_kernel_label(std::string const& str)
{
    static const std::unordered_map<std::string, kernel_l> mp{
            {"uniform",      kernel_l::UNIFORM},
            {"triangular",   kernel_l::TRIANGULAR},
            {"epanechnikov", kernel_l::EPANECHNIKOV},
            {"quartic",      kernel_l::QUARTIC},
            {"triweight",    kernel_l::TRIWEIGHT},
            {"tricube",      kernel_l::TRICUBE},
            {"gaussian",     kernel_l::GAUSSIAN},
            {"cosine",       kernel_l::COSINE},
            {"logistic",     kernel_l::LOGISTIC},
            {"sigmoid",      kernel_l::SIGMOID}
    };

    auto it = mp.find(str);
    assert(it != mp.end());
    return it->second;
}

enum class window_l
{
    FIXED_WINDOW,
    FLOATING_WINDOW
};

window_l str_to_window_label(std::string const& str)
{
    static const std::unordered_map<std::string, window_l> mp{
            {"fixed",    window_l::FIXED_WINDOW},
            {"variable", window_l::FLOATING_WINDOW},
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

distance_function get_distance_fn(distance_l label)
{
    switch (label)
    {
        case distance_l::MANHATTAN:
            return [](auto&& lhs, auto&& rhs) -> precise_t
            {
                precise_t result = 0;
                assert(lhs.size() == rhs.size());
                for (size_t i = 0; i < lhs.size(); ++i)
                    result += std::abs(lhs[i] - rhs[i]);
                return result;
            };
        case distance_l::EUCLIDEAN:
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
        case distance_l::CHEBYSHEV:
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

kernel_function get_kernel_function(kernel_l label)
{
    switch (label)
    {
        case kernel_l::UNIFORM:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? 1 / precise_t(2) : 0;
            };
        case kernel_l::TRIANGULAR:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? 1 - precise_t(temp) : 0;
            };
        case kernel_l::EPANECHNIKOV:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? ((precise_t) 3 / 4) * (1 - temp * temp) : 0;
            };
        case kernel_l::QUARTIC:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                precise_t t2 = 1 - temp * temp;
                return (temp < 1) ? ((precise_t) 15 / 16) * t2 * t2 : 0;
            };
        case kernel_l::TRIWEIGHT:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                precise_t t2 = 1 - temp * temp;
                return (temp < 1) ? ((precise_t) 35 / 32) * t2 * t2 * t2 : 0;
            };
        case kernel_l::TRICUBE:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                precise_t t2 = 1 - temp * temp * temp;
                return (temp < 1) ? ((precise_t) 70 / 81) * t2 * t2 * t2 : 0;
            };
        case kernel_l::GAUSSIAN:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (1 / (std::sqrt(2 * Pi)))
                       * std::exp(((precise_t) -1 / 2) * temp * temp);
            };
        case kernel_l::COSINE:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return (temp < 1) ? Pi / 4 * std::cos(Pi * temp / 2) : 0;
            };
        case kernel_l::LOGISTIC:
            return [](precise_t dist, precise_t window_p) -> precise_t
            {
                auto temp = dist / window_p;
                return 1 / (std::exp(temp) + 2 + std::exp(-temp));
            };
        case kernel_l::SIGMOID:
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
                              distance_l distance_label,
                              kernel_l kernel_label,
                              window_l window_label,
                              size_t window_k,
                              object_features_t const& q)
{
    assert(!objects.empty());
    for (auto&& obj : objects)
    {
        assert(obj.Ys.size() == objects.back().Ys.size());
        (void) obj;
    }
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
        case window_l::FIXED_WINDOW:
            window_p = window_k;
            break;
        case window_l::FLOATING_WINDOW:
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

struct result_accumulator_t
{
    precise_t  f_score{0};
    kernel_l   kernel_label{kernel_l::UNIFORM};
    distance_l distance_label{distance_l::EUCLIDEAN};
    window_l   win_label{window_l::FIXED_WINDOW};
    size_t     window_param{0};
};

void accumulate(result_accumulator_t& a,
                precise_t f_score,
                kernel_l kern_l,
                distance_l dist_l,
                window_l win_l,
                size_t win_p)
{
    if (a.f_score < f_score)
    {
        a.f_score        = f_score;
        a.kernel_label   = kern_l;
        a.distance_label = dist_l;
        a.win_label      = win_l;
        a.window_param   = win_p;
    }
}

std::vector<training_obj> prepare_dataset()
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
                Ys = {0.0, 0.0, 1.0};
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
    return objects;
}

int get_class(object_features_t const& features)
{
    return std::max_element(features.begin(), features.end()) - features.begin();
}

std::ostream& operator<<(std::ostream& os,
                         kernel_l label)
{
    switch (label)
    {
        case kernel_l::UNIFORM:
            os << "uniform";
            break;
        case kernel_l::TRIANGULAR:
            os << "triangular";
            break;
        case kernel_l::EPANECHNIKOV:
            os << "epanechnikov";
            break;
        case kernel_l::QUARTIC:
            os << "quartic";
            break;
        case kernel_l::TRIWEIGHT:
            os << "triweight";
            break;
        case kernel_l::TRICUBE:
            os << "tricube";
            break;
        case kernel_l::GAUSSIAN:
            os << "gaussian";
            break;
        case kernel_l::COSINE:
            os << "cosine";
            break;
        case kernel_l::LOGISTIC:
            os << "logistic";
            break;
        case kernel_l::SIGMOID:
            os << "sigmoid";
            break;
        default:
            assert(false);
    }
    return os;
}

std::ostream& operator<<(std::ostream& os,
                         window_l label)
{
    switch (label)
    {
        case window_l::FIXED_WINDOW:
            os << "fixed_window";
            break;
        case window_l::FLOATING_WINDOW:
            os << "floating_window";
            break;
        default:
            assert(false);
    }
    return os;
}

std::ostream& operator<<(std::ostream& os,
                         distance_l label)
{
    switch (label)
    {
        case distance_l::MANHATTAN:
            os << "manhattan";
            break;
        case distance_l::EUCLIDEAN:
            os << "euclidean";
            break;
        case distance_l::CHEBYSHEV:
            os << "chebyshev";
            break;
        default:
            assert(false);
    }
    return os;
}

int main()
{
    try
    {
        auto const objects_gold = prepare_dataset();

        std::array const kernel_labels{
            kernel_l::UNIFORM,
            kernel_l::TRIANGULAR,
            kernel_l::QUARTIC
        };

        std::array const dist_labels{
            distance_l::EUCLIDEAN,
            distance_l::MANHATTAN,
            distance_l::CHEBYSHEV
        };
        result_accumulator_t accumulator;

        size_t D = sqrt(objects_gold.size());

        /// Default value is num of fixed window params
        size_t tasks_cnt = 2 * D * kernel_labels.size() * dist_labels.size();
        size_t tasks_done = 0;

        std::cerr << "\rDone " << ((double) tasks_done / (double) tasks_cnt) * 100 << "%    ";

        #pragma omp parallel for
        for (size_t kn = 1; kn <= D; ++kn)
        {
            auto objects = objects_gold;
            for (auto kern_l : kernel_labels)
                for (auto dist_l : dist_labels)
                {
                    confusion_m_t confusion_m(3, std::vector<int>(3, 0));
                    for (size_t i = 0; i < objects.size(); ++i)
                    {
                        std::swap(objects[i], objects.back());

                        auto q = std::move(objects.back());
                        objects.pop_back();

                        auto predicted = knn_predict(
                            objects, dist_l, kern_l,
                            window_l::FIXED_WINDOW, kn, q.Xs
                        );
                        
                        auto expected = get_class(q.Ys);
                        auto got = get_class(predicted);
                        confusion_m[expected][got]++;
                        objects.push_back(std::move(q));

                        std::swap(objects[i], objects.back());
                    }

                    precise_t score = 0;
                    switch (f_score_type)
                    {
                    case f_score::MICRO_F:
                        score = micro_macro_f_scores(confusion_m).first;
                        break;
                    case f_score::MACRO_F:
                        score = micro_macro_f_scores(confusion_m).second;
                        break;
                    default:
                        assert(false);
                    }

                    #pragma omp critical
                    {
                        accumulate(
                            accumulator, score, kern_l, dist_l,
                            window_l::FIXED_WINDOW, kn
                        );
                        ++tasks_done;
                        std::cerr << "\rDone " << ((double) tasks_done / (double) tasks_cnt) * 100 << "%    ";
                    }
                }
        }

        for (auto dist_l : dist_labels)
        {
            auto dist_fun = get_distance_fn(dist_l);
            precise_t R = 0;
            for (size_t i = 0; i < objects_gold.size(); ++i)
                for(size_t j = 0; j < objects_gold.size(); ++j)
                    R = std::max(R, dist_fun(objects_gold[i].Xs, objects_gold[j].Xs));

            precise_t step = R / std::sqrt(objects_gold.size());
            #pragma omp parallel for
            for (size_t iter = 1; iter <= D; ++iter)
            {
                auto param = iter * step;
                auto objects = objects_gold;
                for (auto kern_l : kernel_labels)
                {
                    confusion_m_t confusion_m(3, std::vector<int>(3, 0));
                    for (size_t i = 0; i < objects.size(); ++i)
                    {
                        std::swap(objects[i], objects.back());

                        auto q = std::move(objects.back());
                        objects.pop_back();

                        auto predicted = knn_predict(
                            objects, dist_l, kern_l,
                            window_l::FLOATING_WINDOW, param, q.Xs
                        );

                        auto expected = get_class(q.Ys);
                        auto got = get_class(predicted);
                        confusion_m[expected][got]++;
                        objects.push_back(std::move(q));

                        std::swap(objects[i], objects.back());
                    }

                    precise_t score = 0;
                    switch (f_score_type)
                    {
                        case f_score::MICRO_F:
                            score = micro_macro_f_scores(confusion_m).first;
                            break;
                        case f_score::MACRO_F:
                            score = micro_macro_f_scores(confusion_m).second;
                            break;
                        default:
                            assert(false);
                    }

                    #pragma omp critical
                    {
                        accumulate(
                            accumulator, score, kern_l, dist_l,
                            window_l::FLOATING_WINDOW, param
                        );
                        ++tasks_done;
                        std::cerr << "\rDone " << ((double) tasks_done / (double) tasks_cnt) * 100 << "%    ";
                    }
                }
            }
        }

        std::cerr << "\rDone " << ((double) tasks_done / (double) tasks_cnt) * 100 << "%    ";
        std::cerr << std::endl;

        std::cout << "Best score: " << accumulator.f_score << std::endl;
        std::cout << accumulator.kernel_label << " "
                  << accumulator.distance_label << " "
                  << accumulator.win_label << "~" << accumulator.window_param
                  << std::endl;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
