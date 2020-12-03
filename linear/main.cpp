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
#include <map>


#ifdef ENABLE_TESTS
    #include <filesystem>
#endif

using precise_t = double;

using feature_vec = std::vector<precise_t>;

precise_t scalar(feature_vec const& lhs,
                 feature_vec const& rhs)
{
    precise_t sum = 0; // std::transform_reduce is not available on CF
    assert(lhs.size() == rhs.size());

    for (size_t i = 0; i < lhs.size(); ++i)
        sum += lhs[i] * rhs[i];
    return sum;
}

void muls(feature_vec& lhs,
          precise_t rhs)
{
    #pragma omp simd
    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] *= rhs;
}

void subss(feature_vec& lhs,
           feature_vec const& rhs)
{
    assert(lhs.size() == rhs.size());

    #pragma omp simd
    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] -= rhs[i];
}

void divss(feature_vec& lhs,
           feature_vec const& rhs)
{
    assert(lhs.size() == rhs.size());

    #pragma omp simd
    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] /= rhs[i];
}

void minss(feature_vec& sum,
           feature_vec const& Xs)
{
    assert(sum.size() == Xs.size());

    #pragma omp simd
    for (size_t i = 0; i < sum.size(); ++i)
        if (Xs[i] < sum[i])
            sum[i] = Xs[i];
}

void maxss(feature_vec& sum,
           feature_vec const& Xs)
{
    assert(sum.size() == Xs.size());

    #pragma omp simd
    for (size_t i = 0; i < sum.size(); ++i)
        if (Xs[i] > sum[i])
            sum[i] = Xs[i];
}

std::ostream& operator<<(std::ostream& out, feature_vec const& v)
{
    auto first = v.cbegin();
    auto last = v.cend();
    for (auto it = first; it != last; ++it)
    {
        out << std::fixed << std::numeric_limits<precise_t>::digits10 << *it;
        if ((it + 1) != last)
            out << " ";
    }
    return out;
}

struct training_object
{
    feature_vec features;
    precise_t   target;
};

struct single_run_result
{
    precise_t   s; //!< smape error result
    feature_vec w; //!< achieved weights
};

precise_t nz(precise_t x)
{
    return (x == 0) ? 1.0 : x;
}

size_t sample_size(std::vector<training_object> const& objects)
{
    return objects.size();
//    return std::min(objects.size(), (size_t) 100u);
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

precise_t dLoss(feature_vec const &x,
                precise_t E,
                size_t i,
                precise_t F)
{
    assert(i < x.size());
    auto num = x[i] * (copy_sign(std::abs(F) + std::abs(E), F - E) - copy_sign(F - E, F));
    auto a = std::abs(F) + std::abs(E);
    auto den = nz(a * a);
    return num / den;
}

struct normalizer
{
    explicit normalizer(std::vector<training_object> const& objs) noexcept
    {
        assert(objs.size());
        x_mins = objs[0].features;
        x_diff = objs[0].features;
        y_min = objs[0].target;
        y_diff = objs[0].target;
        for (size_t i = 1; i < objs.size(); ++i)
        {
            minss(x_mins, objs[i].features);
            maxss(x_diff, objs[i].features);
            if (objs[i].target < y_min)
                y_min = objs[i].target;
            if (objs[i].target > y_diff)
                y_diff = objs[i].target;
        }

        subss(x_diff, x_mins);
        for (auto&& x : x_diff)
            x = nz(x);
        y_diff -= y_min;
        y_diff = nz(y_diff);
    }

    void normalize(std::vector<training_object>& objs)
    {
        for (size_t i = 0; i < objs.size(); ++i)
        {
            auto& x = objs[i];
            subss(x.features, x_mins);
            divss(x.features, x_diff);

            x.target -= y_min;
            x.target /= y_diff;
        }
    }

    feature_vec denormalize(feature_vec const& w_norm)
    {
        feature_vec w(w_norm.size());
        
        for (size_t i = 0; i < w_norm.size(); ++i)
            w[i] = w_norm[i] * y_diff / x_diff[i];

        precise_t sum = 0;
        for (size_t i = 0; i < w_norm.size() - 1; ++i)
            sum += w_norm[i] * y_diff * x_mins[i] / x_diff[i];
        w.back() += y_min - sum;
        return w;
    }

private:
    feature_vec x_mins;
    feature_vec x_diff;
    precise_t   y_min;
    precise_t   y_diff;
};

struct test_run_context;

feature_vec solution(test_run_context& context,
                     std::vector<training_object> objects,
                     size_t m,
                     std::vector<std::pair<size_t, precise_t>>* plot_data)
{
    normalizer norm(objects);
    norm.normalize(objects);

    std::mt19937_64 rnd_dev{1437687};
    std::uniform_real_distribution<> W_init_dis(-0.5 / (precise_t) m, 0.5 / (precise_t) m);
    std::uniform_int_distribution<size_t> sample_dis((size_t) 0, objects.size() - 1);

    single_run_result result{1000, feature_vec(m)};

    size_t run_cnt = 100;
    auto start = std::chrono::system_clock::now();
    feature_vec W(m, 0);
    std::vector<precise_t> grad(m);

    std::vector<std::pair<size_t, precise_t>> current_plot;
    for (size_t run_id = 0; run_id < run_cnt; ++run_id)
    {
        current_plot.clear();
        std::generate(W.begin(), W.end(), std::bind(W_init_dis, rnd_dev));
        for (size_t iter = 0; iter < 2000; ++iter)
        {
            training_object& obj = objects[sample_dis(rnd_dev)];
            bool brk = true;
            auto learning_rate = 1 / (precise_t) (iter + 1000);
            auto Forecast = scalar(W, obj.features);
            precise_t regularization_tau = 0.009;

            for (size_t i = 0; i < m; ++i)
            {
                precise_t d1 = dLoss(obj.features, obj.target, i, Forecast);
                grad[i] = learning_rate * d1;
                if (grad[i] != 0)
                    brk = false;
            }
            muls(W, 1 - learning_rate * regularization_tau);
            subss(W, grad);

            if (plot_data && iter % 10 == 0)
                current_plot.emplace_back(iter, smape(objects, W));
        }

        assert(W.size() == m);
        auto current_S = smape(objects, W);
        if (current_S < result.s)
        {
            result.s = current_S;
            std::swap(result.w, W);
            if (plot_data)
                std::swap(current_plot, *plot_data);
        }
    }
    return norm.denormalize(result.w);
}

void read_objects(FILE* in,
                  std::vector<training_object>& objects,
                  size_t m, size_t n)
{
    int t;
    objects.resize(n);
    for (auto&& obj : objects)
    {
        obj.features.reserve(m + 1);
        obj.features.resize(m);
        for (auto&& x : obj.features)
        {
            fscanf(in, "%d", &t);
            x = t;
        }

        obj.features.push_back(1);
        fscanf(in, "%d", &t);
        obj.target = t;
    }
}

#ifdef ENABLE_TESTS
struct test_run_context
{
    precise_t smape_sum_on_test{0.0};
    precise_t smape_sum_on_train{0.0};
    std::map<std::filesystem::path, std::string> results{};

    test_run_context& operator>>=(test_run_context&& rhs)
    {
        smape_sum_on_train += rhs.smape_sum_on_train;
        smape_sum_on_test += rhs.smape_sum_on_test;
        std::move(rhs.results.begin(), rhs.results.end(), std::inserter(results, results.end()));
        return *this;
    }
};

test_run_context run_test(std::filesystem::path const& path)
{
    auto test_start = std::chrono::system_clock::now();

    FILE* in = fopen(path.c_str(), "r");
    size_t m, n1, n2;
    std::vector<training_object> train_objects;
    std::vector<training_object> test_objects;

    fscanf(in,"%zu %zu", &m, &n1);
    read_objects(in, train_objects, m, n1);
    fscanf(in,"%zu", &n2);
    read_objects(in, test_objects, m, n2);

    fclose(in);
    ++m;

    test_run_context context;
    std::vector<std::pair<size_t, precise_t>> plot_data;

    auto W = solution(context, train_objects, m, &plot_data);

    std::chrono::duration<double> dur
            = std::chrono::system_clock::now() - test_start;

    auto smape_train = smape(train_objects, W);
    auto smape_test = smape(test_objects, W);

    context.smape_sum_on_test += smape_test;
    context.smape_sum_on_train += smape_train;

    std::stringstream ss;
    ss << "#############\n"
       << "test " << path << " train_objects_num=" << n1 << " test_objects_num=" << n2 << "\n"
       << "Loss on train set = " << smape_train << "\n"
       << "Loss on test set = " << smape_test << "\n"
       << "solution run took " << dur.count() << "s\n";
    if (W.size() <= 5)
        ss << "Weights: " << W << "\n";
    context.results.emplace(path.string(), std::move(ss).str());

    std::error_code ec;
    auto filename = path.filename().string() + ".data";
    auto parent_path = std::filesystem::path("plots");
    std::filesystem::create_directories(parent_path);
    auto data_path = parent_path / filename;
    std::filesystem::remove(data_path, ec);
    std::ofstream plot(data_path);
    for (auto&& [x, y] : plot_data)
        plot << x << "\t" << std::fixed << std::setprecision(std::numeric_limits<precise_t>::max_digits10) << y << "\n";

    return context;
}
#endif

int main(int argc, char** argv)
{
//    std::ios_base::sync_with_stdio(false);

    if (argc == 2 && std::string(argv[1]) == "test")
    {
        try
        {
            std::vector<std::filesystem::path> suites = { "tests/"/*, "simple_tests/" */ };
            for (auto&& suite : suites)
            {
                std::vector<std::string> files;
                test_run_context context;

                std::transform(std::filesystem::directory_iterator(suite),
                               std::filesystem::directory_iterator(),
                               std::back_inserter(files),
                               [] (auto&& f) { return f.path(); });

                #pragma omp parallel for
                for (size_t i = 0; i < files.size(); ++i)
                {
                    auto run_result = run_test(files[i]);
                    #pragma omp critical
                    {
                        context >>= std::move(run_result);
                    }
                }

                for (auto&& result : context.results)
                    std::cout << result.second;

                size_t num_tests = files.size();
                std::cout << "#############\n"
                          << "Overall suite score for " << suite << ":\n"
                          << "\t" << num_tests - context.smape_sum_on_train << "/" << (precise_t) num_tests << " on train\n"
                          << "\t" << num_tests - context.smape_sum_on_test << "/" << (precise_t) num_tests << " on test"
                          << std::endl << std::endl;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            return -1;
        }
    }
}