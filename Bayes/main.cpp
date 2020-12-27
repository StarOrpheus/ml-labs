#include <iostream>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <utility>
#include <functional>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <deque>

using namespace std;

using precise_t = double;
using precise_vec = vector<precise_t>;
using fraction_t = pair<uint64_t, uint64_t>;
using pfraction_t = pair<precise_t, precise_t>;
using word_list = vector<int>;
using class_id_t = int;

struct input_object
{
    input_object(vector<int> subject,
                 vector<int> body,
                 bool spam)
            : subject(move(subject)),
              body(move(body)),
              spam(spam) {}

    static
    optional<input_object> read_object(filesystem::path const& path)
    {
        try
        {
            string line;
            ifstream in;
            in.exceptions(in.exceptions() | ios::failbit);
            in.open(path);
            getline(in, line);
            auto subject = read_subject(line);

            getline(in, line);
            if (!line.empty())
                throw runtime_error("Second line expected to be empty");

            getline(in, line);
            auto body = read_body(line);

            string fname = path.filename().string();
            bool spam = fname.find("spmsg") != fname.npos;
            return {{move(subject), move(body), spam}};
        }
        catch (exception const& e)
        {
            cerr << "Failed to read file " + path.string()
                 << ": " << e.what()
                 << endl;
            return {};
        }
    }

    word_list const& get_subject() const&
    {
        return subject;
    }

    word_list&& get_subject()&&
    {
        return move(subject);
    }

    word_list const& get_body() const&
    {
        return body;
    }

    word_list&& get_body()&&
    {
        return move(body);
    }

    bool is_spam() const
    {
        return spam;
    }

private:
    static word_list read_subject(string const& line)
    {
        constexpr string_view preamble = "Subject:";
        stringstream in(line);
        string first;
        in >> first;
        if (first != preamble)
            throw runtime_error("Bad file preamble");

        return read_ints(in);
    }

    static word_list read_body(string const& line)
    {
        stringstream in(line);
        return read_ints(in);
    }

    static word_list read_ints(istream& in)
    {
        int k;
        word_list result;
        while (in >> k)
            result.push_back(k);

        return result;
    }

private:
    word_list subject;
    word_list body;
    bool spam;
};

struct datapart
{
    datapart(vector<input_object> objects,
             string name)
            : objects(move(objects)), name(move(name)) {}

    vector<input_object> const& get_objects() const&
    {
        return objects;
    }

    vector<input_object>&& get_objects()&&
    {
        return move(objects);
    }

    string const& get_name() const
    {
        return name;
    }

private:
    vector<input_object> objects;
    string name;
};

using dataparts = vector<datapart>;

struct Bayes
{
    explicit
    Bayes(precise_t alpha = 1.0,
          precise_vec lambdas = {1.0, 1.0})
            : n_classes(lambdas.size()),
              alpha(alpha),
              lambdas(move(lambdas)),
              words_prob(n_classes),
              classes_prob(n_classes, {0, n_classes}),
              distinct_words(n_classes, 0) {}

    void fit(vector<word_list> const& Xs,
             vector<class_id_t> const& Ys)
    {
        assert(Xs.size() == Ys.size());

        for (size_t i = 0; i < Xs.size(); ++i)
        {
            auto& x = Xs[i];
            auto y = Ys[i];

            for (auto&& word : x)
            {
                auto& current_prob = words_prob[y];
                auto it = current_prob.find(word);
                if (it == current_prob.end())
                    current_prob.emplace_hint(it, word, fraction_t{0, 0});
                else
                    it->second.first++;
            }
            classes_prob[y].first++;
        }

        for (int target = 0; target < n_classes; ++target)
        {
            auto d = words_prob[target].size();
            distinct_words[target] = d;
            assert(classes_prob[target].second == n_classes);
            for (auto&&[word, _] : words_prob[target])
            {
                words_prob[target][word].second
                        = classes_prob[target].first + d * alpha;
            }
        }
    }

    std::pair<class_id_t, precise_t> classify(word_list const& Xs)
    {
        vector<pfraction_t> metrics;
        metrics.reserve(classes_prob.size());
        transform(classes_prob.begin(), classes_prob.end(),
                  back_inserter(metrics),
                  [](fraction_t const& fr) -> pfraction_t
                  {
                      return {
                              log((precise_t) fr.first),
                              log((precise_t) fr.second)
                      };
                  });

        for (class_id_t target = 0; target < n_classes; ++target)
        {
            metrics[target].first += log(lambdas[target]);
            for (auto word : Xs)
            {
                auto w_prob = get_prob(target, word);
                metrics[target].first += log(w_prob.first);
                metrics[target].second += log(w_prob.second);
            }

            metrics[target].first -= metrics[target].second;
        }

        auto it = max_element(metrics.begin(), metrics.end());
        return {it - metrics.begin(), it->first};
    }

private:
    fraction_t get_prob(class_id_t target,
                        int word) noexcept
    {
        auto num = alpha;
        auto denum =
                classes_prob[target].first + alpha * distinct_words[target];
        auto& class_prob = words_prob[target];
        if (auto it = class_prob.find(word); it != class_prob.end())
            return it->second;
        else
            return {num, denum};
    }

private:
    size_t n_classes;  //!< Number of classes
    precise_t alpha;      //!< Alpha-argument of the classifier
    precise_vec lambdas;    //!< Lambda-arguments of the classifier

    /// ClassID -> Word -> (X, Y) ~ X/Y - word probability
    vector<unordered_map<int, fraction_t>> words_prob;

    /// ClassID -> (X, Y) ~ X/Y - class probability
    vector<fraction_t> classes_prob;

    /// ClassID -> (X, Y) ~ X/Y - number of distinct word in class ClassID
    vector<size_t> distinct_words;
};

struct ngramm_hook
{
    using ngramm_t = vector<int>;

    explicit ngramm_hook(size_t n) : n(n) {}

    ngramm_hook(ngramm_hook&&) noexcept = default;

    ngramm_hook& operator=(ngramm_hook&&) noexcept = default;

    void train(dataparts const& data)
    {
        for (auto&& part : data)
            for (auto&& obj : part.get_objects())
            {
                auto naive_copy = obj.get_subject();
                naive_copy.reserve(naive_copy.size() + obj.get_body().size());
                copy(obj.get_body().begin(), obj.get_body().end(),
                     back_inserter(naive_copy));

                for (size_t i = 0; i + n <= naive_copy.size(); ++i)
                {
                    ngramm_t naive_ngramm(naive_copy.begin() + i,
                                          naive_copy.begin() + i + n);

                    if (auto it = indexes.find(naive_ngramm);
                            it == indexes.end())
                    {
                        indexes.emplace_hint(it, naive_ngramm, last_index++);
                    }
                }
            }
    }

    pair<vector<word_list>, vector<class_id_t>>
    apply(datapart const& part)
    {
        vector<word_list> Xs;
        vector<class_id_t> Ys;

        for (auto&& obj : part.get_objects())
        {
            word_list result;

            auto naive_copy = obj.get_subject();
            naive_copy.reserve(naive_copy.size() + obj.get_body().size());
            copy(obj.get_body().begin(), obj.get_body().end(),
                 back_inserter(naive_copy));

            for (size_t i = 0; i + n <= naive_copy.size(); ++i)
            {
                ngramm_t naive_ngramm(naive_copy.begin() + i,
                                      naive_copy.begin() + i + n);

                if (auto it = indexes.find(naive_ngramm);
                        it == indexes.end())
                {
                    throw runtime_error("Unknown n-gramm");
                } else
                    result.emplace_back(it->second);
            }
            Xs.push_back(move(result));
            Ys.push_back(obj.is_spam());
        }

        return {move(Xs), move(Ys)};
    }

private:
    ngramm_hook(ngramm_hook const&) = delete;

    ngramm_hook& operator=(ngramm_hook const&) = delete;

private:
    struct hasher_
    {
        size_t operator()(ngramm_t const& ng) const
        {
            return reduce(ng.begin(), ng.end(),
                          (size_t) 0,
                          bit_xor<size_t>());
        }
    };

private:
    size_t n;
    size_t last_index{0};
    unordered_map<ngramm_t, size_t, hasher_> indexes{};
};

using confusion_m_t = vector<vector<int>>;

using hyperparam_t = tuple< precise_t //!< alpha
                          , precise_t //!< lambda[0]
                          >;

using roc_pts_t = std::vector<std::pair<precise_t, precise_t>>;

struct run_result_t
{
    hyperparam_t params;
    size_t N;
    precise_t Accuracy;
    roc_pts_t points;
};

roc_pts_t as_roc_plot(std::vector<std::pair<precise_t, bool>>&& mp, size_t positives)
{
//    std::sort(mp.begin(), mp.end());
    roc_pts_t result;
    result.emplace_back(0., 0.);
    precise_t cur_x{0};
    precise_t cur_y{0};

    for (auto&&[possib, guessed] : mp)
    {
        if (guessed)
            cur_y += 1 / (precise_t) positives;
        else
            cur_x += 1 / (precise_t) (mp.size() - positives);

        result.emplace_back(cur_x, cur_y);
    }
    result.emplace_back(1., 1.);

    return result;
}

int main()
{
    constexpr string_view data_path = "./data";
    dataparts data;

    try
    {
        for (auto it = filesystem::directory_iterator(data_path);
             it != filesystem::directory_iterator();
             it++)
        {
            vector<input_object> msgs;
            auto part_path = it->path();
            for (auto it = filesystem::directory_iterator(part_path);
                 it != filesystem::directory_iterator();
                 it++)
            {
                auto object_path = it->path();
                if (auto obj = input_object::read_object(object_path); obj)
                    msgs.emplace_back(move(*obj));
            }

            data.emplace_back(move(msgs), part_path.filename().string());
        }
        size_t const num_parts = data.size();

        vector<hyperparam_t> run_params;
        vector<run_result_t> run_results;

        for (int pow_alpha = -2; pow_alpha <= 2; ++pow_alpha)
            for (int pow_lambda = 1; pow_lambda <= 60; ++pow_lambda)
                run_params.emplace_back(pow(2, pow_alpha), pow(2, pow_lambda));

        for (size_t ngramm_n = 1; ngramm_n < 4; ++ngramm_n)
        {
            ngramm_hook ngrammer{ngramm_n};
            ngrammer.train(data);

            vector<
                    pair<
                            vector<word_list>,
                            vector<class_id_t>
                    >
            > ngrammed_parts;

            for (auto&& part : data)
                ngrammed_parts.emplace_back(ngrammer.apply(part));

            #pragma omp parallel for
            for (size_t iter = 0; iter < run_params.size(); ++iter)
            {
                auto[alpha, zero_lambda] = run_params[iter];

                confusion_m_t cm(2, vector<int>(2, 0));
                std::vector<std::pair<precise_t, bool>> forecasts;

                for (size_t test_id = 0; test_id < num_parts; test_id++)
                {
                    Bayes model{alpha, precise_vec{zero_lambda, 1}};

                    for (size_t i = 0; i < num_parts; ++i)
                        if (i != test_id)
                        {
                            auto&&[Xs, Ys] = ngrammed_parts[i];
                            model.fit(Xs, Ys);
                        }

                    auto&&[test_Xs, test_Ys] = ngrammed_parts[test_id];

                    for (size_t i = 0; i < test_Xs.size(); ++i)
                    {
                        auto E = test_Ys[i];
                        auto[got, p] = model.classify(test_Xs[i]);
                        cm[E][got]++;
                        forecasts.emplace_back(p, E == got);
                    }
                }

                auto total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];
                auto positives = cm[0][0] + cm[1][1];
                auto Accuracy = positives / (precise_t) total;
                auto roc = as_roc_plot(std::move(forecasts), positives);
                #pragma omp critical
                {
                    run_results.push_back(
                            {run_params[iter], ngramm_n, Accuracy, std::move(roc)});
                    cerr << "\rDone "
                         << run_results.size() / (precise_t) run_params.size()
                         << "      ";
                }
            }
        }
        cerr << endl;

        sort(run_results.begin(), run_results.end(),
             [](auto& lhs, auto& rhs)
             {
                 return lhs.Accuracy > rhs.Accuracy;
             });

        cerr << "Top 5 results: " << endl;
        for (size_t i = 0; i < 5; ++i)
        {
            auto& result = run_results[i];
            cerr << " Ngramm's N=" << result.N
                 << " Alpha=" << get<0>(result.params)
                 << " lambda_0=" << get<1>(result.params)
                 << " Acc=" << result.Accuracy
                 << endl;
        }

        ofstream out;
        out.exceptions(out.exceptions() | ios::failbit);
        out.open("result_roc.dat");

        for (auto&&[x, y] : run_results[0].points)
            out << x << "\t" << y << "\n";
        out.flush();
    }
    catch (exception const& e)
    {
        cerr << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
