#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    constexpr float distanceRatioThreshold = 0.75f;

    for (const std::vector<cv::DMatch>& match : matches) {
        if (match.size() > 1) {
            if (match[0].distance / match[1].distance <= distanceRatioThreshold) {
                filtered_matches.push_back(match.front());
            }
        } else if (match.size() == 1) {
            filtered_matches.push_back(match.front());
        } else {
            std::cerr << "Descriptor matcher ratio test: got empty match?" << std::endl;
        }
    }
}

void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches,
                                                   int iterations)
{
    filtered_matches.clear();

    const size_t  total_neighbours  = 5;  // total number of neighbours to test (including candidate)
    const size_t  consistent_matches  = 3;  // minimum number of consistent matches (including candidate)
    const float  radius_limit_scale  = 2.f;  // limit search radius by scaled median

    const int n_matches = matches.size();

    if (n_matches < total_neighbours) {
        throw std::runtime_error("DescriptorMatcher::filterMatchesClusters : too few matches");
    }

    cv::Mat points_query(n_matches, 2, CV_32FC1);
    cv::Mat points_train(n_matches, 2, CV_32FC1);
    for (int i = 0; i < n_matches; ++i) {
        points_query.at<cv::Point2f>(i) = keypoints_query[matches[i].queryIdx].pt;
        points_train.at<cv::Point2f>(i) = keypoints_train[matches[i].trainIdx].pt;
    }

    // размерность всего 2, так что точное KD-дерево
    // NOTE: Точное дерево, может, и можно построить, но похоже, что OpenCV
    // в этом случае этого всё-таки не делает. Для большей точности лишь
    // используем увеличенное количество проверок (на результаты тестов не особо
    // влияет, но при использовании нескольких итераций кластерного фильтра
    // увеличенное число проверок действительно даёт более стабильный результат).
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(4);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(64);

    std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
    std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);

    // для каждой точки найти total neighbors ближайших соседей
    cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_query(n_matches, total_neighbours, CV_32FC1);
    cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_train(n_matches, total_neighbours, CV_32FC1);

    index_query->knnSearch(points_query, indices_query, distances2_query, total_neighbours, *search_params);
    index_train->knnSearch(points_train, indices_train, distances2_train, total_neighbours, *search_params);

    // оценить радиус поиска для каждой картинки
    // NB: radius2_query, radius2_train: квадраты радиуса!
    float radius2_query, radius2_train;
    {
        std::vector<double> max_dists2_query(n_matches);
        std::vector<double> max_dists2_train(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            max_dists2_query[i] = distances2_query.at<float>(i, total_neighbours - 1);
            max_dists2_train[i] = distances2_train.at<float>(i, total_neighbours - 1);
        }

        int median_pos = n_matches / 2;
        std::nth_element(max_dists2_query.begin(), max_dists2_query.begin() + median_pos, max_dists2_query.end());
        std::nth_element(max_dists2_train.begin(), max_dists2_train.begin() + median_pos, max_dists2_train.end());

        radius2_query = max_dists2_query[median_pos] * radius_limit_scale * radius_limit_scale;
        radius2_train = max_dists2_train[median_pos] * radius_limit_scale * radius_limit_scale;
    }

    // метч остается, если левое и правое множества первых total_neighbors соседей в радиусах поиска(radius2_query, radius2_train) имеют как минимум consistent_matches общих элементов
    for (int i_query = 0; i_query < n_matches; ++i_query) {
        size_t curr_consistent_matches = 1;

        for (int j_query = 1; j_query < total_neighbours; ++j_query) {
            if (distances2_query.at<float>(i_query, j_query) > radius2_query) {
                break;
            }

            // поскольку points_query и points_train заполнены по порядку соответствующими точками
            const int target_point_index = indices_query.at<int>(i_query, j_query);
            const int i_train = i_query;

            for (int j_train = 1; j_train < total_neighbours; ++j_train) {
                if (distances2_train.at<float>(i_train, j_train) > radius2_train) {
                    break;
                }

                if (indices_train.at<int>(i_train, j_train) == target_point_index) {
                    ++curr_consistent_matches;
                    break;
                }
            }
        }

        if (curr_consistent_matches >= consistent_matches) {
            filtered_matches.push_back(matches[i_query]);
        }
    }

    // NOTE: реализация применения фильтра в несколько итераций — для
    // минимизации количества изменений в коде задания сделано рекурсивно,
    // в чистовом варианте можно (и лучше) заменить циклом.
    if (iterations > 1) {
        if (filtered_matches.size() >= total_neighbours) {
            std::vector<cv::DMatch> maches_pass2 = filtered_matches;
            filterMatchesClusters(maches_pass2, keypoints_query, keypoints_train, filtered_matches, iterations - 1);
        } else {
            // Слишком мало точек для следующей итерации.
            filtered_matches.clear();
        }
    }
}
