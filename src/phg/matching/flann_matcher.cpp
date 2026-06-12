#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    // NOTE: Поскольку более обоснованных идей у меня нет, здесь используются
    // стандартные значения по умолчанию из OpenCV. Похоже, что они работают
    // вполне неплохо.
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices;
    cv::Mat dists;
    flann_index->knnSearch(query_desc, indices, dists, k);

    matches.clear();
    matches.reserve(indices.rows);

    for (int r = 0; r < indices.rows; ++r) {
        std::vector<cv::DMatch>& match = matches.emplace_back();
        match.reserve(indices.cols);

        for (int c = 0; c < indices.cols; ++c) {
            const float dist = std::sqrt(dists.at<float>(r, c)); // knnSearch возвращает квадрат расстояния
            match.push_back(cv::DMatch(r, indices.at<int>(r, c), dist));
        }
    }
}
