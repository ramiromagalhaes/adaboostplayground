#include <string>
#include <opencv2/core/core.hpp>
#include "haarwavelet.h"

bool loadHaarWavelets(cv::Size * const sampleSize, const std::string &filename, std::vector<HaarWavelet *> & wavelets);

bool writeHaarWavelets(const char * filename, const std::vector<HaarWavelet * > &wavelets);
