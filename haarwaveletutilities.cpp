#include <iostream>
#include <fstream>
#include "haarwaveletutilities.h"



bool loadHaarWavelets(cv::Size * const sampleSize, const std::string &filename, std::vector<HaarWavelet *> & wavelets)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    do
    {
        HaarWavelet * wavelet = new HaarWavelet(sampleSize, ifs);
        if ( !ifs.eof() )
        {
            wavelets.push_back(wavelet);
        }
        else
        {
            break;
        }
    } while (true);

    ifs.close();

    return true;
}



bool writeHaarWavelets(const char * filename, const std::vector<HaarWavelet * > &wavelets)
{
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc);

    if (!ofs.is_open())
    {
        return false;
    }

    std::vector<HaarWavelet * >::const_iterator it = wavelets.begin();
    const std::vector<HaarWavelet * >::const_iterator end = wavelets.end();
    for(; it != end; ++it)
    {
        (*it)->write(ofs);
        ofs << '\n';
    }
    ofs.close();

    return true;
}
