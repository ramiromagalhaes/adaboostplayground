#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/unordered_map.hpp>
#include <boost/filesystem.hpp>

#include "testdatabase.h"

#include "common.h"


/**
 *
 */
int main(int, char **argv) {
    const std::string testImagesIndexFileName = argv[1];
    const std::string groundTruthFileName = argv[2];

    TestDatabase database;
    if ( !database.load(testImagesIndexFileName, groundTruthFileName) )
    {
        return 1;
    }

    std::vector<ImageAndGroundTruth> images = database.getImagesAndGroundTruthAsVector();
    for(std::vector<ImageAndGroundTruth>::iterator it = images.begin(); it != images.end(); ++it)
    {
        cv::Mat showMe(it->image);

        for(std::vector<cv::Rect>::iterator rect = it->faces.begin(); rect != it->faces.end(); ++rect)
        {
            cv::rectangle(showMe, *rect, cv::Scalar( 255, 0, 0 ));
        }

        cv::imshow("Image", showMe);
        cv::waitKey( 0 );
    }

    return 0;
}
