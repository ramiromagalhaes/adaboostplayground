#include "testdatabase.h"



bool TestDatabase::loadImages(const std::string & indexFileName)
{
    std::ifstream indexStream(indexFileName.c_str());
    if (!indexStream.is_open())
    {
        return false;
    }

    while( !indexStream.eof() )
    {
        std::string imagePath;
        std::getline(indexStream, imagePath);
        if (imagePath.empty())
        {
            break;
        }

        if ( !boost::filesystem::exists(imagePath) )
        {
            return false;
        }

        ImageAndGroundTruth iagt;
        iagt.image = cv::imread(imagePath, cv::DataType<unsigned char>::type);
        if ( !iagt.image.data )
        {
            return false;
        }

        //This will extract only the file name from the whole file path.
        //TODO If files in different folders have the same name, this will not work properly. See also loadGroundTruth() method.
        const std::string filename = boost::filesystem::path(imagePath).filename().native();
        images.insert( std::make_pair(filename, iagt) );
    }

    indexStream.close();

    return true;
}



bool TestDatabase::loadGroundTruth(const std::string &grountTruthPath)
{
    std::ifstream gtStream(grountTruthPath.c_str());
    if (!gtStream.is_open())
    {
        return false;
    }

    while( !gtStream.eof() )
    {
        std::string line;
        std::getline(gtStream, line);

        if (line.empty())
        {
            break;
        }

        std::string imageFileName;
        cv::Point2f leftEye, rightEye;

        std::istringstream lineStream(line);
        lineStream >> imageFileName //In the documentation (http://vasc.ri.cmu.edu/idb/html/face/frontal_images/),
                   >> rightEye.x    //they say the first input is the left eye, then the right eye, but they mean
                   >> rightEye.y    //the VIEWER's left, NOT the SUBJECT's.
                   >> leftEye.x
                   >> leftEye.y;

        //Calculate face region from eye position.
        //Here this is done exactly as I extract faces from the BioId database.
        const float distanceBetweenEyes = cv::norm(rightEye-leftEye);
        const float roiWidthHeight = distanceBetweenEyes / 0.5154f;
        cv::Rect faceRegion(rightEye.x - roiWidthHeight * 0.2423f,
                            rightEye.y - roiWidthHeight * 0.25f,
                            roiWidthHeight, roiWidthHeight);

        //It is assumed that imageFileName will only hold the file name, not the full file path.
        //TODO If files in different folders have the same name, this will not work properly. See also loadImages() method.
        if ( images.find(imageFileName) == images.end() )
        {
            return false;
        }

        images[imageFileName].faces.push_back(faceRegion);
        ++totalFaces;
    }

    return true;
}



TestDatabase::TestDatabase()
{
}



bool TestDatabase::load(const std::string &imageIndexPath, const std::string &groundTruthPath)
{
    return loadImages(imageIndexPath) && loadGroundTruth(groundTruthPath);
}



std::vector<ImageAndGroundTruth> TestDatabase::getImagesAndGroundTruthAsVector() const
{
    std::vector<ImageAndGroundTruth> returnMe;

    ImageAndGroundTruthMap::const_iterator it = images.begin();
    const ImageAndGroundTruthMap::const_iterator end = images.end();

    for(; it != end; ++it)
    {
        returnMe.push_back(it->second);
    }

    return returnMe;
}



int TestDatabase::size_annotations() const
{
    return totalFaces;
}



int TestDatabase::size_images() const
{
    return images.size();
}
