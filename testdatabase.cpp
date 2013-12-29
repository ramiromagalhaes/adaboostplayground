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
        lineStream >> imageFileName
                   >> leftEye.x
                   >> leftEye.y
                   >> rightEye.x
                   >> rightEye.y;

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


/*
//Use getTestImages__2 to test the classifier with an image database like those used in trainning, i.e., a huge image composed of many 20x20 pixels images.
bool getTestImages__2(const std::string positivesFile,
                      const std::string negativesFile,
                      std::vector<ImageAndGroundTruth> & images,
                      int & totalFacesInGroundTruth)
{
    std::vector<cv::Mat> samples;

    if ( !SampleExtractor::fromImageFile(positivesFile, samples) )
    {
        return false;
    }

    totalFacesInGroundTruth = samples.size();
    for(std::vector<cv::Mat>::iterator sample = samples.begin(); sample != samples.end(); ++sample)
    {
        ImageAndGroundTruth iagt;
        iagt.image = *sample;
        iagt.faces.push_back(cv::Rect(0, 0, sample->cols, sample->rows));

        images.push_back(iagt);
    }

    samples.clear();

    if ( !SampleExtractor::fromImageFile(negativesFile, samples) )
    {
        return false;
    }
    for(std::vector<cv::Mat>::iterator sample = samples.begin(); sample != samples.end(); ++sample)
    {
        ImageAndGroundTruth iagt;
        iagt.image = *sample;

        images.push_back(iagt);
    }

    return true;
}
*/

//I can use the getTestImages__2 to test the classifier an image database like those used in trainning, i.e.,
//a huge image composed of many 20x20 pixels images.
//If doinf this, remember to also change the name of the ___main function parameters.
//if ( !getTestImages__2(positivesFile, negativesFile, images, totalFacesInGroundTruth) )
