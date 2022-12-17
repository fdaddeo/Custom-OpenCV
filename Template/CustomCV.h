#ifndef CUSTOM_CV_H
#define CUSTOM_CV_H

#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/core/fast_math.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace custom_cv
{
    /**
     * @brief Performs a pixel-wise comparison to check if the two images are identical or not.
     *        Note that the two images have to be of the same type.
     * 
     * @tparam T The datatype contained in the two matrixes.
     * 
     * @param firstSrc The first input image.
     * @param secondSrc The secondo input image.
     * 
     * @return true if the two images are identical, false otherwise.
     * 
    **/
    template <typename T>
    bool equalImages(const cv::Mat & firstSrc, const cv::Mat & secondSrc)
    {
        if (firstSrc.type() != secondSrc.type())
        {
            perror("Error on equalImages function");
            exit(-1);
        }

        if (firstSrc.rows != secondSrc.rows || firstSrc.cols != secondSrc.cols)
        {
            std::cout << "Dimension reason: ";
            return false;
        }
            

        for (int v = 0; v < firstSrc.rows; ++v)
        {
            for (int u = 0; u < firstSrc.cols; ++u)
            {
                if (firstSrc.at<T>(v, u) != secondSrc.at<T>(v, u))
                {
                    std::cout << "Pixel value reason: ";
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * @brief Check if the kernel is an odd kernel.
     * 
     * @param krn Kernel to be checked.
     * 
     * @return true if it is an odd kernel.
     * @return false if it is not an odd kernel.
     * 
    **/
    bool checkOddKernel(const cv::Mat & krn);

    /**
     * @brief Add zero padding to the image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param padded The padded image "returned" of CV_8UC1 type.
     * @param padW The width padding dimension.
     * @param padH The height padding dimension.
     * 
    **/
    void addZeroPadding(const cv::Mat & src, cv::Mat & padded, const int padW, const int padH);

    /**
     * @brief Add padding according to the source image and the the kernel specified.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param kernel The kernel to be used in convolution.
     * @param padded The padded image "returned" of CV_8UC1 type.
     * @param anchor The anchor point in the kernel used to compute the correct padding.
     * @param zeroPad If true add zero padding, if false add 255 as padding.
     * 
    **/
    void addZeroPadding(const cv::Mat & src, const cv::Mat & kernel , cv::Mat & padded, cv::Point anchor, bool zeroPad);

    /**
     * @brief Perform an opertaion of convolution, with zero padded images.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param krn The kernel to be used of CV_32FC1 type.
     * @param dst The "returned" image of CV_32SC1 type.
     * @param stride The stride to be used.
     * 
    **/
    void myfilter2D(const cv::Mat & src, const cv::Mat & krn, cv::Mat & dst, int stride = 1);

    // void myfilter2DBuffer(const cv::Mat & src, const cv::Mat & krn, cv::Mat & dst, int stride);

    /**
     * @brief Computes the histogram of the image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param histogram The array in which store the histogram.
     * 
    **/
    void calculateHistogram(const cv::Mat & src, int (& histogram)[256]);

    /**
     * @brief Computes the best threshold for image binarization according to the OTSU method.
     * 
     *        -- TO BE FIXED with prof's formula --
     * 
     * @param histogram The histogram in which compute the threshold.
     * @param tValue The value of the threshold "returned".
     * @param sigma The value of the variance "returned".
     * 
    **/
    void calculateOTSU(const int (& histogram)[256], int & tValue, float & sigma);

    /**
     * @brief Implements the erosion operation on binary image.
     * 
     * @param src The binarized source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void erosionBinary(const cv::Mat & src, cv::Mat & dst, const cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the dilation operation on binary image.
     * 
     * @param src The binarized source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void dilationBinary(const cv::Mat & src, cv::Mat & dst, const cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the opening operation on binary image.
     * 
     * @param src The binarized source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void openingBinary(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the closing operation on binary image.
     * 
     * @param src The binarized source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void closingBinary(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the erosion operation on greyscale image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void erosionGreyscale(const cv::Mat & src, cv::Mat & dst, const cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the dilation operation on greyscale image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void dilationGreyscale(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the opening operation on greyscale image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void openingGreyscale(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Implements the closing operation on greyscale image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param structuralElement The structural element of CV_8UC1 type.
     * @param anchor The anchor point of the structural element.
     * 
    **/
    void closingGreyscale(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor);

    /**
     * @brief Creates a vertical gaussian kernel for gaussian filtering.
     * 
     * @param sigma The gaussian standard deviation.
     * @param r The kernel radius.
     * @param krnl The vertical gaussian kernel "returned" of CV_32FC1 type.
     * 
    **/
    void createGaussianKrnlVert(float sigma, int r, cv::Mat & krnl);

    /**
     * @brief Performs a gaussian filtering performing a convolution with a gaussian kernel.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param sigma The gaussian standard deviation for the gaussian kernel.
     * @param r The radius of the gaussian kernel.
     * @param dst The filtered image "returned" of CV_32SC1 type.
     * @param stride The stride to be used in the convolution.
     * 
    **/
    void gaussianBlur(const cv::Mat & src, float sigma, int r, cv::Mat & dst, int stride = 1);

    /**
     * @brief Create a Domain Kernel object.
     * 
     * @param krn  The "returned" kernel of CV_32FC1 type.
     * @param radius The radius of the kernel, note that the dimension will be (RADIUS x 2 + 1, RADIUS x 2 + 1).
     * @param sigma The sigma value.
     * 
    **/
    void createDomainKernel(cv::Mat & krn, const int radius, const int sigma);

    /**
     * @brief Create a Range Kernel object, remember that this kernel highly depends on the input image.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param vImage The row index in which center and compute the kernel.
     * @param uImage the column index in which center and compute the kernel.
     * @param radius The radius of the kernel, it has to be correctly computed wrt the diameter.
     * @param diameter The dimension of the kernel.
     * @param sigma The sigma value.
     * 
     * @return The cv::Mat kernel object.
     * 
    **/
    cv::Mat createRangeKernel(const cv::Mat & src, const int vImage, const int uImage, const int radius, const int diameter, const float sigma);

    /**
     * @brief TO FIX!
     * 
     *        Performs a bilateral filtering performing a convolution with a kernel obtained as composition of 
     *        the domain kernel and the range kernel.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_32SC1 type.
     * @param diameter The diameter of the kernel.
     * @param sigmaR The sigma value for the Range Kernel.
     * @param sigmaD The sigma value for the Domain Kernel.
     * 
    **/
    void bilateralFiltering(const cv::Mat & src, cv::Mat & dst, int diameter, float sigmaR, float sigmaD);

    /**
     * @brief Performs a bilateral filtering performing a convolution with a semplified Laplacian Gaussian kernel.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param dst The "returned" image of CV_8UC1 type.
     * @param alpha The alpha value used in the sharpening operation.
     * 
    **/
    void sharpeningFiltering(const cv::Mat & src, cv::Mat & dst, const float alpha);

    /**
     * @brief Computes the magnitude and the orientation of the source image by perform two convolution:
     *        the first one with the vertical sobel filter to obtain the magnitude and the second one 
     *        with the horizontal sobel filter to obtian the orientation.
     * 
     * @param src The source image of CV_8UC1 type.
     * @param magnitude The "returned" magnitude of CV_32FC1 type.
     * @param orientation The "returned" orientation of CV_32FC1 type.
     * 
    **/
    void sobel3x3(const cv::Mat & src, cv::Mat & magnitude, cv::Mat & orientation);

    /**
     * @brief Performs the bilinear interpolation of a point using its four neighbour pixels.
     * 
     * @param src The source image of CV_8UC1 or CV_32FC1 type.
     * @param r The row value, so the y value.
     * @param c The column value, so the x value.
     * 
     * @return The interpolated value of the pixel.
     * 
    **/
    float bilinear(const cv::Mat & src, const float r, const float c);

    /**
     * @brief Performs the operation of Non Maxima Suppression.
     * 
     * @param mangnitude The magnitude of the source image of CV_32FC1 type.
     * @param orientation The orientation of the source image of CV_32FC1 type.
     * @param dst The "returned" image with non maxima suppressed.
     * 
    **/
    void findPeaks(const cv::Mat & mangnitude, const cv::Mat & orientation, cv::Mat & dst);

    /**
     * @brief Performs a threshold operation with hysteresis Canny style.
     * 
     * @param magnitude The magnitude with non-maxima suppressed of the source image of CV_32FC1 type.
     * @param dst The binarized image "returned" of CV_8UC1 type.
     * @param tLow The low threshold.
     * @param tHigh The high threshold.
     * 
    **/
    void doubleTh(const cv::Mat & magnitude, cv::Mat & dst, const float tLow, const float tHigh);

    /**
     * @brief Recursive function used to check if the neighbourhood of a pixel contains an edge.
     * 
     * @param magnitude The magnitude of the source image of CV_32FC1 type.
     * @param vStart The starting v value while checking the neighbourhood.
     * @param uStart The starting u value while checking the neighbourhood.
     * @param tLow The value of low threshold.
     * @param tHigh The value of high threshold.
     * 
     * @return true if the neighbourhood of the pixel, or the neighbourhood of the neighbourhood recursively, contains
     *         an edge.
     * 
    **/
    bool checkNeighborhoodEdge(const cv::Mat & magnitude, const int vStart, const int uStart, const float tLow, const float tHigh);

    /**
     * @brief Performs the Hough Transformation.
     * 
     * @param src The binary source image after a Canny Edge Detector operation of CV_8UC1 type.
     * @param lines The "returned" vector in which there will be stored the value of rho and theta (in degress).
     * @param threshold The threshold value to perform the operation.
     * 
    **/
    void houghTransform(const cv::Mat & src, std::vector<cv::Vec2f> & lines, const int threshold);

    /**
     * @brief Draw lines in the image.
     * 
     * @param src The source content image of CV_8UC1 or CV_8UC3 type.
     * @param dst The "returned" image in which copy the source image and draw the lines.
     * @param lines The corresponding point in parameter of rho and theta (in degrees) which will be translated in lines.
     * 
    **/
    void drawLinesOnImage(const cv::Mat & src, cv::Mat & dst, const std::vector<cv::Vec2f> & lines);

    /**
     * @brief Compute the Disparity matrix using the SAD algorithm. It requires RECTIFIED IMAGES, it uses a window of 7x7 dimension
     *        and it assumes that the maximum disparity value is 128.
     * 
     * @param leftImage The left source image of CV_8UC1 type.
     * @param rightImage The right source image of CV_8UC1 type.
     * @param dst The "returned" disparity matrix of CV_32FC1 type.
     * 
    **/
    void mySAD_Disparity7x7(const cv::Mat & leftImage, const cv::Mat & rightImage, cv::Mat & dst);

    /**
     * @brief Plots the disparity histogram in a binary image. It assumes 128 as maximum disparity value.
     * 
     * @param disparity The source disparity matrix of CV_32FC1 type. 
     * @param dst The "returned" image of CV_16UC1 type.
     *  
    **/
    void VDisparity(const cv::Mat & disparity, cv::Mat & dst);

    /**
     * @brief Calculates the plane passing for threee points, with plane equation equal to: ax + by +cz + d = 0. 
     * 
     * @param p1 The first point.
     * @param p2 The second point.
     * @param p3 The third point.
     * @param a The "returned" value A in the equation.
     * @param b The "returned" value B in the equation.
     * @param c The "returned" value C in the equation.
     * @param d The "returned" value D in the equation.
     * 
    **/
    void plane3points(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, float & a, float & b, float & c ,float & d);

    /**
     * @brief Calculates the distance between a point and a plane.
     * 
     * @param p The point.
     * @param a The A value in the plane equation.
     * @param b The B value in the plane equation.
     * @param c The C value in the plane equation.
     * @param d The D value in the plane equation.
     * 
     * @return The distance between the point and the plane
     * 
    **/
    float distance_plane_point(cv::Point3f p, float a, float b, float c ,float d);

    /**
     * @brief Calculates the corresponding 3D points from a disparity matrix. Internally it has stored the intrinsec calibration
     *        parameters of the camera.
     * 
     * @param disparity The disparity source matrix.
     * @param points The "returned" vector containing the 3D point computed.
     * @param rc The "returned" vector containing the indexes for (row, column) of the corresponding 3D points.
     * 
    **/
    void compute3Dpoints(const cv::Mat & disparity, std::vector<cv::Point3f> & points, std::vector<cv::Point2i> & rc);

    /**
     * @brief TODO
     * 
     * @param points 
     * @param rc 
     * @param inliers_best_points 
     * @param inliers_best_rc 
     * 
    **/
    void computePlane(const std::vector<cv::Point3f> & points, const std::vector<cv::Point2i> & rc, std::vector<cv::Point3f> & inliers_best_points, std::vector<cv::Point2i> & inliers_best_rc);

    /**
     * @brief Create a Image from Inlier Points Vector.
     * 
     * TODO
     * 
     * @param leftSrc 
     * @param dst 
     * @param inliersBestRowColumn 
     * 
    **/
    void createImageFromInliers(const cv::Mat & leftSrc, cv::Mat & dst, const std::vector<cv::Point2i> & inliersBestRowColumn);

    /**
     * @brief TODO
     * 
     * TODO: change the function name?
     * 
     * @param src 
     * @param dst 
     * @param corners_src 
     * 
    **/
    void partialImageReprojection(const cv::Mat & src, cv::Mat & dst, const std::vector<cv::Point2f> & corners_src);
}

#endif