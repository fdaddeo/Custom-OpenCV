#include "CustomCV.h"

// #define RECURSION_FOR_CANNY
#define EIGEN_FOR_HOMOGRAPHY

namespace custom_cv
{
    bool checkOddKernel(const cv::Mat & krn)
    {
        if (krn.cols % 2 != 0 && krn.rows % 2 != 0)
            return true;
        else
            return false;
    }

    void addZeroPadding(const cv::Mat & src, cv::Mat & padded, const int padW, const int padH)
    {
        padded = cv::Mat::zeros(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1);

        for (int v = padH; v < (padded.rows - padH); ++v)
        {
            for (int u = padW; u < (padded.cols - padW); ++u)
            {
                padded.at<uchar>(v, u) = src.at<uchar>((v - padH), (u - padW));
            }
        }
    }

    void addZeroPadding(const cv::Mat & src, const cv::Mat & kernel , cv::Mat & padded, cv::Point anchor, bool zeroPad = true)
    {
        int padW_left = anchor.y;
        int padW_right = kernel.cols - 1 - anchor.y;
        int padH_top = anchor.x;
        int padH_bottom = kernel.rows - 1 - anchor.x;

        if (zeroPad)
        {
            padded = cv::Mat::zeros(src.rows + padH_top + padH_bottom, src.cols + padW_left + padW_right, CV_8UC1);
        }
        else
        {
            padded = cv::Mat(src.rows + padH_top + padH_bottom, src.cols + padW_left + padW_right, CV_8UC1, cv::Scalar(255));
        }
        

        for (int v = padH_top; v < (padded.rows - padH_bottom); ++v)
        {
            for (int u = padW_left; u < (padded.cols - padW_right); ++u)
            {
                padded.at<uchar>(v, u) = src.at<uchar>(v - padH_top, u - padW_left);
            }
        }
    }

    void myfilter2D(const cv::Mat & src, const cv::Mat & krn, cv::Mat & dst, int stride)
    {
        if (!checkOddKernel(krn))
        {
            std::cout << "ERRORE: il kernel deve essere dispari e quadrato.\n";
            exit(-1);
        }

        int padW = krn.cols / 2;
        int padH = krn.rows / 2;

        cv::Mat paddedSrc;
        addZeroPadding(src, paddedSrc, padW, padH);

        dst = cv::Mat((int)((src.rows + 2 * padH - krn.rows) / stride) + 1, (int)((src.cols + 2 * padW - krn.cols) / stride) + 1, CV_32SC1);

        // int * outbuffer = (int *) out.data;
        // float * kernel = (float *) krn.data;
        
        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                float pixelValue = 0.0f;

                for (int k = 0; k < krn.rows; ++k)
                {
                    for (int l = 0; l < krn.cols; ++l)
                    {
                        // pixelValue += paddedSrc.data[(u + l) + (v + k) * paddedSrc.cols] * kernel[l + k * krn.cols];
                        pixelValue += krn.at<float>(k, l) * (int)paddedSrc.at<uchar>(v * stride + k, u * stride + l);
                    }
                }

                // outbuffer[u + v * out.cols] = pixelValue;
                dst.at<int32_t>(v, u) = pixelValue;
            }
        }
    }

    void calculateHistogram(const cv::Mat & src, int (& histogram)[256])
    {
        for (int v = 0; v < src.rows; ++v)
        {
            for (int u = 0; u < src.cols; ++u)
            {
                ++histogram[(int)src.at<uchar>(v, u)];
            }
        }
    }

    void calculateOTSU(const int (& histogram)[256], int & tValue, float & sigma)
    {
        for (int t = 0; t < 256; ++t)
        {
            float w_b = 0.0;
            float w_f = 0.0;
            float mu_b = 0.0;
            float mu_f = 0.0;

            float numPixel_b = 1.0;
            float numPixel_f = 1.0;

            for (int i = 0; i < 256; ++i)
            {
                if (i <= t)
                {
                    w_b += histogram[i];
                    mu_b += (histogram[i] * i);
                    numPixel_b += histogram[i];
                }
                else
                {
                    w_f += histogram[i];
                    mu_f += (histogram[i] * i);
                    numPixel_f += histogram[i];
                }
            }

            w_b = w_b / (numPixel_b + numPixel_f);
            w_f = w_f / (numPixel_b + numPixel_f);

            mu_b = mu_b / numPixel_b;
            mu_f = mu_f / numPixel_f;

            float sigma_tmp = w_f * w_b * std::pow((mu_b - mu_f), 2);

            if (sigma_tmp > sigma)
            {
                sigma = sigma_tmp;
                tValue = t;
            }
        }

        std::cout << "Sigma value: " << sigma << " Value of t: " << tValue << std::endl;
    }

    void erosionBinary(const cv::Mat & src, cv::Mat & dst, const cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat paddedSrc;

        addZeroPadding(src, structuralElement, paddedSrc, anchor);
        
        dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type());

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                bool logicalAnd = true;

                for (int sv = 0; sv < structuralElement.rows; ++sv)
                {
                    for (int su = 0; su < structuralElement.cols; ++su)
                    {
                        if ((int)structuralElement.data[su + sv * structuralElement.cols] == 255)
                        {
                            if ((int)paddedSrc.data[(u + su) + (v + sv) * paddedSrc.cols] != 255)
                            {
                                logicalAnd = false;
                            }
                        }
                    }
                }

                if (logicalAnd)
                {
                    dst.data[u + v * dst.cols] = 255;
                }
                else 
                {
                    dst.data[u + v * dst.cols] = 0;
                }
            }
        }
    }

    void dilationBinary(const cv::Mat & src, cv::Mat & dst, const cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat paddedSrc;
        
        addZeroPadding(src, structuralElement, paddedSrc, anchor);
        
        dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type());

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                bool logicalOr = false;

                for (int sv = 0; sv < structuralElement.rows; ++sv)
                {
                    for (int su = 0; su < structuralElement.cols; ++su)
                    {
                        if ((int)structuralElement.data[su + sv * structuralElement.cols] == 255)
                        {
                            if ((int)paddedSrc.data[(u + su) + (v + sv) * paddedSrc.cols] == 255)
                            {
                                logicalOr = true;
                            }
                        }
                    }
                }

                if (logicalOr)
                {
                    dst.data[u + v * dst.cols] = 255;
                }
                else 
                {
                    dst.data[u + v * dst.cols] = 0;
                }
            }
        }
    }

    void openingBinary(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat eroded;

        erosionBinary(src, eroded, structuralElement, anchor);
        dilationBinary(eroded, dst, structuralElement, anchor);
    }

    void closingBinary(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat dilated;

        dilationBinary(src, dilated, structuralElement, anchor);
        erosionBinary(dilated, dst, structuralElement, anchor);
    }

    void erosionGreyscale(const cv::Mat & src, cv::Mat & dst, const cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat paddedSrc;

        addZeroPadding(src, structuralElement, paddedSrc, anchor, false);
        
        dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type());

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                int minPixelValue = 255;

                for (int sv = 0; sv < structuralElement.rows; ++sv)
                {
                    for (int su = 0; su < structuralElement.cols; ++su)
                    {
                        if ((int)structuralElement.data[su + sv * structuralElement.cols] == 255)
                        {
                            int currentPixelValue = (int)paddedSrc.data[(u + su) + (v + sv) * paddedSrc.cols];
                            
                            if (currentPixelValue < minPixelValue)
                            {
                                minPixelValue = currentPixelValue;
                            }
                        }
                    }
                }

                dst.data[u + v * dst.cols] = minPixelValue;
            }
        }
    }

    void dilationGreyscale(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat paddedSrc;

        addZeroPadding(src, structuralElement, paddedSrc, anchor);
        
        dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), src.type());

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                int maxPixelValue = 0;

                for (int sv = 0; sv < structuralElement.rows; ++sv)
                {
                    for (int su = 0; su < structuralElement.cols; ++su)
                    {
                        if ((int)structuralElement.data[su + sv * structuralElement.cols] == 255)
                        {
                            int currentPixelValue = (int)paddedSrc.data[(u + su) + (v + sv) * paddedSrc.cols];

                            if (currentPixelValue > maxPixelValue)
                            {
                                maxPixelValue = currentPixelValue;
                            }
                        }
                    }
                }

                dst.data[u + v * dst.cols] = maxPixelValue;
            }
        }
    }

    void openingGreyscale(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat eroded;

        erosionGreyscale(src, eroded, structuralElement, anchor);
        dilationGreyscale(eroded, dst, structuralElement, anchor);
    }

    void closingGreyscale(cv::Mat & src, cv::Mat & dst, cv::Mat & structuralElement, cv::Point anchor)
    {
        cv::Mat dilated;

        dilationGreyscale(src, dilated, structuralElement, anchor);
        erosionGreyscale(dilated, dst, structuralElement, anchor);
    }

    void createGaussianKrnlVert(float sigma, int r, cv::Mat & krnl)
    {
        float weight_sum = 0.0f;
        krnl = cv::Mat((2 * r + 1), 1, CV_32FC1);

        for (int i = -r; i <= r; ++i)
        {
            float weight = (1 / sigma * std::sqrt(2 * M_PI)) * std::exp(-(std::pow(i, 2) / (2 * std::pow(sigma, 2))));
            
            krnl.at<float>(i + r, 0) = weight;
            weight_sum += weight;
        }

        krnl /= weight_sum;
    }

    void gaussianBlur(const cv::Mat & src, float sigma, int r, cv::Mat & dst, int stride)
    {
        cv::Mat gaussian_kernel_vertical;
        cv::Mat gaussian_kernel_horizontal;
        cv::Mat out_tmp;

        createGaussianKrnlVert(sigma, r, gaussian_kernel_vertical);
        cv::transpose(gaussian_kernel_vertical, gaussian_kernel_horizontal);
        
        myfilter2D(src, gaussian_kernel_vertical, out_tmp, stride);
        out_tmp.convertTo(out_tmp, CV_8UC1);
        myfilter2D(out_tmp, gaussian_kernel_horizontal, dst, stride);
    }

    void createDomainKernel(cv::Mat & krn, const int radius, const int sigma)
    {
        int diameter = radius * 2 + 1;

        krn = cv::Mat::zeros(diameter, diameter, CV_32FC1);

        for (int v = -radius; v <= radius; ++v)
        {
            for (int u = -radius; u <= radius; ++u)
            {
                krn.at<float>(v + radius, u + radius) = std::exp(- (std::pow(v, 2) + std::pow(u, 2)) / (2 * std::pow(sigma, 2)));
            }
        }
    }

    cv::Mat createRangeKernel(const cv::Mat & src, const int vImage, const int uImage, const int radius, const int diameter, const float sigma)
    {
        cv::Mat krn = cv::Mat::zeros(diameter, diameter, CV_32FC1);

        int imageCenteredPixelValue = src.at<uchar>(vImage, uImage);

        for (int v = -radius; v <= radius; ++v)
        {
            for (int u = -radius; u <= radius; ++u)
            {
                int imageValue = src.at<uchar>(vImage + v, uImage + u);

                krn.at<float>(v + radius, u + radius) = std::exp(- std::pow(imageCenteredPixelValue - imageValue, 2) / (2 * std::pow(sigma, 2)));
            }
        }

        return krn;
    }

    void bilateralFiltering(const cv::Mat & src, cv::Mat & dst, int diameter, float sigmaD, float sigmaR)
    {        
        cv::Mat verticalGaussianKernel;
        cv::Mat horizonalGaussianKernel;
        cv::Mat domainKernel;
        cv::Mat paddedSrc;

        int radius = diameter / 2;

        addZeroPadding(src, paddedSrc, radius, radius);
        createDomainKernel(domainKernel, radius, sigmaD);

        dst = cv::Mat(src.size(), CV_32SC1);
        
        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                cv::Mat rangeKernel = createRangeKernel(paddedSrc, v + radius, u + radius, radius, diameter, sigmaR);
                cv::Mat kernel = domainKernel.mul(rangeKernel);

                float pixelValue = 0.0f;
                float kernelSum = 0.0f;

                for (int k = 0; k < kernel.rows; ++k)
                {
                    for (int l = 0; l < kernel.cols; ++l)
                    {
                        float kernelValue = kernel.at<float>(k, l);
                        
                        pixelValue += kernelValue * (int)paddedSrc.at<uchar>(v + k, u + l);
                        kernelSum += kernelValue;
                    }
                }

                dst.at<int>(v, u) = pixelValue / kernelSum;
            }
        }
    }

    void sharpeningFiltering(const cv::Mat & src, cv::Mat & dst, const float alpha)
    {
        cv::Mat LoG_conv_I;

        float kernelData[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
        cv::Mat kernel = cv::Mat(3, 3, CV_32FC1, kernelData);
        
        dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

        custom_cv::myfilter2D(src, kernel, LoG_conv_I);
        LoG_conv_I.convertTo(LoG_conv_I, CV_32FC1);
        LoG_conv_I *= alpha;

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                float pixelVal = (float)(src.at<uchar>(v, u)) - LoG_conv_I.at<float>(v, u);

                pixelVal = std::max(pixelVal, 0.0f);
                pixelVal = std::min(pixelVal, 255.0f);

                dst.at<uchar>(v, u) = std::round(pixelVal);
            }
        }
    }

    void sobel3x3(const cv::Mat & src, cv::Mat & magnitude, cv::Mat & orientation)
    {
        cv::Mat verticalSobel = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
        cv::Mat horizontalSobel;
        cv::Mat convX;
        cv::Mat convY;

        for (int v = 0; v < verticalSobel.rows; ++v)
        {
            for (int u = 0; u < verticalSobel.cols; ++u)
            {
                if (u == 0 && (v == 0 || v == 2))
                    verticalSobel.at<float>(v, u) = 1;
                
                if (u == 0 && v == 1)
                    verticalSobel.at<float>(v, u) = 2;
                
                if (u == 2 && (v == 0 || v == 2))
                    verticalSobel.at<float>(v, u) = -1;
                
                if (u == 2 && v == 1)
                    verticalSobel.at<float>(v, u) = -2;
            }
        }

        cv::transpose(verticalSobel, horizontalSobel);

        // Compute Sobel Filtering
        myfilter2D(src, verticalSobel, convX);
        myfilter2D(src, horizontalSobel, convY);

        convX.convertTo(convX, CV_32FC1);
        convY.convertTo(convY, CV_32FC1);

        // Compute Magnitude
        cv::pow(convX.mul(convX) + convY.mul(convY), 0.5, magnitude);

        // Compute Orientation
        orientation = cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1);

        for (int v = 0; v < orientation.rows; ++v)
        {
            for (int u = 0; u < orientation.cols; ++u)
            {
                orientation.at<float>(v, u) = atan2f(convY.at<float>(v, u), convX.at<float>(v, u));
            }
        }
    }

    float bilinear(const cv::Mat & src, const float r, const float c)
    {
        if (r > src.rows || c > src.cols)
        {
            std::cout << "ERRORE: valore di (r,c) non corretto\n";
            exit(-1);
        }

        int r_floor = std::floor(r);
        int c_floor = std::floor(c);
        int r_ceil = std::ceil(r);
        int c_ceil = std::ceil(c);

        float t = r - r_floor;
        float s = c - c_floor;

        // Interpolation value
        float output;

        if (src.type() == CV_8UC1)
        {
            output = (float)((int)(src.at<uchar>(r_floor, c_floor)) * (1 - s) * (1 - t) + (int)(src.at<uchar>(r_floor, c_ceil)) * s * (1 - t) + (int)(src.at<uchar>(r_ceil, c_floor)) * (1 - s) * t + (int)(src.at<uchar>(r_ceil, c_ceil)) * s * t);
        }
        else if (src.type() == CV_32FC1)
        {
            output = (float)(src.at<float>(r_floor, c_floor) * (1 - s) * (1 - t) + src.at<float>(r_floor, c_ceil) * s * (1 - t) + src.at<float>(r_ceil, c_floor) * (1 - s) * t + src.at<float>(r_ceil, c_ceil) * s * t);
        }
        else 
        {
            std::cout << "ERRORE: image type non implementato\n";
            exit(-1);
        }
        
        return output;
    }

    void findPeaks(const cv::Mat & magnitude, const cv::Mat & orientation, cv::Mat & dst)
    {        
        dst = cv::Mat::zeros(magnitude.rows, magnitude.cols, CV_32FC1);

        for (int v = 1; v < dst.rows - 1; ++v)
        {
            for (int u = 1; u < dst.cols - 1; ++u)
            {
                float theta = orientation.at<float>(v, u);
                float magnVal = magnitude.at<float>(v, u);

                float e1x = u + 1 * std::cos(theta); 
                float e1y = v + 1 * std::sin(theta);
                float e2x = u - 1 * std::cos(theta);            
                float e2y = v - 1 * std::sin(theta);

                float e1 = bilinear(magnitude, e1y, e1x);
                float e2 = bilinear(magnitude, e2y, e2x);

                if (magnVal > e1 && magnVal > e2)
                {
                    dst.at<float>(v, u) = magnVal;
                }
            }
        }
    }

#ifdef RECURSION_FOR_CANNY
    void doubleTh(const cv::Mat & magnitude, cv::Mat & dst, const float tLow, const float tHigh)
    {
        dst = cv::Mat::zeros(magnitude.size(), CV_8UC1);

        for (int v = 0; v < magnitude.rows; ++v)
        {
            for (int u = 0; u < magnitude.cols; ++u)
            {
                float magnVal = magnitude.at<float>(v, u);

                if (magnVal >= tHigh)
                {
                    dst.at<uchar>(v, u) = 255;
                }
                else if (magnVal > tLow && magnVal < tHigh)
                {
                    if (checkNeighborhoodEdge(magnitude, v, u, tLow, tHigh))
                    {
                        dst.at<uchar>(v, u) = 255;
                    }
                    else
                    {
                        dst.at<uchar>(v, u) = 0;
                    }
                }
                else
                {
                    dst.at<uchar>(v, u) = 0;
                }
            }
        }
    }
#else
    void doubleTh(const cv::Mat & magnitude, cv::Mat & dst, const float tLow, const float tHigh)
    {
        std::vector<cv::Point2i> grownPoints;
        std::vector<cv::Point2i> markedPoints;
        
        dst = cv::Mat::zeros(magnitude.size(), CV_8UC1);

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                float magnVal = magnitude.at<float>(v, u);

                if (magnVal >= tHigh)
                {
                    dst.at<uchar>(v, u) = 255;
                    grownPoints.push_back(cv::Point2i(u, v));
                }
                else if (magnVal >= tLow && magnVal < tHigh)
                {
                    // In order to mark them
                    dst.at<uchar>(v, u) = 100;
                    markedPoints.push_back(cv::Point2i(u, v));
                }
            }
        }

        while (!grownPoints.empty())
        {
            cv::Point2i point = grownPoints.back();
            grownPoints.pop_back();

            for (int r = -1; r <= 1; ++r)
            {
                for (int c = -1; c <= 1; ++c)
                {
                    if ((r == 0 && c == 0))
                        continue;

                    int v = point.y + r;
                    int u = point.x + c;

                    if (v < 0 || u < 0 || v >= dst.rows || u >= dst.cols)
                        continue;

                    if (dst.at<uchar>(v, u) == 100)
                    {
                        dst.at<uchar>(v, u) = 255;
                        grownPoints.push_back(cv::Point2i(u, v));
                    }
                }
            }
        }
        
        for (auto point : markedPoints)
        {
            if (dst.at<uchar>(point.y, point.x) != 255)
            {
                dst.at<uchar>(point.y, point.x) = 0;
            }
        }
    }
#endif

    bool checkNeighborhoodEdge(const cv::Mat & magnitude, const int vStart, const int uStart, const float tLow, const float tHigh)
    {
        if (vStart < 1 || uStart < 1 || vStart >= (magnitude.rows - 1) || uStart >= (magnitude.cols - 1))
        {
            return false;
        }

        for (int v = (vStart - 1); v <= (vStart + 1); ++v)
        {
            for (int u = (uStart - 1); u <= (uStart + 1); ++u)
            {
                float magnVal = magnitude.at<float>(v, u);

                if (magnVal >= tHigh)
                {
                    return true;
                }
                else if (magnVal > tLow && magnVal < tHigh && u >= uStart)
                {
                    if (v != vStart && u != uStart)
                    {
                        if (checkNeighborhoodEdge(magnitude, v, u, tLow, tHigh))
                        {
                            return true;
                        }
                    }                    
                }
            }
        }

        return false;
    }

    void houghTransform(const cv::Mat & src, std::vector<cv::Vec2f> & lines, const int threshold)
    {
        const int diagonal = int(std::pow(std::pow(src.rows, 2) + std::pow(src.cols, 2), 0.5));
        const int degrees = 180;

        int accumulationMatrix[diagonal][degrees] = {0};
        
        for (int v = 0; v < src.rows; ++v)
        {
            for (int u = 0; u < src.cols; ++u)
            {
                if (int(src.at<uchar>(v, u)) == 255)
                {
                    for (int t = 0; t < degrees; ++t)
                    {
                        float theta = t * M_PI / 180;
                        float cos = std::cos(theta);
                        float sin = std::sin(theta);
                        
                        int rho = std::round(u * cos + v * sin);

                        accumulationMatrix[rho][t] += 1;
                    }
                }
            }
        }

        for (int v = 0; v < diagonal; ++v)
        {
            for (int u = 0; u < degrees; ++u)
            {
                if (accumulationMatrix[v][u] > threshold)
                {
                    float theta = u * M_PI / 180;

                    lines.push_back(cv::Vec2f(v, theta));
                }
            }
        }
    }

    void drawLinesOnImage(const cv::Mat & src, cv::Mat & dst, const std::vector<cv::Vec2f> & lines)
    {
        src.copyTo(dst);

        for (size_t i = 0; i < lines.size(); ++i)
		{
			cv::Point pointFirst;
			cv::Point pointLast;

			cv::Vec2f point = lines[i];
			
			float rho = point[0];
			float theta = point[1];

			double a = std::cos(theta);
			double b = std::sin(theta);

			double x0 = a * rho;
			double y0 = b * rho;

			pointFirst.x = cvRound(x0 + 1000 * (-b));
			pointFirst.y = cvRound(y0 + 1000 * (a));
			pointLast.x = cvRound(x0 - 1000 * (-b));
			pointLast.y = cvRound(y0 - 1000 * (a));

			cv::line(dst, pointFirst, pointLast, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
		}
    }

    void mySAD_Disparity7x7(const cv::Mat & leftImage, const cv::Mat & rightImage, cv::Mat & dst)
    {
        cv::Mat paddedLeftImage;
        cv::Mat paddedRightImage;
        
        int padW = 3;
        int padH = 3;

        addZeroPadding(leftImage, paddedLeftImage, padW, padH);
        addZeroPadding(rightImage, paddedRightImage, padW, padH);
        
        dst = cv::Mat::zeros(leftImage.rows, leftImage.cols, CV_32FC1);

        for (int v = 0; v < leftImage.rows; ++v)
        {
            for (int u = 0; u < leftImage.cols; ++u)
            {
                float maxSadValue = 100000000.0f;
                float disparity = 0.0f;

                for (int uRight = ((u >= 128) ? u - 128 : 0); uRight < u; ++uRight)
                {
                    float sadValue = 0.0f;

                    for (int r = -3; r <= 3; ++r)
                    {
                        for (int c = -3; c <= 3; ++c)
                        {
                            sadValue += std::abs((int)(paddedLeftImage.at<uchar>(v + padH + r, u + padW + c)) - (int)(paddedRightImage.at<uchar>(v + padH + r, uRight + padW + c)));
                        }
                    }

                    if (sadValue < maxSadValue)
                    {
                        maxSadValue = sadValue;
                        disparity = u - uRight;
                    }
                }

                dst.at<float>(v, u) = disparity;
            }
        }
    }

    void VDisparity(const cv::Mat & disparity, cv::Mat & dst)
    {
        dst = cv::Mat::zeros(disparity.rows, 128, CV_16UC1);

        for (int v = 0; v < dst.rows; ++v)
        {
            int16_t histogram[128] = {0};

            for (int u = 0; u < disparity.cols; ++u)
            {
                int dispValue = std::floor(disparity.at<float>(v, u));

                if (dispValue > 0)
                {
                    ++histogram[dispValue];
                }
            }

            for (int u = 0; u < dst.cols; ++u)
            {
                dst.at<ushort>(v, u) = histogram[u];
            }
        }
    }

    void plane3points(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, float & a, float & b, float & c ,float & d)
    {
        cv::Point3f p21 = p2 - p1;
        cv::Point3f p31 = p3 - p1;

        a = p21.y * p31.z - p21.z * p31.y;
        b = p21.x * p31.z - p21.z * p31.x;
        c = p21.x * p31.y - p21.y * p31.x;

        d = - (a * p1.x + b * p1.y + c * p1.z);
    }

    float distance_plane_point(cv::Point3f p, float a, float b, float c ,float d)
    {
        return fabs((a * p.x + b * p.y + c * p.z + d)) / (sqrt(a * a + b * b + c * c ));
    }

    void compute3Dpoints(const cv::Mat & disparity, std::vector<cv::Point3f> & points, std::vector<cv::Point2i> & rc)
    {
        // Calibration parameters
        constexpr float focal = 657.475;
        constexpr float baseline = 0.3;
        constexpr float u0 = 509.5;
        constexpr float v0 = 247.15;

        for(int r = 0; r < disparity.rows; ++r)
        {
            for(int c = 0; c < disparity.cols; ++c)
            {
                float disparityVal = disparity.at<float>(r, c);
                
                if(disparityVal > 1)
                {
                    float x = (c - u0) * baseline / disparityVal;
                    float y = - (r - v0) * baseline / disparityVal;
                    float z = - baseline * focal / disparityVal;

                    // For simplicity, only 3D point close less than 30 cm 
                    if(std::abs(z) < 30)
                    {
                        points.push_back(cv::Point3f(x, y, z));
                        rc.push_back(cv::Point2i(r, c));
                    }
                }
            }
        }
    }

    void computePlane(const std::vector<cv::Point3f> & points, const std::vector<cv::Point2i> & rc, std::vector<cv::Point3f> & inliers_best_points, std::vector<cv::Point2i> & inliers_best_rc)
    {
        // RANSAC parameters:
        int N = 10000; // iteration number
        float epsilon = 0.2; // max error for inliers

        int numberPoints = points.size();

        // 50% of probability of being outlier
        float thresholdInliers = (1 - 0.5) * numberPoints;

        std::cout << "RANSAC" << std::endl;
        std::cout << "Points: " << numberPoints << std::endl;

        srand((unsigned)time(NULL));

        for (int iter = 0; iter < N; ++iter)
        {
            int numberInliers = 0;
            
            int i = rand() % numberPoints;
            int j = rand() % numberPoints;
            int k = rand() % numberPoints;

            float a = 0;
            float b = 0;
            float c = 0;
            float d = 0;

            cv::Point3f firstPoint = points[i];
            cv::Point3f secondPoint = points[j];
            cv::Point3f thirdPoint = points[k];

            plane3points(firstPoint, secondPoint, thirdPoint, a, b, c, d);

            for (int index = 0; index < numberPoints; ++index)
            {
                cv::Point3f point = points[index];

                if (index == i || index == j || index == k)
                    continue;

                float dist = distance_plane_point(point, a, b, c, d);

                if (dist < epsilon)
                {
                    inliers_best_points.push_back(point);
                    inliers_best_rc.push_back(cv::Point2i(rc[index].x, rc[index].y));
                    ++numberInliers;
                }
            }

            if (numberInliers >= thresholdInliers)
                break;
            
            inliers_best_points.clear();
            inliers_best_rc.clear();
        }
    }

    void createImageFromInliers(const cv::Mat & leftSrc, cv::Mat & dst, const std::vector<cv::Point2i> & inliersBestRowColumn)
    {
        dst = cv::Mat::zeros(leftSrc.rows, leftSrc.cols, CV_8UC1);

        for (cv::Point2i coordinate : inliersBestRowColumn)
        {
            dst.at<uchar>(coordinate.x, coordinate.y) = leftSrc.at<uchar>(coordinate.x, coordinate.y);
        }
    }

    void imageElementReprojection(const cv::Mat & src, cv::Mat & dst, const std::vector<cv::Point2f> & corners_src)
    {
        std::vector<cv::Point2f> corners_out = {cv::Point2f(0, 0), cv::Point2f(dst.cols - 1, 0), cv::Point2f(dst.cols - 1, dst.rows - 1), cv::Point2f(0, dst.rows - 1)};

        /* nota la posizione dei 4 angoli della copertina in input, questi sono i corrispondenti
         * nell'immagine di uscita.
         * E' importante che l'ordine sia rispettato, quindi top-left, top-right, bottom-right, bottom-left
        */
        cv::Mat homography = cv::findHomography(corners_out, corners_src);

#ifdef EIGEN_FOR_HOMOGRAPHY
        Eigen::Matrix<double, 3, 3> homographyEigen;
        homographyEigen << homography.at<double>(0, 0), homography.at<double>(0, 1), homography.at<double>(0, 2),
                        homography.at<double>(1, 0), homography.at<double>(1, 1), homography.at<double>(1, 2),
                        homography.at<double>(2, 0), homography.at<double>(2, 1), homography.at<double>(2, 2);

        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {            
                Eigen::Matrix<double, 3, 1> coordinateInDstImageEigen; 
                coordinateInDstImageEigen << u, v, 1;

                Eigen::Vector3d coordinateInSrcImageEigen = homographyEigen * coordinateInDstImageEigen;

                int vSrcImage = std::round(coordinateInSrcImageEigen(1) / coordinateInSrcImageEigen(2));
                int uSrcImage = std::round(coordinateInSrcImageEigen(0) / coordinateInSrcImageEigen(2));

                if (vSrcImage >= 0 && vSrcImage < src.rows && uSrcImage >= 0 && uSrcImage < src.cols)
                {
                    dst.at<cv::Vec3b>(v, u) = src.at<cv::Vec3b>(vSrcImage, uSrcImage);
                }
            }
        }
#else
        for (int v = 0; v < dst.rows; ++v)
        {
            for (int u = 0; u < dst.cols; ++u)
            {
                cv::Mat coordinateInDstImage = (cv::Mat_<double>(3, 1) << u, v, 1);
                cv::Mat coordinateInSrcImage = homography * coordinateInDstImage;

                int vSrcImage = std::round(coordinateInSrcImage.at<double>(1, 0) / coordinateInSrcImage.at<double>(2, 0));
                int uSrcImage = std::round(coordinateInSrcImage.at<double>(0, 0) / coordinateInSrcImage.at<double>(2, 0));

                if (vSrcImage >= 0 && vSrcImage < src.rows && uSrcImage >= 0 && uSrcImage < src.cols)
                {
                    dst.at<cv::Vec3b>(v, u) = src.at<cv::Vec3b>(vSrcImage, uSrcImage);
                }
            }
        }
#endif
    }
}