#include <iostream>
#include <vector>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "Eigen/Dense"



int main()
{
	// pyramid: std::vector<cv::Mat> for I and J
	cv::Mat img_i = cv::imread("img1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img_j = cv::imread("img2.jpg", cv::IMREAD_GRAYSCALE);


	// tracking point
  	cv::Point2i p(std::floor(img_i.cols/2), std::floor(img_i.rows/2));


    int Lm = 2; // deepest layer of pyramid
    int omega_x = 2, omega_y = 2;
    int K = 10; // number of iterations for iterative L-K

    // vector for storing image dimensions for different layers
    // pair of [nx ny] for L-th layer: [2*L 2*L+1]
    int dimensions[(Lm+1) * 2];
    dimensions[0] = img_i.cols;
    dimensions[1] = img_i.rows;
    
    // vector for greyscale values of original image
    std::vector<cv::Mat> I;
    std::vector<cv::Mat> J;
    I.push_back(img_i);
    J.push_back(img_j);
 

    // finding dimensions for all layers
    for (size_t L = 1; L <= Lm; L++)
    {
        // L-th [nx ny]: [floor((nx+1)/2) floor((ny+1)/2)]
        dimensions[2*L] = std::floor( (dimensions[2*(L-1)] + 1) / 2 );
        dimensions[2*L + 1] = std::floor( (dimensions[2*(L-1) + 1] + 1) / 2 );

        // making layers of right size to edit their pixels later
        I.push_back(cv::Mat(dimensions[2*L], dimensions[2*L + 1], CV_8UC1)); 
        J.push_back(cv::Mat(dimensions[2*L], dimensions[2*L + 1], CV_8UC1)); 
    }


    // building pyramid layers
    for (int L = 1; L <= Lm; L++)
    {
        for (int y = 1; y < dimensions[2 * L]; y++)
        {
            for (int x = 1; x < dimensions[2 * L + 1]; x++)
            {
                I[L].at<uchar>(x-1,y-1) = 1/4 * I[L-1].at<uchar>(2*x,2*y) +
                                   		  1/8 * (I[L-1].at<uchar>(2*x-1,2*y) + I[L-1].at<uchar>(2*x+1,2*y) + I[L-1].at<uchar>(2*x,2*y-1) +
                                        	I[L-1].at<uchar>(2*x,2*y+1)) +
                                      	  1/16 * (I[L-1].at<uchar>(2*x-1,2*y-1) + I[L-1].at<uchar>(2*x+1,2*y+1) +
                                        	I[L-1].at<uchar>(2*x-1,2*y+1) + I[L-1].at<uchar>(2*x+1,2*y+1));
                
                J[L].at<uchar>(x-1,y-1) = 1/4 * J[L-1].at<uchar>(2*x,2*y) +
                                      	  1/8 * (J[L-1].at<uchar>(2*x-1,2*y) + J[L-1].at<uchar>(2*x+1,2*y) + J[L-1].at<uchar>(2*x,2*y-1) +
                                        	J[L-1].at<uchar>(2*x,2*y+1)) +
                                      	  1/16 * (J[L-1].at<uchar>(2*x-1,2*y-1) + J[L-1].at<uchar>(2*x+1,2*y+1) +
                                        	J[L-1].at<uchar>(2*x-1,2*y+1) + J[L-1].at<uchar>(2*x+1,2*y+1));
            }
        }
    }

    std::vector<Eigen::Vector2d> g(Lm); // pyramidal guess [0, 0]
    std::vector<int> u((Lm+1) * 2); // vector for point u for all layers
    u[0] = p.x;
    u[1] = p.y;
    cv::Point2i v; // point v
    Eigen::Vector2d bk;
    bk << 0, 0; // image mismatch vector
    Eigen::Vector2d etak; // guess for next iteration
    std::vector<Eigen::Vector2d> d(Lm); // final optical flow

    for (int L = Lm; L >= 0; L--)
    {
        // finding u
        u[2*L] = (u[0]) / std::pow(2, L);
        u[2*L + 1] = (u[1]) / std::pow(2, L);

        // derivative Ix, Iy
        std::vector<std::vector<int>> Ix;
        std::vector<std::vector<int>> Iy;
        for (int y = p.y - omega_y; y < p.y + omega_y; y++)
        {
            for (int x = p.x - omega_x; x < p.x + omega_x; x++)
            {
               Ix[x][y] = (I[L].at<uchar>(x+1,y) - I[L].at<uchar>(x-1,y)) / 2;
               Iy[x][y] = (I[L].at<uchar>(x,y+1) - I[L].at<uchar>(x,y-1)) / 2;
            }
        }

        // spatial gradient matrix
        Eigen::Matrix2d G;
        G << 0,0,0,0;
        for (int y = p.y - omega_y; y <= p.y + omega_y; y++)
        {
            for (int x = p.x - omega_x; x <= p.x + omega_x; x++)
            {
                G(0,0) += std::pow(Ix[x][y], 2);
                G(0,1) += Ix[x][y] * Iy[x][y];
                G(1,0) += Ix[x][y] * Iy[x][y];
                G(1,1) += std::pow(Iy[x][y], 2);
            }
        }
        


        std::vector<std::vector<int>> deltaIk;

        // initialization of iterative L-K
        std::vector<Eigen::Vector2d> nu(K);
        nu[0] << 0,0; // [0, 0]

        for (int k = 1; k <= K; k++)
        {
            // image difference
            for (int y = p.y - omega_y; y <= p.y + omega_y; y++)
        	{
            	for (int x = p.x - omega_x; x <= p.x + omega_x; x++)
            	{
                    deltaIk[x][y] = I[L].at<uchar>(x,y) - 
                    				J[L].at<uchar>(x + g[L](0,0) + nu[k-1](0,0), y + g[L](1,0) + nu[k-1](1,0));
                }
            }

            // image mismatch vector
            for (int y = p.y - omega_y; y <= p.y + omega_y; y++)
        	{
            	for (int x = p.x - omega_x; x <= p.x + omega_x; x++)
            	{
                    bk(0,0) += deltaIk[x][y] * Ix[x][y];
                    bk(1,0) += deltaIk[x][y] * Iy[x][y];
                }
            }

            // optical flow
            etak = G.inverse() * bk;
            
            // guess for next iteration
            nu[k](0,0) = nu[k-1](0,0) + etak(0,0);
            nu[k](1,0) = nu[k-1](1,0) + etak(1,0);
                        
        }

        // final optical flow at level L
        d[L](0,0) = nu[K](0,0);
        d[L](1,0) = nu[K](1,0);

        // guess for next level L-1
        g[L-1](0,0) = 2 * (g[L](0,0) + d[L](0,0));
        g[L-1](1,0) = 2 * (g[L](1,0) + d[L](1,0));

    }

    // final optical flow vector
    Eigen::Vector2d d_final;
    d_final(0,0) = g[0](0,0) + d[0](0,0);
    d_final(1,0) = g[0](1,0) + d[0](1,0);

    // location of point on J
    v.x = u[0] + d_final(0,0);
    v.y = u[1] + d_final(1,0);


    return 0;
}