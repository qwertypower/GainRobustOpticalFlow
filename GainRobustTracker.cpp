//
//  GainRobustTracker.cpp
//  OnlinePhotometricCalibration
//
//  Created by Paul on 17.11.17.
//  Copyright (c) 2017-2018 Paul Bergmann and co-authors. All rights reserved.
//
//  See LICENSE.txt
//

#include "GainRobustTracker.h"

#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>
//#define EIGEN_DEFAULT_TO_ROW_MAJOR 
#include <Eigen/dense>

GainRobustTracker::GainRobustTracker(int patch_size, int pyramid_levels)
{
	// Initialize patch size and pyramid levels
	m_patch_size = (patch_size - 1) / 2 - 1;
	m_pyramid_levels = pyramid_levels;
}

// Todo: change frame_1 frame 2 to ref (or const ref), pts_1 to ref
double GainRobustTracker::trackImagePyramids(cv::Mat frame_1,
	cv::Mat frame_2,
	std::vector<cv::Point2f> pts_1,
	std::vector<cv::Point2f>& pts_2,
	std::vector<uchar>& point_status)
{
	// All points valid in the beginning of tracking
	std::vector<uchar> point_validity(pts_1.size());
	for (int i = 0; i < pts_1.size(); i++)
	{
		point_validity[i] = 1;
	}

	// Calculate image pyramid of frame 1 and frame 2
	std::vector<cv::Mat> new_pyramid;
	cv::buildPyramid(frame_2, new_pyramid, m_pyramid_levels);

	std::vector<cv::Mat> old_pyramid;
	cv::buildPyramid(frame_1, old_pyramid, m_pyramid_levels);

	// Temporary vector to update tracking estiamtes over time
	std::vector<cv::Point2f> tracking_estimates = pts_1;

	double all_exp_estimates = 0.0;
	int nr_estimates = 0;

	// Iterate all pyramid levels and perform gain robust KLT on each level (coarse to fine)
	for (int level = (int)new_pyramid.size() - 1; level >= 0; level--)
	{
		// Scale the input points and tracking estimates to the current pyramid level
		std::vector<cv::Point2f> scaled_tracked_points;
		std::vector<cv::Point2f> scaled_tracking_estimates;
		for (int i = 0; i < pts_1.size(); i++)
		{
			cv::Point2f scaled_point;
			float level_inv = 1.f / (1 << level);
			scaled_point.x = (float)(pts_1.at(i).x * level_inv);
			scaled_point.y = (float)(pts_1.at(i).y * level_inv);
			scaled_tracked_points.push_back(scaled_point);

			cv::Point2f scaled_estimate;
			scaled_estimate.x = (float)(tracking_estimates.at(i).x * level_inv);
			scaled_estimate.y = (float)(tracking_estimates.at(i).y * level_inv);
			scaled_tracking_estimates.push_back(scaled_estimate);
		}

		// Perform tracking on current level
		double exp_estimate = trackImageExposurePyr(old_pyramid.at(level),
			new_pyramid.at(level),
			scaled_tracked_points,
			scaled_tracking_estimates,
			point_validity);

		// Optional: Do something with the estimated exposure ratio
		// std::cout << "Estimated exposure ratio of current level: " << exp_estimate << std::endl;

		// Average estimates of each level later
		all_exp_estimates += exp_estimate;
		nr_estimates++;

		// Update the current tracking result by scaling down to pyramid level 0
		for (int i = 0; i < scaled_tracking_estimates.size(); i++)
		{
			if (point_validity.at(i) == 0)
				continue;

			cv::Point2f scaled_point;
			scaled_point.x = (float)(scaled_tracking_estimates.at(i).x * (1 << level));
			scaled_point.y = (float)(scaled_tracking_estimates.at(i).y * (1 << level));

			tracking_estimates.at(i) = scaled_point;
		}
	}

	// Write result to output vectors passed by reference
	pts_2 = tracking_estimates;
	point_status = point_validity;

	// Average exposure ratio estimate
	double overall_exp_estimate = all_exp_estimates / nr_estimates;
	return overall_exp_estimate;
}


int __float_as_int(float a)
{
	int r;
	memcpy(&r, &a, sizeof(r));
	return r;
}

float __int_as_float(int a)
{
	float r;
	memcpy(&r, &a, sizeof(r));
	return r;
}
float my_faster_logf(float a)
{
	float m, r, s, t, i, f;
	int32_t e;

	e = (__float_as_int(a) - 0x3f2aaaab) & 0xff800000;
	m = __int_as_float(__float_as_int(a) - e);
	i = (float)e * 1.19209290e-7f; // 0x1.0p-23
	/* m in [2/3, 4/3] */
	f = m - 1.0f;
	s = f * f;
	/* Compute log1p(f) for f in [-1/3, 1/3] */
	r = fmaf(0.230836749f, f, -0.279208571f); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
	t = fmaf(0.331826031f, f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2
	r = fmaf(r, s, t);
	r = fmaf(r, s, f);
	r = fmaf(i, 0.693147182f, r); // 0x1.62e430p-1 // log(2) 
	return r;
}

/**
 * For a reference on the meaning of the optimization variables and the overall concept of this function
 * refer to the photometric calibration paper
 * introducing gain robust KLT tracking by Kim et al.
 */
 // Todo: change Mat and vector to ref

void solveEigen(cv::Mat& A, cv::Mat& b, cv::Mat& c)
{
	Eigen::Map<Eigen::MatrixXd> A_Eigen(A.ptr<double>(), A.rows, A.cols);
	Eigen::Map<Eigen::MatrixXd> b_Eigen(b.ptr<double>(), b.rows, b.cols);
	Eigen::MatrixXd x = A_Eigen.ldlt().solve(b_Eigen);
	c = cv::Mat(x.rows(), x.cols(), CV_64F, (void*)x.data());
}

void mulMatTEigen(cv::Mat& a, cv::Mat& b, cv::Mat& c)
{
	Eigen::Map<Eigen::MatrixXd> A_Eigen(a.ptr<double>(), a.rows, a.cols);
	Eigen::Map<Eigen::MatrixXd> b_Eigen(b.ptr<double>(), b.rows, b.cols);
	Eigen::MatrixXd x = -A_Eigen.transpose() * b_Eigen;
	c = cv::Mat(x.rows(), x.cols(), CV_64F, (void*)x.data());
}


#define MAT_ELEM_PTR_FAST( mat, row, col, pix_size ) ( (mat).data + (size_t)(mat).step*(row) + (pix_size)*(col) )
#define MAT_ELEM( mat, elemtype, row, col ) ( *(elemtype*)MAT_ELEM_PTR_FAST( mat, row, col, sizeof(elemtype)) )

/*double GainRobustTracker::trackImageExposurePyr(cv::Mat old_image,
	cv::Mat new_image,
	std::vector<cv::Point2f> input_points,
	std::vector<cv::Point2f>& output_points,
	std::vector<uchar>& point_validity)
{
	// Number of points to track
	int nr_points = static_cast<int>(input_points.size());

	// Updated point locations which are updated throughout the iterations
	if (output_points.size() == 0)
	{
		output_points = input_points;
	}
	else if (output_points.size() != input_points.size())
	{
		std::cout << "ERROR - OUTPUT POINT SIZE != INPUT POINT SIZE!" << std::endl;
		return -1;
	}

	// Input image dimensions
	int image_rows = new_image.rows;
	int image_cols = new_image.cols;

	// Final exposure time estimate
	double K_total = 0.0;
	Eigen::setNbThreads(1);
	for (int round = 0; round < 1; round++)
	{
		// Get the currently valid points
		int nr_valid_points = getNrValidPoints(point_validity);

		// Allocate space for W,V matrices
		W.create(2 * nr_valid_points, 1, CV_64F);
		V.create(2 * nr_valid_points, 1, CV_64F);
		// Allocate space for U_INV and the original Us
		U_INV.create(2 * nr_valid_points, 2 * nr_valid_points, CV_64F);
		memset(U_INV.data, 0, U_INV.rows * U_INV.cols * sizeof(double));
		memset(W.data, 0, W.rows * W.cols * sizeof(double));
		memset(V.data, 0, V.rows * V.cols * sizeof(double));
		std::vector<cv::Mat> Us;
		Us.reserve(input_points.size());

		double lambda = 0;
		double m = 0;

		int absolute_point_index = -1;

		for (int p = 0; p < input_points.size(); p++)
		{
			if (point_validity[p] == 0)
			{
				continue;
			}

			absolute_point_index++;

			// Build U matrix
			cv::Mat U(2, 2, CV_64F, 0.0);
			double* U_ptr = (double*)U.data;

			// Bilinear image interpolation
			cv::Mat patch_intensities_1;
			cv::Mat patch_intensities_2;
			int absolute_patch_size = ((m_patch_size + 1) * 2 + 1);  // Todo: why m_patch_size+1?
			cv::getRectSubPix(new_image, cv::Size(absolute_patch_size, absolute_patch_size), output_points[p], patch_intensities_2, CV_32F);
			cv::getRectSubPix(old_image, cv::Size(absolute_patch_size, absolute_patch_size), input_points[p], patch_intensities_1, CV_32F);
			
			int sz = 2 * m_patch_size + 1;
			// Go through image patch around this point
			for (int r = 0; r < sz; r++)
			{
				for (int c = 0; c < sz; c++)
				{
					// Fetch patch intensity values
					double i_frame_1 = MAT_ELEM(patch_intensities_1, float, 1 + r, 1 + c);//patch_intensities_1.at<float>(1 + r, 1 + c);
					double i_frame_2 = MAT_ELEM(patch_intensities_2, float, 1 + r, 1 + c);//patch_intensities_2.at<float>(1 + r, 1 + c);

					if (i_frame_1 < 1)
						i_frame_1 = 1;
					if (i_frame_2 < 1)
						i_frame_2 = 1;

					// Estimate patch gradient values
					double grad_1_x = (MAT_ELEM(patch_intensities_1, float, 1 + r, 1 + c) - MAT_ELEM(patch_intensities_1, float, 1 + r, c)) * 0.5;//(patch_intensities_1.at<float>(1 + r, 1 + c + 1) - patch_intensities_1.at<float>(1 + r, 1 + c - 1)) / 2;
					double grad_1_y = (MAT_ELEM(patch_intensities_1, float, r + 2, 1 + c) - MAT_ELEM(patch_intensities_1, float, r, 1 + c)) * 0.5;//(patch_intensities_1.at<float>(1 + r + 1, 1 + c) - patch_intensities_1.at<float>(1 + r - 1, 1 + c)) / 2;

					double grad_2_x = (MAT_ELEM(patch_intensities_2, float, 1 + r, c + 2) - MAT_ELEM(patch_intensities_2, float, 1 + r, c)) * 0.5;//(patch_intensities_2.at<float>(1 + r, 1 + c + 1) - patch_intensities_2.at<float>(1 + r, 1 + c - 1)) / 2;
					double grad_2_y = (MAT_ELEM(patch_intensities_2, float, r + 2, 1 + c) - MAT_ELEM(patch_intensities_2, float, r, 1 + c)) * 0.5;//(patch_intensities_2.at<float>(1 + r + 1, 1 + c) - patch_intensities_2.at<float>(1 + r - 1, 1 + c)) / 2;

					double a = grad_2_x / i_frame_2 + grad_1_x / i_frame_1;//(1.0 / i_frame_2) * grad_2_x + (1.0 / i_frame_1) * grad_1_x;
					double b = grad_2_y / i_frame_2 + grad_1_y / i_frame_1;//(1.0 / i_frame_2) * grad_2_y + (1.0 / i_frame_1) * grad_1_y;
					double beta = my_faster_logf(i_frame_2 / i_frame_1);

					
					*U_ptr++ += 0.5 * a * a;
					*U_ptr++ += 0.5 * a * b;
					*U_ptr++ += 0.5 * a * b;
					*U_ptr++ += 0.5 * b * b;
					U_ptr -= 4;

					int idx = 2 * absolute_point_index;
					MAT_ELEM(W, double, idx, 0) -= a;
					MAT_ELEM(W, double, idx + 1, 0) -= b;

					MAT_ELEM(V, double, idx, 0) -= beta * a;
					MAT_ELEM(V, double, idx + 1, 0) -= beta * b;

					lambda += 2;
					m += 2 * beta;
				}
			}

			//Back up U for re-substitution
			Us.push_back(U);

			//Invert matrix U for this point and write it to diagonal of overall U_INV matrix
			cv::Mat U_INV_p = U.inv();
			
			int idx = 2 * absolute_point_index;
			MAT_ELEM(U_INV, double, idx, idx) = MAT_ELEM(U_INV_p, double, 0, 0);
			MAT_ELEM(U_INV, double, idx, idx + 1) = MAT_ELEM(U_INV_p, double, 0, 1);
			MAT_ELEM(U_INV, double, idx + 1, idx) = MAT_ELEM(U_INV_p, double, 1, 0);
			MAT_ELEM(U_INV, double, idx + 1, idx + 1) = MAT_ELEM(U_INV_p, double, 1, 1);
		}

		// Todo: check if opencv utilizes the sparsity of U
		//solve for the exposure
		cv::Mat K_MAT;
		cv::Mat WtU_INV; mulMatTEigen(W, U_INV, WtU_INV);// = -W.t() * U_INV;
		//cv::solve(WtU_INV * W + lambda, WtU_INV * V + m, K_MAT);
		cv::Mat WtU_INV_W = WtU_INV * W + lambda;
		cv::Mat WtU_INV_V = WtU_INV * V + m;
		solveEigen(WtU_INV_W, WtU_INV_V, K_MAT);
		double K = K_MAT.at<double>(0, 0);

		// Solve for the displacements
		absolute_point_index = -1;
		for (int p = 0; p < nr_points; p++)
		{
			if (point_validity[p] == 0)
				continue;

			absolute_point_index++;
			int idx = 2 * absolute_point_index;
			cv::Mat U_p = Us.at(absolute_point_index);
			cv::Mat V_p = V(cv::Rect(0, idx, 1, 2));
			cv::Mat W_p = W(cv::Rect(0, idx, 1, 2));

			cv::Mat displacement;
			V_p = V_p - K * W_p;
			//cv::solve(U_p, V_p - K * W_p, displacement);
			solveEigen(U_p, V_p, displacement);

			output_points.at(p).x += MAT_ELEM(displacement, double, 0, 0);
			output_points.at(p).y += MAT_ELEM(displacement, double, 1, 0);

			// Filter out this point if too close at the boundaries
			int filter_margin = 2;
			double x = output_points.at(p).x;
			double y = output_points.at(p).y;
			// Todo: the latter two should be ">=" ?
			if (x < filter_margin || y < filter_margin || x > image_cols - filter_margin || y > image_rows - filter_margin)
			{
				point_validity[p] = 0;
			}
		}

		K_total += K;
	}

	return exp(K_total);
}*/


double GainRobustTracker::trackImageExposurePyr(cv::Mat old_image,
	cv::Mat new_image,
	std::vector<cv::Point2f> input_points,
	std::vector<cv::Point2f>& output_points,
	std::vector<uchar>& point_validity)
{
	// Number of points to track
	int nr_points = static_cast<int>(input_points.size());
	Eigen::setNbThreads(1);
	// Updated point locations which are updated throughout the iterations
	if (output_points.size() == 0)
	{
		output_points = input_points;
	}
	else if (output_points.size() != input_points.size())
	{
		std::cout << "ERROR - OUTPUT POINT SIZE != INPUT POINT SIZE!" << std::endl;
		return -1;
	}

	// Input image dimensions
	int image_rows = new_image.rows;
	int image_cols = new_image.cols;

	// Final exposure time estimate
	double K_total = 0.0;

	for (int round = 0; round < 1; round++)
	{
		// Get the currently valid points
		int nr_valid_points = getNrValidPoints(point_validity);

		// Allocate space for W,V matrices
		W.resize(2 * nr_valid_points, 1);
		memset(W.data(), 0, W.rows() * W.cols() * sizeof(double));//W.setZero();
		V.resize(2 * nr_valid_points, 1);
		memset(V.data(), 0, V.rows() * V.cols() * sizeof(double));//V.setZero();

		// Allocate space for U_INV and the original Us
		U_INV.resize(2 * nr_valid_points, 2 * nr_valid_points);
		memset(U_INV.data(), 0, U_INV.rows() * U_INV.cols() * sizeof(double));
		//U_INV.setZero();

		std::vector<Eigen::Matrix2d> Us;
		Us.reserve(input_points.size());

		double lambda = 0;
		double m = 0;

		int absolute_point_index = -1;

		for (int p = 0; p < input_points.size(); p++)
		{
			if (point_validity[p] == 0)
			{
				continue;
			}

			absolute_point_index++;
			int idx = 2 * absolute_point_index;

			// Build U matrix
			Eigen::Matrix2d U;
			U.setZero();

			// Bilinear image interpolation
			cv::Mat patch_intensities_1;
			cv::Mat patch_intensities_2;
			int absolute_patch_size = ((m_patch_size + 1) * 2 + 1);  // Todo: why m_patch_size+1?
			cv::getRectSubPix(new_image, cv::Size(absolute_patch_size, absolute_patch_size), output_points[p], patch_intensities_2, CV_32F);
			cv::getRectSubPix(old_image, cv::Size(absolute_patch_size, absolute_patch_size), input_points[p], patch_intensities_1, CV_32F);
			int sz = 2 * m_patch_size + 1;
			// Go through image patch around this point
			for (int r = 0; r < sz; r++)
			{
				for (int c = 0; c < sz; c++)
				{
					// Fetch patch intensity values
					double i_frame_1 = MAT_ELEM(patch_intensities_1, float, 1 + r, 1 + c);
					double i_frame_2 = MAT_ELEM(patch_intensities_2, float, 1 + r, 1 + c);

					if (i_frame_1 < 1)
						i_frame_1 = 1;
					if (i_frame_2 < 1)
						i_frame_2 = 1;

					// Estimate patch gradient values
					double grad_1_x = (MAT_ELEM(patch_intensities_1, float, 1 + r, 1 + c) - MAT_ELEM(patch_intensities_1, float, 1 + r, c)) * 0.5;
					double grad_1_y = (MAT_ELEM(patch_intensities_1, float, r + 2, 1 + c) - MAT_ELEM(patch_intensities_1, float, r, 1 + c)) * 0.5;

					double grad_2_x = (MAT_ELEM(patch_intensities_2, float, 1 + r, c + 2) - MAT_ELEM(patch_intensities_2, float, 1 + r, c)) * 0.5;
					double grad_2_y = (MAT_ELEM(patch_intensities_2, float, r + 2, 1 + c) - MAT_ELEM(patch_intensities_2, float, r, 1 + c)) * 0.5;

					double a = grad_2_x / i_frame_2 + grad_1_x / i_frame_1;
					double b = grad_2_y / i_frame_2 + grad_1_y / i_frame_1;
					double beta = my_faster_logf(i_frame_2 / i_frame_1);

					U(0, 0) += 0.5 * a * a;
					U(0, 1) += 0.5 * a * b;
					U(1, 0) += 0.5 * a * b;
					U(1, 1) += 0.5 * b * b;
					
					W(idx, 0) -= a;
					W(idx + 1, 0) -= b;

					V(idx, 0) -= beta * a;
					V(idx + 1, 0) -= beta * b;

					lambda += 2;
					m += 2 * beta;
				}
			}

			//Back up U for re-substitution
			Us.push_back(U);

			//Invert matrix U for this point and write it to diagonal of overall U_INV matrix
			Eigen::Matrix2d U_INV_p = U.inverse();
			
			U_INV(idx, idx) = U_INV_p(0, 0);
			U_INV(idx, idx + 1) = U_INV_p(0, 1);
			U_INV(idx + 1, idx) = U_INV_p(1, 0);
			U_INV(idx + 1, idx + 1) = U_INV_p(1, 1);
		}

		// Todo: check if opencv utilizes the sparsity of U
		//solve for the exposure
		Eigen::MatrixXd K_MAT;
		Eigen::MatrixXd Wt = -W.transpose();
		Eigen::MatrixXd WtUINV = Wt * U_INV;
		Eigen::MatrixXd A; 
		A.noalias() = (WtUINV * W);
		A.noalias() = (A.array() + lambda).matrix();
		Eigen::MatrixXd b;
		b.noalias() = (WtUINV * V);
		b.noalias() = (b.array() + m).matrix();
		//solve
		K_MAT.noalias() = A.ldlt().solve(b);
		double K = K_MAT(0, 0);

		// Solve for the displacements
		absolute_point_index = -1;
		for (int p = 0; p < nr_points; p++)
		{
			if (point_validity[p] == 0)
				continue;

			absolute_point_index++;
			int idx = 2 * absolute_point_index;
			Eigen::MatrixXd U_p = Us.at(absolute_point_index);

			Eigen::MatrixXd V_p = V.block(idx, 0, 2, 1);
			Eigen::MatrixXd W_p = W.block(idx, 0, 2, 1);

			Eigen::MatrixXd displacement;
			Eigen::MatrixXd B;
			B.noalias() = V_p - K * W_p;
			displacement = U_p.ldlt().solve(B);
			output_points.at(p).x += displacement(0, 0);
			output_points.at(p).y += displacement(1, 0);

			// Filter out this point if too close at the boundaries
			int filter_margin = 2;
			double x = output_points.at(p).x;
			double y = output_points.at(p).y;
			// Todo: the latter two should be ">=" ?
			if (x < filter_margin || y < filter_margin || x > image_cols - filter_margin || y > image_rows - filter_margin)
			{
				point_validity[p] = 0;
			}
		}

		K_total += K;
	}

	return exp(K_total);
}


int GainRobustTracker::getNrValidPoints(std::vector<uchar> validity_vector)
{
	// Simply sum up the validity vector
	return std::accumulate(validity_vector.begin(), validity_vector.end(), 0);
}
