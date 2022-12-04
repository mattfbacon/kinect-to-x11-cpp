#include <fstream>
#include <iostream>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "opencv2/highgui.hpp"

using namespace libfreenect2;

constexpr float SCALE = 0.00392f;
constexpr float CONFIDENCE_THRESHOLD = 0.5f;

int main() {
	auto net = cv::dnn::readNetFromDarknet("yolo.cfg", "yolo.weights");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
	auto const out_names = net.getUnconnectedOutLayersNames();

	namedWindow("window uwu", cv::WINDOW_NORMAL);

	Freenect2 ctx;

	if (ctx.enumerateDevices() == 0) {
		std::cerr << "no devices found\n";
		return 1;
	}

	auto* const dev = ctx.openDefaultDevice();

	SyncMultiFrameListener listener(Frame::Color | Frame::Depth);
	FrameMap frames;

	dev->setColorFrameListener(&listener);
	dev->setIrAndDepthFrameListener(&listener);

	if (!dev->start()) {
		std::cerr << "could not start device\n";
		return 1;
	}

	auto* const registration = new Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
	Frame undistorted(512, 424, 4), registered(512, 424, 4), bigdepth(1920, 1082, 4);
	bigdepth.format = Frame::Format::Float;

	for (;;) {
		if (!listener.waitForNewFrame(frames, 10 * 1000)) {
			std::cerr << "timeout waiting for new frame\n";
			return 1;
		}

		auto const rgb = frames[libfreenect2::Frame::Color];
		auto const depth = frames[libfreenect2::Frame::Depth];

		registration->apply(rgb, depth, &undistorted, &registered, true, &bigdepth);

		// cv::Mat rgb_mat{ static_cast<int>(rgb->height), static_cast<int>(rgb->width), CV_8UC3, rgb->data };
		cv::Mat rgb_mat{ static_cast<int>(rgb->height), static_cast<int>(rgb->width), CV_8UC3, rgb->data };
		for (int row = 0; row < rgb_mat.rows; ++row) {
			for (int col = 0; col < rgb_mat.cols; ++col) {
				auto const* image_ent = &rgb->data[(row * rgb_mat.cols + col) * 4];
				auto& mat_ent = rgb_mat.at<std::array<uint8_t, 3>>(col, row);
				mat_ent[0] = image_ent[0];
				mat_ent[1] = image_ent[1];
				mat_ent[2] = image_ent[2];
			}
		}
		cv::Mat depth_mat{ static_cast<int>(depth->height), static_cast<int>(depth->width), CV_32FC1, depth->data };

		std::vector<cv::Mat> outs;
		{
			cv::Mat blob;
			// swap B and R if frame is RGB because OpenCV uses BGR
			cv::dnn::blobFromImage(rgb_mat, blob, 1.0, cv::Size{ 608, 608 }, cv::Scalar(), rgb->format == libfreenect2::Frame::RGBX, false, CV_8U);
			net.setInput(blob, "", SCALE);
			if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
				cv::Mat image_info = (cv::Mat_<float>(1, 3) << rgb_mat.rows, rgb_mat.cols, 1.6f);
				net.setInput(image_info, "im_info");
			}

			net.forward(outs, out_names);
			assert(net.getLayer(net.getUnconnectedOutLayers()[0])->type == "Region");
			for (size_t i = 0; i < outs.size(); ++i) {
				auto* data = reinterpret_cast<float*>(outs[i].data);
				for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
					auto const scores = outs[i].row(j).colRange(5, outs[i].cols);
					cv::Point class_id_point;
					double confidence;
					cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
					if (confidence > CONFIDENCE_THRESHOLD) {
						auto const center_x = static_cast<int>(data[0] * static_cast<float>(rgb_mat.cols));
						auto const center_y = static_cast<int>(data[1] * static_cast<float>(rgb_mat.rows));
						auto const width = static_cast<int>(data[2] * static_cast<float>(rgb_mat.cols));
						auto const height = static_cast<int>(data[3] * static_cast<float>(rgb_mat.rows));
						auto const left = center_x - width / 2;
						auto const top = center_y - height / 2;
						std::cerr << "at rect " << width << "x" << height << "+" << left << "x" << top << " there is " << class_id_point.x << " with confidence "
											<< static_cast<float>(confidence) << '\n';
					}
				}
			}
		}
		std::cerr << "---\n";

		listener.release(frames);
	}
}
