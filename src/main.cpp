#include <cstdlib>
#include <fstream>
#include <iostream>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

constexpr float SCALE = 0.00392f;
constexpr float CONFIDENCE_THRESHOLD = 0.5f;

// converts RGBX|BGRX to BGR mat
static void load_color(size_t rows, size_t cols, unsigned char const* data, bool is_rgb, cv::Mat& mat) {
	mat.create(static_cast<int>(rows), static_cast<int>(cols), CV_8UC3);

	for (int row = 0; row < mat.rows; ++row) {
		for (int col = 0; col < mat.cols; ++col) {
			auto const* image_ent = &data[(row * mat.cols + col) * 4];
			auto& mat_ent = mat.at<std::array<uint8_t, 3>>(row, col);
			if (is_rgb) {
				mat_ent[0] = image_ent[2];
				mat_ent[2] = image_ent[0];
			} else {
				mat_ent[0] = image_ent[0];
				mat_ent[2] = image_ent[2];
			}
			mat_ent[1] = image_ent[1];
		}
	}
}

static void load_depth(size_t rows, size_t cols, unsigned char* data, cv::Mat& mat) {
	mat = cv::Mat{ static_cast<int>(rows), static_cast<int>(cols), CV_32FC1, data };
}

static void read_from_bin(char const* const path, cv::Mat& mat) {
	std::ifstream file(path);
	alignas(size_t) char buf[sizeof(size_t)];

	file.read(buf, sizeof(size_t));
	auto const cols = *(reinterpret_cast<size_t const*>(buf));

	file.read(buf, sizeof(size_t));
	auto const rows = *(reinterpret_cast<size_t const*>(buf));

	file.read(buf, sizeof(size_t));
	auto const bytes_per_pixel = *(reinterpret_cast<size_t const*>(buf));

	file.read(buf, sizeof(size_t));
	auto const format = static_cast<libfreenect2::Frame::Format>(*(reinterpret_cast<size_t const*>(buf)));

	auto* const data = new unsigned char[cols * rows * bytes_per_pixel];
	file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(cols * rows * bytes_per_pixel));

	file.close();

	switch (format) {
		case libfreenect2::Frame::BGRX:
		case libfreenect2::Frame::RGBX:
			load_color(rows, cols, data, format == libfreenect2::Frame::RGBX, mat);
			delete[] data;
			break;
		case libfreenect2::Frame::Float:
			load_depth(rows, cols, data, mat);
			break;

		default:
			std::cerr << "unknown frame type: " << format << '\n';
			return;
	}
}

[[maybe_unused]] static void read_from_kinect(cv::Mat& color_mat, cv::Mat& depth_mat) {
	libfreenect2::Freenect2 ctx;

	if (ctx.enumerateDevices() == 0) {
		std::cerr << "no devices found\n";
		std::exit(1);
	}

	auto* const dev = ctx.openDefaultDevice();

	libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
	libfreenect2::FrameMap frames;

	dev->setColorFrameListener(&listener);
	dev->setIrAndDepthFrameListener(&listener);

	if (!dev->start()) {
		std::cerr << "could not start device\n";
		std::exit(1);
	}

	auto* const registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
	libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), bigdepth(1920, 1082, 4);
	bigdepth.format = libfreenect2::Frame::Format::Float;
	if (!listener.waitForNewFrame(frames, 10 * 1000)) {
		std::cerr << "timeout waiting for new frame\n";
		std::exit(1);
	}

	auto const rgb = frames[libfreenect2::Frame::Color];
	auto const depth = frames[libfreenect2::Frame::Depth];

	registration->apply(rgb, depth, &undistorted, &registered, true, &bigdepth);

	load_color(rgb->height, rgb->width, rgb->data, rgb->format == libfreenect2::Frame::RGBX, color_mat);
	load_depth(depth->height, depth->width, depth->data, depth_mat);

	// TODO - do we need to save the listener and free all the frames at the end?
	// listener.release(frames);
}

class Processor {
private:
	cv::dnn::Net net;
	cv::Mat blob;

public:
	Processor() : net(cv::dnn::readNetFromDarknet("yolo.cfg", "yolo.weights")) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
	}

	void process_bgr(cv::Mat const& mat) {
		std::vector<cv::String> out_names = net.getLayerNames();
		std::vector<cv::Mat> outs;
		{
			// it calls for a BGR frame; we don't swap B and R here because the caller provides a BGR frame
			cv::dnn::blobFromImage(mat, blob, 1.0, cv::Size{ 608, 608 }, cv::Scalar(), false, false, CV_8U);
			net.setInput(blob, "", SCALE);
			if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
				cv::Mat image_info = (cv::Mat_<float>(1, 3) << mat.rows, mat.cols, 1.6f);
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
						auto const center_x = static_cast<int>(data[0] * static_cast<float>(mat.cols));
						auto const center_y = static_cast<int>(data[1] * static_cast<float>(mat.rows));
						auto const width = static_cast<int>(data[2] * static_cast<float>(mat.cols));
						auto const height = static_cast<int>(data[3] * static_cast<float>(mat.rows));
						auto const left = center_x - width / 2;
						auto const top = center_y - height / 2;
						std::cerr << "at rect " << width << "x" << height << "+" << left << "x" << top << " there is " << class_id_point.x << " with confidence "
											<< static_cast<float>(confidence) << '\n';
					}
				}
			}
		}
		std::cerr << "---\n";
	}
};

int main() {
	Processor p;

	cv::Mat color;
	read_from_bin("color.bin", color);

	cv::imshow("bgr", color);
	cv::waitKey(0);

	p.process_bgr(color);
}
