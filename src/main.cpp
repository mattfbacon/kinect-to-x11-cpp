#include <fstream>
#include <iostream>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <span>

#include "opencv2/highgui.hpp"

using namespace libfreenect2;

// static void write(char const* const path, Frame const& frame) {
// 	std::ofstream file(path);
// 	file.write(reinterpret_cast<char const*>(&frame.width), sizeof(frame.width));
// 	file.write(reinterpret_cast<char const*>(&frame.height), sizeof(frame.height));
// 	file.write(reinterpret_cast<char const*>(&frame.bytes_per_pixel), sizeof(frame.bytes_per_pixel));
// 	auto const format = static_cast<size_t>(frame.format);
// 	file.write(reinterpret_cast<char const*>(&format), sizeof(format));
// 	file.write(reinterpret_cast<char const*>(frame.data), static_cast<std::streamsize>(frame.width * frame.height * frame.bytes_per_pixel));
// 	file.flush();
// }

int main() {
	Freenect2 ctx;

	if (ctx.enumerateDevices() == 0) {
		std::cerr << "no devices found\n";
		return 1;
	}

	// PacketPipeline* const pipeline = new CpuPacketPipeline();
	auto* const dev = ctx.openDefaultDevice();

	SyncMultiFrameListener listener(Frame::Color | Frame::Depth);
	FrameMap frames;

	dev->setColorFrameListener(&listener);
	dev->setIrAndDepthFrameListener(&listener);

	if (!dev->start()) {
		std::cerr << "could not start device\n";
		return 1;
	}

	// auto* const registration = new Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
	Frame undistorted(512, 424, 4), registered(512, 424, 4), bigdepth(1920, 1082, 4);
	bigdepth.format = Frame::Format::Float;

	for (;;) {
		if (!listener.waitForNewFrame(frames, 10 * 1000)) {
			std::cerr << "timeout waiting for new frame\n";
			return 1;
		}

		auto const rgb = frames[libfreenect2::Frame::Color];
		// auto const depth = frames[libfreenect2::Frame::Depth];

		// registration->apply(rgb, depth, &undistorted, &registered, true, &bigdepth);

		cv::Mat rgb_mat(static_cast<int>(rgb->height), static_cast<int>(rgb->width), CV_8UC4, rgb->data);  //, depth_mat(depth->height, depth->width, CV_32FC1, depth->data);

		cv::imshow("RGB Frame", rgb_mat);

		cv::waitKey(0);

		return 0;

		listener.release(frames);
	}
}
