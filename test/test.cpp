#include <string>
#include <opencv2/cudaimgproc.hpp>
#include "../src/prand.hpp"

cv::Size LimitSize(const cv::Size &in, int nMaxSize) {
	cv::Size out = in;
	if (out.width >= out.height && out.width > nMaxSize) {
		out.height = int((out.height * nMaxSize) / (float)out.width);
		out.width = nMaxSize;
	}
	if (out.height >= out.width && out.height > nMaxSize) {
		out.width = int((out.width * nMaxSize) / (float)out.height);
		out.height = nMaxSize;
	}
	return out;
}

int main(int nArgCnt, char *ppArgs[]) {
	FLAGS_alsologtostderr = 1; 
	google::InitGoogleLogging(ppArgs[0]);

	CHECK_GE(nArgCnt, 2) << "Usage: " << ppArgs[0] << " <RTSP_URL> [GPU_ID]";
	int nDevID = 0;
	if (nArgCnt > 2) {
		nDevID = std::atoi(ppArgs[2]);
		CHECK_GE(nDevID, 0) << "Invalid GPU_ID";
	}
	const int nMaxSize = 480;

	Prand prand(ppArgs[1], nDevID);
	prand.SetJpegQuality(75);
	prand.Start();
	cv::cuda::GpuMat gpuImg;
	cv::Mat img1, img2;
	std::string strJpegData;
	for (int64_t nLastFrame = 0; ; ) {
		int64_t nCurFrame = prand.GetFrame(gpuImg, &strJpegData);
		if (nCurFrame < 0) {
			for (; !prand.Start(); ) {
				usleep(100 * 1000);
			}
		}
		if (nCurFrame > nLastFrame) {
			nLastFrame = nCurFrame;
			gpuImg.download(img1);
			std::vector<uint8_t> bytes(strJpegData.size());
			memcpy(bytes.data(), strJpegData.data(), strJpegData.size());
			img2 = cv::imdecode(bytes, cv::IMREAD_COLOR);
			cv::resize(img1, img1, LimitSize(img1.size(), nMaxSize));
			cv::resize(img2, img2, LimitSize(img2.size(), nMaxSize));
			cv::imshow("Downloaded from GPU", img1);
			cv::imshow("Decode From JPEG", img2);
			LOG(INFO) << "Frame " << nCurFrame;
			int nKey = cv::waitKey(1) & 0xFF;
			if (nKey == 27) {
				break;
			}
		} else {
			usleep(1000);
		}
	}
	return 0;
}
