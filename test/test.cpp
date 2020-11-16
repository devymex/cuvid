#include <string>
#include <opencv2/cudaimgproc.hpp>

#include "../src/prand.hpp"

int main(int nArgCnt, char *ppArgs[]) {
	FLAGS_alsologtostderr = 1; 
	google::InitGoogleLogging(ppArgs[0]);

	const std::string strURL = "rtsp://10.201.105.94/user=admin&password=&channel=1&stream=0.sdp";

	Prand prand(strURL, 0);
	prand.Start();
	cv::cuda::GpuMat gpuImg;
	cv::Mat img1, img2;
	std::string strJpegData;
	for (int nLastFrame = 0; ; ) {
		int nCurFrame = prand.GetFrame(gpuImg, &strJpegData);
		if (nCurFrame != nLastFrame) {
			nLastFrame = nCurFrame;
			gpuImg.download(img1);
			std::vector<uint8_t> bytes(strJpegData.size());
			memcpy(bytes.data(), strJpegData.data(), strJpegData.size());
			img2 = cv::imdecode(bytes, cv::IMREAD_COLOR);
			cv::resize(img1, img1, img1.size() / 4);
			cv::resize(img2, img2, img2.size() / 4);
			cv::imshow("Downloaded from GPU", img1);
			cv::imshow("Decode From JPEG", img2);
			int nKey = cv::waitKey(1) & 0xFF;
			if (nKey == 27) {
				break;
			}
		}
	}
	return 0;
}
