#include <string>
#include <opencv2/cudaimgproc.hpp>
#include "../src/prand.hpp"

int main(int nArgCnt, char *ppArgs[]) {
	const std::string strURL = "rtsp://10.201.105.94/user=admin&password=&channel=1&stream=0.sdp";

	Prand prand(strURL, 0);
	prand.Start();
	for (int nLastFrame = 0; ; ) {
		cv::cuda::GpuMat gpuImg;
		int nCurFrame = prand.GetFrame(gpuImg);
		if (nCurFrame != nLastFrame) {
			nLastFrame = nCurFrame;
			cv::cuda::GpuMat gpuBGR;
			cv::cuda::cvtColor(gpuImg, gpuBGR, cv::COLOR_BGRA2BGR);

			cv::Mat img;
			gpuBGR.download(img);
			cv::imshow("img", img);
			int nKey = cv::waitKey(1) & 0xFF;
			if (nKey == 27) {
				break;
			}
		}
	}
	return 0;
}
