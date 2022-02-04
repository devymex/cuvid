#include "../include/cuvid.hpp"
#include "../src/logging.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <string>
#include <unistd.h>
#include <regex>
#include <fstream>

#include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;

std::vector<std::string> EnumerateFiles(const std::string &strPath,
		const std::string &strPattern) {
	std::vector<std::string> filenames;
	for (auto &entry : stdfs::recursive_directory_iterator(strPath,
			stdfs::directory_options::skip_permission_denied)) {
		auto strFilename = entry.path().filename().string();
		std::smatch match;
		if (std::regex_match(strFilename, match, std::regex(strPattern))) {
			filenames.emplace_back(entry.path().string());
		}
	}
	return filenames;
}

int main(int nArgCnt, char *ppArgs[]) {
	CHECK_GE(nArgCnt, 2) << "Usage: " << ppArgs[0] << " <SOURCE> [GPU_ID]";
	const std::string strURL = ppArgs[1];
	int nDevID = 0;
	if (nArgCnt > 2) {
		nDevID = std::atoi(ppArgs[2]);
		CHECK_GE(nDevID, 0) << "Invalid GPU_ID";
	}
	const int nMaxSize = 480;

	std::string strSource {ppArgs[1]};
	std::vector<std::string> filenames;
	if (stdfs::is_directory(strSource)) {
		filenames = EnumerateFiles(strSource, ".*\\.mp4");
	} else if (stdfs::is_regular_file(strSource)) {
		if (std::regex_match(strSource, std::regex(".*\\.txt"))) {
			std::ifstream inFile(strSource);
			CHECK(inFile.is_open());
			for (std::string strLine; std::getline(inFile, strLine); ) {
				filenames.emplace_back(std::move(strLine));
			}
		} else {
			filenames.emplace_back(std::move(strSource));
		}
	}

	Cuvid cuvid(nDevID);

	for (uint32_t i = 0; i < filenames.size(); ++i) {
		cuvid.close();
		CHECK(cuvid.open(filenames[i]));
		auto dFPS = cuvid.get(cv::CAP_PROP_FPS);
		auto dFrmCnt = cuvid.get(cv::CAP_PROP_FRAME_COUNT);
		cv::Size frameSize(cuvid.get(cv::CAP_PROP_FRAME_WIDTH),
						cuvid.get(cv::CAP_PROP_FRAME_HEIGHT));
		for (cv::cuda::GpuMat gpuImg; ; ) {
			auto [nFrameId, nTimeStamp] = cuvid.read(gpuImg);
			if (nFrameId < 0) {
				break;
			}
			if (gpuImg.empty()) {
				throw "Empty frame!";
			}
			LOG(INFO) << "[" << nFrameId << "] time: " << nTimeStamp
				<< ", size: (" << gpuImg.cols << "x" << gpuImg.rows << ")";
		}
	}
	return 0;
}
