#include "../include/cuvid.hpp"
#include "../src/logging.hpp"
#include <string>
#include <unistd.h>
#include <regex>
#include <fstream>

int main(int nArgCnt, char *ppArgs[]) {
	CHECK_GE(nArgCnt, 2) << "Usage: " << ppArgs[0] << " <URL> [GPU_ID]";
	const std::string strURL = ppArgs[1];
	int nDevID = 0;
	if (nArgCnt > 2) {
		nDevID = std::atoi(ppArgs[2]);
		CHECK_GE(nDevID, 0) << "Invalid GPU_ID";
	}

	std::unique_ptr<Cuvid> pCuvid(new Cuvid(nDevID));
	CHECK(pCuvid->open(ppArgs[1]));
	std::cout << "Num frames: " << pCuvid->get(7)
			  << ", Resolution: " << pCuvid->get(3) << "x" << pCuvid->get(4)
			  << ", FPS: " << pCuvid->get(5) << std::endl;
	for (GpuBuffer frameBuf; ; ) {
		auto [nFrameId, nTimeStamp] = pCuvid->read(frameBuf);
		if (nFrameId < 0) {
			break;
		}
		if (frameBuf.empty()) {
			throw "Empty frame!";
		}
		LOG(INFO) << "[" << nFrameId << "] time: " << nTimeStamp
				  << ", size: " << frameBuf.size();
	}
	pCuvid->close();
	pCuvid.reset(nullptr);
	return 0;
}
