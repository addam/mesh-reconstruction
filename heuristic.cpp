#include "recon.hpp"

Heuristic::Heuristic(Configuration *iconfig)
{
	config = iconfig;
	iteration = 0;
}
bool Heuristic::notHappy(const Mat points)
{
	iteration ++;
	return (iteration <= 1);
}
void Heuristic::filterPoints(Mat points)
{
	// tohle výjimečně tak může zůstat
}
void Heuristic::chooseCameras()
{
	chosenCameras.clear();
	for (int i=0; i < config->frameCount(); i += 20) {
		std::vector<int> linked;
		if (i - 15 >= 0)
			linked.push_back(i-15);
		if (i - 5 >= 0)
			linked.push_back(i-5);
		if (i + 5 < config->frameCount())
			linked.push_back(i+5);
		if (i + 15 < config->frameCount())
			linked.push_back(i+15);
		chosenCameras.push_back(numberedVector(i, linked));
	}
}
int Heuristic::beginMain()
{ // initialize and return frame number for first main camera
	if (chosenCameras.size() == 0)
		return Heuristic::sentinel;
	else
		return chosenCameras[mainIdx = 0].first;
}
int Heuristic::nextMain()
{ // return frame number for next main camera
	if (++mainIdx < chosenCameras.size())
		return chosenCameras[mainIdx].first;
	else
		return Heuristic::sentinel;
}
int Heuristic::beginSide(int imain)
{ // initialize and return frame number for first side camera
	if (imain != chosenCameras[mainIdx].first || chosenCameras[mainIdx].second.size() == 0)
		return Heuristic::sentinel;
	else
		return chosenCameras[mainIdx].second[sideIdx = 0];
}
int Heuristic::nextSide(int imain)
{ // return frame number for next side camera
	if (imain != chosenCameras[mainIdx].first || ++sideIdx >= chosenCameras[mainIdx].second.size())
		return Heuristic::sentinel;
	else
		return chosenCameras[mainIdx].second[sideIdx];
}
