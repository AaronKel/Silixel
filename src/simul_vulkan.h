#pragma once

#include <vector>
#include <string>
#include <map>
#include "read.h"

void simulInit_vulkan(const std::vector<t_lut>& luts, const std::vector<int>& ones);
void simulBegin_vulkan(const std::vector<t_lut>& luts, const std::vector<int>& step_starts, const std::vector<int>& step_ends, const std::vector<int>& ones);
void simulCycle_vulkan(const std::vector<t_lut>& luts, const std::vector<int>& step_starts, const std::vector<int>& step_ends);
bool simulReadback_vulkan();
void simulTerminate_vulkan();
