#ifndef GROUP_BOX_H_
#define GROUP_BOX_H_
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;
struct Box {
    double x_min, x_max, y_min, y_max, y_center, height, width;
};
std::vector<Box> groupTextBox(std::vector<Box>& polys, double slope_ths = 0.1, double ycenter_ths = 0.5, double height_ths = 0.5, double width_ths = 0.5, double add_margin = 0.1, double min_size = 20);
#endif