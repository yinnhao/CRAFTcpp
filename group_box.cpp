#include "group_box.h"
std::vector<Box> groupTextBox(std::vector<Box>& polys, double slope_ths, double ycenter_ths,
                               double height_ths, double width_ths, double add_margin, double min_size) {
    // std::vector<Box> horizontal_list, free_list;
    std::vector<Box> new_box, merged_list;
    std::vector<std::vector<Box>> combined_list;

    // 按照y方向上的中心点坐标的大小关系对Box进行排序
    std::sort(polys.begin(), polys.end(), [](const Box& a, const Box& b) {
        return a.y_center < b.y_center;
    });
    

    //对box进行分组，将y方向上的中心点坐标相近的box分到一组，后续进行合并操作
    for (const Box& poly : polys) {
        if (new_box.empty()) {
            new_box.push_back(poly);
        } else {
            double b_height = new_box[0].height;
            double b_ycenter = new_box[0].y_center;

            if (std::abs(b_ycenter - poly.y_center) < ycenter_ths * (b_height / new_box.size())) {
                b_height += poly.height;
                b_ycenter += poly.y_center;
                new_box.push_back(poly);
            } else {
                combined_list.push_back(new_box);
                new_box.clear();
                new_box.push_back(poly);
            }
        }
    }
    combined_list.push_back(new_box);

    // for (const std::vector<Box>& group : combined_list) {
    //     std::cout << "Group:" << std::endl;
    //     for (const Box& box : group) {
    //         std::cout << "x_min: " << box.x_min << ", x_max: " << box.x_max
    //                   << ", y_min: " << box.y_min << ", y_max: " << box.y_max
    //                   << ", y_center: " << box.y_center << ", height: " << box.height
    //                   << ", width: " << box.width << std::endl;
    //     }
    // }


    for (std::vector<Box>& boxes : combined_list) {
        if (boxes.size() == 1) {
            Box& box = boxes[0];
            double margin = add_margin * std::min(box.x_max - box.x_min, box.height);
            box.x_min -= margin;
            box.x_max += margin;
            box.y_min -= margin;
            box.y_max += margin;
            if((box.x_max-box.x_min)>min_size || (box.y_max-box.y_min)>min_size)
                merged_list.push_back(box);
        } else {
            std::vector<Box> sorted_boxes = boxes;
            // 对分到一组的box，按照x_min进行排序
            std::sort(sorted_boxes.begin(), sorted_boxes.end(), [](const Box& a, const Box& b) {
                return a.x_min < b.x_min;
            });

            std::vector<Box> new_box;
            std::vector<std::vector<Box>> merged_box;
            double x_max = 0.0;
            for (const Box& box : sorted_boxes) {
                if (new_box.empty()) {
                    new_box.push_back(box);
                    x_max = box.x_max;
                } else {
                    double b_height = new_box[0].height;
                    // 如果两个框在x方向上离得很近，那么这两个框需要合并
                    if ((box.x_min - x_max) < width_ths * box.width) {
                        new_box.push_back(box);
                        x_max = box.x_max;
                    } else {
                        merged_box.push_back(new_box);
                        new_box.clear();
                        new_box.push_back(box);
                        x_max = box.x_max;
                    }
                }
            }
            if (!new_box.empty()) {
                merged_box.push_back(new_box);
            }
            // 对框进行合并
            for (std::vector<Box>& mbox : merged_box) {
                if (mbox.size() != 1) {
                    double x_min = std::min_element(mbox.begin(), mbox.end(), [](const Box& a, const Box& b) {
                        return a.x_min < b.x_min;
                    })->x_min;
                    double x_max = std::max_element(mbox.begin(), mbox.end(), [](const Box& a, const Box& b) {
                        return a.x_max < b.x_max;
                    })->x_max;
                    double y_min = std::min_element(mbox.begin(), mbox.end(), [](const Box& a, const Box& b) {
                        return a.y_min < b.y_min;
                    })->y_min;
                    double y_max = std::max_element(mbox.begin(), mbox.end(), [](const Box& a, const Box& b) {
                        return a.y_max < b.y_max;
                    })->y_max;

                    double box_width = x_max - x_min;
                    double box_height = y_max - y_min;
                    double margin = add_margin * std::min(box_width, box_height);
                    Box merge_box;
                    merge_box.x_min = x_min - margin;
                    merge_box.x_max = x_max + margin;
                    merge_box.y_min = y_min - margin;
                    merge_box.y_max = y_max + margin;
                    

                    if((merge_box.x_max-merge_box.x_min)>min_size || (merge_box.y_max-merge_box.y_min)>min_size)
                        merged_list.push_back(merge_box);
                } else {
                    Box& box = mbox[0];
                    double box_width = box.x_max - box.x_min;
                    double box_height = box.y_max - box.y_min;
                    double margin = add_margin * std::min(box_width, box_height);
                    box.x_min -= margin;
                    box.x_max += margin;
                    box.y_min -= margin;
                    box.y_max += margin;
                    if((box.x_max-box.x_min)>min_size || (box.y_max-box.y_min)>min_size)
                        merged_list.push_back(box);
                }
            }
        }
    }

    // for (Box& box : merged_list) {
    //     cout<<"combine"<<endl;
    //     std::cout << "x_min: " << box.x_min << ", x_max: " << box.x_max << ", y_min: " << box.y_min << ", y_max: " << box.y_max << "\n";
    //     cout<<"combine end"<<endl;
    // }


    return merged_list;
}
/*
int main() {

    Box box1 = {1050, 1500, 310, 410, 360, 100, 450};
    Box box2 = {500, 1000, 300, 400, 350, 100, 500};
    Box box3 = {500, 1000, 700, 800, 750, 100, 500};
    
    // Define your input polygons here
    std::vector<Box> polys = {box1, box2, box3
        // Fill in your polygon data here
    };
    for (const Box& box : polys) {
        std::cout << "x_min: " << box.x_min << ", x_max: " << box.x_max << ", y_min: " << box.y_min << ", y_max: " << box.y_max << "\n";
    }

    // Call the function
    std::vector<Box> merged_list = groupTextBox(polys);

    // Print the results
    for (const Box& box : merged_list) {
        std::cout << "x_min: " << box.x_min << ", x_max: " << box.x_max << ", y_min: " << box.y_min << ", y_max: " << box.y_max << "\n";
    }

    return 0;
}
*/