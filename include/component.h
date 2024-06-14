#ifndef COMPONENT_H_
#define COMPONENT_H_
struct ConnectedComponent {
    int label;
    int startX, startY;
    int endX, endY;
    int area;
};
#endif