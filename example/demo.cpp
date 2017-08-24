#include "header.hpp"

int main()
{
    //Calibration file name
    //  string calib = "../intrinsics_fisheye.yml";
    string calib = "try it fun";
    MOD mod(0, calib, false, NONE);
    mod.setmargin(2);
    mod.process();
    return 0;
}
