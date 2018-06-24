#include "utility/measurement.h"
#include <string>

int main (){
    std::string data_path = "/media/pang/Plus/dataset/MH_01_easy";
    EurocDatasetReader eurocDatasetReader(data_path);
    eurocDatasetReader.readData();

    return 0;
}