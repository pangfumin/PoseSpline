#include "PoseSpline/VectorSplineSampleAutoError.hpp"



VectorSplineSampleAutoError::VectorSplineSampleAutoError(const double& t_meas,
                                                 const Eigen::Vector3d& V_meas):
        t_meas_(t_meas),V_Meas_(V_meas){

};

VectorSplineSampleAutoError::~VectorSplineSampleAutoError(){

}
