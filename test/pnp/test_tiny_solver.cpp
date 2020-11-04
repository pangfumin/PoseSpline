//
// Created by pang on 2020/11/2.
//


#include "tiny_solver.h"
#include "tiny_solver_multiple_function.h"
#include <iostream>



typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::VectorXd VecX;

bool EvaluateResidualsAndJacobians(const double* parameters,
                                   double* residuals,
                                   double* jacobian) {
    double x = parameters[0];
    double y = parameters[1];
    double z = parameters[2];

    residuals[0] = x + 2*y + 4*z;
    residuals[1] = y * z;

    if (jacobian) {
        jacobian[0 * 2 + 0] = 1;
        jacobian[0 * 2 + 1] = 0;

        jacobian[1 * 2 + 0] = 2;
        jacobian[1 * 2 + 1] = z;

        jacobian[2 * 2 + 0] = 4;
        jacobian[2 * 2 + 1] = y;
    }
    return true;
}

class ExampleStatic {
public:
    typedef double Scalar;
    enum {
        // Can also be Eigen::Dynamic.
        NUM_RESIDUALS = 2,
        NUM_PARAMETERS = 3,
    };
    bool operator()(const double* parameters,
                    double* residuals,
                    double* jacobian) const {
        return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
    }
};




class ExampleStatic2 {
public:
    typedef double Scalar;
    enum {
        // Can also be Eigen::Dynamic.
        NUM_RESIDUALS = 1,
        NUM_PARAMETERS = 3,
    };
    ExampleStatic2(double x, double y):x_(x), y_(y) {

    }

    bool EvaluateResidualsAndJacobians2(const double* parameters,
                                        double* residuals,
                                        double* jacobian) const  {
        double a = parameters[0];
        double b = parameters[1];
        double c = parameters[2];


        double exp_y = std::exp( a*x_*x_ + b*x_ + c );
        residuals[0] =  exp_y - y_;


        if (jacobian) {
            Eigen::Map<Eigen::Matrix<double, 1, 3>> jaco_abc(jacobian);  // 误差为1维，状态量 3 个，所以是 1x3 的雅克比矩阵
            jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;


        }
        return true;
    }


    bool operator()(const double* parameters,
                    double* residuals,
                    double* jacobian) const {
        return EvaluateResidualsAndJacobians2(parameters, residuals, jacobian);
    }
    double x_, y_;
};



template <typename Function, typename Vector>
void TestHelper(const Function& f, const Vector& x0) {
    Vector x = x0;
    Vec2 residuals;
    f(x.data(), residuals.data(), NULL);
//    EXPECT_GT(residuals.squaredNorm() / 2.0, 1e-10);

    std::cout << "residuals: " << residuals.transpose() << std::endl;

    solver::TinySolver<Function> solver;
    auto other_x = x;
    solver::TinySolverMultipleFunction<Function> solver_multiple;
    solver.Solve(f, &x);

    //    EXPECT_NEAR(0.0, solver.summary.final_cost, 1e-10);
    std::cout << "solver.summary.final_cost: " << solver.summary.final_cost << std::endl;
    std::cout << "opt x: " << x.transpose() << std::endl;


//    std::cout << "--------------------------------------------" << std::endl;
//    std::vector<Function> f_vec={f};
//
//    solver_multiple.Solve(f_vec, &other_x);
//
//
//    std::cout << "solver_multiple.summary.final_cost: " << solver_multiple.summary.final_cost<< std::endl;
//    std::cout << "opt other_x: " << other_x.transpose() << std::endl;

}

void TestHelper2() {

    solver::TinySolverMultipleFunction<ExampleStatic2> solver_multiple;

    int N = 30;
    std::vector<ExampleStatic2> f_vec;
    for (int i = 0; i < N; ++i) {

        double x = (double)i / (N);
        // 观测 y
        double a=1.0, b=2.0, c=1.0;
        double y = std::exp( a*x*x + b*x + c ) ;

        ExampleStatic2  e(x,y);
        f_vec.push_back(e);
    }
    Vec3 other_x(0.76026643, 0.01799744, 0.55192142);
    solver_multiple.Solve(f_vec, &other_x);
    std::cout << "solver_multiple.summary.final_cost: " << solver_multiple.summary.final_cost<< std::endl;
    std::cout << "opt other_x: " << other_x.transpose() << std::endl;
}


int main() {
    Vec3 x0(0.76026643, 0.01799744, 0.55192142);
    ExampleStatic f;
//    TestHelper(f, x0);
    TestHelper2();
    return 0;
}

