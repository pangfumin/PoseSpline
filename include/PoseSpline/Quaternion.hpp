#ifndef QUATERNION_H
#define QUATERNION_H


#include <map>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>
/*
 * R.f.:
 * [1] Pose estimation using linearized rotations and quaternion algebra.
 * [2] Indirect Kalman filter for 3D attitude estimation.
 *
 */

typedef  Eigen::Matrix<double,4,1> Quaternion;
template <typename T>
using QuaternionTemplate = Eigen::Matrix<T,4,1>;
typedef  Quaternion* QuatPtr;
typedef  Eigen::Map<Quaternion> QuaternionMap;
typedef  Eigen::Matrix<double,3,3> RotMat;
typedef  double real_t;



template<typename T>
Eigen::Matrix<T,4,1> unitQuat(){
    return (Eigen::Matrix<T,4,1>() 
        << T(0.0),T(0.0),T(0.0),T(1.0)).finished();
};

template<typename T>
Eigen::Matrix<T,4,1> quatMap(T* ptr){
    Eigen::Map<Eigen::Matrix<T,4,1>> j(ptr);
    Eigen::Matrix<T,4,1> jacobian = j;
    return jacobian;
 };

template<typename T>
inline Eigen::Matrix<T,3,3> crossMat(const Eigen::Matrix<T,3,1>& vec)
{
    return (Eigen::Matrix<T,3,3>() << T(0),       -vec[2], vec[1],
                                    vec[2],  T(0),       -vec[0],
                                    -vec[1], vec[0],  T(0)).finished();
}


template<typename T>
bool isLessThenEpsilons4thRoot(T x){
    static const T epsilon4thRoot = pow(std::numeric_limits<T>::epsilon(), T(1.0/4.0));
    return x < epsilon4thRoot;
}


template<typename T>
Eigen::Matrix<T,4,1> quatExp(const Eigen::Matrix<T,3,1>& dx) {
    // Method of implementing this function that is accurate to numerical precision from
    // Grassia, F. S. (1998). Practical parameterization of rotations using the exponential map.
    // journal of graphics, gpu, and game tools, 3(3):29–48.

    T theta = dx.norm();
    // na is 1/theta sin(theta/2)
    T na;
    if(isLessThenEpsilons4thRoot(theta)){
        static const T one_over_48 = T(1.0/48.0);
        na = T(0.5) + (theta * theta) * one_over_48;
    } else {
        na = sin(theta*T(0.5)) / theta;
    }
    T ct = cos(theta*T(0.5));
    return Eigen::Matrix<T,4,1>(dx[0]*na,
                      dx[1]*na,
                      dx[2]*na,
                      ct);
}

template<typename T>
T fabsT(const T scale){
    if( scale > T(0)) return scale;
    else return -scale;
}
template<typename T>
T arcSinXOverX(T x) {
    if(isLessThenEpsilons4thRoot(fabsT(x))){
        return T(1.0) + x * x * T(1.0/6.0);
    }
    return asin(x) / x;
}



template<typename T>
Eigen::Matrix<T,3,1> quatLog(Eigen::Matrix<T,4,1> & q){


    const Eigen::Matrix<T, 3, 1> a = q.head(3);
    const T na = a.norm();
    const T eta = q[3];
    T scale;
    if(fabsT(eta) < na){ // use eta because it is more precise than na to calculate the scale. No singularities here.
        // check sign of eta so that we can be sure that log(-q) = log(q)
        if (eta >= T(0)) {
            scale = acos(eta) / na;
        } else {
            scale = -acos(-eta) / na;
        }
    } else {
        /*
         * In this case more precision is in na than in eta so lets use na only to calculate the scale:
         *
         * assume first eta > 0 and 1 > na > 0.
         *               u = asin (na) / na  (this implies u in [1, pi/2], because na i in [0, 1]
         *    sin (u * na) = na
         *  sin^2 (u * na) = na^2
         *  cos^2 (u * na) = 1 - na^2
         *                              (1 = ||q|| = eta^2 + na^2)
         *    cos^2 (u * na) = eta^2
         *                              (eta > 0,  u * na = asin(na) in [0, pi/2] => cos(u * na) >= 0 )
         *      cos (u * na) = eta
         *                              (u * na in [ 0, pi/2] )
         *                 u = acos (eta) / na
         *
         * So the for eta > 0 it is acos(eta) / na == asin(na) / na.
         * From some geometric considerations (mirror the setting at the hyper plane q==0)
         * it follows for eta < 0 that (pi - asin(na)) / na = acos(eta) / na.
         */
        if(eta > T(0)) {
            // For asin(na)/ na the singularity na == 0 can be removed. We can ask (e.g. Wolfram alpha)
            // for its series expansion at na = 0. And that is done in the following function.
            scale = arcSinXOverX(na);
        } else {
            // the negative is here so that log(-q) == log(q)
            scale = -arcSinXOverX(na);
        }
    }
    return a * ((T(2.0)) * scale);
}

template<typename T>
Eigen::Matrix<T,3,3> quatL(Eigen::Matrix<T,4,1> &q){
    Eigen::Matrix<T,3,3> Jac_log;
    Jac_log.setIdentity();
    if(std::abs(q(3) - T(1)) < T(1e-5)){
        return Jac_log;
    }else{
        Eigen::Matrix<T,3,1> phi = quatLog(q);
        Eigen::Matrix<T,3,3> I;
        I.setIdentity();
        T nphi = phi.norm();
        Eigen::Matrix<T,3,1> a = phi/nphi;
        Eigen::Matrix<T,3,3> squareA= crossMat(a)*crossMat(a);


        Jac_log = I + T(0.5)*crossMat<T>(phi)
                  + (T(1) - nphi/(T(2)*tan(T(0.5)*nphi)))*squareA;
        return Jac_log;
    }
}
template<typename T>
Eigen::Matrix<T,3,3> quatS(Eigen::Matrix<T,3,1> &phi){
    Eigen::Matrix<T,3,3> Jac_exp;
    Jac_exp.setIdentity();
    T nphi = phi.norm();
    if(nphi < T(1e-5)){
        return Jac_exp;
    }else{
        Eigen::Matrix<T,3,3> I;
        I.setIdentity();
        Eigen::Matrix<T,3,1> a = phi/nphi;
        T squareSin = sin(T(0.5)*nphi)*sin((0.5)*nphi);
        Eigen::Matrix<T,3,3> squareA= crossMat(a)*crossMat(a);
        Jac_exp = I - T(2.0)/nphi*squareSin*crossMat(a)
                  + (T(1) - T(1.0)/(nphi)*sin(nphi))*squareA;
        return Jac_exp;

    }
}

template<typename T>
Eigen::Matrix<T,4,1> QuatFromEuler(T* euler)
{
    Eigen::Matrix<T,4,1> quat;
    T cr2 = cos(euler[0]*T(0.5));
    T cp2 = cos(euler[1]*T(0.5));
    T cy2 = cos(euler[2]*T(0.5));
    T sr2 = sin(euler[0]*T(0.5));
    T sp2 = sin(euler[1]*T(0.5));
    T sy2 = sin(euler[2]*T(0.5));


    quat[0] = sr2*cp2*cy2 - cr2*sp2*sy2;
    quat[1] = cr2*sp2*cy2 + sr2*cp2*sy2;
    quat[2] = cr2*cp2*sy2 - sr2*sp2*cy2;
    quat[3] = cr2*cp2*cy2 + sr2*sp2*sy2;
    return quat;
}


template<typename T>
Eigen::Matrix<T,4,1> deltaQuat( Eigen::Matrix<T,3,1> &deltaTheta )
{
    Eigen::Matrix<T,4,1>  r;
    Eigen::Matrix<T,3,1> deltaq = T(0.5) * deltaTheta;

    r.head(3) = deltaq;
    r(3) = T(1.0);

    return r;
}


template<typename T>
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
null(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& A)
{
    int r = 0;
    Eigen::JacobiSVD<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>>
            svd(A.transpose(), Eigen::ComputeFullV);

    /* Get the V matrix */
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
            V((int)svd.matrixV().rows(), (int)svd.matrixV().cols());
    V = svd.matrixV();
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> S = svd.singularValues();

    T tol = std::max(A.rows(), A.cols()) * S.maxCoeff() * T(2.2204e-016);
    for (int i = 0; i < S.size(); i++)
    {
        if (S.coeff(i) > tol)
        {
            r++;
        }
    }
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
            Z = V.block(0, r, V.rows(), V.cols()-r);
    return Z;
}


template<typename T>
Eigen::Matrix<T,4,1> quatInv(const Eigen::Matrix<T,4,1> q)
{
    //assert( abs( q.norm() - 1.0) <= std::numeric_limits<double>::epsilon() );
    Eigen::Matrix<T,4,1> qinv =  q/q.norm();
    qinv(0,0) = -qinv(0,0);
    qinv(1,0) = -qinv(1,0);
    qinv(2,0) = -qinv(2,0);
    qinv(3,0) = qinv(3,0);

    return qinv;
}

template<typename T>
inline Eigen::Matrix<T,4,1> quatNorm( Eigen::Matrix<T,4,1> q){
    return q/q.norm();
}


template<typename T>
inline Eigen::Matrix<T,4,4> quatLeftComp( const Eigen::Matrix<T,4,1> q )
{
    // [  q3,  q2, -q1, q0]
    // [ -q2,  q3,  q0, q1]
    // [  q1, -q0,  q3, q2]
    // [ -q0, -q1, -q2, q3]
    Eigen::Matrix<T,4,4> Q;
    Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
    Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
    Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
    Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];

    return Q;
}

template<typename T>
inline Eigen::Matrix<T,4,4> quatRightComp( const Eigen::Matrix<T,4,1> q )
{
    // [  q3, -q2,  q1, q0]
    // [  q2,  q3, -q0, q1]
    // [ -q1,  q0,  q3, q2]
    // [ -q0, -q1, -q2, q3]

    Eigen::Matrix<T,4,4> Q;
    Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
    Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
    Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
    Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];

    return Q;
}



template<typename T>
Eigen::Matrix<T,4,1> quatMult( const Eigen::Matrix<T,4,1> q,const Eigen::Matrix<T,4,1> p)
{
    Eigen::Matrix<T,4,1> qplus_p;
    // p0*q3 + p1*q2 - p2*q1 + p3*q0
    qplus_p[0] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0];
    // p2*q0 - p0*q2 + p1*q3 + p3*q1
    qplus_p[1] = p[2]*q[0] - p[0]*q[2] + p[1]*q[3] + p[3]*q[1];
    // p0*q1 - p1*q0 + p2*q3 + p3*q2
    qplus_p[2] = p[0]*q[1] - p[1]*q[0] + p[2]*q[3] + p[3]*q[2];
    // p3*q3 - p1*q1 - p2*q2 - p0*q0
    qplus_p[3] = p[3]*q[3] - p[1]*q[1] - p[2]*q[2] - p[0]*q[0];

    
    return qplus_p;
}


template<typename T>
Eigen::Matrix<T,3,3> renormalizeRotMat( Eigen::Matrix<T,3,3> &m )
{
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3> > jsvd(m, Eigen::ComputeFullU | Eigen::ComputeFullV );

    Eigen::Matrix<T,3,3> VT = jsvd.matrixV().transpose();

    return jsvd.matrixU() * Eigen::Matrix<T,3,3>::Identity() * VT;
}

template<typename T>
Eigen::Matrix<T,3,3> quatToRotMat( Eigen::Matrix<T,4,1> q )
{
    // https://github.com/ethz-asl/maplab/blob/master/common/maplab-common/include/maplab-common/quaternion-math-inl.h
    q = q / q.norm();

    T one = static_cast<T>(1.0);
    T two = static_cast<T>(2.0);
    Eigen::Matrix<T,3,3> rot_matrix;

    (rot_matrix)(0, 0) = one - two * (q(1) * q(1) + q(2) * q(2));
    (rot_matrix)(0, 1) = two * (q(0) * q(1) + q(2) * q(3));
    (rot_matrix)(0, 2) = two * (q(0) * q(2) - q(1) * q(3));

    (rot_matrix)(1, 0) = two * (q(0) * q(1) - q(2) * q(3));
    (rot_matrix)(1, 1) = one - two * (q(0) * q(0) + q(2) * q(2));
    (rot_matrix)(1, 2) = two * (q(1) * q(2) + q(0) * q(3));

    (rot_matrix)(2, 0) = two * (q(0) * q(2) + q(1) * q(3));
    (rot_matrix)(2, 1) = two * (q(1) * q(2) - q(0) * q(3));
    (rot_matrix)(2, 2) = one - two * (q(0) * q(0) + q(1) * q(1));

    return rot_matrix;
}



template<typename T>
Eigen::Matrix<T,4,1> rotMatToQuat( Eigen::Matrix<T,3,3> &R )
{
    Eigen::Matrix<T,4,1> q;
    Eigen::Matrix<T,3,3> R_T = R.transpose();

    T Rxx = R_T(0,0); T Rxy = R_T(0,1); T Rxz = R_T(0,2);
    T Ryx = R_T(1,0); T Ryy = R_T(1,1); T Ryz = R_T(1,2);
    T Rzx = R_T(2,0); T Rzy = R_T(2,1); T Rzz = R_T(2,2);

    T w = sqrt( R.trace() + T(1.0) ) / T(2.0);
    T x = sqrt( T(1.0) + Rxx - Ryy - Rzz ) / T(2.0);
    T y = sqrt( T(1.0) + Ryy - Rxx - Rzz ) / T(2.0);
    T z = sqrt( T(1.0) + Rzz - Ryy - Rxx ) / T(2.0);

    T i = std::max( std::max(w,x), std::max(y,z) );
    if( i == w )
    {
        x = ( Rzy - Ryz ) / (T(4.0)*w);
        y = ( Rxz - Rzx ) / (T(4.0)*w);
        z = ( Ryx - Rxy ) / (T(4.0)*w);
    }
    else if( i == x )
    {
        w = ( Rzy - Ryz ) / (T(4.0)*x);
        y = ( Rxy + Ryx ) / (T(4.0)*x);
        z = ( Rzx + Rxz ) / (T(4.0)*x);
    }
    else if( i == y )
    {
        w = ( Rxz - Rzx ) / (T(4.0)*y);
        x = ( Rxy + Ryx ) / (T(4.0)*y);
        z = ( Ryz + Rzy ) / (T(4.0)*y);
    }
    else if( i == z )
    {
        w = ( Ryx - Rxy ) / (T(4.0)*z);
        x = ( Rzx + Rxz ) / (T(4.0)*z);
        y = ( Ryz + Rzy ) / (T(4.0)*z);
    }

    q(0) = x;
    q(1) = y;
    q(2) = z;
    q(3) = w;

    return q;
}



template<typename T>
Eigen::Matrix<T,3,3> axisAngleToRotMat(Eigen::Matrix<T,3,1> aa){
    T phi = aa.norm();
    Eigen::Matrix<T,3,1> a = aa/phi;
    Eigen::Matrix<T,3,3> I;
    I.setIdentity();
    T cos_phi = cos(phi);

    return cos_phi*I + (1 - cos_phi)*a*a.transpose() - sin(phi)*crossMat<T>(a);

}

template<typename T>
Eigen::Matrix<T,4,1> randomQuat() {
    // Create a random unit-length axis.
    Eigen::Matrix<T, 3, 1> axis = T(M_PI) * Eigen::Matrix<T, 3, 1>::Random();

    Eigen::Matrix<T,3,3> C_
            = Eigen::AngleAxis<T>(axis.norm(), axis.normalized()).toRotationMatrix();
    Eigen::Matrix<T,4,1> q_ = rotMatToQuat<T>(C_);
    return q_;
}



/*
 * From Barfoot's book.
 */


template<typename T>
Eigen::Matrix<T,3,3> rotX(T x){
    Eigen::Matrix<T,3,3> Rx ;
    Rx<<   T(1),      T(0),     T(0),
            T(0), cos(x),sin(x),
            T(0),-sin(x),cos(x);
    return Rx;
}


template<typename T>
Eigen::Matrix<T,3,3> rotY(T y){
    Eigen::Matrix<T,3,3> Ry ;
    Ry<<  cos(y),   T(0),   -sin(y),
            T(0),   T(1),         T(0),
            sin(y),  T(0),    cos(y);
    return Ry;

}


template<typename T>
Eigen::Matrix<T,3,3> rotZ(double z){
    Eigen::Matrix<T,3,3> Rz ;
    Rz<<  cos(z),sin(z),T(0),
            -sin(z),cos(z),T(0),
            T(0),     T(0),   T(1);
    return Rz;

}

/*
 * Kinematics
 */


template<typename T>
Eigen::Matrix<T,4,4> OmegaMat(const Eigen::Matrix<T,3,1>& vec){
    return (Eigen::Matrix<T,4,4>()<< 0, vec[2], -vec[1], vec[0],
            -vec[2], 0, vec[0], vec[1],
            vec[1], -vec[0], 0, vec[2],
            -vec[0], -vec[1], -vec[2], 0).finished();
};


template<typename  T>
inline Eigen::Matrix<T,3,1> errorRotationPropagate(Eigen::Matrix<T,3,1>& omega,
                                            Eigen::Matrix<T,3,1>& deltaTheta0 ){
   return (Eigen::Matrix<T,3,3>::setIdentity() - crossMat(omega))*deltaTheta0;
};


/*
 * q2 = Exp(omega*dt)*q1
 */
template <typename T>
Eigen::Matrix<T,3,1> getOmegaFromTwoQuaternion(const Eigen::Matrix<T,4,1>& q1,
                                               const Eigen::Matrix<T,4,1>& q2,
                                               T dt){
    Eigen::Matrix<T,4,1> temp = quatLeftComp<T>(q2)*quatInv<T>(q1);
    return   quatLog<T>(temp)/dt;
};

#endif