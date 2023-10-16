#define _USE_MATH_DEFINES
// good tutorial to build casadi: https://github.com/zehuilu/Tutorial-on-CasADi-with-CPP
// eigen repo: https://gitlab.com/libeigen/eigen

#include <iostream>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace Eigen;
using namespace casadi;
using namespace std;

// Function Prototype
MatrixXd Skew(VectorXd);
MatrixXd q2R(MatrixXd);
SX q2R(SX, SX);
MatrixXd f(MatrixXd, MatrixXd, double, double);
void shift(int, double, MatrixXd &, MatrixXd,MatrixXd &, double, double);

int main() {

    // Declare states + controls
    SX thrust = SX::sym("thrust",3,1);
    SX torque = SX::sym("torque",3,1);
    SX controls = vertcat(thrust,torque);

    SX u1 = SX::sym("u1");
    SX u2 = SX::sym("u2");
    SX u3 = SX::sym("u3");
    SX u4 = SX::sym("u4");
    SX u5 = SX::sym("u5");
    SX u6 = SX::sym("u6");

    SX controls1 = vertcat(u1,u2,u3,u4,u5,u6);

    SX x = SX::sym("x");
    SX y = SX::sym("y");
    SX z = SX::sym("z");
    SX dx = SX::sym("dx");
    SX dy = SX::sym("dy");
    SX dz = SX::sym("dz");
    SX sq = SX::sym("sq");
    SX vq = SX::sym("vq",3,1);
    SX dw = SX::sym("dw",3,1);


    SX states = vertcat(x,y,z,dx,dy,dz);
    states = vertcat(states,sq,vq,dw);

    // Number of differential states
    const int numStates = states.size1();

    // Number of controls
    const int numControls = controls.size1();

    // Bounds and initial guess for the control
    double thrustMax = 1e-2;
    double thrustMin = -thrustMax;
    double torqueMax = 1e-4;
    double torqueMin = -torqueMax;
    std::vector<double> u_min =  { thrustMin, thrustMin, thrustMin, torqueMin, torqueMin, torqueMin };
    std::vector<double> u_max  = { thrustMax, thrustMax, thrustMax, torqueMax, torqueMax, torqueMax };
    std::vector<double> u_init = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Initial Satellite Conditions + Problem Parameters
    double tPeriod = 92.68 * 60; // ISS orbital period (seconds)
    double n = -2*M_PI/tPeriod; // Mean motion of ISS (rad/s)
    double mc = 12; // mass of the chaser

    double xPos = 1.5;
    double xVel = 0.001;
    double yPos = (2/n)*xVel;
    double yVel = -2*n*xPos;
    double zPos = 1.5;
    double zVel = 0.001;
    std::vector<double> q0 = { 0.771517, 0.46291, 0.308607, 0.308607 }; // Normalized [0.5,0.3,0.2,0.2]
    std::vector<double> wc0 = { 0.0, 0.0, -0.005 };

    // Bounds and initial guess for the state
    std::vector<double> x0_min = {  xPos, yPos, zPos, xVel, yVel, zVel }; // initial position and velocity
    x0_min.insert(x0_min.end(),q0.begin(),q0.end()); // append initial quaternion
    x0_min.insert(x0_min.end(),wc0.begin(),wc0.end()); // append initial angular velocity

    std::vector<double> x0_max = {  xPos, yPos, zPos, xVel, yVel, zVel };
    x0_max.insert(x0_max.end(),q0.begin(),q0.end()); // append initial quaternion
    x0_max.insert(x0_max.end(),wc0.begin(),wc0.end()); // append initial angular velocity

    std::vector<double> x_min  = { -inf, -inf, -inf, -inf, -inf, -inf, -1, -1, -1, -1, -inf, -inf, -inf };
    std::vector<double> x_max  = { inf, inf, inf, inf, inf, inf, 1, 1, 1, 1, inf, inf, inf };
    std::vector<double> xf_min = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> xf_max = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> x_init = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // MPC Horizon and Sampling Period
    const int N = 500; // Prediction Horizon
    double ts = 3.0; // sampling period
    double tf = ts*N;

    // ODE right hand side

    SX Rtc = q2R(sq,vq);
    SX f = mtimes(Rtc,thrust);
    SX Jd = SX::zeros(3,3); // Moment of inertia for the chaser
    Jd(0,0) = 0.2734;
    Jd(1,1) = 0.2734;
    Jd(2,2) = 0.3125;

    SX wt = SX::zeros(3,1); // angular velocity of the target
    wt(2) = n;

    SX Kd = solve(Jd,SX::eye(Jd.size1())); // Kd = Jd^-1
    SX Ko = mtimes(mtimes(Rtc,Kd),Rtc);



    SX ode = vertcat(dx, dy, dz,
                    3* pow(n,2)*x+2*n*dy+f(0)/mc,
                    -2*n*dx+f(1)/mc,
                    -pow(n,2)*z+f(2)/mc );
    ode = vertcat(ode,
                  -0.5*dot(vq,dw),
                  0.5 * mtimes( (sq * SX::eye(3) + skew(vq)), dw),
                  mtimes(skew(dw),wt) + mtimes(Ko,torque)
                  - mtimes(Ko, ( mtimes(skew(dw), mtimes( mtimes(Jd,Rtc.T()), dw))
                                + mtimes(skew(dw), mtimes( mtimes(Jd,Rtc.T()), wt))
                                + mtimes(skew(wt), mtimes( mtimes(Jd,Rtc.T()), dw))
                                + mtimes(skew(wt), mtimes( mtimes(Jd,Rtc.T()), wt))))
                  );

    SXDict dae = {{"x", states}, {"p", controls}, {"ode", ode}};

    // Create an integrator (rk4)
    Function F = integrator("integrator", "rk", dae, 0, tf/N);

    // Total number of NLP variables
    const int numVars = numStates*(N+1) + numControls*N;

    // Declare variable vector for the NLP
    MX V = MX::sym("V",numVars);

    // NLP variable bounds and initial guess
    std::vector<double> v_min,v_max,v_init;

    // Offset in V
    int offset=0;

    // State at each shooting node and control for each shooting interval
    std::vector<MX> X, U;
    for(int k=0; k<N; ++k){
        // Local state
        X.push_back( V.nz(Slice(offset,offset + numStates)));
        if(k==0){
            v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
            v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
        } else {
            v_min.insert(v_min.end(), x_min.begin(), x_min.end());
            v_max.insert(v_max.end(), x_max.begin(), x_max.end());
        }
        v_init.insert(v_init.end(), x_init.begin(), x_init.end());
        offset += numStates;

        // Local control
        U.push_back( V.nz(Slice(offset,offset + numControls)));
        v_min.insert(v_min.end(), u_min.begin(), u_min.end());
        v_max.insert(v_max.end(), u_max.begin(), u_max.end());
        v_init.insert(v_init.end(), u_init.begin(), u_init.end());
        offset += numControls;
    }

    // State at end
    X.push_back(V.nz(Slice(offset,offset+numStates)));
    v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
    v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
    v_init.insert(v_init.end(), x_init.begin(), x_init.end());
    offset += numStates;

    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
    casadi_assert(offset==numVars, "");

    // Initialize Objective Function and Weighting Matrices
    MX J = 0; // Objective Function
    MX Q = MX::zeros(numStates,numStates);
    Q(0,0) = 1e1; // xPos
    Q(1,1) = 1e1; // yPos
    Q(2,2) = 1e1; // zPos
    Q(3,3) = 1e-3; // dx
    Q(4,4) = 1e-3; // dy
    Q(5,5) = 1e-3; // dz
    Q(6,6) = 1e8; // sq
    Q(7,7) = 1e8; // vq
    Q(8,8) = 1e8; // vq
    Q(9,9) = 1e8; // vq
    Q(10,10) = 1e8; // dw
    Q(11,11) = 1e8; // dw
    Q(12,12) = 1e8; // dw


    MX R = MX::zeros(numControls,numControls);
    R(0,0) = 1e3;
    R(1,1) = 1e3;
    R(2,2) = 1e3;
    R(3,3) = 1e10;
    R(4,4) = 1e10;
    R(5,5) = 1e10;

    MX xd = MX::zeros(numStates);
    xd(0) = 0.0; xd(1) = 0.0; xd(2) = 0.0;
    xd(3) = 0.0; xd(4) = 0.0; xd(5) = 0.0;
    xd(6) = 1.0; xd(7) = 0.0; xd(8) = 0.0; xd(9) = 0.0;
    xd(10) = 0.0; xd(11) = 0.0; xd(12) = 0.0;

    //Constraint function and bounds
    std::vector<MX> g;

    // Loop over shooting nodes
    for(int k=0; k<N; ++k){
        // Create an evaluation node
        MXDict I_out = F(MXDict{{"x0", X[k]}, {"p", U[k]}});

        // Save continuity constraints
        g.push_back( I_out.at("xf") - X[k+1] );

        // Add objective function contribution
        J += mtimes(mtimes((I_out.at("xf")-xd).T(),Q),(I_out.at("xf")-xd)) + mtimes(mtimes(U[k].T(),R),U[k]);
    }

    // NLP
    MXDict nlp = {{"x", V}, {"f", J}, {"g", vertcat(g)}};

    // Set options
    Dict opts;
    opts["ipopt.tol"] = 1e-5;
    opts["ipopt.max_iter"] = 3;
    opts["ipopt.print_level"] = 0;
    opts["ipopt.acceptable_tol"] = 1e-8;
    opts["ipopt.acceptable_obj_change_tol"] = 1e-6;
    opts["ipopt.file_print_level"] = 3;
    opts["ipopt.print_timing_statistics"] = "yes";
    opts["ipopt.output_file"] = "timing.csv";

    // Create an NLP solver and buffers
    Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
    std::map<std::string, DM> arg, res, sol;

    // Bounds and initial guess
    arg["lbx"] = v_min;
    arg["ubx"] = v_max;
    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["x0"] = v_init;

    //---------------------//
    //      MPC Loop       //
    //---------------------//
//    double xPos = 1.5;
//    double xVel = 0.001;
//    double yPos = (2/n)*xVel;
//    double yVel = -2*n*xPos;
//    double zPos = 1.5;
//    double zVel = 0.001;
//    std::vector<double> q0 = { 0.771517, 0.46291, 0.308607, 0.308607 }; // Normalized [0.5,0.3,0.2,0.2]
//    std::vector<double> wc0 = { 0.0, 0.0, -0.005 };

    Eigen::MatrixXd wcInit(3,1);
    Eigen::MatrixXd wtInit(3,1);
    Eigen::MatrixXd dwInit(3,1);
    Eigen::MatrixXd qInit(4,1);
    Eigen::MatrixXd RtcInit(3,3);

    wcInit << wc0[0] , wc0[1], wc0[2];
    wtInit << 0 , 0, n;
    qInit << q0[0],q0[1],q0[2],q0[3];
    RtcInit = q2R(qInit);
    dwInit = wcInit - RtcInit.transpose()*wtInit;

    Eigen::MatrixXd x0(numStates,1);
    x0 << xPos, yPos, zPos, xVel, yVel, zVel, qInit, dwInit;
    double oneNormX = 100;

    Eigen::MatrixXd xs(numStates,1);
    xs << 0,0,0,0,0,0,1,0,0,0,0,0,0;

    Eigen::MatrixXd xx(numStates, N+1);
    xx.col(0) = x0;

    Eigen::MatrixXd xx1(numStates, N+1);
    Eigen::MatrixXd X0(numStates,N+1);
    Eigen::MatrixXd u0;
    Eigen::MatrixXd uwu(numControls,N);
    Eigen::MatrixXd u_cl(numControls,N);

    vector<vector<double> > MPCstates(numStates);
    vector<vector<double> > MPCcontrols(numControls);

    // Start MPC
    int iter = 0;
    double epsilon = 1e-3;
    Eigen::VectorXf t(N);
    t(0) = 0;
    cout <<numVars<<endl;

    double infNorm = 100;

    while( infNorm > epsilon && iter <= N)
    {
        // Solve NLP
        sol = solver(arg);

        std::vector<double> V_opt(sol.at("x"));
        Eigen::MatrixXd V = Eigen::Map<Eigen::Matrix<double, 9513, 1> >(V_opt.data()); // convert vector to eigen matrix
        //Eigen::MatrixXd V = Eigen::Map<Eigen::Matrix<double, numVars, 1> >(V_opt.data());

        // Store Solution
        for(int i=0; i<=N; ++i)
        {
            xx1(0,i) = V(i*(numStates+numControls));
            xx1(1,i) = V(1+i*(numStates+numControls));
            xx1(2,i) = V(2+i*(numStates+numControls));
            xx1(3,i) = V(3+i*(numStates+numControls));
            xx1(4,i) = V(4+i*(numStates+numControls));
            xx1(5,i) = V(5+i*(numStates+numControls));
            xx1(6,i) = V(6+i*(numStates+numControls));
            xx1(7,i) = V(7+i*(numStates+numControls));
            xx1(8,i) = V(8+i*(numStates+numControls));
            xx1(9,i) = V(9+i*(numStates+numControls));
            xx1(10,i) = V(10+i*(numStates+numControls));
            xx1(11,i) = V(11+i*(numStates+numControls));
            xx1(12,i) = V(12+i*(numStates+numControls));
            if(i < N)
            {
                uwu(0,i)= V(numStates + i*(numStates+numControls));
                uwu(1,i) = V(1+numStates + i*(numStates+numControls));
                uwu(2,i) = V(2+numStates + i*(numStates+numControls));
                uwu(3,i) = V(3+numStates + i*(numStates+numControls));
                uwu(4,i) = V(4+numStates + i*(numStates+numControls));
                uwu(5,i) = V(5+numStates + i*(numStates+numControls));
            }
        }
        cout << "NLP States:" << endl << xx1 << endl;
        cout <<endl;
        cout << "NLP Controls:" << endl <<  uwu << endl;
        cout <<endl;

        // Get solution Trajectory
        u_cl.col(iter) = uwu.col(0); // Store first control action from optimal sequence
        t(iter+1) = t(iter) + ts;

        // Apply control and shift solution
        shift(N,ts,x0,uwu,u0,n,mc);
        xx(Eigen::placeholders::all,iter+1)=x0;

        // Shift trajectory to initialize next step
        std::vector<int> ind(N) ; // vector with N-1 integers to be filled
        std::iota (std::begin(ind), std::end(ind), 1); // fill vector with N integers starting at 1
        X0 = xx1(Eigen::placeholders::all,ind); // assign X0 with columns 1-(N) of xx1
        X0.conservativeResize(X0.rows(), X0.cols()+1);
        X0.col(X0.cols()-1) = xx1(Eigen::placeholders::all,Eigen::placeholders::last);

        cout << "MPC States:" << endl << xx << endl;
        cout <<endl;
        cout << "MPC Controls:" << endl << u_cl << endl << endl;

        for(int j=0; j<numStates; j++)
        {
            MPCstates[j].push_back(x0(j));
        }

        for(int j=0; j<numControls; j++)
        {
            MPCcontrols[j].push_back(u_cl(j,iter));
        }

        // Re-initialize Problem Parameters
        v_min.erase(v_min.begin(),v_min.begin()+numStates);
        v_min.insert(v_min.begin(),x0(12));
        v_min.insert(v_min.begin(),x0(11));
        v_min.insert(v_min.begin(),x0(10));
        v_min.insert(v_min.begin(),x0(9));
        v_min.insert(v_min.begin(),x0(8));
        v_min.insert(v_min.begin(),x0(7));
        v_min.insert(v_min.begin(),x0(6));
        v_min.insert(v_min.begin(),x0(5));
        v_min.insert(v_min.begin(),x0(4));
        v_min.insert(v_min.begin(),x0(3));
        v_min.insert(v_min.begin(),x0(2));
        v_min.insert(v_min.begin(),x0(1));
        v_min.insert(v_min.begin(),x0(0));

        v_max.erase(v_max.begin(),v_max.begin()+numStates);
        v_max.insert(v_max.begin(),x0(12));
        v_max.insert(v_max.begin(),x0(11));
        v_max.insert(v_max.begin(),x0(10));
        v_max.insert(v_max.begin(),x0(9));
        v_max.insert(v_max.begin(),x0(8));
        v_max.insert(v_max.begin(),x0(7));
        v_max.insert(v_max.begin(),x0(6));
        v_max.insert(v_max.begin(),x0(5));
        v_max.insert(v_max.begin(),x0(4));
        v_max.insert(v_max.begin(),x0(3));
        v_max.insert(v_max.begin(),x0(2));
        v_max.insert(v_max.begin(),x0(1));
        v_max.insert(v_max.begin(),x0(0));

        arg["lbx"] = v_min;
        arg["ubx"] = v_max;
        arg["x0"] = V_opt;

        infNorm = max((x0-xs).lpNorm<Eigen::Infinity>(),u_cl.col(iter).lpNorm<Eigen::Infinity>()); // l-infinity norm of current state and control
        cout << infNorm << endl;

        iter++;
        cout << iter <<endl;
    }

    ofstream fout; // declare fout variable
    fout.open("MPCresults.csv", std::ofstream::out | std::ofstream::trunc ); // open file to write to
    fout << "MPC States:" << endl;
    for(int i=0; i < MPCstates.size(); i++)
    {
        fout << MPCstates[i] << endl;
    }
    fout << "MPC Controls:" << endl;
    for(int i=0; i < MPCcontrols.size(); i++)
    {
        fout << MPCcontrols[i] << endl;
    }
    fout.close();

//    plt::figure();
//    plt::plot(MPCstates[0],MPCstates[1]);
//
//    plt::figure();
//    plt::plot(MPCstates[2]);
//
//    plt::figure();
//    plt::plot(MPCcontrols[0]);
//    plt::plot(MPCcontrols[1]);
//
//    plt::show();



    return 0;
}

MatrixXd Skew(VectorXd f)
{
    MatrixXd S(3,3);
    S << 0, -f(2), f(1),
         f(2), 0, -f(0),
         -f(1), f(0), 0;

    return S;
}

MatrixXd q2R(MatrixXd q){
    MatrixXd R(3,3);
    double sq = q(0);
    VectorXd vq(3);
    vq << q(1),q(2),q(3);

    R(0,0) = pow(sq,2) + pow(vq(0),2) - pow(vq(1),2) - pow(vq(2),2);
    R(0,1) = 2*(vq(0)*vq(1)-sq*vq(2));
    R(0,2) = 2*(vq(0)*vq(2)+sq*vq(1));

    R(1,0) = 2*(vq(0)*vq(1)+sq*vq(2));
    R(1,1) = pow(sq,2) - pow(vq(0),2) + pow(vq(1),2) - pow(vq(2),2);
    R(1,2) = 2*(vq(1)*vq(2)-sq*vq(0));

    R(2,0) =  2*(vq(0)*vq(2)-sq*vq(2));
    R(2,1) =  2*(vq(1)*vq(2)+sq*vq(0));
    R(2,2) =  pow(sq,2) - pow(vq(0),2) - pow(vq(1),2) + pow(vq(2),2);

    return R;
};

SX q2R(SX sq, SX vq)
{
    SX R = SX::sym("R",3,3);
    R(0,0) = pow(sq,2) + pow(vq(0),2) - pow(vq(1),2) - pow(vq(2),2);
    R(0,1) = 2*(vq(0)*vq(1)-sq*vq(2));
    R(0,2) = 2*(vq(0)*vq(2)+sq*vq(1));

    R(1,0) = 2*(vq(0)*vq(1)+sq*vq(2));
    R(1,1) = pow(sq,2) - pow(vq(0),2) + pow(vq(1),2) - pow(vq(2),2);
    R(1,2) = 2*(vq(1)*vq(2)-sq*vq(0));

    R(2,0) =  2*(vq(0)*vq(2)-sq*vq(2));
    R(2,1) =  2*(vq(1)*vq(2)+sq*vq(0));
    R(2,2) =  pow(sq,2) - pow(vq(0),2) - pow(vq(1),2) + pow(vq(2),2);

    return R;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: f
// Description: This function is used to implement the dynamics of our system
//              once a control action is implemented
// Inputs: MatrixXd st - current state, MatrixXd con - current control action
// Outputs: MatrixXd xDot - time derivative of the current state
//////////////////////////////////////////////////////////////////////////////
MatrixXd f(MatrixXd st, MatrixXd con, double n, double mc)
{
    double x = st(0);
    double y = st(1);
    double z = st(2);
    double dx = st(3);
    double dy = st(4);
    double dz = st(5);
    double sq = st(6);
    double v1 = st(7);
    double v2 = st(8);
    double v3 = st(9);
    double dw1 = st(10);
    double dw2 = st(11);
    double dw3 = st(12);

    VectorXd dw(3);
    dw << dw1,dw2,dw3;
    VectorXd vq(3);
    vq << v1,v2,v3;
    VectorXd q(4);
    q << sq,vq;

    double u1 = con(0);
    double u2 = con(1);
    double u3 = con(2);
    double u4 = con(3);
    double u5 = con(4);
    double u6 = con(5);

    VectorXd thrust(3);
    thrust << u1,u2,u3;
    VectorXd torque(3);
    torque << u4,u5,u6;

    MatrixXd Rtc = q2R(q);
    VectorXd f = Rtc*thrust;
    Matrix3d Jd = Matrix3d::Zero(); // Moment of inertia for the chaser
    Jd(0,0) = 0.2734;
    Jd(1,1) = 0.2734;
    Jd(2,2) = 0.3125;

    VectorXd wt(3); // angular velocity of the target
    wt(0) = 0;
    wt(1) = 0;
    wt(2) = n;

    MatrixXd Kd = Jd.inverse(); // Kd = Jd^-1
    MatrixXd Ko = Rtc*Kd*Rtc;

    MatrixXd xDot(13,1);
    xDot << dx, dy, dz,
            3* pow(n,2)*x + 2*n*dy + f(0)/mc,
            -2*n*dx + f(1)/mc,
            -pow(n,2)*z + f(2)/mc,
            -0.5*vq.dot(dw),
            0.5*(sq * MatrixXd::Identity(3,3) + Skew(vq)) * dw,
            Skew(dw)*wt + Ko*torque - Ko*(Skew(dw)*(Jd*Rtc.transpose()*dw) + Skew(dw)*(Jd*Rtc.transpose()*wt) + Skew(wt)*(Jd*Rtc.transpose()*dw) + Skew(wt)*(Jd*Rtc.transpose()*wt));
    return xDot;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: shift
// Description: This function is used to shift our MPC states and control inputs
//              in time so that we can re-initialize our optimization problem
//              with the new current state of the system
// Inputs: N - Prediction Horizon, ts - sampling time, x0 - initial state,
//             uwu - optimal control sequence from NLP, u0 - shifted
//             control sequence, n - mean motion of target satellite,
//             mc - mass of chaser satellite
// Outputs: None
//////////////////////////////////////////////////////////////////////////////
void shift(int N, double ts, MatrixXd& x0, MatrixXd uwu, MatrixXd& u0, double n, double mc)
{
    // Shift State
    MatrixXd st = x0;
    MatrixXd con = uwu.col(0);

    MatrixXd k1 = f(st,con,n,mc);
    MatrixXd k2 = f(st + (ts/2)*k1,con,n,mc);
    MatrixXd k3 = f(st + (ts/2)*k2,con,n,mc);
    MatrixXd k4 = f(st + ts*k3,con,n,mc);

    st = st + ts/6*(k1 + 2*k2 + 2*k3 + k4);
    x0 = st;

    // Shift Control
    std::vector<int> ind(N-1) ; // vector with N-1 integers to be filled
    std::iota (std::begin(ind), std::end(ind), 1); // fill vector with N-1 integers starting at 1
    u0 = uwu(Eigen::placeholders::all,ind); // assign u0 with columns 1-(N-1) of uwu
    u0.conservativeResize(u0.rows(), u0.cols()+1);
    u0.col(u0.cols()-1) = uwu(Eigen::placeholders::all,Eigen::placeholders::last); // copy last column and append it
}