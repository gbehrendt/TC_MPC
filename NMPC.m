% NEW NEW NEW Dynamics

clc; close all; clear all;

%addpath(':/home/corelab/Documents/Gabe/casadi')
addpath('C:\Users\gbehrendt\OneDrive - University of Florida\Research\myPapers\Paper4\casadi2')
import casadi.*


% Tunable Parameters
T = 3; % sampling time
N = 1000; % prediction horizon
Q = diag([1e1*ones(1,3),1e-3*ones(1,3),1e8*ones(1,4),1e8*ones(1,3)]); % cost matrix for states
R = diag([1e3*ones(1,3),1e10*ones(1,3)]); % cost matrix for inputs

% Problem parameters
tPeriod = 92.68 * 60; %ISS orbital period (seconds)
% tRad = 400; % ISS distance from Earth (km)
n = -2*pi/tPeriod; % Mean motion of ISS (rad/s)
Jd = diag([0.2734 0.2734 0.3125]); % Moment of inertia matrix of chaser
mc = 12; % mass of chaser (kg)
wt = [0;0;n]; % constant angular velocity of target in target frame
xs = [zeros(6,1); 1; zeros(6,1)]; % desired state

% input limits
thrustMax = 1e-2; thrustMin = -thrustMax; % Newtons
torqueMax = 1e-4; torqueMin = -torqueMax; % rad/sec^2

% Initial conditions
xPos = 1.5;
xVel = 0.001;
yPos = (2/n)*xVel;
yVel = -2*n*xPos;
zPos = 3;
zVel = 0;

initPos = [xPos;yPos;zPos]; % (km)
initVel = [xVel; yVel; zVel]; % (km/s)
q0 = [0.5; 0.3; 0.2; 0.2]; q0 = q0/norm(q0); % initial unit quaternion
wc0 = [0.0;0.0;-0.005]; % initial angular velocity about the chaser frame (rad/s)


% Declare states + control inputs
x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z'); % relative distances
dx = SX.sym('dx'); dy = SX.sym('dy'); dz = SX.sym('dz'); % relative velocities
sq = SX.sym('sq'); vq = SX.sym('vq',1,3); % unit quaternion for chaser orientation wrt target frame  (s,i,j,k)
dw = SX.sym('dw',1,3); % angular velocity of chaser in the target frame
q = [sq vq];

states = [x,y,z,dx,dy,dz,sq,vq,dw];
numStates = length(states);

% Declare control inputs
u1 = SX.sym('u1'); u2 = SX.sym('u2'); u3 = SX.sym('u3');
u4 = SX.sym('u4'); u5 = SX.sym('u5'); u6 = SX.sym('u6');
controls = [u1;u2;u3;u4;u5;u6]; numControls = length(controls);

% Declare xdot equations
Rtc = q2R(q); % orientation of the chaser frame relative to the target
f = Rtc*[u1;u2;u3]; % force generated by thrusters wrt target frame
J1 = Jd(1,1); J2 = Jd(2,2); J3 = Jd(3,3); % diagonol elements of J
tau = [u4;u5;u6]; % torque inputs
W = Rtc'*wt;
Kd = eye(3)/Jd;
Ko = Rtc*Kd*Rtc; 

rhs = [dx; dy; dz;
       3*n^2*x+2*n*dy+f(1)/mc; -2*n*dx+f(2)/mc; -n^2*z+f(3)/mc;
       
        -(1/2) * vq*dw';
        (1/2) * (sq * eye(3) + skew(vq)) * dw';
%        (1/2)*QQ*dw';
        
        skew(dw)*wt + Ko*tau - Ko*(skew(dw)*(Jd*Rtc'*dw') + skew(dw)*(Jd*Rtc'*wt) + skew(wt)*(Jd*Rtc'*dw') + skew(wt)*(Jd*Rtc'*wt))
       
       ]; % xDot equations / System RHS

% Declare problem structures
f = Function('f',{states,controls},{rhs}); % nonlinear mapping function xDot = f(x,u)
U = SX.sym('U',numControls, N); % Decision variables (controls)
P = SX.sym('P',numStates + numStates); % parameters that include initial state and desired state
X = SX.sym('X',numStates, N+1); % matrix that contains the states over the problem
obj = 0;
g = []; % Constraint vector




%% Construct constraints
st = X(:,1); %initial state
g = [g;st-P(1:numStates)];
for k = 1:N
    st = X(:,k); con = U(:,k);
    obj = obj + (st-xs)'*Q*(st-xs) + con'*R*con; % compute objective
    stNext = X(:,k+1);
    % 4th order Runge-Kutta
    k1 = f(st, con);
    k2 = f(st + T/2*k1, con);
    k3 = f(st + T/2*k2, con);
    k4 = f(st + T*k3, con);

    stNextRK4 = st + T/6*(k1 + 2*k2 + 2*k3 + k4);
    g = [g;stNext-stNextRK4]; % Multiple shooting path continuity constraint
end



for k = 1:numStates
    g = [g; X(k,N+1)]; % terminal equality constraint x(N) = 0
end

% Assemble optimization problem using defined structures
OPTvars = [reshape(X,numStates*(N+1),1); reshape(U,numControls*N,1)]; % make decision variables into column vector
nlp_prob = struct('f', obj, 'x', OPTvars, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2;
opts.ipopt.print_level = 0;
opts.print_time = 0;

opts.ipopt.acceptable_iter = 1; % 15

opts.ipopt.acceptable_tol = 1e10; % 1e-6
opts.ipopt.acceptable_constr_viol_tol = 1e-4; % 1e-2
opts.ipopt.acceptable_compl_inf_tol = 1e10; % 1e-2
opts.ipopt.expect_infeasible_problem = 'yes';

%opts.ipopt.acceptable_dual_inf_tol = 1e10; % 1e10
%opts.ipopt.acceptable_obj_change_tol = 1e20; % 1e20





% opts.ipopt.alpha_for_y = 'min-dual-infeas';
% opts.ipopt.gamma_theta = 9e-1;
% opts.ipopt.gamma_phi = 10e-14;


solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

% Define Constraints
args = struct;

    % Equality Constraints
    args.lbg(1:numStates*(N+1)) = 0; % State path equality constriants 
    args.ubg(1:numStates*(N+1)) = 0; % lbg = ubg for equality constraints

    % terminal constraint x(N) = xd
    args.lbg(numStates*(N+1) + 1 : numStates*(N+1) + numStates) = xs; % pos + vel + Vq + wc x,y
    args.ubg(numStates*(N+1) + 1 : numStates*(N+1) + numStates) = xs; % pos + vel + Vq + wc x,y
    
    % State limits
    args.ubx(1:numStates:numStates*(N+1),1) = inf; % state x upper bound
    args.ubx(2:numStates:numStates*(N+1),1) = inf; % state y upper bound
    args.ubx(3:numStates:numStates*(N+1),1) = inf; % state z upper bound
    args.ubx(4:numStates:numStates*(N+1),1) = inf; % state dx upper bound
    args.ubx(5:numStates:numStates*(N+1),1) = inf; % state dy upper bound
    args.ubx(6:numStates:numStates*(N+1),1) = inf; % state dz upper bound
    args.ubx(7:numStates:numStates*(N+1),1) = 1; % state sq upper bound
    args.ubx(8:numStates:numStates*(N+1),1) = 1; % state qi upper bound
    args.ubx(9:numStates:numStates*(N+1),1) = 1; % state qj upper bound
    args.ubx(10:numStates:numStates*(N+1),1) = 1; % state qk upper bound
    args.ubx(11:numStates:numStates*(N+1),1) = inf; % state \delta wcx upper bound
    args.ubx(12:numStates:numStates*(N+1),1) = inf; % state \delta wcy upper bound
    args.ubx(13:numStates:numStates*(N+1),1) = inf; % state \delta wcz upper bound

    args.lbx(1:numStates:numStates*(N+1),1) = -inf; % state x lower bound
    args.lbx(2:numStates:numStates*(N+1),1) = -inf; % state y lower bound
    args.lbx(3:numStates:numStates*(N+1),1) = -inf; % state z lower bound
    args.lbx(4:numStates:numStates*(N+1),1) = -inf; % state dx lower bound
    args.lbx(5:numStates:numStates*(N+1),1) = -inf; % state dy lower bound
    args.lbx(6:numStates:numStates*(N+1),1) = -inf; % state dz lower bound
    args.lbx(7:numStates:numStates*(N+1),1) = -1; % state s lower bound
    args.lbx(8:numStates:numStates*(N+1),1) = -1; % state qi lower bound
    args.lbx(9:numStates:numStates*(N+1),1) = -1; % state qj lower bound
    args.lbx(10:numStates:numStates*(N+1),1) = -1; % state qk lower bound
    args.lbx(11:numStates:numStates*(N+1),1) = -inf; % state \delta wcx lower bound
    args.lbx(12:numStates:numStates*(N+1),1) = -inf; % state \delta wcy lower bound
    args.lbx(13:numStates:numStates*(N+1),1) = -inf; % state \delta wcz lower bound


    
    % input constraints    
    args.lbx(numStates*(N+1) + 1 : numControls : numStates*(N+1) + numControls*N,1) = thrustMin; % u1 (thrust) lower bound
    args.lbx(numStates*(N+1) + 2 : numControls : numStates*(N+1) + numControls*N,1) = thrustMin; % u2 (thrust) lower bound
    args.lbx(numStates*(N+1) + 3 : numControls : numStates*(N+1) + numControls*N,1) = thrustMin; % u3 (thrust) lower bound
    args.lbx(numStates*(N+1) + 4 : numControls : numStates*(N+1) + numControls*N,1) = torqueMin; % u4 (angular acceleration) lower bound
    args.lbx(numStates*(N+1) + 5 : numControls : numStates*(N+1) + numControls*N,1) = torqueMin; % u5 (angular acceleration) lower bound
    args.lbx(numStates*(N+1) + 6 : numControls : numStates*(N+1) + numControls*N,1) = torqueMin; % u6 (angular acceleration) lower bound

    args.ubx(numStates*(N+1) + 1 : numControls : numStates*(N+1) + numControls*N,1) = thrustMax; % u1 (thrust) upper bound
    args.ubx(numStates*(N+1) + 2 : numControls : numStates*(N+1) + numControls*N,1) = thrustMax; % u2 (thrust) upper bound
    args.ubx(numStates*(N+1) + 3 : numControls : numStates*(N+1) + numControls*N,1) = thrustMax; % u3 (thrust) upper bound
    args.ubx(numStates*(N+1) + 4 : numControls : numStates*(N+1) + numControls*N,1) = torqueMax; % u4 (angular acceleration) upper bound
    args.ubx(numStates*(N+1) + 5 : numControls : numStates*(N+1) + numControls*N,1) = torqueMax; % u5 (angular acceleration) upper bound
    args.ubx(numStates*(N+1) + 6 : numControls : numStates*(N+1) + numControls*N,1) = torqueMax; % u6 (angular acceleration) upper bound

%------------------------------%
%       SIMULATION LOOP        %
%------------------------------%

t0 = 0;

%initial conditions
Rtc0 = q2R(q0);
x0 = [initPos; initVel; q0; wc0 - Rtc0'*wt]; %initial condition

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;
u0 = zeros(N,numControls); % six control inputs
X0 = repmat(x0,1,N+1)'; % initilization of states decision variables
sim_time = N*T + T; % Max simulation time
tol = 1e-6;
eps = 1e-3;
stop = 0;

% Start MPC
iter = 0;
xx1 = []; % stores predicted state
u_cl = []; % stores control actions
normq(1) = norm(q0,2);

% initial error
posError(:,1) = [norm(x0(1)-xs(1),2);
                 norm(x0(2)-xs(2),2);
                 norm(x0(3)-xs(3),2)];
velError(:,1) = [norm(x0(4)-xs(4),2);
                 norm(x0(5)-xs(5),2);
                 norm(x0(6)-xs(6),2)];
qError(:,1) = [norm(x0(7)-xs(7),2);
               norm(x0(8)-xs(8),2);
               norm(x0(9)-xs(9),2);
               norm(x0(10)-xs(10),2);];
dwError(:,1) = [norm(x0(11)-xs(11),2);
                norm(x0(12)-xs(12),2);
                norm(x0(13)-xs(13),2);];

while(norm((x0-xs),2) > tol && iter < sim_time/T && stop == 0)
    args.p = [x0;xs]; % set values of parameter vector
    args.x0 = [reshape(X0',numStates*(N+1),1); reshape(u0',numControls*N,1)]; % intial value of optimization variables
    
    tic;
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, 'lbg', ...
        args.lbg, 'ubg', args.ubg, 'p', args.p);
    times(iter+1) = toc; % store execution time
    
    
    % Store solution
    uPlot = reshape(full(sol.x(numStates*(N+1)+1:end))',numControls,N)'; % Reshaping u from a vector to a matrix (only get controls from the solution)
    xx1(:,1:numStates,iter+1) = reshape(full(sol.x(1:numStates*(N+1)))',numStates,N+1)'; % get solution trajectory
    objVal(iter+1)=full(sol.f);
    gVal(:,iter+1) = full(sol.g);
    lamg(:,iter+1) = full(sol.lam_g);
    lamp(:,iter+1) = full(sol.lam_p);
    lamx(:,iter+1) = full(sol.lam_x);
    
    u_cl = [u_cl; uPlot(1,:)]; % store first control action from optimal sequence
    t(iter+1) = t0;

    % Calculate quaternion norm
    qk = [xx(7,iter+1); xx(8,iter+1); xx(9,iter+1); xx(10,iter+1)];
    normq(iter+2) = norm(qk,2)^2;

    % Apply control and shift solution
    [t0, x0, u0] = shift(T, t0, x0, uPlot, f);
    xx(:,iter+2) = x0;

    % Shift trajectory to initilize next step
    X0 = xx1(:,1:numStates,iter+1);
    X0 = [X0(2:end,:); X0(end,:)];

    % Error calculations
    posError(:,iter+2) = [norm(x0(1)-xs(1),2);
                          norm(x0(2)-xs(2),2);
                          norm(x0(3)-xs(3),2)];
    velError(:,iter+2) = [norm(x0(4)-xs(4),2);
                          norm(x0(5)-xs(5),2);
                          norm(x0(6)-xs(6),2)];
    qError(:,iter+2) = [norm(x0(7)-xs(7),2);
                        norm(x0(8)-xs(8),2);
                        norm(x0(9)-xs(9),2);
                        norm(x0(10)-xs(10),2);];
    dwError(:,iter+2) = [norm(x0(11)-xs(11),2);
                         norm(x0(12)-xs(12),2);
                         norm(x0(13)-xs(13),2)];
    curError(iter+1) = norm(x0-xs,2);
    Error(:,iter+1) = x0-xs;
    chck(:,iter+1) = abs(x0-xs) > eps;
    
    % check for stopping condition
    if any(chck(:,iter+1) == 1)
        stop = 0;
    else
        stop = 1;
    end
    
    % Cost Calculations
    cost(iter+1) = (x0 - xs)'* Q *(x0 - xs) + u_cl(iter+1,:) * R * u_cl(iter+1,:)';

    iter = iter + 1
    x0(1:6)
end

if(iter >= sim_time/T)
    disp("timed out")
end
if(norm((x0-xs),2)<tol)
    disp("tolerance requirement met")
end
if stop == 1
   disp("epsilon met") 
end
%%
totError = norm((x0-xs),2);
sprintf('%.12f',norm((x0-xs),2))
avg_MPC_time = mean(times);
total_MPC_time = sum(times);

tspan = T*[0:iter];
tspan1 = 0:iter*T-1;
uPlot = [];

for i = 1:numControls
    row=[];
    for j = 1:iter
        row = [row u_cl(j,i)*ones(1,T)];
    end
    uPlot = [uPlot; row];
end

% Individual Costs

for j = 1:size(xx,2)-1
    for i = 1:numStates
        indCost(i,j) =  (xx(i,j) - xs(i))'* Q(i,i) *(xx(i,j) - xs(i));
    end 
end

figure(1)
tiledlayout(2,1)
nexttile
plot(tspan1,uPlot(1:3,:))
title('Deputy Thrust Inputs, $u^{\mathcal{D}}_d$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Thrust ($N$)','Interpreter','latex','FontSize',12)
xlim([0 iter*T])
%ylim([-0.5e-2 0.5e-2])
legend('$u^{\mathcal{D}}_{d,x}$','$u^{\mathcal{D}}_{d,y}$','$u^{\mathcal{D}}_{d,z}$','Interpreter','latex','Location','southeast','FontSize',9)
grid on;

nexttile
plot(tspan1,uPlot(4:6,:))
title('Deputy Torque Inputs, $\tau^{\mathcal{D}}$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Angular Acceleration ($rad/s^2$)','Interpreter','latex','FontSize',12)
xlim([0 iter*T])
%ylim([-0.5e-3 0.5e-3])
legend('$\tau^{\mathcal{D}}_x$', '$\tau^{\mathcal{D}}_y$','$\tau^{\mathcal{D}}_z$','Interpreter','latex','Location','southeast','FontSize',9)
grid on;


figure(2)
tiledlayout(2,1)
nexttile
plot(tspan, xx(1:3,:))
title('Deputy Relative Position, $\delta r^{\mathcal{O}}$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Deputy Position ($km$)','Interpreter','latex','FontSize',12)
legend('$\delta x$', '$\delta y$','$\delta z$','Interpreter','latex','FontSize',10)
grid on;

nexttile
plot(tspan, posError)
title('Deputy Relative Position Error, $\Vert \delta r^{\mathcal{O}} \Vert_2$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Deputy Position ($km$)','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
legend('$\delta x$', '$\delta y$','$\delta z$','Interpreter','latex','FontSize',10)
grid on;

figure(3)
tiledlayout(2,1)
nexttile
plot(tspan, xx(4:6,:))
title('Deputy Relative Velocity, $\delta \dot{r}^{\mathcal{O}}$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Deputy Velocity ($km/s$)','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
ylim([-0.05 0.05])
legend('$\delta \dot{x}$', '$\delta \dot{y}$','$\delta \dot{z}$','Interpreter','latex','FontSize',10)
grid on;

nexttile
plot(tspan, velError)
title('Deputy Relative Velocity Error, $\Vert \delta \dot{r}^{\mathcal{O}} \Vert_2$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Deputy Velocity ($km/s$)','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
ylim([-0.05 0.05])
legend('$\delta \dot{x}$', '$\delta \dot{y}$','$\delta \dot{z}$','Interpreter','latex','FontSize',10)
grid on;

figure(4)
tiledlayout(2,1)
nexttile
plot(tspan, xx(7:10,:))
title('Quaternion, $q^{\mathcal{O}}_{\mathcal{D}}$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('$q^{\mathcal{O}}_{\mathcal{D}}$','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
ylim([-0.5 3.25])
legend('$\eta^{\mathcal{O}}_{\mathcal{D}}$', '$\rho^{\mathcal{O}}_{\mathcal{D},1}$','$\rho^{\mathcal{O}}_{\mathcal{D},2}$','$\rho^{\mathcal{O}}_{\mathcal{D},3}$','Interpreter','latex','FontSize',9)
grid on;

nexttile
plot(tspan, normq)
title('Quaternion Norm, $\Vert q^{\mathcal{O}}_{\mathcal{D}} \Vert_2$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('$\Vert q^{\mathcal{O}}_{\mathcal{D}} \Vert_2$','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
ylim([0 2])
grid on;

figure(5)
subplot(2,1,1)
plot(tspan, xx(11:13,:))
title('Deputy Relative Angular Velocity, $\omega^{\mathcal{O}}_{\mathcal{O}\mathcal{D}}$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Angular Velocity ($rad/s$)','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
legend('$\delta \omega_1$','$\delta \omega_2$','$\delta \omega_3$','Interpreter','latex','FontSize',9,'Location','southeast')
grid on;

subplot(2,1,2)
plot(tspan, dwError)
title('Deputy Relative Angular Velocity Error, $\Vert \omega^{\mathcal{O}}_{\mathcal{O}\mathcal{D}} \Vert_2$','Interpreter','latex','FontSize',14)
xlabel('Time ($s$)','Interpreter','latex','FontSize',12)
ylabel('Angular Velocity ($rad/s$)','Interpreter','latex','FontSize',12)
%xlim([0 iter*T])
legend('$\delta \omega_1$','$\delta \omega_2$','$\delta \omega_3$','Interpreter','latex','FontSize',9)
grid on;

figure(6)
semilogy(tspan(1:end-1),cost)
title('Objective Value','Interpreter','latex','FontSize',16)
xlabel('Time ($s$)','Interpreter','latex','FontSize',15)
ylabel('$J(x_u(k),u(k))$','Interpreter','latex','FontSize',15)
grid on;

figure(7)
semilogy(tspan(1:end-1),curError)
title('Error','Interpreter','latex','FontSize',16)
xlabel('Time ($s$)','Interpreter','latex','FontSize',15)
ylabel('$\Vert x(k) - x_d \Vert_2$','Interpreter','latex','FontSize',15)
grid on;

%% THIS IS FOR SAVING MY DATA
% addpath('/home/corelab/Documents/Gabe/CCC_IFAC_unperturb_data/');
% addpath('/home/corelab/Documents/Gabe/CCC_IFAC_unperturb_plots/');
% saveDir = "/home/corelab/Documents/Gabe/CCC_IFAC_unperturb_data/";
% figPath = "/home/corelab/Documents/Gabe/CCC_IFAC_unperturb_plots/max"+string(opts.ipopt.max_iter)+"/";
% filename = saveDir + "IFACmaxIter"+string(opts.ipopt.max_iter)+".mat";
% save(filename);
% 
% figName = ["inputs"; "position"; "velocity"; "quaternion"; "angular_vel"; "cost"; "error"];
% for ii = 1:7
%     saveas(figure(ii), fullfile(figPath,(figName(ii) + '.png')));
% end
% 
% iter
% sprintf('%.12f',sum(times))
% sprintf('%.12f',sum(times)/iter)

%%


%% THIS FUNCTION SHIFTS THE PROBLEM TO THE NEXT TIME STEP
function [t0, x0, u0] = shift(T, t0, x0, u, f)
    st = x0;
    con = u(1,:)';

    k1 = f(st, con);
    k2 = f(st + T/2*k1, con);
    k3 = f(st + T/2*k2, con);
    k4 = f(st + T*k3, con);

    st = st + T/6*(k1 + 2*k2 + 2*k3 + k4);


    x0 = full(st);

    t0 = t0 + T;
    u0 = [u(2:size(u,1),:); u(size(u,1),:)];
end

function R = q2R(q)
    R = eye(3) + 2*skew(q(2:4))* skew(q(2:4)) + 2*q(1)*skew(q(2:4));
end

%% THIS FUNCTION WAS FOR PREVIOUS DEBUGGING AND HAS NO USE NOW
function [dw1,dw2] = check(W,dw,tau,J)
    K = inv(J);
    J1 = J(1,1); J2 = J(2,2); J3 = J(3,3);
    
    dw1 = [(tau(1)/J1) + W(3)*dw(2) - W(2)*dw(3) + ((J2-J3)/J1)*dw(2)*dw(3) + ((J2-J3)/J1)*W(2)*W(3) + dw(3)*W(2)*(J2/J1) - dw(2)*W(3)*(J3/J1) + dw(2)*W(3)*(J2/J1) - dw(3)*W(2)*(J3/J1);
           (tau(2)/J2) + W(1)*dw(3) - W(3)*dw(1) + ((J3-J1)/J2)*dw(1)*dw(3) + ((J3-J1)/J2)*W(1)*W(3) + dw(1)*W(3)*(J3/J1) - dw(3)*W(1)*(J1/J2) + dw(3)*W(1)*(J3/J2) - dw(1)*W(3)*(J1/J2);
           (tau(3)/J3) + W(2)*dw(1) - W(1)*dw(2) + ((J1-J2)/J3)*dw(1)*dw(2) + ((J1-J2)/J3)*W(1)*W(2) + dw(2)*W(1)*(J1/J3) - dw(1)*W(2)*(J2/J3) + dw(1)*W(2)*(J1/J3) - dw(2)*W(1)*(J2/J3);];

    dw2 = -K*(skew(dw)*(J*dw) + skew(dw)*(J*W) +  skew(W)*(J*dw) + skew(W)*(J*W)) + skew(dw)*W + K*tau;




end

%% THIS FUNCTION TURNS A QUATERNION INTO A ROTATION MATRIX
function R2 = q2R2(q)
    s = q(1);
    v1 = q(2); 
    v2 = q(3); 
    v3 = q(4);



    R2 = [s^2+v1^2-v2^2-v3^2 2*(v1*v2-s*v3) 2*(v1*v3+s*v2);
         2*(v1*v2+s*v3) s^2-v1^2+v2^2-v3^2, 2*(v2*v3-s*v1);
         2*(v1*v3-s*v3) 2*(v2*v3+s*v1) s^2-v1^2-v2^2+v3^2;];
end

function S = skew(f)
    S = [0 -f(3) f(2);
         f(3) 0 -f(1);
         -f(2) f(1) 0];
end
