%
% FEM Group Project 1
%
% By Chan Chi Hin

clc
clear all
close all

% Length of Rod
L = 4;
% Number of elements
N = 10;
dx = L/(N+1);
x = -2:dx:2;

% Exact Solution
T_ex = -f(x) + 50*sinh(2).*x + 100*(2+cosh(2));

% Basis Functions
phi_1 = (2-x).*(2+x);
phi_2 = x.*(2-x).*(2+x);

% Collocation Method
% Setting system of equations to solve for coefficients
col_pt = 2/3;
a(1,1) = 2;
a(2,1) = 2;
a(1,2) = 6*col_pt;
a(2,2) = 6*-col_pt;
b(1,1) = f(col_pt);
b(2,1) = f(-col_pt);

c = a\b;

T_col = 200 + c(1).*phi_1 + c(2).*phi_2;
col_error = T_ex - T_col;

% Galerkin Method
% Coefficients obtained by doing integration by hand
g1 = 73.079;
g2 = 22;

T_gal = 200 + g1.*phi_1 + g2 .*phi_2;
gal_error = T_ex - T_gal;

% Solutions
plot(x,T_ex,'--', x, T_col, '-o', x, T_gal, '-o');
legend("Exact", "Collocation Method", "Galerkin Method");
title("Temperature Distribution Along Rod for different methods");
xlabel("x");
ylabel("T");
grid on

% Error
figure;
plot(x, col_error, '-o', x, gal_error, '-o');
legend("Collocation Method", "Garlerkin Method");
title("Difference between analytical method and Garlerkin/Collocation");
xlabel("x");
ylabel("Error");
grid on


function y=f(x)
    y = 100*exp(x);
end
