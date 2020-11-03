function test_xgl(n, nn, k)
% function test_xgl(n, nn, k)
%
% Test of the CGL and LGL barycentric interpolation and differentiation
% routines on [-1, +1], using (n+1) nodes each, and test of the quadrature
% weights associated with the grids. Also test of OGL barycentric
% and quadrature routines for nodes on open interval (-1, +1), and 
% RGL barycentric and quadrature for nodes on semi-open set [-1, +1).
%
% An analytical function is approximated and differentiated and
% interpolated on a uniform grid with nn points.
%   CGL = Chebyshev-Gauss-Lobatto    [-1, +1]
%   LGL = Legendre-Gauss-Lobatto     [-1, +1]
%   OGL = Ordinary Gauss-Legendre    (-1, +1)
%   RGL = Radau-Gauss-Legendre       [-1, +1)
%
% if k is provided, check the approximation errors as a funtion of the
% number of nodes (max n), in steps of k. The respective quadrature
% errors are also evaluated.
%
% EXAMPLE:
%   test_xgl(32, 500, 1);
%

flipflag = round(rand);
xx = linspace(-1, +1, nn)';

[x1, w1, l1, D1] = cgl(n, flipflag);
F1 = baryxgl(x1, l1, xx);

[x2, w2, l2, D2] = lgl(n, flipflag);
F2 = baryxgl(x2, l2, xx);

[x3, w3, l3, D3] = ogl(n, flipflag);
F3 = baryxgl(x3, l3, xx);

[x4, w4, l4, D4] = rgl(n, flipflag);
F4 = baryxgl(x4, l4, xx);

m1 = 5*randn;
m2 = randn;
m3 = randn;
fx = @(x)(m3*cos(m1*x) + sin(m2*x));
dfx = @(x)(-m3*m1*sin(m1*x) + m2*cos(m2*x));
Ix = @(x)((m3/m1)*sin(m1*x) - (1/m2)*cos(m2*x)); % assuming m1 ~= 0 and m2 ~= 0

f1 = fx(x1);
f2 = fx(x2);
f3 = fx(x3);
f4 = fx(x4);

df1 = dfx(x1);
df2 = dfx(x2);
df3 = dfx(x3);
df4 = dfx(x4);

assert(abs(m1) > 0 && abs(m2) > 0);
Iq = quad(fx, -1, +1, 1e-12);  % Illustration of quadrature integration 
I0 = Ix(1) - Ix(-1);
I1 = sum(w1.*f1); % which is just a simple scalar product using weights w
I2 = sum(w2.*f2);
I3 = sum(w3.*f3);
I4 = sum(w4.*f4);

disp(['flipflag=',num2str(flipflag)]);

fprintf(1, 'I(quad) = %.12f\n', Iq); % replace with "true" value
fprintf(1, 'I(true) = %.12f\n', I0);
fprintf(1, 'I(cgl)  = %.12f\n', I1);
fprintf(1, 'I(lgl)  = %.12f\n', I2);
fprintf(1, 'I(ogl)  = %.12f\n', I3);
fprintf(1, 'I(rgl)  = %.12f\n', I4);

figure; hold on;
plot(x1, f1, 'bo'); % CGL
plot(x2, f2, 'rs'); % LGL
plot(x3, f3, 'gx'); % OGL
plot(x4, f4, 'cx'); % RGL
plot(xx, fx(xx), 'k-'); % True function
plot(xx, F1*f1, 'b--');
plot(xx, F2*f2, 'r-.');
plot(xx, F3*f3, 'g:');
plot(xx, F4*f4, 'c-.');
xlabel('x');
ylabel('f(x)');
legend('CGL', 'LGL', 'OGL', 'RGL', 'f(x)', 'f(CGL)', 'f(LGL)', 'f(OGL)', 'f(RGL)');
title('nodes and interpolation of f(x)');

figure; hold on;
plot(x1, df1, 'bo'); % CGL
plot(x2, df2, 'rs'); % LGL
plot(x3, df3, 'gx'); % OGL
plot(x4, df4, 'cx'); % RGL
plot(xx, dfx(xx), 'k-'); % True function
plot(xx, F1*D1*f1, 'b--');
plot(xx, F2*D2*f2, 'r-.');
plot(xx, F3*D3*f3, 'g:');
plot(xx, F4*D4*f4, 'c:');
xlabel('x');
ylabel('df(x)/dx');
legend('CGL', 'LGL', 'OGL', 'RGL', 'df(x)/dx', 'df(CGL)/dx', 'df(LGL)/dx', 'df(OGL)/dx', 'df(RGL)/dx');
title('nodes and interpolation of df(x)/dx');

if nargin == 3 && k >= 1 && k < n
  % Optional plot of the error vs. the number of nodes,
  % in steps of k: jj = 2:k:n; and evaluate the error for
  % both f(x) and f'(x).
  fxx = fx(xx);
  dfxx = dfx(xx);
  nvec = (2:k:n)';
  err123 = zeros(length(nvec), 4);
  derr123 = zeros(length(nvec), 4);
  ierr123 = zeros(length(nvec), 4);
  for jj=1:length(nvec)
    n = nvec(jj);
    
    [x1, w1, l1, D1] = cgl(n, flipflag); % CGL
    F1 = baryxgl(x1, l1, xx);
    
    [x2, w2, l2, D2] = lgl(n, flipflag); % LGL
    F2 = baryxgl(x2, l2, xx);
    
    [x3, w3, l3, D3] = ogl(n, flipflag); % OGL
    F3 = baryxgl(x3, l3, xx);
    
    [x4, w4, l4, D4] = rgl(n, flipflag); % RGL
    F4 = baryxgl(x4, l4, xx);
    
    f1 = fx(x1);
    f2 = fx(x2);
    f3 = fx(x3);
    f4 = fx(x4);
    
    err123(jj, 1) = norm(fxx - F1*f1, 2);
    err123(jj, 2) = norm(fxx - F2*f2, 2);
    err123(jj, 3) = norm(fxx - F3*f3, 2);
    err123(jj, 4) = norm(fxx - F4*f4, 2);
    
    derr123(jj, 1) = norm(dfxx - F1*D1*f1, 2);
    derr123(jj, 2) = norm(dfxx - F2*D2*f2, 2);
    derr123(jj, 3) = norm(dfxx - F3*D3*f3, 2);
    derr123(jj, 4) = norm(dfxx - F4*D4*f4, 2);
    
    ierr123(jj, 1) = abs(sum(w1.*f1) - I0);
    ierr123(jj, 2) = abs(sum(w2.*f2) - I0);
    ierr123(jj, 3) = abs(sum(w3.*f3) - I0);
    ierr123(jj, 4) = abs(sum(w4.*f4) - I0);
  end

  figure;
  plot(nvec + 1, log10(ierr123), 'o--');
  legend('CGL f', 'LGL f', 'OGL f', 'RGL f');
  xlabel(sprintf('#nodes (flipflag=%i)', flipflag));
  ylabel('log10(error)');
  title('Convergence of integral[f(x)] (quadrature)');
  grid on;

  figure;
  plot(nvec + 1, log10([err123, derr123]), 'o--');
  legend('CGL f', 'LGL f', 'OGL f', 'RGL f', 'CGL df', 'LGL df', 'OGL df', 'RGL df');
  xlabel(sprintf('#nodes (flipflag=%i)', flipflag));
  ylabel('log10(error)');
  title('Convergence of f(x) and df(x)/dx');
  grid on;
end

end
