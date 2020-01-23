function [x, w, l, D] = ogl(n, flipflag)
%
% function [x, w, l, D] = ogl(n, flipflag)
%
% Compute n+1 ordinary Gauss-Legendre abscissae x, quadrature weights w,
% barycentric interpolation weights l, and the associated diff. matrix D.
% If flipflag is provided and is positive, the grid x will be in
% increasing order (w, l, D will be set consistently), otherwise
% the grid will be in decreasing order. The nodes x lie in standard
% (-1, +1) range (no endpoints).
%

if nargin < 2, flipflag = 0; end

x1 = -1;
x2 = +1;
n1 = n + 1;
x = ones(n1, 1) * (-1);
w = ones(n1, 1) * (-1);
EPS = eps; %0.5e-14;
m = floor((n1 + 1)/2);
xm = (x2 + x1)/2;
xl = (x2 - x1)/2;
for ii = 0:(m - 1)
  z = cos(pi*(ii + .75)/(n1 + .5));
  z1 = Inf;
  while abs(z - z1) > EPS
    p1 = 1.0; p2 = 0.0;
    for jj = 0:(n1 - 1)
      p3 = p2; p2 = p1;
      p1 = ((2*jj + 1)*z*p2 - jj*p3)/(jj + 1);
    end
    pp = n1*(z*p1 - p2)/(z*z - 1.0);
    z1 = z;
    z = z1 - p1/pp;
  end
  x(ii + 1) = xm - xl*z;
  x(1 + n1 - (ii + 1)) = xm + xl*z;
  w(ii + 1) = 2*xl/((1 - z*z)*pp*pp);
  w(1 + n1 - (ii + 1)) = w(ii + 1);
end

l = ((-1).^(0:n)').*sqrt((1 - x.*x).*w);

if nargout >= 4
  P = NaN(n1, n1); P(:, 1) = ones(n1, 1); P(:, 2) = x;
  Pp = NaN(n1, n1); Pp(:, 1) = zeros(n1, 1); Pp(:, 2) = ones(n1, 1);
  for kk = 1:(n - 1)
    P(:, kk + 1 + 1) = ((2*kk + 1)*x.*P(:, kk + 1) - kk*P(:, kk - 1 + 1))/(kk + 1);
    Pp(:, kk + 1 + 1) = (kk + 1)*P(:, kk + 1) + x.*Pp(:, kk + 1);
  end
  aP = repmat([(1 + 2*(0:n)')/2], [1, n1]).*repmat(w', [n1, 1]).*P';
  D = Pp*aP;
end

if flipflag <= 0 % need to unflip to decreasing order
  x = flipud(x);
  w = flipud(w);
  l = flipud(l);
  if nargout >= 4
    D = flipud(fliplr(D));
  end
end

end
