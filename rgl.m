function [x, w, l, D] = rgl(n, flipflag)
%
% function [x, w, l, D] = rgl(n, flipflag)
%
% Compute n+1 Radau-Gauss-Legendre abscissae x, quadrature weights w,
% barycentric interpolation weights l, and the associated diff. matrix D.
% If flipflag is provided and is positive, the grid x will be in
% increasing order (w, l, D will be set consistently), otherwise
% the grid will be in decreasing order. The nodes x lie in standard
% [-1, +1) range (endpoint at -1 included).
%

% See lgl.m for listing of references.

if nargin < 2, flipflag = 0; end

n1 = n + 1;
x = -cos(2*pi*(0:n)'/(2*n + 1));  % initial guess
P = zeros(n1, n1 + 1);
% Newton's method
xold = ones(n1, 1) * 2;
absc = 2:n1;
while max(abs(x - xold)) > eps
  xold = x;
  P(1, :) = (-1).^(0:n1);
  P(absc, 1) = 1;
  P(absc, 2) = x(absc);
  for k = 2:n1
    P(absc, k + 1) = ((2*k - 1)*x(absc).*P(absc, k) - (k - 1)*P(absc, k - 1))/k;
  end
  x(absc) = xold(absc) - ...
    ((1 - xold(absc))/n1) .* (P(absc, n1) + P(absc, n1 + 1)) ./ ...
    (P(absc, n1) - P(absc, n1 + 1));
end
P0 = P(1:n1, 1:n1); % Vandermonde matrix (discarded, it is recalculated below)

% Quadrature weights
w = zeros(n1, 1);
w(1) = 2/n1^2;
w(absc) = (1 - x(absc))./(n1*P(absc, n1)).^2;

% Interpolation weight (for baryxgl)
l = ((-1).^(0:n)').*sqrt((1 - x).*w);

if nargout >= 4
  P = NaN(n1, n1); P(:, 1) = ones(n1, 1); P(:, 2) = x;
  Pp = NaN(n1, n1); Pp(:, 1) = zeros(n1, 1); Pp(:, 2) = ones(n1, 1);
  for kk = 1:(n - 1)
    P(:, kk + 1 + 1) = ((2*kk + 1)*x.*P(:, kk + 1) - kk*P(:, kk - 1 + 1))/(kk + 1);
    Pp(:, kk + 1 + 1) = (kk + 1)*P(:, kk + 1) + x.*Pp(:, kk + 1);
  end
  aP = repmat([(1 + 2*(0:n)')/2], [1, n1]).*repmat(w', [n1, 1]).*P';
  D = Pp*aP;
  % disp(norm(P - P0, 'fro')); 
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
