function [x, w, l, D] = cgl(n, flipflag)
% function [x, w, l, D] = cgl(n, flipflag)
% Compute (n+1) Chebyshev-Gauss-Lobatto nodes x, their Clenshaw-Curtis 
% quadrature weights w, their Barycentric interpolation weights l,
% and the associated square differentiation matrix D.
% If flipflag is provided and is positive, the grid x will be
% in increasing order (w,l,D will be set consistently),
% otherwise the grid will be in decreasing order.
%

%
% Based on L. Trefethen's codes clencurt.m and cheb.m (SIAM, 2000).
%

nvec = (0:n)'; theta = pi*nvec/n; x = cos(theta);
w = zeros(1,n+1); ii = 2:n; v = ones(n-1,1);
if mod(n,2)==0
  w(1) = 1/(n^2-1); w(n+1) = w(1);
  for k=1:n/2-1, v = v - 2*cos(2*k*theta(ii))/(4*k^2-1); end
  v = v - cos(n*theta(ii))/(n^2-1);
else
  w(1) = 1/n^2; w(n+1) = w(1);
  for k=1:(n-1)/2, v = v - 2*cos(2*k*theta(ii))/(4*k^2-1); end
end
w(ii) = 2*v/n;
w = w';
l = [0.5;ones(n-1,1);0.5].*(-1).^nvec; % Barycentric weights
if nargout>=4
  % Differentiation matrix
  c = [2;ones(n-1,1);2].*(-1).^nvec;
  X = repmat(x,1,n+1); dX = X-X';
  D = (c*(1./c)')./(dX+eye(n+1));
  D = D - diag(sum(D')); %#ok<UDIM>
end
% Flip order of LGL nodes if requested
if nargin>1 && flipflag>0
  x = flipud(x); w = flipud(w); l = flipud(l);
  if nargout>=4
    D = flipud(fliplr(D)); %#ok<FLUDLR>
  end
end

end
