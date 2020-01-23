function [x, w, l, D] = lgl(n, flipflag)
% function [x, w, l, D] = lgl(n, flipflag)
% Compute (n+1) Legendre-Gauss-Lobatto nodes x, their quadrature weights w,
% their Barycentric interpolation weights l, and the associated square
% differentiation matrix D. If flipflag is provided and is positive, the
% grid x will be in increasing order (w,l,D will be set consistently),
% otherwise the grid will be in decreasing order.
%

%
% Reference on LGL nodes and weights and differentiation matrix: 
%   C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral Methods
%   in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
% (Greg von Winckel - 04/17/2004, MatlabCentral)
%
% Reference on Barycentric interpolation weights:
%   H. Wang, D. Huybrechs, S. Vandewalle, "Explicit barycentric weights
%   for polynomial interpolation in the roots or extrema of classical
%   orthogonal polynomials", Dec. 3 2012, arXiv:
%   http://arxiv.org/pdf/1202.0154v3.pdf
%   Journal: Math. Comp. 83 (2014), 2893-2914 
%

n1=n+1;
% Use the Chebyshev-Gauss-Lobatto nodes as the first guess
xc=cos(pi*(0:n)/n)';
% Uniform nodes
xu=linspace(-1,1,n1)';
% Make a close first guess to reduce iterations
if n<3
  x=xc;
else
  x=xc+sin(pi*xu)./(4*n);
end
% The Legendre Vandermonde Matrix
P=zeros(n1);
% Compute P_(n) using the recursion relation
% Compute its first and second derivatives and 
% update x using the Newton-Raphson method.
xold=2;
while max(abs(x-xold))>eps
  xold=x;
  P(:,1)=1; P(:,2)=x;
  for k=2:n
    P(:,k+1)=((2*k-1)*x.*P(:,k)-(k-1)*P(:,k-1))/k;
  end
  x=xold-(x.*P(:,n1)-P(:,n))./(n1*P(:,n1));
end
% Compute quadrature weights
w=2./(n*n1*P(:,n1).^2);
% Compute Barycentric interpolation weights
l=((-1).^(0:n)').*sqrt(w);
% Compute differentiation matrix, if requested
if nargout>=4
  X=repmat(x,1,n1);
  Xdiff=X-X'+eye(n1);
  L=repmat(P(:,n1),1,n1);
  L(1:(n1+1):n1*n1)=1;
  D=(L./(Xdiff.*L'));
  D(1:(n1+1):n1*n1)=0;
  D(1)=(n1*n)/4;
  D(n1*n1)=-(n1*n)/4;
end
% Flip order of LGL nodes if requested
if nargin>1 && flipflag>0
  x = flipud(x);
  w = flipud(w);
  l = flipud(l);
  if nargout>=4
    %F = flipud(eye(n1)); D = F*D*F;
    D = flipud(fliplr(D)); %#ok<FLUDLR>
  end
end

end
