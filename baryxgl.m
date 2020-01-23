function F = baryxgl(x, l, xx)
% function F = baryxgl(x, l, xx)
% Calculate a matrix F that gives the Barycentric interpolation on the grid
% xx, from the vector of function values at the OGL, LGL or CGL grid
% x, with the interpolation weights l. F*D, F*D*D, etc would be the
% matrices that gives the 1st and 2nd derivatives, and so on.
% Example usage:
%   [x,~,l]=cgl(n) (or with ogl or lgl)
%   xx=linspace(-1,+1,nn)'; F=baryxgl(x,l,xx);
%   ff=F*f(x)

n1 = size(x,1);
assert(size(x,2)==1);
assert(size(l,1)==n1);
assert(size(l,2)==1);

% It is assumed that x is now a column with the (x)GL nodes, and that
% l is a column of the associated Lagrange Barycentric interpolation
% weights.

nxx = size(xx,1);
assert(size(xx,2)==1);
assert(min(xx)>=-1);
assert(max(xx)<=+1);
assert(length(unique(xx))==nxx);

F = zeros(nxx,n1);
E = zeros(n1,1);
for jj=1:n1
  xdiff = xx-x(jj);
  F(:,jj) = l(jj)./xdiff;
  tmp = find(xdiff==0,1); % at most 1 index will exist here
  if numel(tmp)==1
    E(jj) = tmp;
  end
end
d = sum(F,2);   % collect row sums of F
for jj=1:nxx
  F(jj,:) = F(jj,:)/d(jj);    % normalize row coefficients
end
% Check for the special cases of exact collocation; avoid NaNs
for jj=1:n1
  if E(jj)>0
    Fj = zeros(1,n1);
    Fj(jj) = 1;
    F(E(jj),:) = Fj;
  end
end

end
