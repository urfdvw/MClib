function [z_RSed,ind]=resample(z,w)
% resample a vector z by weight w, keeping the sample size
% Input:
%     z: N*D or D*N matrix: data
%     w: N*1 or 1*N matrix: unnomalized weights
% Output:
%     z_RSed: same size as z: resampled data
%     ind: resampled index: 1*N index vector
N=length(w(:));
ind=datasample(1:N,N,'weights',w);
if size(z,1)==N
  z_RSed=z(ind,:);
else
  z_RSed=z(:,ind);
end