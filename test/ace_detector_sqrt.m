function [ace_out,mu,siginv] = ace_detector_sqrt(hsi_img,target,mask,mu,siginv)
if ~exist('mask','var'), mask = []; end
if ~exist('mu','var'), mu = []; end
if ~exist('siginv','var'), siginv = []; end
% [row,col,bands] = size(hsi_img);
% hsi_img = reshape(hsi_img,bands,row*col);
tgt_sig = target';
[ace_out,mu,siginv] = img_det(@ace_det,hsi_img,tgt_sig,mask,mu,siginv);
end

function [ace_data,mu,siginv] = ace_det(hsi_data,tgt_sig,mu,siginv)
% hsi_data = hsi_img;
% n_pix = size(hsi_data,2);
if isempty(mu)
    mu = mean(hsi_data,2);
end
if isempty(siginv)
    siginv = pinv(cov(hsi_data'));
end

s = tgt_sig - mu;
z = bsxfun(@minus,hsi_data,mu);

st_siginv = s'*siginv;
st_siginv_s = s'*siginv*s;


A = sum(st_siginv*z,1);
B = sqrt(st_siginv_s);
C = sqrt(sum(z.*(siginv*z),1));

ace_data = A./(B.*C);
% ace_out = ace_data;
end