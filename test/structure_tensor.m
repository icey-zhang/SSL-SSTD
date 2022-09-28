function [J_rho,im_sigma]=structure_tensor(im,sigma,rho)
%   This function computes the structure tensor matrix associated to the 
%   image im.
%   
%   INPUT:
%   im: mxn matrix corresponding to the image
%   sigma: the standard deviation (in pixels) of the Gaussian kernel
%   applied to the image
%   rho: the standard deviation (in pixels) of the Gaussian kernel applied 
%   to the strcture tensor to average the directions of the eigenvectors.
%
%   OUTPUT:
%   J_rho: the structure tensor a 2*n by 2*m matrix. T_rho is a 4 blocks 2D 
%   matrix [A1,A2;A3,A4] each block of size n by m. For example, the first 
%   block should contain the first component of the structure tensor for 
%   each pixel of the image.
%   grad: the gradient matrix associated to the image. n x m x 2 matrix.
%   grad(:,:,1) is the fist component of the gradient vector for each pixel
%   of the image.

% %  Written by Wenxing Zhang, Omar Dounia and Pierre Weiss, 
% %  ITAV, Toulouse, France, July 2014.
% %  Troubleshooting: wenxing84@gmail.com, pierre.armand.weiss@gmail.com


    % Gaussian kernel convolution applied to the image 高斯核卷积应用于图像
        [nx,ny]=size(im);
        if sigma==0
            im_sigma=im; 
        else
            gk_sigma=gaussian_kernel(ny,nx,sigma);      
            im_sigma=ifft2(fft2(gk_sigma).*fft2(im));   
        end
    % In order to find parallel structures we need to replace the gradient by
    % its tensor product so that the structure descriptor is invariant under
    % sign changes:  为了找到平行结构，我们需要用它的张量积来代替梯度，这样结构描述符在符号变化下是不变的:      

       grad = zeros(nx,ny,2);        
       h = [-3 0 3; -10 0 10; -3 0 3]/32;    %%% gradient B:        
       grad(:,:,1) = imfilter(im_sigma,h','symmetric'); 
       grad(:,:,2) = imfilter(im_sigma,h,'symmetric'); 
        
        %%%%%%% contrast invariant
        Tensor = zeros(nx,ny,4);
        Tensor(:,:,1)= grad(:,:,1).^2; 
        Tensor(:,:,2)= grad(:,:,1).*grad(:,:,2);
        Tensor(:,:,3)= Tensor(:,:,2);
        Tensor(:,:,4)= grad(:,:,2).^2;        
        
        clear im gk_sigma  grad
        gk_rho = gaussian_kernel(ny,nx,rho);
        J_rho  = zeros(size(Tensor));
        Fgk_rho= fft2(gk_rho);
        J_rho(:,:,1) = ifft2(Fgk_rho.*fft2(Tensor(:,:,1)));
        J_rho(:,:,2) = ifft2(Fgk_rho.*fft2(Tensor(:,:,2)));
        J_rho(:,:,3) = J_rho(:,:,2);
        J_rho(:,:,4) = ifft2(Fgk_rho.*fft2(Tensor(:,:,4)));
      
                
end