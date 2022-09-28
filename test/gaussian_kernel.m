function gk=gaussian_kernel(nx,ny,sigma)
%function gk=gaussian_kernel(nx,ny,sigma)
%   this function computes the gaussian kernel with standard 
%   deviation sigma expressed in pixels. The output is an nx by ny matrix.  
%
% %  Written by Dr. Pierre Weiss, ITAV, Toulouse, France

    if (sigma==0)
        sigma=1e-3;
    end
    [X,Y]=meshgrid(linspace(-nx/2+1,nx/2,nx),linspace(-ny/2+1,ny/2,ny));
    gk=exp(-(X.^2+Y.^2)/(2*sigma^2));
    gk=gk/sum(gk(:)); % normalization
    gk=fftshift(gk);
end
