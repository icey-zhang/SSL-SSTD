function outputs = hCEM(data,target)
%% hCEM_demo
    [row,col,bands] = size(data);
    D = bands;
    N = row*col;
    HIM = reshape(data,[N,D]);
    X=HIM';
    d=target';
%     N = size(data,1)*size(data,2); % pixel number
%     D = size(data,3); % band number
%% add 30 dB Gaussian white noise
%     SNR = 30; 
%     for i = 1:size(X,2)
%            X(:,i) = awgn(X(:,i), SNR);
%     end
    % show groundtruth
%     figure; subplot(121); imshow(groundtruth); 
%     title('groundtruth'); hold on;


%% parameter settings
    % To obtain the optimal performances, the parameters lambda and epsilon 
    % should be optimized for each hyperspectral image.
    lambda = 200;
    epsilon = 1e-6;

%% hCEM algorithm
    % initialization 
    Weight = ones(1,N);
    y_old = ones(1,N);
    max_it = 100;
    Energy = [];
    
    for T = 1:max_it

         for pxlID = 1:N
             X(:,pxlID) = X(:,pxlID).*Weight(pxlID);
         end
         R = X*X'/N;

         % To inrease stability, a small diagnose matrix is added 
         % before the matrix inverse process.
         w = inv(R+0.0001*eye(D)) * d / (d'*inv(R+0.0001*eye(D)) *d); %eye返回n*n单位矩阵

         y = w' * X;
         Weight = 1 - 2.71828.^(-lambda*y);
         Weight(Weight<0) = 0;

         res = norm(y_old)^2/N - norm(y)^2/N; %norm范数
%          fprintf('ITERATION: %5d, RES: %.5u \n', T, res);
         Energy = [Energy, norm(y)^2/N];
         y_old = y;

         % stop criterion:
         if (abs(res)<epsilon)
             break;
         end
         
%          hCEMMap = reshape(mat2gray(y),[row,col]);
%          subplot(122); imshow(hCEMMap); 
    end
  outputs = reshape(mat2gray(y),[row,col]);
end