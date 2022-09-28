function output = CR_SUNSAL_Detection_Local(Data, TargetTrain, win_out, win_in, lambda, beta)
%
% Using SR and CR weights to produce class label
%


[a b c] = size(Data);        
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;

% padding avoid edges
DataTest = zeros(3*a, 3*b, c);
DataTest(a+1:2*a, b+1:2*b, :) = Data;
DataTest(a+1:2*a, 1:b, :) = Data(:, b:-1:1, :);
DataTest(a+1:2*a, 2*b+1:3*b, :) = Data(:, b:-1:1, :);
DataTest(1:a, :, :) = DataTest(2*a:-1:(a+1), :, :);
DataTest(2*a+1:3*a, :, :) = DataTest(2*a:-1:(a+1), :, :);

index = 1;
for i = 1+b: 2*b 
    for j = 1+a: 2*a
        block = DataTest(j-t: j+t, i-t: i+t, :);
        y = squeeze(DataTest(j, i, :)).';  %1 x 205
        block(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;
        block = reshape(block, M, c);
        block(isnan(block(:, 1)), :) = [];
        H = block';  % num_dim x num_sam  205x96
        
        % L1-minimization
        weight = sunsal(TargetTrain,y,'lambda',0,'ADDONE','yes','POSITIVITY','no', ...
            'AL_iters',300,'TOL', 1e-6,'verbose','no');
        y_hat_1 = (weight'*TargetTrain')';  % 1 x num_dim
        
        % L2-minimization
        norms = sum((H - repmat(y', [1 size(H,2)])).^2);   %H-y的2范数
        G = diag(beta*norms);
        %warning off;
        weights = ((H'*H + G)+0.0001*eye(size(G)))\(H'*y');  %左除等于inv(A)*B
        y_hat_2 = (H*weights(:))';  % 1 x num_dim
        
        y = y./norm(y);
        y_hat_1 = y_hat_1./norm(y_hat_1);
        y_hat_2 = y_hat_2./norm(y_hat_2);
        
        D_Y(1) = norm(y - y_hat_1);
        D_Y(2) = norm(y - y_hat_2);
        
        output(index, :) = [D_Y(2), D_Y(1)];
        index = index + 1;
    end
end