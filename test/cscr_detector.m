function [r1] = cscr_detector(data,d)
    [a b c] = size(data);
    Data_Ori=data;
%     mask=map;
%     figure; imagesc(groundtruth); 
%     TargetTrain=generateD(Data_Ori,mask);
    TargetTrain = d;
    lambda = 1e-1; beta = 1e-2;
    output = CR_SUNSAL_Detection_Local(Data_Ori, TargetTrain, 11, 5, lambda, beta);
    output3 = reshape(output, a, b, 2);
    tmp1 = output3(:,:,1); tmp2 = output3(:,:,2);
    output3(:,:,1) = (tmp1./max(tmp1(:)));
    output3(:,:,2) = (tmp2./max(tmp2(:)));
%     r1 = output3(:,:,1)-output3(:,:,2);
    r1 = output3(:,:,1);
%     r2 = reshape(r1, 1, a*b);
end