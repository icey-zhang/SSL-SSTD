function [mix_final,O,Dde,S,x_e,x_r] = ourmethoddetection(x_input,x_a,d,number,delta)
        [row,col,bands] = size(x_input);
    %% 光谱融合
        rho = 0.1;
        data_mix = x_a;
        b=size(data_mix,3);
        mix = zeros(row,col);
        mc = zeros(row,col);
        for i=1:b
            EigInfo(i) = coherence_orientation(data_mix(:,:,i),rho);
            lamda(i) = sum(EigInfo(i).nu2(:));
            t(i) = sum(EigInfo(i).Trace(:)); %迹
        end
        t=t/max(t(:));
        for i=1:b
          mix(:,:) = mix(:,:)+data_mix(:,:,i).*t(i);
        end
        mix = hyperNormalize(mix);
%         mc=1-mix;
%         mc=mix;
        mc=1-mix;
     %% 属性引导滤波
        O = shuxing(mc);
    %% 选取波段 做CEM   
%         number = round(1/2*bands);
        for bb=1:1:bands
            mssim(bb) = SSIM(mix, x_input(:,:,bb));
        end
        [B_2,x] = sort(mssim,'descend');
        for j = 1:1:number
            x_1(1,j) = x(1,j);
        end
        [B_3] = sort(x_1,'ascend');
        for k = 1:1:number
            d_cho(1,k) = d(1,B_3(k));
            x_e(:,:,k) = x_input(:,:,B_3(k));
        end
%         for k = 1:1:number
%           d_cho(1,k) = d(1,B_3(k));
%           x_e(:,k) = X_input(:,B_3(k));
%         end
        %% show删除波段情况
        for j = number+1:1:bands
            x_2(1,j-number) = x(1,j);
        end
        [B_4] = sort(x_2,'ascend');
        for k = 1:1:bands-number
            x_r(:,:,k) = x_input(:,:,B_4(k));
        end
%% CEM
%         X_test = x_e';
%         d_test = d_cho';
%         Dde = CEMour(X_test,d_test,row,col);
        Dde = CEM189(x_e,d_cho);       
        Dde = hyperNormalize(Dde);
        O = hyperNormalize(O);
    %% 融合
        result_mix = zeros(row,col,2);
        result_mix(:,:,1) = reshape(O,row,col,1);
        result_mix(:,:,2) = reshape(Dde,row,col,1);
    %% 光谱融合 自适应
        mix_final = zeros(row,col);
        for i=1:2
            EigInfo(i) = coherence_orientation(result_mix(:,:,i),rho);
            t_1(i) = sum(EigInfo(i).Trace(:));
        end
        t_1=t_1/max(t_1(:));
        for i=1:2
          mix_final(:,:) = mix_final(:,:)+result_mix(:,:,i).*t_1(i);
        end
%        mix_final = average_fusion(result_mix,1);
       S = hyperNormalize(mix_final);
%% (1-e^(-bataT1))*T
    chi = (1-exp(-delta.*O));
    mix_final = chi.*S;
%     mix_final = chi.*S;
end