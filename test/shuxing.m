function [O] = shuxing(data)
    %morphological attribute pro?les (APs), morphological pro?les (MPs)    
    PCs= data;  %��Ҫ
    Lambda = [25];
    PC1_int16 = ConvertFromZeroToOneThousand(PCs,false);
    PC1_int16 = int16(PC1_int16);           
    Lambda = double(sort(nonzeros(Lambda))');
    AP1 = attribute_profile(PC1_int16,'a', Lambda);%�����˲���C����
    AP1 = double(AP1);
    %% ���
    D(:,:,1) = AP1(:,:,2)-AP1(:,:,3);%+AP1(:,:,1)-AP1(:,:,2);%2��ԭͼ
    %% ƽ���ں�
    img=average_fusion(D,1);
    %% ��Ե�����˲���
    if length(size(data))>2  
    img=rgb2gray(img);  
    end  
    img = double(img) / 255; 
    p = img;  
    r = 3;   
    eps = 0.5;   
    O = guidedfilter(img, p, r, eps);  
end
