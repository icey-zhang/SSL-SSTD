function target=generateD(HIM,groundtruth)
    [h,w,p]=size(HIM);
    ground=reshape(groundtruth,[1,h*w]);
    HIM2D=hyperConvert2d(HIM);
    target=ground*HIM2D'./sum(ground(:));
end