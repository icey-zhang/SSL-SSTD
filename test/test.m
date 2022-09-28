clear all
%% detection
    file = 'G:\open\data_selected';
    files = dir(file);
%     clear result
figure('Units','centimeter','Position',[0 0 10 10]);
for tfirst = 4
    clear cem_out cscr_out hcem_out ace_out ecem_out our_out our_out1 our_out2
    str = files(tfirst).name;
    file_path = sprintf('%s\\%s',file,str);
    load(file_path);
%     x_input = data_1;
    d = d_1;
    [row,col,bands] = size(x_input);
    %%
%     smalld = makesmalld(x_input,map);
    index_o = find('.'==str);
    name = str(1:(index_o(1)-1)); 
    fprintf(name);
    %%
%     b = sprintf('%s\\%s_data.mat',file,name);
%     save(b,'X_input','d_1','new_d','x_input','map','smalld');
%% ACE  
%     tic
%     [ace_out_1,mu,siginv] = ace_detector_sqrt(x_input,d);
%     toc
% %     ace_out_2 = ace_out_1 + 1;
%     ace_out = hyperNormalize(ace_out_1);
%     [FPR_1_ace,TPR_1_ace,thre_1_ace,auc_1_ace] = myPlot3DROC(map, ace_out);    
%     FAR_1_ace =-trapz(FPR_1_ace,thre_1_ace);
% %     
%     savefile_ACE = 'D:\zjq\aae\comparing algorithm\result_1\ACE_selected';
%     save_pathACE = sprintf('%s\\mat\\%s',savefile_ACE,str);
%     save(save_pathACE,'ace_out');
%     
%     colormap('jet');imagesc(ace_out);
%     set(gca,'position',[0 0 1 1]);  % È¥°×±ß
%     axis('off')
%     print(gcf,'-djpeg','-r1000',sprintf('%s\\resultmap\\%s.png',savefile_ACE,name));

%  %% CSCR
%     tic
%     cscr_out_1 = cscr_detector(x_input,d);
%     toc
%     cscr_out = hyperNormalize(cscr_out_1);
%     [FPR_1_cscr,TPR_1_cscr,thre_1_cscr,auc_1_cscr] = myPlot3DROC(map, cscr_out);
%     FAR_1_cscr =-trapz(FPR_1_cscr,thre_1_cscr);
%     
%     savefile_CSCR = 'D:\zjq\aae\comparing algorithm\result_1\CSCR_selected';
%     save_pathCSCR = sprintf('%s\\mat\\%s',savefile_CSCR,str);
%     save(save_pathCSCR,'cscr_out');
%     
%     colormap('jet');imagesc(cscr_out);
%     set(gca,'position',[0 0 1 1]);  % È¥°×±ß
%     axis('off')
%     print(gcf,'-djpeg','-r1000',sprintf('%s\\resultmap\\%s.png',savefile_CSCR,name));
%% CEM
%     tic
%     cem_out = CEM189(x_input,d);
%     toc
%     cem_out = hyperNormalize(cem_out);
%     [FPR_1_cem,TPR_1_cem,thre_1_cem,auc_1_cem] = myPlot3DROC(map,cem_out);
%     FAR_1_cem =-trapz(FPR_1_cem,thre_1_cem);
% %     
%     savefile_CEM = 'D:\zjq\aae\comparing algorithm\result_1\CEM_selected';
%     save_pathCEM = sprintf('%s\\mat\\%s',savefile_CEM,str);
%     save(save_pathCEM,'cem_out');
%     
%     colormap('jet');imagesc(cem_out);
%     set(gca,'position',[0 0 1 1]);  % È¥°×±ß
%     axis('off')
%     print(gcf,'-djpeg','-r1000',sprintf('%s\\resultmap\\%s.png',savefile_CEM,name));
%% hCEM
%     tic
%     hcem_out = hCEM(x_input,d);
%     toc
%     hcem_out = hyperNormalize(hcem_out);
%     [FPR_1_hcem,TPR_1_hcem,thre_1_hcem,auc_1_hcem] = myPlot3DROC(map,hcem_out);
%     FAR_1_hcem =-trapz(FPR_1_hcem,thre_1_hcem);
%     
%     savefile_hCEM = 'D:\zjq\aae\comparing algorithm\result_1\hCEM_selected';
%     save_pathhCEM = sprintf('%s\\mat\\%s',savefile_hCEM,str);
%     save(save_pathhCEM,'hcem_out');
%     
%     colormap('jet');imagesc(hcem_out);
%     set(gca,'position',[0 0 1 1]);  % È¥°×±ß
%     axis('off')
%     print(gcf,'-djpeg','-r1000',sprintf('%s\\resultmap\\%s.png',savefile_hCEM,name));
%% E_CEM
%    savefile_ECEM = 'D:\zjq\aae\comparing algorithm\result_1\ECEM_selected';
%    file_ECEM = sprintf('%s\\mat\\%s',savefile_ECEM,str);
%    load(file_ECEM); 
%    ecem_out = hyperNormalize(ecem_out);
%    
%    [FPR_1_ecem,TPR_1_ecem,thre_1_ecem,auc_1_ecem] = myPlot3DROC(map,ecem_out);
%    FAR_1_ecem =-trapz(FPR_1_ecem,thre_1_ecem);
%    
%     colormap('jet');imagesc(ecem_out);
%     set(gca,'position',[0 0 1 1]);  % È¥°×±ß
%     axis('off')
%     print(gcf,'-djpeg','-r1000',sprintf('%s\\resultmap\\%s.png',savefile_ECEM,name)); 

%% our method_1000
   clear x_encoder x_a
   epsilon = 0.7;
   number = round(epsilon*bands);
   encoderfile_our = 'G:\open\endedata1000';
   file_encoder = sprintf('%s\\%s_encoder10000.mat',encoderfile_our,name);
   load(file_encoder); 
   x_a = hyperNormalize(reshape(x_encoder,row,col,size(x_encoder,2)));
   tic
   delta = 5;
   our_out = ourmethoddetection(x_input,x_a,d,number,delta);
   toc
   our_out = hyperNormalize(our_out);
   
   [FPR_1_our,TPR_1_our,thre_1_our,auc_1_our] = myPlot3DROC(map,our_out);
   FAR_1_our =-trapz(FPR_1_our,thre_1_our);
   savefile_our = 'G:\open\our_selected_1000';
    colormap('jet');imagesc(our_out);
    set(gca,'position',[0 0 1 1]);  % È¥°×±ß
    axis('off')
    print(gcf,'-djpeg','-r1000',sprintf('%s\\resultmap\\%s.png',savefile_our,name)); 
    
 
%% ´æÖµ
%     result(tfirst-2,1) = max(max(max(x_input)));
%     result(tfirst-2,2) = bands;
% %     result(tfirst-2,3) = auc_1_cscr;
% %     result(tfirst-2,4) = FAR_1_cscr;
%     
%     result(tfirst-2,5) = auc_1_ace;
%     result(tfirst-2,6) = FAR_1_ace;
%     
%     result(tfirst-2,7) = auc_1_cem;
%     result(tfirst-2,8) = FAR_1_cem;
%     
%     result(tfirst-2,9) = auc_1_hcem;
%     result(tfirst-2,10) = FAR_1_hcem;
%     
%     result(tfirst-2,11) = auc_1_ecem;
%     result(tfirst-2,12) = FAR_1_ecem;
%     
%     result(tfirst-2,13) = auc_1_our;
%     result(tfirst-2,14) = FAR_1_our;
%     
%     result(tfirst-2,15) = auc_1_our1;
%     result(tfirst-2,16) = FAR_1_our1;
%     
% %     result(tfirst-2,17) = auc_1_our2;
% %     result(tfirst-2,18) = FAR_1_our2;
%     
%     figure,plot(FPR_1_ace,TPR_1_ace);    
%     xlabel('FPR'); ylabel('TPR');   
%     hold on;plot(FPR_1_cscr,TPR_1_cscr);
%     hold on;plot(FPR_1_cem,TPR_1_cem);
%     hold on;plot(FPR_1_hcem,TPR_1_hcem);
%     hold on;plot(FPR_1_ecem,TPR_1_ecem);
%     hold on;plot(FPR_1_our,TPR_1_our);    
%     legend('ace','cscr','cem','hcem','ecem','our');
end