function createfigure(X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6)
%CREATEFIGURE(X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6)
%  X1:  x 数据的矢量
%  Y1:  y 数据的矢量
%  X2:  x 数据的矢量
%  Y2:  y 数据的矢量
%  X3:  x 数据的矢量
%  Y3:  y 数据的矢量
%  X4:  x 数据的矢量
%  Y4:  y 数据的矢量
%  X5:  x 数据的矢量
%  Y5:  y 数据的矢量
%  X6:  x 数据的矢量
%  Y6:  y 数据的矢量
%  由 MATLAB 于 21-Apr-2020 23:21:32 自动生成

% 创建 figure
figure('InvertHardcopy','off','Color',[1 1 1],'Renderer','painters');

% 创建 axes
axes1 = axes('Position',...
    [0.13882863340564 0.143608811732495 0.766171366594362 0.768236918295054]);
hold(axes1,'on');

% 创建 semilogx
semilogx(X1,Y1,'DisplayName','ACE','LineWidth',2,'Color',[0 1 1]);

% 创建 semilogx
semilogx(X2,Y2,'DisplayName','CSCR','Marker','x','Color',[1 0 0]);

% 创建 semilogx
semilogx(X3,Y3,'DisplayName','CEM','LineWidth',3,'LineStyle','-.',...
    'Color',[0 0 1]);

% 创建 semilogx
semilogx(X4,Y4,'DisplayName','hCEM','Marker','+','LineWidth',1,...
    'Color',[1 0 1]);

% 创建 semilogx
semilogx(X5,Y5,'DisplayName','ECEM','MarkerSize',10,'Marker','.',...
    'LineWidth',2,...
    'Color',[0.400000005960464 1 0.200000002980232]);

% 创建 semilogx
semilogx(X6,Y6,'DisplayName','SSTD','MarkerSize',2,'Marker','>',...
    'LineWidth',2,...
    'Color',[1 1 0]);

% 创建 xlabel
xlabel('FPR','FontWeight','bold','FontName','Times New Roman');

% 创建 ylabel
ylabel('TPR','FontWeight','bold','FontName','Times New Roman');

% 取消以下行的注释以保留坐标轴的 X 范围
% xlim(axes1,[0.0001 1]);
% 取消以下行的注释以保留坐标轴的 Y 范围
% ylim(axes1,[0 1]);
% 取消以下行的注释以保留坐标轴的 Z 范围
% zlim(axes1,[-1 1]);
box(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontName','Times New Roman','FontSize',12,'FontWeight','bold',...
    'XGrid','on','XMinorTick','on','XScale','log','YGrid','on');
