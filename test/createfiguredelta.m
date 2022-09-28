function createfiguredelta(X1, YMatrix1)
%CREATEFIGURE1(X1, YMATRIX1)
%  X1:  x 数据的矢量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 22-Apr-2020 17:38:51 自动生成

% 创建 figure
figure1 = figure('InvertHardcopy','off','Color',[1 1 1],...
    'Renderer','painters');

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
plot1 = plot(X1,YMatrix1,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','Pavia','MarkerSize',10,'Marker','x');
set(plot1(2),'DisplayName','San Diego','Marker','pentagram');
set(plot1(3),'DisplayName','Texas Coast-1','Marker','v');
set(plot1(4),'DisplayName','Texas Coast-2','MarkerSize',15,'Marker','.');
set(plot1(5),'DisplayName','Los Angeles','Marker','diamond');
set(plot1(6),'DisplayName','Cuprite','Marker','*');

% 创建 xlabel
xlabel('\delta','FontWeight','bold','FontSize',11,...
    'FontName','Times New Roman');

% 创建 ylabel
ylabel('AUC','FontWeight','bold','FontSize',11,'FontName','Times New Roman');

% 取消以下行的注释以保留坐标轴的 X 范围
% xlim(axes1,[0 20]);
% 取消以下行的注释以保留坐标轴的 Y 范围
% ylim(axes1,[0.98 1]);
% 取消以下行的注释以保留坐标轴的 Z 范围
% zlim(axes1,[-1 1]);
box(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontName','Times New Roman','FontWeight','bold');
