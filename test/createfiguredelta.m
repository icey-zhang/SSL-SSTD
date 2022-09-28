function createfiguredelta(X1, YMatrix1)
%CREATEFIGURE1(X1, YMATRIX1)
%  X1:  x ���ݵ�ʸ��
%  YMATRIX1:  y ���ݵľ���

%  �� MATLAB �� 22-Apr-2020 17:38:51 �Զ�����

% ���� figure
figure1 = figure('InvertHardcopy','off','Color',[1 1 1],...
    'Renderer','painters');

% ���� axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% ʹ�� plot �ľ������봴������
plot1 = plot(X1,YMatrix1,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','Pavia','MarkerSize',10,'Marker','x');
set(plot1(2),'DisplayName','San Diego','Marker','pentagram');
set(plot1(3),'DisplayName','Texas Coast-1','Marker','v');
set(plot1(4),'DisplayName','Texas Coast-2','MarkerSize',15,'Marker','.');
set(plot1(5),'DisplayName','Los Angeles','Marker','diamond');
set(plot1(6),'DisplayName','Cuprite','Marker','*');

% ���� xlabel
xlabel('\delta','FontWeight','bold','FontSize',11,...
    'FontName','Times New Roman');

% ���� ylabel
ylabel('AUC','FontWeight','bold','FontSize',11,'FontName','Times New Roman');

% ȡ�������е�ע���Ա���������� X ��Χ
% xlim(axes1,[0 20]);
% ȡ�������е�ע���Ա���������� Y ��Χ
% ylim(axes1,[0.98 1]);
% ȡ�������е�ע���Ա���������� Z ��Χ
% zlim(axes1,[-1 1]);
box(axes1,'on');
% ������������������
set(axes1,'FontName','Times New Roman','FontWeight','bold');
