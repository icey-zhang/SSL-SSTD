function createfigure(X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6)
%CREATEFIGURE(X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6)
%  X1:  x ���ݵ�ʸ��
%  Y1:  y ���ݵ�ʸ��
%  X2:  x ���ݵ�ʸ��
%  Y2:  y ���ݵ�ʸ��
%  X3:  x ���ݵ�ʸ��
%  Y3:  y ���ݵ�ʸ��
%  X4:  x ���ݵ�ʸ��
%  Y4:  y ���ݵ�ʸ��
%  X5:  x ���ݵ�ʸ��
%  Y5:  y ���ݵ�ʸ��
%  X6:  x ���ݵ�ʸ��
%  Y6:  y ���ݵ�ʸ��
%  �� MATLAB �� 21-Apr-2020 23:21:32 �Զ�����

% ���� figure
figure('InvertHardcopy','off','Color',[1 1 1],'Renderer','painters');

% ���� axes
axes1 = axes('Position',...
    [0.13882863340564 0.143608811732495 0.766171366594362 0.768236918295054]);
hold(axes1,'on');

% ���� semilogx
semilogx(X1,Y1,'DisplayName','ACE','LineWidth',2,'Color',[0 1 1]);

% ���� semilogx
semilogx(X2,Y2,'DisplayName','CSCR','Marker','x','Color',[1 0 0]);

% ���� semilogx
semilogx(X3,Y3,'DisplayName','CEM','LineWidth',3,'LineStyle','-.',...
    'Color',[0 0 1]);

% ���� semilogx
semilogx(X4,Y4,'DisplayName','hCEM','Marker','+','LineWidth',1,...
    'Color',[1 0 1]);

% ���� semilogx
semilogx(X5,Y5,'DisplayName','ECEM','MarkerSize',10,'Marker','.',...
    'LineWidth',2,...
    'Color',[0.400000005960464 1 0.200000002980232]);

% ���� semilogx
semilogx(X6,Y6,'DisplayName','SSTD','MarkerSize',2,'Marker','>',...
    'LineWidth',2,...
    'Color',[1 1 0]);

% ���� xlabel
xlabel('FPR','FontWeight','bold','FontName','Times New Roman');

% ���� ylabel
ylabel('TPR','FontWeight','bold','FontName','Times New Roman');

% ȡ�������е�ע���Ա���������� X ��Χ
% xlim(axes1,[0.0001 1]);
% ȡ�������е�ע���Ա���������� Y ��Χ
% ylim(axes1,[0 1]);
% ȡ�������е�ע���Ա���������� Z ��Χ
% zlim(axes1,[-1 1]);
box(axes1,'on');
% ������������������
set(axes1,'FontName','Times New Roman','FontSize',12,'FontWeight','bold',...
    'XGrid','on','XMinorTick','on','XScale','log','YGrid','on');
