function Zcem = CEM189(data,target)
    [row,col,bands] = size(data);
    z = bands;
    xy = row*col;
    HIM = reshape(data,[xy,z]);
    %R=zeros(z);             %����z*z��ȫ0����
    r=HIM';         %��rת��
    d=target';
    R=r*r';
    R=R/xy;
 %  w=(pinv(R)*d)/(transpose(d)*pinv(R)*d);
    [qq,qrr]=qr(R);
    R_inv=pinv(qrr)*qq';
    w=(R_inv*d)/(transpose(d)*R_inv*d);
    for i=1:xy
        Z(i)=transpose(w)*r(:,i);
    end
    Zcem=reshape(Z,row,col);
    Zcem=abs(Zcem);
end

 