function [M] = MVEC_VFL_update_M(view,data,cluster,dis,beta,nbFoc)
% Update the belief matrix M
[row,~] = size(data{1});
% M=zeros([row,2^cluster]);
M=zeros([row,nbFoc]);

D = zeros(row, nbFoc);
for i = 1:view
   D = D + dis{i}; 
end

m = zeros(row,nbFoc-1);
for i=1:row
    vect0 = D(i,2:end);
    for j=2:nbFoc
        if sum(D(:, j)) ~= 0
            vect1 = ((D(i,j)*ones(1,nbFoc-1))./vect0).^(1/(beta-1));
            vect1(vect1 == inf) = 0;
    %         vect2 =  ((c(j)^(alpha/(beta-1)))*ones(1,nbFoc-1))./(c.^(alpha/(beta-1)));
            vect3 = vect1;
            m(i,j)=1/(sum(vect3)+(D(i,j)/D(i,1))^(1/(beta-1)));
        end
    end
end
m = [ones(row,1)-sum(m,2) m(:, 2:end)]; 
M = m;
M(M(:, 1)<0) = 0;
end

