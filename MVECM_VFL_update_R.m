function [R] = MVEC_VFL_update_R(view,dis)
R = repmat(1/view,[1,view]);
temp = [];
for i = 1:view
    F_i = sum(sum(dis{i},1));
    temp = [temp F_i];
end
for i = 1:view
    R(i) = prod(temp)^(1/view)/temp(i);
end
end

