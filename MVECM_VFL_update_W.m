function [W] = MVEC_VFL_update_W(view,dis,features)
W = cell(1,view);
for i = 1:view
    W{i} = prod(dis{i}.^(1/features(i)))./dis{i};
end
end

