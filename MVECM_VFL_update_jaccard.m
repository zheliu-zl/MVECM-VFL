function [jaccard] =  MVEC_VFL_update_jaccard(view,R,dis)
part_1 = 0;
for i= 1:view
    part_1 = part_1 + R(i)*sum(sum(dis{i}));
end

jaccard = part_1;
end