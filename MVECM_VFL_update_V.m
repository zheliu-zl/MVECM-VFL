function [V] = MVECM_VFL_update_V(view,cluster,alpha,beta,data,M,F_update,features)
V = cell(1,view);
for i = 1:view
    B = zeros([cluster,1]); % shape:Cx1
    B1 = zeros([cluster,features(i)]); % shape:Cxp
    % Calculate B
    for j = 1:cluster
        pos = []; % Record the indices
        for k = 1:length(F_update{i})
            if F_update{i}(k,j) == 1
                pos(end+1) = k;
            end
        end
        B_X1 = zeros(1,features(i));
        for n=1:length(pos)
            card = sum(F_update{i}(pos(n),:));
            aj_dis1 = card^(alpha-1)*data{i}.* (M(:,pos(n)).^beta);
            B_X1 = B_X1 + sum(aj_dis1);
        end
        B1(j,:) = B_X1;
    end
    %  Calculate H
    H = zeros([cluster,cluster]);
    for c=1:cluster
        for k=1:cluster
            loc = [];
            for n = 1:length(F_update{i})
                if F_update{i}(n,c) == 1 && F_update{i}(n,k)== 1
                    loc(end+1) = n;
                end
            end
            H_ck = 0;
            for n=1:length(loc)
                card = sum(F_update{i}(loc(n),:));
                aj_all_dis = card^(alpha-2) * M(:,loc(n)).^beta;
                H_ck = H_ck + sum(aj_all_dis);
            end
            H(c,k) = H_ck;
        end
    end
    v1 = pinv(H) * B1;
    V{i} = v1;
end
end