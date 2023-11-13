%% Multi-View Evidential c-Means Clustering With View-Weight and Feature-Weight Learning
% @Date: 2023.11.13
clear all; clc;
%% Load Data Set

Data = load('foresttype.txt');  Y = Data(:,end);
Data = mapminmax(Data',0,1); Data = Data';Data(:,end) = Y;
view = 2; features = [9 18];
left = 1;right=0; X=cell(1,view);
for i=1:view
    X{i} = Data(:, left:features(i)+right);
    left = features(i)+right+1;
    right = features(i)+right;
end

%% Initialization
delta = repmat(10,1,view);
n_max = 50; epsilon = 1e-3; cluster = max(Y); 
center = cell(1,view); Aj = cell(1,view); F_update=cell(1,view); W = cell(1,view);
%% Test
metrics = [];
for alpha = 1:0.2:3  % alpha -> {1,1.2,1.4, ...,2.6,2.8,3}
    for beta = 1.1:0.1:2  % beta -> {1.1,1.2,...,1.9,2}
        for i = 1:view
            center{i} = Centroids_Initialization(X{i},cluster);
        end
        %-------------------- Initialization ----------------------%
        % construction of the focal set matrix %
        ii=1:2^cluster;
        F=zeros(length(ii),cluster);
        for i=1:cluster
              F(:,i)=bitget(ii'-1,i);
        end

        if cluster>3
            truc = sum(F,2);
            if cluster >= 7
                ind = find((cluster>truc) & (truc>1));
            else
                ind = find(truc>2);
                ind(end)=[];
            end
            F(ind,:)=[];
        end

        if cluster~=2
            if cluster >= 7
                nbFoc = cluster + 2 ; % with empty set and total ignorance
            else
                nbFoc = cluster + cluster*(cluster-1)/2 + 2 ; % with empty set
            end
        else
            nbFoc=4;
        end
        
        % r Initialization %
        R = repmat(1/view,[1,view]);
        
        % w Initialization %
        for i=1:view
           W{i} = repmat(1/features(i),[1,features(i)]);
        end
        %-------------------- MVECM-VFL ----------------------%
        n = 0; flag = 1; J_value = [];
        while n < n_max && flag
            n = n + 1;
            % Aj Initialization %
            [Aj,~] = update_Aj(view,cluster,features,Aj,F,center,nbFoc);
            % Update M %
            F_update = [];
            for i = 1:view
                F_update = [F_update {F}];
            end
            dis = MVECM_VFL_get_distance(1,view,X,nbFoc,Aj,F_update,alpha,beta,delta,features,R,W);
            M = MVECM_VFL_update_M(view,X,cluster,dis,beta,nbFoc);
            % Update r %
            dis = MVECM_VFL_get_distance(2,view,X,nbFoc,Aj,F_update,alpha,beta,delta,features,M,W);
            R = MVECM_VFL_update_R(view,dis);  
            % Update W %
            dis = MVECM_VFL_get_distance(3,view,X,nbFoc,Aj,F_update,alpha,beta,delta,features,R,M);
            W = MVECM_VFL_update_W(view,dis,features);
            % Update V %
            center = MVECM_VFL_update_V(view,cluster,alpha,beta,X,M,F_update,features);
            % Update Jaccard %    
            dis = MVECM_VFL_get_distance(2,view,X,nbFoc,Aj,F_update,alpha,beta,delta,features,M,W);
            jaccard = MVECM_VFL_update_jaccard(view,R,dis);
            J_value(end+1) = jaccard;
            % update epsilon %
            if n > 1
                eps_m = abs(jaccard - J_value(end-1));
                if eps_m < epsilon
                    flag = 0;
                end
            end
        end
        %-------------------- BetP ----------------------%
        K = cluster; mass = M; real_label = Y;
        result_label = [];
        for i =1:size(Y,1)
             [~,result_label(i,1)]=max(mass(i,:));
        end
        result_label = result_label+K;
        mm = 1;
        save_loca = [];
        for i = 1:size(F,1)
            loca = find(F(i,:) == 1);
            if length(loca) == 1
                save_loca(mm,1) = i+K;
                mm = mm+1;
            end
        end
        row = {};
        row_outier = [];
        nn = 1;
        for i = 1:K
             row{i,1} = find(result_label == save_loca(i,1));
        end
        row_outier = find(result_label == 1+K); 

        for i = 1:K
            result_label(row{i,1},:) = i;
        end
        result_label(row_outier) = 1000;
        [result_label,obj] = label_map( result_label, real_label );

        BetP = zeros([length(M),cluster]);
        for i = 1:cluster
            pos=[];
            for z = 1:length(F)
                if F(z,i) == 1
                   pos(end+1) = z; 
                end
            end
            for j = 1:length(M)
                betp = 0;
                for x = 1:length(pos)
                    card = sum(F(pos(x),:));
                    betp = betp + M(j,pos(x))/(card * (1 - M(j,1))); 
                    BetP(j,i) = betp;
                end
            end
        end
        %------------------- Evaluate -------------------------%
        mass = M; K = cluster;real_label = Y;
        % Calculate NMI and Evaluate 
        row = size(real_label,1);
        res = zeros(row,1);
        for i = 1:size(real_label,1)
            [~,idx] = max(BetP(i,:));
            res(i,1) = idx;
        end
        predY = bestMap(Y, res);
        result = CalcMeasures(Y, res);
        nmi = NMI(Y,predY);
        [P,R1,F1,RI,FM,J] = Evaluate(Y,predY);
        fprintf("ACC:%f,NMI:%f,P:%f,R:%f,F:%f,RI:%f,FM:%f,J:%f\n",result(1),nmi,P,R1,F1,RI,FM,J);
        metrics = [metrics;[alpha,beta,result(1),impr,outlier,error01,nmi,P,R1,F1,RI,FM,J]];
    end
end
