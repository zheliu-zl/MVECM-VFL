%% Multi-View Evidential c-Means Clustering With View-Weight and Feature-Weight Learning
% @Date: 2023.11.13
clear all; clc;
%% Load Data Set
load 3sources.mat
Y = truelabel{1}';
view = length(data); features = [3560,3631,3068];label = Y;
for i = 1:view
    X1 = mapminmax(data{i},0,1);
    X{i} = X1';
end

% load MNIST-10k.mat % 13579
% idx1 = find(Y==1); idx3 = find(Y==3); idx5 = find(Y==5); idx7 = find(Y==7); idx9 = find(Y==9);
% Y = Y([idx1;idx3;idx5;idx7;idx9]);
% Y(find(Y==1)) = 1;Y(find(Y==3)) = 2;Y(find(Y==5)) = 3;Y(find(Y==7)) = 4;Y(find(Y==9)) = 5;
% view = 3;features = [30 9 30];
% for i =1:view
%     X{i} = X{i}(:,[idx1;idx3;idx5;idx7;idx9]);
%     X1 = mapminmax(X{i},0,1);
%     X{i} = X1';
% end

% load MNIST-10k.mat % 246810
% idx2 = find(Y==2); idx4 = find(Y==4); idx6 = find(Y==6); idx8 = find(Y==8); idx10 = find(Y==10);
% Y = Y([idx2;idx4;idx6;idx8;idx10]);
% Y(find(Y==2)) = 1;Y(find(Y==4)) = 2;Y(find(Y==6)) = 3;Y(find(Y==8)) = 4;Y(find(Y==10)) = 5;
% view = 3;features = [30 9 30];
% for i =1:view
%     X{i} = X{i}(:,[idx2;idx4;idx6;idx8;idx10]);
%     X1 = mapminmax(X{i},0,1);
%     X{i} = X1';
% end

% Data = load('foresttype.txt');  Y = Data(:,end);
% Data = mapminmax(Data',0,1); Data = Data';Data(:,end) = Y;
% view = 2; features = [9 18];
% left = 1;right=0; X=cell(1,view);
% for i=1:view
%     X{i} = Data(:, left:features(i)+right);
%     left = features(i)+right+1;
%     right = features(i)+right;
% end

% Data = load('IS.txt');  Y = Data(:,end); Data(:,end) = [];
% Data = mapminmax(Data',0,1); Data = Data';
% view = 2; features = [9 10];
% left = 1;right=0; X=cell(1,view);
% for i=1:view
%     X{i} = Data(:, left:features(i)+right);
%     left = features(i)+right+1;
%     right = features(i)+right;
% end

% load prokaryotic.mat
% data = [{gene_repert'} {proteome_comp'} {text'}];
% Y = truth; 
% view = 3; features = [393 3 438];
% for i =1:view
%     X1 = mapminmax(data{i},0,1);
%     X{i} = X1';
% end

% load Caltech101-7.mat
% features = [48,40,254,1984,512,928];view = size(X,2);
% for i = 1:view
%     X1 = mapminmax(X{i}',0,1);
%     X{i} = X1';
% end

% load COIL20.mat
% features = [30,19,30];view = size(X,2);
% for i = 1:view
%     X1 = mapminmax(X{i},0,1);
%     X{i} = X1';
% end

% load Mfeat.mat    % 3view 13579
% Y = truelabel{1};
% Y = Y([1:200 401:600 801:1000 1201:1400 1601:1800]);
% Y(find(Y==1)) = 1;Y(find(Y==3)) = 2;Y(find(Y==5)) = 3;Y(find(Y==7)) = 4;Y(find(Y==9)) = 5;
% view = 3; features = [216,64,240];
% for i =1:view
%     if i==1
%         X1 = mapminmax(data{i}(:,[1:200 401:600 801:1000 1201:1400 1601:1800]),0,1);
%     elseif i==2
%         X1 = mapminmax(data{3}(:,[1:200 401:600 801:1000 1201:1400 1601:1800]),0,1);
%     else
%         X1 = mapminmax(data{5}(:,[1:200 401:600 801:1000 1201:1400 1601:1800]),0,1);
%     end
%     X{i} = X1';
% end

% load Mfeat.mat    % 3view 246810
% Y = truelabel{1};
% Y = Y([201:400 601:800 1001:1200 1401:1600 1801:2000]); 
% Y(find(Y==2)) = 1;Y(find(Y==4)) = 2;Y(find(Y==6)) = 3;Y(find(Y==8)) = 4;Y(find(Y==10)) = 5;
% view = 3; features = [216,64,240];
% for i =1:view
%     if i==1
%         X1 = mapminmax(data{i}(:,[201:400 601:800 1001:1200 1401:1600 1801:2000]),0,1);
%     elseif i==2
%         X1 = mapminmax(data{3}(:,[201:400 601:800 1001:1200 1401:1600 1801:2000]),0,1);
%     else
%         X1 = mapminmax(data{5}(:,[201:400 601:800 1001:1200 1401:1600 1801:2000]),0,1);
%     end
%     X{i} = X1';
% end
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

        tt = 1;
        n_impr = 0; 
        n_error = 0; 
        n_outlier = 0; 
        for i = 1:size(Y, 1)
            if result_label(i) ~= 1000
                if result_label(i) > K
                    label_single = find(F(result_label(i)-K,:)~=0);
                    label_single_x = label_single;        
                for t = 1:size(label_single,2)
                    label_single(t) = find(obj == label_single_x(t));
                end          
                    if ~isempty (find(label_single == real_label(i))) 
                        n_impr = n_impr+1;
                    else
                        n_error = n_error+1;
                        ERR(tt,1) = i;
                        tt = tt +1;
                    end
                elseif result_label(i) ~= real_label(i)
                    n_error = n_error+1;
                    ERR(tt,1) = i;
                    tt = tt +1;
                end
            else
                n_outlier = n_outlier+1;
            end
        end

        error01  = n_error/size(Y, 1);
        impr   = (n_impr)/size(Y, 1);
        outlier = n_outlier/size(Y, 1);

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
        fprintf("ACC:%f,IMP:%f,OUT:%f,ERR:%f,NMI:%f,P:%f,R:%f,F:%f,RI:%f,FM:%f,J:%f\n",result(1),impr,outlier,error01,nmi,P,R1,F1,RI,FM,J);
        metrics = [metrics;[alpha,beta,result(1),impr,outlier,error01,nmi,P,R1,F1,RI,FM,J]];
    end
end