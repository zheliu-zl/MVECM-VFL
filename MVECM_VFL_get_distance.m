function [Dis] = MVEC_VFL_get_distance(mode,varargin)
% Calculate the distance in 3 modes
%   mode = 1 -> Calculate the distance applied for update M or Jaccard
%   mode = 2 -> Calculate the distance applied for update R
%   mode = 3 -> Calculate the distance applied for update W
view = varargin{1};data = varargin{2}; nbFoc = varargin{3}; Aj = varargin{4};F_update = varargin{5};
alpha = varargin{6}; beta = varargin{7}; delta = varargin{8}; features = varargin{9}; 
Dis = cell(1,view);
if mode == 1
    R = varargin{10};
    W = varargin{11};
end
if mode == 2
    M = varargin{10};
    W = varargin{11};
end
if mode == 3
    R = varargin{10};
    M = varargin{11};
end

for i=1:view
    [Row,~] = size(data{i});
    [ROW,~] = size(F_update{i});
    dis_temp = zeros(Row,ROW);
    dis_temp1 = zeros(1,features(i));
    % process non-empty sets %
    [row,~] = size(Aj{i});
    for j=2:row
        temp = (data{i}-Aj{i}(j,:)).* (data{i}-Aj{i}(j,:));
        % calculate the cardinary of each focal set in Aj
        if sum(F_update{i}(j,:)) > 0
            card = sum(F_update{i}(j,:));
        else
            card = 0; % empty set and invalid class are set to 0
        end
        if mode == 1
            temp = sum(transpose(W{i}.*temp));
            temp = temp';
            temp(temp == 0) = 1e-10;
            dis_temp(:,j)=(temp .* R(i) .* card^alpha)';
        end
        
        if mode == 2
            temp = sum(transpose(W{i}.*temp)); 
            temp = temp';
            temp(temp == 0) = 1e-10;
            dis_temp(:,j)=(temp.* (M(:,j).^beta) * card^alpha)';
        end
        
        if mode == 3
            dis_temp1 = dis_temp1+sum(R(i)*temp.* (M(:,j).^beta) * card^alpha);
        end
    end
    % process empty sets %
    if mode == 1 
        dis_temp(:,1) = delta(i)^2 .* R(i);
        Dis{i}=dis_temp;
    end
    if mode == 2 
        dis_temp(:,1) = delta(i)^2 .* M(:,1).^beta;
        Dis{i}=dis_temp; 
    end
    if mode == 3 
        dis_temp1(dis_temp1 == 0) = 1e-10;
        Dis{i}=dis_temp1; 
    end
end
end

