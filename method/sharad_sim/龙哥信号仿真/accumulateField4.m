function E_sum = accumulateField4(centroid, hcenter, V, distance)

% 计算天线到每个三角形中心的距离
dis = vecnorm(hcenter - centroid, 2, 2).';

% % 计算最大和最小距离
% max_distance = max(distances);
% min_distance = min(distances);
min_distance = min(dis);
max_distance = max(dis);
min_ind = find(distance < min_distance,1,'last');
max_ind = find(distance > max_distance,1,'first');

% 初始化 E_sum
E_sum = zeros(length(distance), 3);

% 遍历每一段
for i = min_ind:max_ind
    % 找到在这一段距离内的三角形中心
    mask = (dis >= distance(i)) & (dis < distance(i+1));
    
    % 将这一段内的 Er 相加
    E_sum(i, :) = sum(V(mask,:), 1);
end

end