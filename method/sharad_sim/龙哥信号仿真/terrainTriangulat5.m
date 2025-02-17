function [centroid, normal, points,triangles,area] = terrainTriangulat5(X_sufi, Y_sufi, Dem_sufii)
[X,Y] = meshgrid(X_sufi,Y_sufi);
points = [X(:), Y(:), Dem_sufii(:)];
% 获取 x 和 y 的长度
m = length(X_sufi);
kk = length(Y_sufi);
% 初始化三角形索引
len = 2*(m-1)*(kk-1);
triangles = zeros(len, 3);
% 创建三角形索引
j = (1:(kk-1)).';
for i = 1:(m-1)    
    index = ((i-1)*(kk-1)+j);
    p1 = (i-1)*kk + j;
    p2 = i*kk + j;
    p3 = (i-1)*kk + j + 1;
    p4 = i*kk + j + 1;
    triangles(index, :) = [p1, p2, p3];
    triangles(len/2+index, :) = [p3, p2, p4];    
end

% 计算三角形的中心点
centroid = (points(triangles(:,1), :) + points(triangles(:,2), :) + points(triangles(:,3), :)) / 3;

% 计算三角形的法线
normal = -cross(points(triangles(:,2), :) - points(triangles(:,1), :), points(triangles(:,3), :) - points(triangles(:,1), :));

% 计算三角形的面积
area = vecnorm(normal, 2, 2) / 2;

normal = normal./sqrt(normal(:,1).^2 + normal(:,2).^2 + normal(:,3).^2); % 归一化


end