function Ar = sph_to_cartvec2(As, az, el)
% 输入：
% As - 3xN 的矩阵，每一列代表一个球坐标系中的向量 [az_hat; el_hat; r_hat]
% az - 1xN 的行向量，每个元素是一个方位角
% el - 1xN 的行向量，每个元素是一个仰角

% 将角度向量转换成列向量
az = az(:);
el = el(:);

% 计算各角度的余弦和正弦值
cosAz = cosd(az);
sinAz = sind(az);
cosEl = cosd(el);
sinEl = sind(el);

% 构造M矩阵的每个组件
r_vec = [cosEl.*cosAz, cosEl.*sinAz, sinEl];
el_vec = [-sinEl.*cosAz, -sinEl.*sinAz, cosEl];
az_vec = [-sinAz, cosAz, zeros(size(az))];

% 调整M矩阵以适应每个向量
M = zeros(3, 3, length(az));
M(:,1,:) = az_vec';
M(:,2,:) = el_vec';
M(:,3,:) = r_vec';

% 计算结果
Ar = pagemtimes(M, reshape(As, [3, 1, size(As, 2)]));
Ar = squeeze(Ar);

end
