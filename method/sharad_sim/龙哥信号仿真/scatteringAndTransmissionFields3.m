function Es = scatteringAndTransmissionFields3(centroid, normal,area, hcenter, epsilon_r, Ei,k,R)
% 这个是错的，看博士论文后面火星大尺度仿真部分的（第四章）公式重新搞（可以对照下之前的仿真程序，那个"有可能"是对的），
% 博士论文的公式有些也不对，需要你反复确认
% 初始化散射场和透射场矢量数组
% Es = zeros(size(centroid));
% Et = zeros(size(centroid));
k1 = k * sqrt(epsilon_r);
% 计算入射场的方向
ki = centroid - hcenter;
ki = ki ./ vecnorm(ki, 2, 2); % 归一化

% 检查法线与射线之间的夹角
% dot_product = -dot(ki, normal, 2);
% in = dot_product > 0;
% ki = ki(in,:);
% normal = normal(in,:);
% area = area(in,:);

theta = acos(-dot(ki, normal, 2)); % 计算角度theta

[Rh, Rv] = fresnelCoefficients(theta, k,k1,epsilon_r);

hi = cross(ki,normal,2);
hi = hi ./ vecnorm(hi, 2, 2); % 归一化
vi = cross(hi,ki,2);
E = Ei.';
F = -dot(E,hi,2).*dot(normal,ki,2).*(1-Rh).*hi +...
    dot(E,vi,2).*cross(normal,hi,2).*(1+Rv) +...
    dot(E,hi,2).*cross(-ki,cross(normal,hi,2),2).*(1+Rh) +...
    dot(E,vi,2).*dot(normal,ki,2).*cross(-ki,hi,2).*(1-Rv);
IA = exp(-2j*k*R.');
Es = (1j*k*F)./(4*pi*R.').^2 .* IA .*area;
end

function [Rh, Rv] = fresnelCoefficients(theta, k,k1,epsilon_r)

% 计算入射角的正弦和余弦
sin_theta_i = sin(theta);
cos_theta_i = cos(theta);



% 计算Fresnel反射系数
Rh = (k*cos_theta_i - sqrt(k1^2-k^2*sin_theta_i.^2))./ (k*cos_theta_i + sqrt(k1^2-k^2*sin_theta_i.^2));
Rv = (epsilon_r*k*cos_theta_i - sqrt(k1^2-k^2*sin_theta_i.^2))./ (epsilon_r*k*cos_theta_i + sqrt(k1^2-k^2*sin_theta_i.^2));



end
