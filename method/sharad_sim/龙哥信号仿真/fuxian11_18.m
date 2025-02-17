%% 建立表面
Dem = zeros(101,101); % (1001,1001)为中心，分辨率10m，用zero表示高度为0
X_sufi = linspace(-10000,10000,101);
Y_sufi = linspace(-10000,10000,101);
%%
c = 2.99792458e+08; % 光速
fc = 5e6; % 中心频率
hspac = 50e3; % 卫星高度
hcenter = [0,0,hspac]; % 卫星位置
L = 30;
I = 4;
%%
epsilon_r = 4; % 介电常数
k = 2*pi*fc/c; % 波数
distance = linspace(50e3,53e3,3001);
%%
[centroid, normal, points,triangles,R,ang,area] = find_thesepath5(hcenter,X_sufi,Y_sufi,Dem);
pax = [0;0;1];
resp = shortdipole(pax,ang);
y_h = resp.H;y_v = resp.V;
As = [y_h.'; y_v.'; zeros(1, size(y_h, 1))];
y = sph_to_cartvec2(As, ang(1, :), ang(2, :));
rot = [0, 0, 1; 0, 1, 0; -1, 0, 0]; % 这个是beam2body，正好和body2beam一样了
Ei = rot*y*L*I;
Es = scatteringAndTransmissionFields3(centroid, normal,area, hcenter, epsilon_r, Ei,k,R);
E_sumhh = accumulateField4(centroid, hcenter, Es, distance);
E=sqrt(abs(E_sumhh(:,1)).^2+abs(E_sumhh(:,2)).^2+abs(E_sumhh(:,3)).^2);
figure;plot(E)
 

