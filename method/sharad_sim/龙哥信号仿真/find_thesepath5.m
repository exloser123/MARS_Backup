function [centroid, normal, points,triangles,R,ang,area] = find_thesepath5(hcenter,X_sufi,Y_sufi,Dem_sufii)
%normal的计算应该不对，你可能是后面求基尔霍夫近似掩盖掉了（有可能是对的）需要你验证一下老师说的角度逆顺方向确定值
%上面这个是之前的判断，normal的计算大概率对，波束在非中心会偏嘛（1.飞机会倾斜，2.即使不倾斜，非正好x轴也会偏），那么就存在和法向量夹角大于90的情况
[centroid, normal, points,triangles,area] = terrainTriangulat5(X_sufi, Y_sufi, Dem_sufii);
%% 获取其他参数
patchLos = centroid - hcenter;
Rot = [0, 0, -1; 0, 1, 0; 1, 0, 0];
patchLos = Rot *patchLos.'; % 矢量从场景转换到平台
% Get angle of departure and arrival
hypotxy = sqrt(abs(patchLos(1,:)).^2 + abs(patchLos(2,:)).^2);
R = sqrt(abs(patchLos(1,:)).^2 + abs(patchLos(2,:)).^2 + abs(patchLos(3,:)).^2);
elev = atan2(patchLos(3,:),hypotxy); % ang(2,:)
az = atan2(patchLos(2,:),patchLos(1,:)); % ang(1,:)
ang = [az;elev];
ang = ang*180/pi;

end