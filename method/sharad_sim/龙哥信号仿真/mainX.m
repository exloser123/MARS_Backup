clear
%% 读取地图信息
% GrIMP
tif = 'D:\DYL\Icebridge\GrIMP_DEM\tile_1_5_30m_dem_v02.0.tif';
[Dem_suf,Y_suf,X_suf] = openGrIMP2dem(tif,0);
tif = 'D:\DYL\Icebridge\GrIMP_DEM\tile_1_4_30m_dem_v02.0.tif';
[Dem_suf1_4,Y_suf1_4,~] = openGrIMP2dem(tif,0);
Y_suf = [Y_suf Y_suf1_4];
Dem_suf = [Dem_suf; Dem_suf1_4];
tif = 'D:\DYL\Icebridge\GrIMP_DEM\tile_2_5_30m_dem_v02.0.tif';
[Dem_suf2_5,~,X_suf2_5] = openGrIMP2dem(tif,0);
tif = 'D:\DYL\Icebridge\GrIMP_DEM\tile_2_4_30m_dem_v02.0.tif';
[Dem_suf2_4,~,~] = openGrIMP2dem(tif,0);
Dem_suf2_5 = [Dem_suf2_5; Dem_suf2_4];
X_suf = [X_suf X_suf2_5];
Dem_suf = [Dem_suf Dem_suf2_5];
% figure;imagesc(Dem_suf)
clear tif Dem_suf1_4 Dem_suf2_4 Dem_suf2_5 X_suf2_5 Y_suf1_4
%% 发射信号
c = 2.99792458e+08; % 光速
fs = 1200e6; % 采样频率
Tpd = 100e-06; % 脉冲持续时间
f0 = 40e6; % 起始频率
f1 = 50e6; % 截止频率
Nt_raw = 40960; % 原始文件跨轨道采样数
hspac = 400e3; % 卫星高度
hmola = 3e3; % 地理高度上限(图像第一行高度)
% 信号生成
chirp_rate = (f1-f0) / Tpd; % 调频率
fc = (f1+f0)/2; % 中心频率
dt = 1/fs;
pad_length = floor(Tpd * fs);
Nt = Nt_raw + pad_length;
time = dt*(0:Nt-1).' + hspac*2/c - hmola*2/c;
itime = dt*(0:Nt-1).'-pad_length/2*dt;
distance = time*c/2;
%% 生成chrip信号
chrip = tukeyfun(itime/Tpd+0.5, 1).*exp(1j*pi*(2*fc*itime+chirp_rate*itime.^2)); % 加载频
chrip2=tukeyfun(itime/Tpd+0.5, 1) .*exp(1j*pi*(chirp_rate*itime.^2)); % 不加载频
%%
epsilon_r = 3.15; % 冰介电常数,EXCEL给出
lambda = c/fc; % 雷达波长
k = 2*pi*fc/c; % 波数
%% 
xtrack = 2400:10:4490;
ytrack = 13500;
xlen = 801;
ylen = 801;
summ = zeros(length(time),length(ytrack));
if isempty(gcp('nocreate'))
     parpool(32); % 启动并行池，MATLAB会根据您的硬件和许可证自动选择工作线程数
end
parfor i = 1:length(xtrack)
    xtracki = xtrack(i);
    Dem_sufi = Dem_suf((ytrack-(ylen-1)/2):(ytrack+(ylen-1)/2), (xtracki-(xlen-1)/2):(xtracki+(xlen-1)/2));  
    ll=2;
    newXline = xlen*ll+1;
    newYline = ylen*ll+1;
    Dem_sufii = inttt(Dem_sufi,newXline,newYline);
    X_sufi = ((-(xlen*ll)/2):((xlen*ll)/2))*30/ll;
    Y_sufi = ((-(ylen*ll)/2):((ylen*ll)/2))*30/ll;
    hcenter = [0,0,hspac];
    % figure;surf(X_sufi,Y_sufi,Dem_sufi);
    %normal的计算应该不对，你可能是后面求基尔霍夫近似掩盖掉了（有可能是对的）需要你验证一下老师说的角度逆顺方向确定值
    %上面这个是之前的判断，normal的计算大概率对，波束在非中心会偏嘛（1.飞机会倾斜，2.即使不倾斜，非正好x轴也会偏），那么就存在和法向量夹角大于90的情况
    [centroid, normal, points,triangles,R,ang,area] = find_thesepath5(hcenter,X_sufi,Y_sufi,Dem_sufii);
    pax = [0;1;0];
    resp = shortdipole(pax,ang);
    y_h = resp.H;y_v = resp.V;
    As = [y_h.'; y_v.'; zeros(1, size(y_h, 1))];
    y = sph_to_cartvec2(As, ang(1, :), ang(2, :));
    rot = [0, 0, 1; 0, 1, 0; -1, 0, 0]; % 这个是beam2body，正好和body2beam一样了
    Ei = rot*y;
    Es = scatteringAndTransmissionFields3(centroid, normal,area, hcenter, epsilon_r, Ei,k,R);
    Ei_e = reshape(Ei,[3,1,length(area)]);
    Es_e = reshape(Es.',[1,3,length(area)]);
    V = pagemtimes(Es_e,Ei_e);
    V = squeeze(V);
    E_sumhh = accumulateField3(centroid, hcenter, V, distance);
    % figure;plot(abs(E_sumhh))
    % E_sum = ifft(fft(E_sumhh).*fft(chrip));
    % E_sum_ji = E_sum.*exp(-1j*pi*(2*fc*itime));
    % E_summ=ifft(fft(E_sum_ji).*conj(fft(chrip2)));
    % figure;plot(abs(E_summ))
    summ(:,i) = E_sumhh;
    i
end
E_sum = ifft(fft(summ).*fft(chrip));
E_sum_ji = E_sum.*exp(-1j*pi*(2*fc*itime));
E_summ=ifft(fft(E_sum_ji).*conj(fft(chrip2)));